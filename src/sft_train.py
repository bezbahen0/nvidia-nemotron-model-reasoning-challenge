import os
import time
import argparse
import torch
import pandas as pd
import wandb

import json
import torch.nn.functional as F

from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint

from trl import SFTTrainer, SFTConfig
from src.metric import verify, extract_final_answer
from src.log import logger

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--wandb_run_basename", type=str)

    parser.add_argument("--model_id", type=str, default="nemotron-3-nano-30b-a3b-bf16")
    parser.add_argument("--data", type=str, required=True, help="dataset name for log W&B")
    parser.add_argument("--train_path", type=str, required=True, help="train.csv")
    parser.add_argument("--val_path", type=str, required=True, help="val.csv")
    parser.add_argument("--output_dir", type=str, default="./models/sft_baseline_v1")
    
    # train params
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_hours", type=float, required=True)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.00)

    # loss
    parser.add_argument(
        "--loss_impl",
        type=str,
        default="default",
        choices=["default", "cce"],
        help="default = regular model CE loss, cce = manual Cut Cross Entropy loss with telemetry support",
    )
    parser.add_argument(
        "--cce_impl",
        type=str,
        default="cce",
        choices=["cce", "torch_compile"],
        help="Cut Cross Entropy backend implementation",
    )
    parser.add_argument(
        "--cce_compare_default_loss_once",
        action="store_true",
        help="Compare CCE loss with default logits CE loss once on the first train batch. This materializes logits once.",
    )
    parser.add_argument(
        "--cce_loss_diff_warn_threshold",
        type=float,
        default=1e-2,
        help="Warn if default CE and CCE loss differ by more than this value in one-batch comparison",
    )

    # telemetry
    parser.add_argument("--train_telemetry", action="store_true")
    parser.add_argument("--eval_telemetry", action="store_true")

    # generation eval accuracy
    parser.add_argument(
        "--eval_generation_max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens for honest full-answer generation eval.",
    )
    parser.add_argument(
        "--eval_generation_batch_size",
        type=int,
        default=0,
        help="Generation eval batch size. 0 means use per_device_eval_batch_size.",
    )
    parser.add_argument(
        "--eval_accuracy_save_predictions",
        action="store_true",
        help="Save generated eval predictions as jsonl files for debugging.",
    )
    
    return parser.parse_args()

class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_hours=11.5):
        self.max_seconds = max_hours * 3600
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_seconds:
            logger.info(f"\n[ВНИМАНИЕ] Достигнут лимит времени ({elapsed_time/3600:.2f} часов). Мягкая остановка...")
            control.should_training_stop = True
            control.should_save = True


def load_metadata(data_path):
    df = pd.read_csv(data_path)

    metadata = {}

    for source_row_index, row in df.iterrows():
        metadata[int(source_row_index)] = {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "label": str(row["label"]),
        }

    return metadata


def prepare_dataset(csv_path, eval=False):
    df = pd.read_csv(csv_path)

    if eval:
        before_len = len(df)
        
        df = df[df.computed_answer.notna()]
        df = df[df.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)]


        logger.info(
            f"Eval dataset filtered by is_correct=True: {before_len} -> {len(df)}"
        )

    instruction_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

    formatted_data = []

    for source_row_index, row in df.iterrows():
        user_text = str(row["prompt"]) + instruction_suffix

        computed_answer = str(row["computed_answer"]).strip()

        assistant_text = (
            f"<think>\n{row['generated_cot']}\n</think>\n"
            f"Final Answer: \\boxed{{{computed_answer}}}"
        )

        formatted_data.append({
            "source_row_index": int(source_row_index),
            "prompt": [
                {"role": "user", "content": user_text}
            ],
            "completion": [
                {"role": "assistant", "content": assistant_text}
            ],
        })
    return Dataset.from_list(formatted_data)



def load_eval_generation_rows(data_path, prompt_dataset):
    """Build raw validation rows for generation accuracy eval.

    IMPORTANT: this intentionally reuses prompt_dataset items produced by prepare_dataset(..., eval=False),
    so eval generation uses the same conversational prompt object as train/eval SFT data.
    prepare_dataset itself is not modified.
    """
    df = pd.read_csv(data_path)
    rows = []

    for item in prompt_dataset:
        source_row_index = int(item["source_row_index"])
        row = df.loc[source_row_index]

        answer = str(row["answer"]).strip()
        if not answer or answer.lower() == "nan":
            continue

        rows.append({
            "source_row_index": source_row_index,
            "id": str(row["id"]),
            "source": str(row["source"]),
            "label": str(row["label"]),
            "answer": answer,
            "prompt": item["prompt"],
        })

    return rows


def sanitize_metric_name(value):
    value = str(value)
    for old, new in (
        ("/", "_"),
        (" ", "_"),
        (".", "_"),
        (":", "_"),
        ("\\", "_"),
    ):
        value = value.replace(old, new)
    return value


def import_linear_cross_entropy():
    try:
        from cut_cross_entropy import linear_cross_entropy
    except ImportError as exc:
        raise ImportError(
            "cut_cross_entropy is not installed. "
            "Install it before using --loss_impl cce. "
            "Example: pip install \"cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git\""
        ) from exc

    return linear_cross_entropy


class TelemetrySFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        train_telemetry_metadata=None,
        eval_telemetry_metadata=None,
        train_telemetry_output_dir=None,
        eval_telemetry_output_dir=None,
        telemetry_save_steps=50,
        train_telemetry_enabled=True,
        eval_telemetry_enabled=True,
        telemetry_tokenizer=None,
        loss_impl="default",
        cce_impl="cce",
        cce_compare_default_loss_once=False,
        cce_loss_diff_warn_threshold=1e-2,
        eval_generation_rows=None,
        eval_generation_max_new_tokens=2048,
        eval_generation_batch_size=0,
        eval_accuracy_save_predictions=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.train_telemetry_metadata = train_telemetry_metadata or {}
        self.eval_telemetry_metadata = eval_telemetry_metadata or {}
        
        self.train_telemetry_output_dir = train_telemetry_output_dir
        self.eval_telemetry_output_dir = eval_telemetry_output_dir
        
        self.telemetry_save_steps = int(telemetry_save_steps)
        self.train_telemetry_enabled = bool(train_telemetry_enabled)
        self.eval_telemetry_enabled = bool(eval_telemetry_enabled)
        
        self.telemetry_tokenizer = telemetry_tokenizer
        self.telemetry_top_bad_tokens_count = 5
        self.telemetry_context_window_tokens = 8

        self.loss_impl = str(loss_impl)
        self.cce_impl = str(cce_impl)
        self.linear_cross_entropy = None
        self.cce_compare_default_loss_once = bool(cce_compare_default_loss_once)
        self.cce_loss_diff_warn_threshold = float(cce_loss_diff_warn_threshold)
        self.cce_default_loss_compared = False
        self.cce_base_model = None

        self.eval_generation_rows = list(eval_generation_rows or [])
        self.eval_generation_max_new_tokens = int(eval_generation_max_new_tokens)
        self.eval_generation_batch_size = int(eval_generation_batch_size)
        self.eval_accuracy_save_predictions = bool(eval_accuracy_save_predictions)

        if self.loss_impl == "cce":
            self.linear_cross_entropy = import_linear_cross_entropy()
            logger.info(f"[CCE] Cut Cross Entropy enabled, impl={self.cce_impl}")
        
        self.telemetry_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        
        if self.train_telemetry_output_dir is not None:
            os.makedirs(self.train_telemetry_output_dir, exist_ok=True)
        
        if self.eval_telemetry_output_dir is not None:
            os.makedirs(self.eval_telemetry_output_dir, exist_ok=True)
                
        base_data_collator = self.data_collator

        def telemetry_data_collator(features):
            source_row_indexes = []
            cleaned_features = []

            for feature in features:
                feature = dict(feature)

                source_row_indexes.append(int(feature.pop("source_row_index", -1)))

                feature.pop("id", None)
                feature.pop("source", None)
                feature.pop("label", None)

                cleaned_features.append(feature)

            batch = base_data_collator(cleaned_features)
            batch["source_row_index"] = torch.tensor(source_row_indexes, dtype=torch.long)

            return batch

        self.data_collator = telemetry_data_collator

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        source_row_index = inputs.pop("source_row_index", None)

        token_nll = None

        if self.loss_impl == "cce":
            loss, outputs, token_nll = self._compute_cce_loss(model, inputs)
        else:
            try:
                loss, outputs = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                loss, outputs = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                )

        is_train = bool(model.training)

        if is_train:
            telemetry_enabled = self.train_telemetry_enabled
            telemetry_split = "train"
            telemetry_metadata = self.train_telemetry_metadata
            telemetry_output_dir = self.train_telemetry_output_dir
        else:
            telemetry_enabled = self.eval_telemetry_enabled
            telemetry_split = "eval"
            telemetry_metadata = self.eval_telemetry_metadata
            telemetry_output_dir = self.eval_telemetry_output_dir

        if telemetry_enabled and source_row_index is not None:
            self._write_batch_telemetry(
                outputs=outputs,
                labels=inputs.get("labels"),
                input_ids=inputs.get("input_ids"),
                source_row_index=source_row_index,
                split=telemetry_split,
                metadata=telemetry_metadata,
                output_dir=telemetry_output_dir,
                token_nll=token_nll,
            )

        if return_outputs:
            return loss, outputs

        return loss

    def _compute_cce_loss(self, model, inputs):
        if self.linear_cross_entropy is None:
            raise RuntimeError("linear_cross_entropy is not initialized")

        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if labels is None:
            raise RuntimeError("CCE loss requires labels")

        if input_ids is None:
            raise RuntimeError("CCE loss requires input_ids")

        base_model = self._get_cce_base_model(model)

        backbone_kwargs = {}

        for key in (
            "position_ids",
            "past_key_values",
            "use_cache",
            "cache_position",
        ):
            if key in inputs:
                backbone_kwargs[key] = inputs[key]

        backbone_out = base_model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **backbone_kwargs,
        )

        hidden_states = backbone_out[0]
        lm_weight, lm_bias = self._get_lm_weight_and_bias(base_model.lm_head, hidden_states)

        if self.cce_compare_default_loss_once and not self.cce_default_loss_compared:
            self._compare_default_loss_once(
                model=model,
                inputs=inputs,
                cce_hidden_states=hidden_states,
                cce_lm_weight=lm_weight,
                cce_lm_bias=lm_bias,
                labels=labels,
            )
            self.cce_default_loss_compared = True

        per_token_ce = self._linear_cross_entropy(
            hidden_states=hidden_states,
            lm_weight=lm_weight,
            labels=labels,
            lm_bias=lm_bias,
        )

        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)

        if per_token_ce.shape == labels.shape:
            per_token_ce = per_token_ce[:, 1:]

        if per_token_ce.shape != shift_labels.shape:
            raise RuntimeError(
                f"Unexpected CCE per_token_ce shape: {tuple(per_token_ce.shape)}, "
                f"expected {tuple(shift_labels.shape)}"
            )

        loss = per_token_ce[valid_mask].mean()

        outputs = {
            "loss": loss,
        }

        return loss, outputs, per_token_ce

    def _linear_cross_entropy(self, hidden_states, lm_weight, labels, lm_bias=None):
        kwargs = {
            "shift": 1,
            "reduction": "none",
            "ignore_index": -100,
        }

        if self.cce_impl is not None:
            kwargs["impl"] = self.cce_impl

        if lm_bias is not None:
            kwargs["bias"] = lm_bias

        try:
            return self.linear_cross_entropy(
                hidden_states,
                lm_weight,
                labels,
                **kwargs,
            )
        except TypeError:
            kwargs.pop("impl", None)

            try:
                return self.linear_cross_entropy(
                    hidden_states,
                    lm_weight,
                    labels,
                    **kwargs,
                )
            except TypeError:
                kwargs.pop("bias", None)

                try:
                    return self.linear_cross_entropy(
                        hidden_states,
                        lm_weight,
                        labels,
                        **kwargs,
                    )
                except TypeError:
                    shifted_hidden_states = hidden_states[:, :-1, :]
                    shifted_labels = labels[:, 1:]

                    return self.linear_cross_entropy(
                        shifted_hidden_states,
                        lm_weight,
                        shifted_labels,
                        reduction="none",
                        ignore_index=-100,
                    )

    def _compare_default_loss_once(
        self,
        model,
        inputs,
        cce_hidden_states,
        cce_lm_weight,
        cce_lm_bias,
        labels,
    ):
        if not bool(model.training):
            return

        was_training = model.training

        with torch.no_grad():
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels.ne(-100)

            per_token_ce = self._linear_cross_entropy(
                hidden_states=cce_hidden_states.detach(),
                lm_weight=cce_lm_weight.detach(),
                labels=labels,
                lm_bias=None if cce_lm_bias is None else cce_lm_bias.detach(),
            )

            if per_token_ce.shape == labels.shape:
                per_token_ce = per_token_ce[:, 1:]

            cce_loss = per_token_ce[valid_mask].mean()

            default_outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=labels,
            )

            default_loss = default_outputs.loss.detach()
            diff = float(torch.abs(default_loss.float() - cce_loss.float()).item())

            logger.info(
                f"[CCE] default_loss={float(default_loss.float().item()):.8f}, "
                f"cce_loss={float(cce_loss.float().item()):.8f}, "
                f"abs_diff={diff:.8f}"
            )

            if diff > self.cce_loss_diff_warn_threshold:
                logger.warning(
                    f"[CCE] default loss and CCE loss differ by {diff:.8f}, "
                    f"threshold={self.cce_loss_diff_warn_threshold:.8f}. "
                    "Check shift/masking/lm_head LoRA handling."
                )

        if was_training:
            model.train()

    def _get_cce_base_model(self, model):
        if self.cce_base_model is not None:
            return self.cce_base_model

        current = model
        visited = set()

        while current is not None and id(current) not in visited:
            visited.add(id(current))

            if hasattr(current, "backbone") and hasattr(current, "lm_head"):
                self.cce_base_model = current
                logger.info(f"[CCE] Found base model for CCE: {type(current)}")
                self._validate_lm_head_for_cce(current.lm_head)
                return current

            candidates = []

            for attr_name in (
                "base_model",
                "model",
                "module",
            ):
                if hasattr(current, attr_name):
                    next_current = getattr(current, attr_name)

                    if next_current is not current and next_current is not None:
                        candidates.append(next_current)

            found = None

            for candidate in candidates:
                if hasattr(candidate, "backbone") and hasattr(candidate, "lm_head"):
                    found = candidate
                    break

            if found is not None:
                current = found
            elif len(candidates) > 0:
                current = candidates[0]
            else:
                break

        raise RuntimeError(
            "Could not find base model with .backbone and .lm_head for CCE. "
            "Nemotron remote-code structure may have changed."
        )

    def _validate_lm_head_for_cce(self, lm_head):
        if not hasattr(lm_head, "lora_dropout"):
            return

        lora_dropout = getattr(lm_head, "lora_dropout")

        if not isinstance(lora_dropout, dict):
            return

        for adapter_name, dropout_module in lora_dropout.items():
            p = getattr(dropout_module, "p", 0.0)

            if float(p) != 0.0:
                raise RuntimeError(
                    "CCE manual lm_head weight merge is only exact when lm_head LoRA dropout is 0. "
                    f"Found lm_head lora_dropout[{adapter_name}]={p}. "
                    "Set --lora_dropout 0.0 or remove lm_head from LoRA target modules."
                )

    def _get_active_lora_adapter_name(self, lm_head):
        for attr_name in ("active_adapter", "active_adapters"):
            if hasattr(lm_head, attr_name):
                value = getattr(lm_head, attr_name)

                if isinstance(value, str):
                    return value

                if isinstance(value, (list, tuple)) and len(value) > 0:
                    return str(value[0])

        if hasattr(lm_head, "lora_A") and isinstance(lm_head.lora_A, dict):
            if "default" in lm_head.lora_A:
                return "default"

            keys = list(lm_head.lora_A.keys())

            if len(keys) > 0:
                return str(keys[0])

        return "default"

    def _get_lm_weight_and_bias(self, lm_head, hidden_states):
        if (
            hasattr(lm_head, "base_layer")
            and hasattr(lm_head, "lora_A")
            and hasattr(lm_head, "lora_B")
        ):
            adapter_name = self._get_active_lora_adapter_name(lm_head)

            if adapter_name in lm_head.lora_A and adapter_name in lm_head.lora_B:
                base_layer = lm_head.base_layer
                base_w = base_layer.weight

                lora_A = lm_head.lora_A[adapter_name].weight
                lora_B = lm_head.lora_B[adapter_name].weight
                scaling = lm_head.scaling[adapter_name]

                lora_delta = lora_B.to(base_w.dtype) @ lora_A.to(base_w.dtype)
                lm_weight = base_w + scaling * lora_delta

                lm_bias = getattr(base_layer, "bias", None)

                if lm_bias is not None and lm_bias.dtype != hidden_states.dtype:
                    lm_bias = lm_bias.to(hidden_states.dtype)

                if lm_weight.dtype != hidden_states.dtype:
                    lm_weight = lm_weight.to(hidden_states.dtype)

                return lm_weight, lm_bias

        lm_weight = lm_head.weight
        lm_bias = getattr(lm_head, "bias", None)

        if lm_weight.dtype != hidden_states.dtype:
            lm_weight = lm_weight.to(hidden_states.dtype)

        if lm_bias is not None and lm_bias.dtype != hidden_states.dtype:
            lm_bias = lm_bias.to(hidden_states.dtype)

        return lm_weight, lm_bias

    def _get_token_texts(self, token_ids):
        if self.telemetry_tokenizer is None:
            return [str(int(token_id)) for token_id in token_ids]

        return [
            self.telemetry_tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        ]

    def _get_token_sections(self, token_texts):
        full_text = "".join(token_texts)

        think_start = full_text.find("<think>")
        think_content_start = full_text.find("\n", think_start)
        think_end = full_text.find("</think>")
        final_start = full_text.find("Final Answer:")
        boxed_start = full_text.find("\\boxed{")

        boxed_content_start = -1
        boxed_content_end = -1

        if boxed_start != -1:
            boxed_content_start = boxed_start + len("\\boxed{")
            boxed_content_end = full_text.find("}", boxed_content_start)

        sections = []

        char_pos = 0

        for token_text in token_texts:
            token_start = char_pos
            token_end = char_pos + len(token_text)
            token_mid = token_start if token_end == token_start else token_start + (token_end - token_start) // 2

            section = "format"

            if boxed_content_start != -1 and token_mid >= boxed_content_start:
                if boxed_content_end == -1 or token_mid < boxed_content_end:
                    section = "boxed_answer"
                elif final_start != -1 and token_mid >= final_start:
                    section = "final_answer"
            elif final_start != -1 and token_mid >= final_start:
                section = "final_answer"
            elif think_end != -1:
                if think_content_start != -1 and token_mid > think_content_start and token_mid < think_end:
                    section = "cot"
            elif think_start != -1:
                if think_content_start != -1 and token_mid > think_content_start:
                    section = "cot"

            sections.append(section)
            char_pos = token_end

        return sections

    def _build_section_metrics(self, values, sections):
        result = {}

        section_names = [
            "cot",
            "final_answer",
            "boxed_answer",
            "format",
        ]

        for section_name in section_names:
            section_indexes = [
                index
                for index, section in enumerate(sections)
                if section == section_name
            ]

            if len(section_indexes) == 0:
                result[f"{section_name}_num_tokens"] = 0
                result[f"{section_name}_mean_nll"] = None
                result[f"{section_name}_min_logprob"] = None
                result[f"{section_name}_p05_logprob"] = None
                result[f"{section_name}_hard_token_ratio"] = None
                result[f"{section_name}_near_zero_token_ratio"] = None
                continue

            section_values = values[section_indexes]

            result[f"{section_name}_num_tokens"] = int(section_values.numel())
            result[f"{section_name}_mean_nll"] = float((-section_values).mean().item())
            result[f"{section_name}_min_logprob"] = float(section_values.min().item())
            result[f"{section_name}_p05_logprob"] = float(torch.quantile(section_values, 0.05).item())
            result[f"{section_name}_hard_token_ratio"] = float((section_values < -1.0).float().mean().item())
            result[f"{section_name}_near_zero_token_ratio"] = float((section_values > -0.05).float().mean().item())

        return result

    def _empty_section_metrics(self):
        result = {}

        section_names = [
            "cot",
            "final_answer",
            "boxed_answer",
            "format",
        ]

        for section_name in section_names:
            result[f"{section_name}_num_tokens"] = 0
            result[f"{section_name}_mean_nll"] = None
            result[f"{section_name}_min_logprob"] = None
            result[f"{section_name}_p05_logprob"] = None
            result[f"{section_name}_hard_token_ratio"] = None
            result[f"{section_name}_near_zero_token_ratio"] = None

        return result

    def _build_top_bad_tokens(self, values, token_ids, token_texts, sections):
        top_bad_tokens = []

        if values.numel() == 0:
            return top_bad_tokens

        top_count = min(self.telemetry_top_bad_tokens_count, int(values.numel()))

        _, top_indexes = torch.topk(
            -values,
            k=top_count,
            largest=True,
        )

        top_indexes = top_indexes.detach().cpu().tolist()

        for token_index in top_indexes:
            context_start = max(0, int(token_index) - self.telemetry_context_window_tokens)
            context_end = min(len(token_texts), int(token_index) + self.telemetry_context_window_tokens + 1)

            context = "".join(token_texts[context_start:context_end])

            top_bad_tokens.append({
                "completion_token_pos": int(token_index),
                "token_id": int(token_ids[token_index]),
                "token_text": token_texts[token_index],
                "logprob": float(values[token_index].item()),
                "nll": float((-values[token_index]).item()),
                "section": sections[token_index],
                "context": context,
            })

        return top_bad_tokens

    def _write_batch_telemetry(
        self,
        outputs,
        labels,
        input_ids,
        source_row_index,
        split,
        metadata,
        output_dir,
        token_nll=None,
    ):
        if output_dir is None:
            return

        if labels is None:
            return

        if input_ids is None:
            return

        global_step = int(self.state.global_step)

        if split == "train":
            if self.telemetry_save_steps <= 0:
                checkpoint_bucket = global_step
            else:
                checkpoint_bucket = ((global_step // self.telemetry_save_steps) + 1) * self.telemetry_save_steps
        else:
            checkpoint_bucket = global_step

        if split == "train":
            file_name = f"train_until_checkpoint-{checkpoint_bucket}_rank-{self.telemetry_rank}.jsonl"
        else:
            file_name = f"eval_at_checkpoint-{checkpoint_bucket}_rank-{self.telemetry_rank}.jsonl"

        path = os.path.join(output_dir, file_name)

        with torch.no_grad():
            labels = labels.detach()
            input_ids = input_ids.detach()

            shift_labels = labels[:, 1:]
            shift_input_ids = input_ids[:, 1:]
            valid_mask = shift_labels.ne(-100)

            if token_nll is None:
                logits = None

                if isinstance(outputs, dict):
                    logits = outputs.get("logits")

                if logits is None and hasattr(outputs, "logits"):
                    logits = outputs.logits

                if logits is None:
                    return

                logits = logits.detach().float()
                shift_logits = logits[:, :-1, :]

                token_nll = F.cross_entropy(
                    shift_logits.transpose(1, 2),
                    shift_labels,
                    reduction="none",
                    ignore_index=-100,
                )
            else:
                token_nll = token_nll.detach()

                if token_nll.shape == labels.shape:
                    token_nll = token_nll[:, 1:]

                if token_nll.shape != shift_labels.shape:
                    raise RuntimeError(
                        f"Unexpected token_nll shape: {tuple(token_nll.shape)}, "
                        f"expected {tuple(shift_labels.shape)}"
                    )

            token_log_probs = -token_nll

            if isinstance(source_row_index, torch.Tensor):
                source_row_indexes = source_row_index.detach().cpu().tolist()
            else:
                source_row_indexes = list(source_row_index)

            records = []

            batch_size = shift_labels.shape[0]

            for batch_index in range(batch_size):
                current_valid_mask = valid_mask[batch_index]
                valid_values = token_log_probs[batch_index][current_valid_mask]
                valid_token_ids = shift_input_ids[batch_index][current_valid_mask].detach().cpu().tolist()

                row_index = int(source_row_indexes[batch_index])
                metadata_item = metadata.get(
                    row_index,
                    {
                        "id": str(row_index),
                        "source": "unknown",
                        "label": "unknown",
                    },
                )

                if valid_values.numel() == 0:
                    record = {
                        "step": int(self.state.global_step),
                        "epoch": None if self.state.epoch is None else float(self.state.epoch),
                        "checkpoint_bucket": int(checkpoint_bucket),
                        "split": split,
                        "source_row_index": row_index,
                        "id": metadata_item["id"],
                        "source": metadata_item["source"],
                        "label": metadata_item["label"],
                        "num_loss_tokens": 0,
                        "total_loss": None,
                        "mean_nll": None,
                        "min_logprob": None,
                        "p05_logprob": None,
                        "hard_token_ratio": None,
                        "near_zero_token_ratio": None,
                        **self._empty_section_metrics(),
                        "top_bad_tokens": [],
                    }
                else:
                    values = valid_values.detach().cpu()
                    token_texts = self._get_token_texts(valid_token_ids)
                    sections = self._get_token_sections(token_texts)
                    section_metrics = self._build_section_metrics(values, sections)
                    top_bad_tokens = self._build_top_bad_tokens(
                        values=values,
                        token_ids=valid_token_ids,
                        token_texts=token_texts,
                        sections=sections,
                    )

                    num_loss_tokens = int(values.numel())
                    total_loss = float((-values).sum().item())
                    mean_nll = float((-values).mean().item())
                    min_logprob = float(values.min().item())
                    p05_logprob = float(torch.quantile(values, 0.05).item())
                    hard_token_ratio = float((values < -1.0).float().mean().item())
                    near_zero_token_ratio = float((values > -0.05).float().mean().item())

                    record = {
                        "step": int(self.state.global_step),
                        "epoch": None if self.state.epoch is None else float(self.state.epoch),
                        "checkpoint_bucket": int(checkpoint_bucket),
                        "split": split,
                        "source_row_index": row_index,
                        "id": metadata_item["id"],
                        "source": metadata_item["source"],
                        "label": metadata_item["label"],
                        "num_loss_tokens": num_loss_tokens,
                        "total_loss": total_loss,
                        "mean_nll": mean_nll,
                        "min_logprob": min_logprob,
                        "p05_logprob": p05_logprob,
                        "hard_token_ratio": hard_token_ratio,
                        "near_zero_token_ratio": near_zero_token_ratio,
                        **section_metrics,
                        "top_bad_tokens": top_bad_tokens,
                    }

                records.append(record)

        with open(path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = self._compute_generation_eval_accuracy(metric_key_prefix=metric_key_prefix)
        self.log(metrics)

        if self.is_world_process_zero():
            logger.info(
                "[generation_eval] "
                + json.dumps(metrics, ensure_ascii=False, sort_keys=True)
            )

        return metrics

    def _build_generation_prompt_text(self, row):
        if self.telemetry_tokenizer is None:
            raise RuntimeError("Generation eval requires telemetry_tokenizer")

        tokenizer = self.telemetry_tokenizer
        prompt_messages = row["prompt"]

        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "Tokenizer has no apply_chat_template. The training data is conversational, "
                "so generation eval must use the tokenizer chat template instead of a custom fallback."
            )

        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _compute_generation_eval_accuracy(self, metric_key_prefix="eval"):
        start_time = time.time()

        rows = self.eval_generation_rows
        total_rows = len(rows)

        if total_rows == 0:
            metrics = {
                f"{metric_key_prefix}_accuracy": 0.0,
                f"{metric_key_prefix}_correct": 0,
                f"{metric_key_prefix}_total": 0,
                f"{metric_key_prefix}_not_found": 0,
                f"{metric_key_prefix}_runtime": 0.0,
                f"{metric_key_prefix}_samples_per_second": 0.0,
            }
            return metrics

        world_size = int(getattr(self.args, "world_size", 1))
        process_index = int(getattr(self.args, "process_index", 0))

        local_rows = [
            row
            for index, row in enumerate(rows)
            if index % max(world_size, 1) == process_index
        ]

        label_names = sorted({str(row.get("label", "unknown")) for row in rows})
        label_to_index = {label: index for index, label in enumerate(label_names)}

        local_correct = 0
        local_total = 0
        local_not_found = 0
        local_label_correct = [0 for _ in label_names]
        local_label_total = [0 for _ in label_names]

        predictions_file = None
        if self.eval_accuracy_save_predictions:
            predictions_dir = os.path.join(self.args.output_dir, "eval_accuracy_predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            predictions_file = os.path.join(
                predictions_dir,
                f"step-{int(self.state.global_step)}_rank-{process_index}.jsonl",
            )

        model = self.model
        tokenizer = self.telemetry_tokenizer
        was_training = bool(model.training)
        old_padding_side = getattr(tokenizer, "padding_side", "right")
        old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)

        eval_batch_size = self.eval_generation_batch_size
        if eval_batch_size <= 0:
            eval_batch_size = int(self.args.per_device_eval_batch_size)
        eval_batch_size = max(1, int(eval_batch_size))

        model.eval()
        tokenizer.padding_side = "left"

        if old_use_cache is not None:
            model.config.use_cache = True

        records_to_write = []

        try:
            with torch.no_grad():
                for batch_start in range(0, len(local_rows), eval_batch_size):
                    batch_rows = local_rows[batch_start: batch_start + eval_batch_size]
                    prompt_texts = [self._build_generation_prompt_text(row) for row in batch_rows]

                    batch = tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.args.max_length,
                    )
                    batch = batch.to(self.args.device)

                    generated = model.generate(
                        **batch,
                        max_new_tokens=self.eval_generation_max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                    prompt_token_length = int(batch["input_ids"].shape[1])
                    generated_tails = generated[:, prompt_token_length:]

                    for row, tail_ids in zip(batch_rows, generated_tails):
                        generated_text = tokenizer.decode(
                            tail_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        predicted_answer = extract_final_answer(generated_text)
                        gold_answer = str(row["answer"]).strip()
                        is_correct = bool(verify(gold_answer, predicted_answer))

                        label = str(row.get("label", "unknown"))
                        label_index = label_to_index[label]

                        local_total += 1
                        local_correct += int(is_correct)
                        local_not_found += int(predicted_answer == "NOT_FOUND")
                        local_label_total[label_index] += 1
                        local_label_correct[label_index] += int(is_correct)

                        if predictions_file is not None:
                            records_to_write.append({
                                "step": int(self.state.global_step),
                                "source_row_index": int(row["source_row_index"]),
                                "id": str(row["id"]),
                                "source": str(row["source"]),
                                "label": label,
                                "gold_answer": gold_answer,
                                "predicted_answer": predicted_answer,
                                "is_correct": is_correct,
                                "generated_text": generated_text,
                            })
        finally:
            if old_use_cache is not None:
                model.config.use_cache = old_use_cache
            tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

        if predictions_file is not None and len(records_to_write) > 0:
            with open(predictions_file, "a", encoding="utf-8") as file:
                for record in records_to_write:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

        counts = torch.tensor(
            [local_correct, local_total, local_not_found],
            dtype=torch.long,
            device=self.args.device,
        )
        label_correct_tensor = torch.tensor(
            local_label_correct,
            dtype=torch.long,
            device=self.args.device,
        )
        label_total_tensor = torch.tensor(
            local_label_total,
            dtype=torch.long,
            device=self.args.device,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(label_correct_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(label_total_tensor, op=torch.distributed.ReduceOp.SUM)

        correct = int(counts[0].item())
        total = int(counts[1].item())
        not_found = int(counts[2].item())
        runtime = float(time.time() - start_time)

        metrics = {
            f"{metric_key_prefix}_accuracy": correct / max(total, 1),
            f"{metric_key_prefix}_correct": correct,
            f"{metric_key_prefix}_total": total,
            f"{metric_key_prefix}_not_found": not_found,
            f"{metric_key_prefix}_runtime": runtime,
            f"{metric_key_prefix}_samples_per_second": total / max(runtime, 1e-12),
        }

        label_correct_values = label_correct_tensor.detach().cpu().tolist()
        label_total_values = label_total_tensor.detach().cpu().tolist()

        for label, label_correct, label_total in zip(
            label_names,
            label_correct_values,
            label_total_values,
        ):
            safe_label = sanitize_metric_name(label)
            metrics[f"{metric_key_prefix}_label_accuracy/{safe_label}"] = (
                int(label_correct) / max(int(label_total), 1)
            )
            metrics[f"{metric_key_prefix}_label_correct/{safe_label}"] = int(label_correct)
            metrics[f"{metric_key_prefix}_label_total/{safe_label}"] = int(label_total)

        return metrics


class MemoryStatsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if not torch.cuda.is_available():
            return

        device_index = torch.cuda.current_device()

        logs["gpu/max_memory_allocated_gb"] = round(
            torch.cuda.max_memory_allocated(device_index) / 1024**3,
            4,
        )
        logs["gpu/max_memory_reserved_gb"] = round(
            torch.cuda.max_memory_reserved(device_index) / 1024**3,
            4,
        )

def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    os.makedirs(args.output_dir, exist_ok=True)
    run_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
    
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"
        logger.info(f"Возобновление графика W&B (Run ID: {run_id})")
    else:
        run_id = wandb.util.generate_id()
        with open(run_id_file, "w") as f:
            f.write(run_id)
        os.environ["WANDB_RUN_ID"] = run_id
        logger.info(f"Новый запуск W&B (Run ID: {run_id})")

    logger.info(f"Загрузка токенизатора для {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(args.train_path)
    logger.info(f"train dataset final len: {len(train_dataset)}")
    train_metadata = load_metadata(args.train_path)

    
    val_dataset = prepare_dataset(args.val_path, eval=False)
    logger.info(f"val dataset final len without correctness filtering: {len(val_dataset)}")
    eval_metadata = load_metadata(args.val_path)
    eval_generation_rows = load_eval_generation_rows(args.val_path, val_dataset)
    logger.info(f"generation eval rows final len: {len(eval_generation_rows)}")

    logger.info("Загрузка модели в bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False
    )

    logger.info(
        f"Training loss implementation: loss_impl={args.loss_impl}, cce_impl={args.cce_impl}"
    )


    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|gate_up_proj|down_proj|in_proj|out_proj|embed_tokens|lm_head)$", 
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=10,
        
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2, 
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=args.max_seq_len,
        completion_only_loss=True, 
        assistant_only_loss=False,

        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataset_num_proc=8,
        remove_unused_columns=False,

        report_to="wandb",
        run_name=f"{args.wandb_run_basename}-SFT-{args.data}-ep{args.epochs}-lr{args.lr}-{args.loss_impl}"
    )

    train_telemetry_output_dir = os.path.join(args.output_dir, "train_telemetry")
    eval_telemetry_output_dir = os.path.join(args.output_dir, "eval_telemetry")

    trainer = TelemetrySFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        callbacks=[TimeLimitCallback(max_hours=args.max_hours), MemoryStatsCallback()],
        train_telemetry_metadata=train_metadata,
        eval_telemetry_metadata=eval_metadata,
        train_telemetry_output_dir=train_telemetry_output_dir,
        eval_telemetry_output_dir=eval_telemetry_output_dir,
        telemetry_save_steps=args.eval_steps,
        train_telemetry_enabled=args.train_telemetry,
        eval_telemetry_enabled=args.eval_telemetry,
        telemetry_tokenizer=tokenizer,
        loss_impl=args.loss_impl,
        cce_impl=args.cce_impl,
        cce_compare_default_loss_once=args.cce_compare_default_loss_once,
        cce_loss_diff_warn_threshold=args.cce_loss_diff_warn_threshold,
        eval_generation_rows=eval_generation_rows,
        eval_generation_max_new_tokens=args.eval_generation_max_new_tokens,
        eval_generation_batch_size=args.eval_generation_batch_size,
        eval_accuracy_save_predictions=args.eval_accuracy_save_predictions,
    )

    last_checkpoint = None
    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        logger.info(f"Возобновление обучения с чекпоинта: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Запуск обучения с нуля.")
        trainer.train()

    logger.info(f"Save adapters to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
