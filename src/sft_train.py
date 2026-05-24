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
from src.metric import verify
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

    # telemetry
    parser.add_argument("--train_telemetry", action="store_true")
    parser.add_argument("--eval_telemetry", action="store_true")
    
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
            )

        if return_outputs:
            return loss, outputs

        return loss

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

    def _write_batch_telemetry(self, outputs, labels, input_ids, source_row_index, split, metadata, output_dir):
        if output_dir is None:
            return

        if labels is None:
            return

        if input_ids is None:
            return

        if not hasattr(outputs, "logits"):
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
            logits = outputs.logits.detach().float()
            labels = labels.detach()
            input_ids = input_ids.detach()

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_input_ids = input_ids[:, 1:]

            token_nll = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
                ignore_index=-100,
            )

            token_log_probs = -token_nll
            valid_mask = shift_labels.ne(-100)

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

    
    val_dataset = prepare_dataset(args.val_path, eval=True)
    logger.info(f"val dataset final len: {len(val_dataset)}")
    eval_metadata = load_metadata(args.val_path)

    logger.info("Загрузка модели в bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        run_name=f"{args.wandb_run_basename}-SFT-{args.data}-ep{args.epochs}-lr{args.lr}"
    )

    train_telemetry_output_dir = os.path.join(args.output_dir, "train_telemetry")
    eval_telemetry_output_dir = os.path.join(args.output_dir, "eval_telemetry")

    trainer = TelemetrySFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        callbacks=[TimeLimitCallback(max_hours=args.max_hours)],
        train_telemetry_metadata=train_metadata,
        eval_telemetry_metadata=eval_metadata,
        train_telemetry_output_dir=train_telemetry_output_dir,
        eval_telemetry_output_dir=eval_telemetry_output_dir,
        telemetry_save_steps=args.eval_steps,
        train_telemetry_enabled=args.train_telemetry,
        eval_telemetry_enabled=args.eval_telemetry,
        telemetry_tokenizer=tokenizer,
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
