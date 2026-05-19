import argparse
import os
import time
import types

import pandas as pd
import torch
import wandb
from datasets import Dataset

from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from unsloth import FastLanguageModel

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
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention backend for Unsloth/Transformers model loading.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_hours", type=float, required=True)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Keep 0.0 when CCE is enabled because the CCE patch fuses lm_head LoRA weights.",
    )

    # Cut Cross Entropy
    parser.add_argument(
        "--use_cce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Cut Cross Entropy for the LM head loss. Use --no-use_cce to disable.",
    )
    parser.add_argument(
        "--cce_impl",
        type=str,
        default="cce",
        choices=["cce", "cce_kahan", "cce_kahan_full_c", "cce_exact", "torch_compile"],
        help="Implementation passed to cut_cross_entropy.linear_cross_entropy.",
    )

    return parser.parse_args()


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_hours=11.5):
        self.max_seconds = max_hours * 3600
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_seconds:
            logger.info(
                f"\n[ВНИМАНИЕ] Достигнут лимит времени ({elapsed_time / 3600:.2f} часов). "
                f"Мягкая остановка..."
            )
            control.should_training_stop = True
            control.should_save = True


def prepare_dataset(csv_path, eval=False):
    df = pd.read_csv(csv_path)

    instruction_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

    formatted_data = []
    for _, row in df.iterrows():
        user_text = str(row["prompt"]) + instruction_suffix

        # Use computed answer, reduce noise, we compare results on previous stage
        computed_answer = str(row["computed_answer"]).strip()

        assistant_text = (
            f"<think>\n{row['generated_cot']}\n</think>\n"
            f"Final Answer: \\boxed{{{computed_answer}}}"
        )

        formatted_data.append(
            {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
        )

    return Dataset.from_list(formatted_data)


def _resolve_causal_lm_module(model):
    """
    Unwrap PEFT/Unsloth wrappers until the actual Nemotron causal LM module is found.
    For Nemotron-H this module is expected to expose `.backbone` and `.lm_head`.
    """
    current = model
    visited = set()

    for _ in range(30):
        if id(current) in visited:
            break
        visited.add(id(current))

        if hasattr(current, "backbone") and hasattr(current, "lm_head"):
            return current

        next_module = None
        if hasattr(current, "get_base_model"):
            candidate = current.get_base_model()
            if candidate is not current:
                next_module = candidate
        if next_module is None and hasattr(current, "base_model") and getattr(current, "base_model") is not current:
            next_module = getattr(current, "base_model")
        if next_module is None and hasattr(current, "model") and getattr(current, "model") is not current:
            next_module = getattr(current, "model")

        if next_module is None:
            break
        current = next_module

    raise RuntimeError(
        "Could not find causal LM module with `.backbone` and `.lm_head`. "
        "This CCE patch is tailored for Nemotron-H style models. "
        "Run with --no-use_cce to fall back to the model default loss."
    )


def _normalize_active_adapter(value):
    if callable(value):
        value = value()
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else "default"
    if isinstance(value, str):
        return value
    return "default"


def _get_active_lora_adapter(model, lm_head):
    for obj in (lm_head, model):
        if hasattr(obj, "active_adapter"):
            return _normalize_active_adapter(getattr(obj, "active_adapter"))
        if hasattr(obj, "active_adapters"):
            return _normalize_active_adapter(getattr(obj, "active_adapters"))
    return "default"


def _get_lm_head_weight_and_bias_for_cce(model, lm_head):
    """
    Return a classifier weight matrix compatible with linear_cross_entropy.

    If lm_head has LoRA, fuse base weight + LoRA delta for the classifier matrix.
    This is why lora_dropout must be 0.0 when CCE is enabled: dropout on the LoRA branch
    cannot be represented by a single fused classifier matrix.
    """
    adapter = _get_active_lora_adapter(model, lm_head)

    has_lora = (
        hasattr(lm_head, "base_layer")
        and hasattr(lm_head, "lora_A")
        and hasattr(lm_head, "lora_B")
        and adapter in lm_head.lora_A
        and adapter in lm_head.lora_B
    )

    if not has_lora:
        weight = lm_head.weight
        bias = getattr(lm_head, "bias", None)
        return weight, bias

    base_layer = lm_head.base_layer
    base_weight = base_layer.weight
    bias = getattr(base_layer, "bias", None)

    lora_a = lm_head.lora_A[adapter].weight
    lora_b = lm_head.lora_B[adapter].weight
    scaling = lm_head.scaling[adapter]

    if not torch.is_tensor(scaling):
        scaling = torch.tensor(scaling, device=base_weight.device, dtype=base_weight.dtype)
    else:
        scaling = scaling.to(device=base_weight.device, dtype=base_weight.dtype)

    lora_delta = (lora_b @ lora_a).to(dtype=base_weight.dtype)
    weight = base_weight + scaling * lora_delta
    return weight, bias


def patch_forward_with_cce_for_sfttrainer(model, impl="cce"):
    """
    Patch the underlying Nemotron causal LM forward to compute loss with Cut Cross Entropy.

    Compatible with TRL SFTTrainer:
    - labels are NOT pre-shifted by the dataloader;
    - linear_cross_entropy receives shift=1;
    - labels == -100 are preserved for completion-only loss / padding ignore mask;
    - forward returns CausalLMOutputWithPast, not a raw scalar tensor;
    - labels-free calls fall back to the original forward for generation/inference.
    """
    from cut_cross_entropy import linear_cross_entropy
    from transformers.modeling_outputs import CausalLMOutputWithPast

    base = _resolve_causal_lm_module(model)
    original_forward = base.forward

    def patched_causal_forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        num_items_in_batch=None,
        **kwargs,
    ):
        if labels is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        backbone_kwargs = {}
        if position_ids is not None:
            backbone_kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            backbone_kwargs["past_key_values"] = past_key_values
        if inputs_embeds is not None:
            backbone_kwargs["inputs_embeds"] = inputs_embeds
        if output_attentions is not None:
            backbone_kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            backbone_kwargs["output_hidden_states"] = output_hidden_states

        # For training loss computation we do not want KV cache.
        backbone_kwargs["use_cache"] = False

        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **backbone_kwargs,
        )

        hidden_states = backbone_out[0]
        lm_weight, lm_bias = _get_lm_head_weight_and_bias_for_cce(model, self.lm_head)

        if num_items_in_batch is not None:
            loss_sum = linear_cross_entropy(
                hidden_states,
                lm_weight,
                labels,
                bias=lm_bias,
                shift=1,
                reduction="sum",
                impl=impl,
            )
            if not torch.is_tensor(num_items_in_batch):
                num_items_in_batch = torch.tensor(
                    num_items_in_batch,
                    device=loss_sum.device,
                    dtype=loss_sum.dtype,
                )
            else:
                num_items_in_batch = num_items_in_batch.to(device=loss_sum.device, dtype=loss_sum.dtype)
            loss = loss_sum / num_items_in_batch.clamp_min(1)
        else:
            loss = linear_cross_entropy(
                hidden_states,
                lm_weight,
                labels,
                bias=lm_bias,
                shift=1,
                reduction="mean",
                impl=impl,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=getattr(backbone_out, "past_key_values", None),
            hidden_states=getattr(backbone_out, "hidden_states", None),
            attentions=getattr(backbone_out, "attentions", None),
        )

    base._original_forward_before_cce = original_forward
    base.forward = types.MethodType(patched_causal_forward, base)
    logger.info(f"Cut Cross Entropy patch enabled for SFTTrainer. impl={impl}")
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.use_cce and abs(args.lora_dropout) > 1e-12:
        raise ValueError(
            "CCE is enabled, but --lora_dropout is not 0.0. "
            "Set --lora_dropout 0.0 or disable CCE with --no-use_cce."
        )

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

    logger.info(
        "Подготовка тренировочного датасета, так же не используем для обучения промпты где solver не дал правильный ответ..."
    )
    train_dataset = prepare_dataset(args.train_path)
    logger.info(f"train dataset final len: {len(train_dataset)}")

    logger.info(
        "Подготовка валидационного датасета, так же не используем для валидации промпты где solver не дал правильный ответ..."
    )
    val_dataset = prepare_dataset(args.val_path, eval=True)
    logger.info(f"val dataset final len: {len(val_dataset)}")

    logger.info(f"Загрузка модели и токенизатора через Unsloth FastLanguageModel: {args.model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation=args.attn_implementation,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Навешиваем LoRA через Unsloth FastLanguageModel.get_peft_model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj",
            "out_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    if args.use_cce:
        model = patch_forward_with_cce_for_sfttrainer(model, impl=args.cce_impl)

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
        gradient_checkpointing=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=args.max_seq_len,
        completion_only_loss=True,
        prediction_loss_only=args.use_cce,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataset_num_proc=8,
        report_to="wandb",
        run_name=f"{args.wandb_run_basename}-SFT-{args.data}-ep{args.epochs}-lr{args.lr}",
    )

    training_args.group_by_length = True

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[TimeLimitCallback(max_hours=args.max_hours)],
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
