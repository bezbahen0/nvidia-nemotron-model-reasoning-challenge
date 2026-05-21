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
    parser.add_argument("--train_telemetry", action="store_true")
    
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


def load_train_metadata(train_path):
    df = pd.read_csv(train_path)

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
        telemetry_metadata=None,
        telemetry_output_dir=None,
        telemetry_save_steps=50,
        telemetry_enabled=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.telemetry_metadata = telemetry_metadata or {}
        self.telemetry_output_dir = telemetry_output_dir
        self.telemetry_save_steps = int(telemetry_save_steps)
        self.telemetry_enabled = bool(telemetry_enabled)

        self.telemetry_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

        if self.telemetry_output_dir is not None:
            os.makedirs(self.telemetry_output_dir, exist_ok=True)

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

        if self.telemetry_enabled and model.training and source_row_index is not None:
            self._write_train_batch_telemetry(
                outputs=outputs,
                labels=inputs.get("labels"),
                source_row_index=source_row_index,
            )

        if return_outputs:
            return loss, outputs

        return loss

    def _write_train_batch_telemetry(self, outputs, labels, source_row_index):
        if self.telemetry_output_dir is None:
            return

        if labels is None:
            return

        if not hasattr(outputs, "logits"):
            return

        if self.telemetry_save_steps <= 0:
            checkpoint_bucket = int(self.state.global_step)
        else:
            checkpoint_bucket = (
                (int(self.state.global_step) // self.telemetry_save_steps) + 1
            ) * self.telemetry_save_steps

        path = os.path.join(
            self.telemetry_output_dir,
            f"train_until_checkpoint-{checkpoint_bucket}_rank-{self.telemetry_rank}.jsonl",
        )

        with torch.no_grad():
            logits = outputs.logits.detach().float()
            labels = labels.detach()

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

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
                valid_values = token_log_probs[batch_index][valid_mask[batch_index]]

                row_index = int(source_row_indexes[batch_index])
                metadata = self.telemetry_metadata.get(
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
                        "source_row_index": row_index,
                        "id": metadata["id"],
                        "source": metadata["source"],
                        "label": metadata["label"],
                        "num_loss_tokens": 0,
                        "total_loss": None,
                        "mean_nll": None,
                        "min_logprob": None,
                        "p05_logprob": None,
                        "hard_token_ratio": None,
                        "near_zero_token_ratio": None,
                    }
                else:
                    values = valid_values.detach().cpu()

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
                        "source_row_index": row_index,
                        "id": metadata["id"],
                        "source": metadata["source"],
                        "label": metadata["label"],
                        "num_loss_tokens": num_loss_tokens,
                        "total_loss": total_loss,
                        "mean_nll": mean_nll,
                        "min_logprob": min_logprob,
                        "p05_logprob": p05_logprob,
                        "hard_token_ratio": hard_token_ratio,
                        "near_zero_token_ratio": near_zero_token_ratio,
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

    logger.info("Подготовка тренировочного датасета, так же не используем для обучения промпты где solver не дал правильный ответ...")
    train_dataset = prepare_dataset(args.train_path)
    logger.info(f"train dataset final len: {len(train_dataset)}")
    train_metadata = load_train_metadata(args.train_path)

    
    logger.info("Подготовка валидационного датасета, так же не используем для валидации промпты где solver не дал правильный ответ...")
    val_dataset = prepare_dataset(args.val_path, eval=True)
    logger.info(f"val dataset final len: {len(val_dataset)}")

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

    telemetry_output_dir = os.path.join(args.output_dir, "train_telemetry")

    trainer = TelemetrySFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        callbacks=[TimeLimitCallback(max_hours=args.max_hours)],
        telemetry_metadata=train_metadata,
        telemetry_output_dir=telemetry_output_dir,
        telemetry_save_steps=args.eval_steps,
        telemetry_enabled=args.train_telemetry,
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