import os
import time
import argparse
import torch
import pandas as pd
import wandb
from datasets import Dataset
from peft import LoraConfig
import torch.nn.functional as F

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
    parser = argparse.ArgumentParser(description="NVIDIA Nemotron SFT Baseline Trainer")
    
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
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
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

class FocalLossSFTTrainer(SFTTrainer):
    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1), 
            reduction='none',
            ignore_index=-100
        )
        
        ce_loss = ce_loss.float()
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        valid_mask = shift_labels.view(-1) != -100
        
        loss = focal_loss[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
        
        return (loss, outputs) if return_outputs else loss


def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)

    instruction_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
    
    formatted_data = []
    for _, row in df.iterrows():
        # Skip all task vere solver can't find result
        if not verify(row["computed_answer"], row["answer"]):
            continue

        user_text = str(row['prompt']) + instruction_suffix

        # Use computed answer, reduce noice, we compare results on previosue stage
        computed_answer = str(row['computed_answer']).strip()
        
        # } break compputed metric provided by orgs, reg search only to first }
        # matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
        if '}' in computed_answer:
            meta_reasoning = (
                "\n\nWait, the calculated answer contains a closing curly brace '}'. "
                "Wrapping it in a standard LaTeX \\boxed{} tag would create malformed syntax and ambiguity "
                "where the bounding box terminates prematurely. To avoid formatting errors and preserve the exact answer, "
                "I will use the plain text format instead."
            )
            
            assistant_text = (
                f"<think>\n{row['generated_cot']}{meta_reasoning}\n</think>\n"
                f"The final answer is: {computed_answer}"
            )
        else:
            assistant_text = (
                f"<think>\n{row['generated_cot']}\n</think>\n"
                f"Final Answer: \\boxed{{{computed_answer}}}"
            )
        
        formatted_data.append({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ]
        })
        
    return Dataset.from_list(formatted_data)

def main():
    args = parse_args()
    set_seed(args.seed)

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

    
    logger.info("Подготовка валидационного датасета, так же не используем для валидации промпты где solver не дал правильный ответ...")
    val_dataset = prepare_dataset(args.val_path)
    logger.info(f"val dataset final len: {len(val_dataset)}")

    logger.info("Загрузка модели в bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False 
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj|out_proj|embed_tokens|lm_head)$", 
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

        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataset_num_proc=8,

        report_to="wandb",
        run_name=f"{args.wandb_run_basename}-SFT-{args.data}-ep{args.epochs}-lr{args.lr}"
    )

    trainer = FocalLossSFTTrainer(
        gamma=2.0,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        callbacks=[TimeLimitCallback(max_hours=args.max_hours)]
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