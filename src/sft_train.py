import os
import argparse
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed
)

from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="NVIDIA Nemotron SFT Baseline Trainer")
    
    # Пути
    parser.add_argument("--model_id", type=str, default="metric/nemotron-3-nano-30b-a3b-bf16", help="HF ID или локальный путь к модели")
    parser.add_argument("--data", type=str, required=True, help="Имя датасета для логов W&B")
    parser.add_argument("--train_path", type=str, required=True, help="Путь к обучающей выборке (train.csv)")
    parser.add_argument("--val_path", type=str, required=True, help="Путь к валидационной выборке (val.csv)")
    parser.add_argument("--output_dir", type=str, default="./models/sft_baseline_v1", help="Папка для сохранения адаптера")
    
    # Гиперпараметры обучения
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча на один GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="Шаги аккумуляции градиента")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=50, help="Шаги между валидациями")
    
    # Параметры LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    return parser.parse_args()

def prepare_dataset(csv_path):
    """
    Загрузка CSV и приведение к стандартному формату диалогов.
    Новый SFTTrainer сам применит Chat Template к колонке 'messages'.
    """
    df = pd.read_csv(csv_path)

    instruction_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
    
    formatted_data = []
    for _, row in df.iterrows():
        user_text = str(row['prompt']) + instruction_suffix
        assistant_text = (
            f"<think>\n{row['generated_cot']}\n</think>\n"
            f"Final Answer: \\boxed{{{row['answer']}}}"
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

    print(f"Загрузка токенизатора для {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Подготовка тренировочного датасета...")
    train_dataset = prepare_dataset(args.train_path)
    
    print("Подготовка валидационного датасета...")
    val_dataset = prepare_dataset(args.val_path)

    print("Загрузка модели в bfloat16...")
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        load_best_model_at_end=True,
        
        optim="adamw_torch",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=args.max_seq_len,
        completion_only_loss=True, 
        group_by_length=True,

        report_to="wandb",
        run_name=f"SFT-{args.data}-ep{args.epochs}-lr{args.lr}"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
    )

    print("Запуск обучения...")
    trainer.train()

    print(f"Сохранение адаптера в {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()