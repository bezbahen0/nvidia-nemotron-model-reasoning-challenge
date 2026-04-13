import json
import torch
import shutil
import wandb
import argparse
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM

def parse_args():
    # ... (код parse_args без изменений) ...
    parser = argparse.ArgumentParser(description="Отбор лучшего чекпоинта и создание Weighted Model Soup")
    parser.add_argument("--base_model", type=str, default="metric/nemotron-3-nano-30b-a3b-bf16", help="Путь к базовой модели")
    parser.add_argument("--checkpoints_dir", type=str, default="./models/sft_baseline_v1", help="Папка с чекпоинтами")
    parser.add_argument("--scores_path", type=str, default="validation_scores.json", help="JSON файл с оценками")
    parser.add_argument("--output_best", type=str, default="./models/lora_best", help="Куда скопировать лучший адаптер")
    parser.add_argument("--output_soup", type=str, default="./models/lora_soup", help="Куда сохранить усредненный адаптер")
    parser.add_argument("--top_k", type=int, default=5, help="Сколько лучших чекпоинтов использовать для Model Soup")
    return parser.parse_args()

def main():
    args = parse_args()

    wandb.init(
        project="nemotron-competition",
        job_type="blending",
        name="model-soup-creation",
        config=vars(args) # <-- ДОБАВЛЕНО: логируем аргументы запуска
    )

    if not Path(args.scores_path).exists():
        raise FileNotFoundError(f"Файл {args.scores_path} не найден. Сначала запустите evaluate_checkpoints.py")

    with open(args.scores_path, "r") as f:
        scores = json.load(f)

    sorted_checkpoints = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_checkpoints:
        print("Оценки не найдены.")
        return

    best_ckpt_name, best_score = sorted_checkpoints[0]
    best_ckpt_path = Path(args.checkpoints_dir) / best_ckpt_name
    
    # <-- ДОБАВЛЕНО: сохраняем лучший результат в summary (сводку) проекта
    wandb.summary["best_checkpoint_name"] = best_ckpt_name
    wandb.summary["best_checkpoint_accuracy"] = best_score
    
    print(f"🏆 Лучший чекпоинт: {best_ckpt_name} с Accuracy = {best_score:.4f}")
    print(f"Копирование лучшего адаптера в {args.output_best}...")
    
    if Path(args.output_best).exists():
        shutil.rmtree(args.output_best)
    shutil.copytree(best_ckpt_path, args.output_best)
    print("✅ Лучший чекпоинт сохранен.\n")

    top_k_checkpoints = sorted_checkpoints[:args.top_k]
    top_k_checkpoints = [(name, score) for name, score in top_k_checkpoints if score > 0]
    
    if len(top_k_checkpoints) < 2:
        print("Недостаточно чекпоинтов с Accuracy > 0 для создания Model Soup.")
        wandb.finish()
        return

    print(f"🍲 Создание Model Soup из Топ-{len(top_k_checkpoints)} чекпоинтов:")
    
    total_score = sum(score for _, score in top_k_checkpoints)
    adapters_to_merge = []
    weights = []
    
    wandb_weights_log = {}

    for name, score in top_k_checkpoints:
        weight = score / total_score
        adapters_to_merge.append(str(Path(args.checkpoints_dir) / name))
        weights.append(weight)
        wandb_weights_log[name] = weight # <-- Сохраняем вес
        print(f" -> {name}: score = {score:.4f} | вес = {weight:.4f}")

    wandb.log({"soup_weights": wandb_weights_log})

    print("\nЗагрузка базовой модели на CPU (для экономии видеопамяти)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
        trust_remote_code=True
    )

    print("Загрузка основного адаптера...")
    peft_model = PeftModel.from_pretrained(
        base_model, 
        adapters_to_merge[0], 
        adapter_name="adapter_0"
    )

    adapter_names = ["adapter_0"]
    for i, adapter_path in enumerate(adapters_to_merge[1:], start=1):
        adapter_name = f"adapter_{i}"
        print(f"Загрузка {adapter_name}...")
        peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)

    print("\nСлияние весов (Weighted Linear Blend)...")
    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="soup",
        combination_type="linear"
    )

    peft_model.set_adapter("soup")
    
    print(f"Сохранение Model Soup в {args.output_soup}...")
    peft_model.save_pretrained(args.output_soup)
    print("✅ Model Soup успешно создан!")

    wandb.finish()

if __name__ == "__main__":
    main()