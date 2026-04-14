import os
import re
import math
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import wandb

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.log import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRa by solving real tasks")
    
    parser.add_argument("--wandb_run_basename", type=str)
   
    parser.add_argument("--base_model", type=str, default="nemotron-3-nano-30b-a3b-bf16")
    parser.add_argument("--val_data", type=str, required=True, help="val_split.csv")
    parser.add_argument("--checkpoints_dir", type=str)
    parser.add_argument("--output_scores", type=str)
    
    # default gen params
    parser.add_argument("--max_tokens", type=int, default=3584)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_lora_rank", type=int, default=32)
    return parser.parse_args()

### As orgs

def extract_final_answer(text: str | None) -> str:
    if text is None:
        return 'NOT_FOUND'

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'

def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()

def main():
    args = parse_args()

    wandb.init(
        job_type="evaluation",
        name=f"{args.wandb_run_basename}-eval-{args.checkpoints_dir.split('/')[-1]}",
        config=vars(args)
    )

    checkpoints_path = Path(args.checkpoints_dir)
    checkpoints = sorted([d for d in checkpoints_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint")])

    if not checkpoints:
        logger.info(f"В {args.checkpoints_dir} не найдено чекпоинтов.")
        return

    logger.info(f"Найдено чекпоинтов для оценки: {len(checkpoints)}")

    val_df = pd.read_csv(args.val_data)
    
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['TRANSFORMERS_NO_FLAX'] = '1'

    logger.info(f"Загрузка базовой модели {args.base_model} в vLLM...")
    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=1,
        dtype='auto',
        max_model_len=4096,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,

        gpu_memory_utilization=0.80,
        max_num_seqs=32,
    )

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    prompts = []
    logger.info("Форматирование промптов...")
    for item in val_df.itertuples(index=False):
        user_content = (
            item.prompt
            + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
        )
        try:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)

    results = {}
    max_accuracy = 0

    # 5. Итерация по чекпоинтам и замер
    for i, ckpt_path in enumerate(checkpoints):
        ckpt_name = ckpt_path.name
        logger.info(f"\n[{i+1}/{len(checkpoints)}] Оценка чекпоинта: {ckpt_name}")
        
        # Генерация ответов через LoRA
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest(f"adapter_{i}", i + 1, str(ckpt_path)),
            use_tqdm=True
        )

        num_correct = 0
        debug_records = []

        for item, output in zip(val_df.itertuples(index=False), outputs):
            raw_text = output.outputs[0].text
            extracted_answer = extract_final_answer(raw_text)
            ground_truth = str(item.answer)

            is_correct = verify(ground_truth, extracted_answer)
            if is_correct:
                num_correct += 1

            debug_records.append({
                "prompt": item.prompt,
                "ground_truth": ground_truth,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct
            })

        accuracy = num_correct / len(val_df)
        results[ckpt_name] = accuracy
        max_accuracy = max(max_accuracy, accuracy)
        
        wandb.log({
            "step": i,
            "checkpoint_accuracy": accuracy,
            "checkpoint_name": ckpt_name
        })
        logger.info(f"-> Accuracy ({ckpt_name}): {accuracy:.4f}")

        debug_df = pd.DataFrame(debug_records)
        debug_df.to_csv(ckpt_path / "eval_results.csv", index=False)
        
    wandb.summary["best_val_accuracy"] = max_accuracy
    wandb.finish()

    with open(args.output_scores, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"\nВсе оценки завершены. Итоговые результаты сохранены в {args.output_scores}")

if __name__ == "__main__":
    main()