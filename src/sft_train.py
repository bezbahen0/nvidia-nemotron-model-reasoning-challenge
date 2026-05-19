import os
import re
import time
import math
import argparse
import random
import zipfile
import shutil
import gc
from collections import defaultdict, Counter

import unsloth
from unsloth import FastLanguageModel

import torch
import pandas as pd
import wandb
import bitsandbytes as bnb
from transformers import AutoTokenizer, set_seed
from cut_cross_entropy import linear_cross_entropy
from peft import load_peft_weights

try:
    from src.log import logger
except Exception:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb_run_basename", type=str, default="nemotron")
    parser.add_argument("--model_id", type=str, default="nemotron-3-nano-30b-a3b-bf16")
    parser.add_argument("--data", type=str, required=True, help="dataset name for W&B")
    parser.add_argument("--train_path", type=str, required=True, help="train.csv")
    parser.add_argument("--val_path", type=str, required=True, help="val.csv")
    parser.add_argument("--output_dir", type=str, default="./models/sft_unsloth_cce")

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4, help="real micro batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--save_total_limit", type=int, default=6)
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto", help="auto, none, or explicit checkpoint path")
    parser.add_argument("--max_hours", type=float, required=True)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means epochs * full pass")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--prompt_col", type=str, default="prompt")
    parser.add_argument("--cot_col", type=str, default="generated_cot")
    parser.add_argument("--answer_col", type=str, default="computed_answer")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--id_col", type=str, default="id")

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--drop_duplicate_prompts", action="store_true")
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--no_stratified_batches", action="store_true")
    parser.add_argument("--no_sort_within_batch_by_length", action="store_true")

    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", choices=["paged_adamw_8bit", "adamw"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1e9)

    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj,out_proj,embed_tokens,lm_head",
    )

    return parser.parse_args()


def safe_metric_name(value: str) -> str:
    value = str(value).strip().replace(" ", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_boxed_answer(answer: str) -> str:
    answer = clean_text_value(answer)
    if "\\boxed" in answer:
        return answer
    return f"\\boxed{{{answer}}}"


def get_completion_end_text(tokenizer) -> str:
    try:
        vocab = tokenizer.get_vocab()
        if "<|im_end|>" in vocab:
            return "<|im_end|>"
    except Exception:
        pass

    return tokenizer.eos_token or ""


def tokenize_prompt(tokenizer, user_text: str) -> list[int]:
    messages = [{"role": "user", "content": user_text}]

    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer(f"USER: {user_text}\n\nASSISTANT:", add_special_tokens=True)["input_ids"]

    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    return [int(x) for x in ids]


def tokenize_completion(tokenizer, completion_text: str) -> list[int]:
    ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    return [int(x) for x in ids]


def row_to_example(tokenizer, row: pd.Series, args) -> dict | None:
    instruction_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

    prompt = clean_text_value(row[args.prompt_col])
    user_text = prompt + instruction_suffix

    cot = clean_text_value(row[args.cot_col])
    boxed = make_boxed_answer(row[args.answer_col])

    if "<think>" in cot and "</think>" in cot:
        assistant_text = f"{cot}\nFinal Answer: {boxed}"
    else:
        assistant_text = f"<think>\n{cot}\n</think>\nFinal Answer: {boxed}"

    end_text = get_completion_end_text(tokenizer)
    if end_text and not assistant_text.endswith(end_text):
        assistant_text += end_text

    prompt_ids = tokenize_prompt(tokenizer, user_text)
    completion_ids = tokenize_completion(tokenizer, assistant_text)

    full_ids = prompt_ids + completion_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)

    if len(full_ids) > args.max_seq_len:
        full_ids = full_ids[:args.max_seq_len]
        weights = weights[:args.max_seq_len]

    if len(full_ids) < 2:
        return None

    target_weights = weights[1:]
    if sum(target_weights) == 0:
        return None

    label = clean_text_value(row[args.label_col]) if args.label_col in row.index else "unknown"
    problem_id = clean_text_value(row[args.id_col]) if args.id_col in row.index else ""

    return {
        "problem_id": problem_id,
        "category": label or "unknown",
        "tokens": full_ids[:-1],
        "targets": full_ids[1:],
        "weights": target_weights,
        "token_count": len(full_ids) - 1,
        "loss_token_count": int(sum(target_weights)),
    }


def load_examples(csv_path: str, tokenizer, args, split_name: str) -> list[dict]:
    df = pd.read_csv(csv_path)

    required_cols = [args.prompt_col, args.cot_col, args.answer_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"{split_name}: missing columns: {missing}")

    if args.drop_duplicate_prompts:
        before = len(df)
        df = df.drop_duplicates(subset=[args.prompt_col]).reset_index(drop=True)
        logger.info(f"{split_name}: dropped duplicate prompts: {before} -> {len(df)}")

    examples = []
    skipped = 0

    for _, row in df.iterrows():
        example = row_to_example(tokenizer, row, args)
        if example is None:
            skipped += 1
            continue
        examples.append(example)

    total_tokens = sum(e["token_count"] for e in examples)
    loss_tokens = sum(e["loss_token_count"] for e in examples)
    lengths = sorted(e["token_count"] for e in examples)
    by_label = Counter(e["category"] for e in examples)
    by_label_loss_tokens = defaultdict(int)

    for e in examples:
        by_label_loss_tokens[e["category"]] += e["loss_token_count"]

    logger.info(
        f"{split_name}: examples={len(examples)}, skipped={skipped}, "
        f"tokens={total_tokens:,}, loss_tokens={loss_tokens:,}"
    )

    if lengths:
        logger.info(
            f"{split_name}: len p50={lengths[len(lengths)//2]}, "
            f"p90={lengths[int(len(lengths)*0.9)]}, max={lengths[-1]}"
        )

    logger.info(f"{split_name}: label stats:")
    for label, count in sorted(by_label.items()):
        logger.info(f"  - {label}: examples={count}, loss_tokens={by_label_loss_tokens[label]:,}")

    if not examples:
        raise RuntimeError(f"{split_name}: no examples loaded")

    return examples


def sanity_check_gpu_and_kernels():
    import causal_conv1d
    import mamba_ssm
    from causal_conv1d import causal_conv1d_fn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cc = torch.cuda.get_device_capability(0)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    logger.info(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    logger.info(f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(x, w, None, activation="silu")
    logger.info("causal_conv1d CUDA kernel: OK")


def patch_mamba_fast_path():
    nemotron_mod = None

    for module_name, module in sys.modules.items():
        if "modeling_nemotron_h" in module_name and hasattr(module, "is_fast_path_available"):
            nemotron_mod = module
            break

    if nemotron_mod is None:
        logger.warning("Could not find modeling_nemotron_h module; skip Mamba fast-path patch")
        return

    logger.info(f"is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    nemotron_mod.is_fast_path_available = True
    logger.info("Patched is_fast_path_available = True")


def patch_forward_with_cce(model):
    base = model

    while hasattr(base, "model"):
        base = base.model

    def patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        backbone_out = base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )

        hidden_states = backbone_out[0]
        lm_head = base.lm_head

        if hasattr(lm_head, "base_layer") and hasattr(lm_head, "lora_A") and "default" in lm_head.lora_A:
            base_w = lm_head.base_layer.weight
            lora_a = lm_head.lora_A["default"].weight
            lora_b = lm_head.lora_B["default"].weight
            scaling = lm_head.scaling["default"]
            lm_weight = base_w + scaling * lora_b @ lora_a
        else:
            lm_weight = lm_head.weight

        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states,
                lm_weight,
                labels,
                reduction="none",
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None

        model._cached_per_token_ce = per_token_ce
        return loss

    base.forward = patched_causal_forward
    logger.info("Patched CausalLM.forward with Cut Cross-Entropy")


def make_batch_tensors(batch, device):
    max_len = max(len(e["tokens"]) for e in batch)
    n = len(batch)

    input_ids = torch.zeros(n, max_len, dtype=torch.long, device=device)
    labels = torch.zeros(n, max_len, dtype=torch.long, device=device)
    weights = torch.zeros(n, max_len, dtype=torch.float32, device=device)
    attention_mask = torch.zeros(n, max_len, dtype=torch.long, device=device)

    for i, ex in enumerate(batch):
        seq_len = len(ex["tokens"])
        input_ids[i, :seq_len] = torch.tensor(ex["tokens"], dtype=torch.long, device=device)
        labels[i, :seq_len] = torch.tensor(ex["targets"], dtype=torch.long, device=device)
        weights[i, :seq_len] = torch.tensor(ex["weights"], dtype=torch.float32, device=device)
        attention_mask[i, :seq_len] = 1

    return input_ids, labels, weights, attention_mask


def make_stratified_batches(examples, effective_batch_size, seed, epoch, shuffle=True):
    rng = random.Random(seed + epoch)
    labels = [e["category"] for e in examples]
    lengths = [e["token_count"] for e in examples]

    bucket_size = max(effective_batch_size * 64, effective_batch_size)

    indices = list(range(len(examples)))
    indices.sort(key=lambda i: lengths[i])

    buckets = [
        indices[start:start + bucket_size]
        for start in range(0, len(indices), bucket_size)
    ]

    all_batches = []

    for bucket in buckets:
        by_label = defaultdict(list)

        for idx in bucket:
            by_label[labels[idx]].append(idx)

        for label in by_label:
            rng.shuffle(by_label[label])

        label_names = sorted(by_label.keys())
        pos = {label: 0 for label in label_names}
        current = []

        while True:
            progressed = False

            for label in label_names:
                if pos[label] >= len(by_label[label]):
                    continue

                current.append(by_label[label][pos[label]])
                pos[label] += 1
                progressed = True

                if len(current) == effective_batch_size:
                    current.sort(key=lambda i: lengths[i])
                    all_batches.append(current)
                    current = []

            if not progressed:
                break

    if shuffle:
        rng.shuffle(all_batches)

    used = Counter()
    for batch in all_batches:
        for idx in batch:
            used[labels[idx]] += 1

    logger.info(f"epoch={epoch + 1}: built {len(all_batches)} stratified batches")
    logger.info(f"epoch={epoch + 1}: used by label: {dict(sorted(used.items()))}")

    return all_batches


def make_plain_batches(examples, effective_batch_size, seed, epoch, shuffle=True):
    rng = random.Random(seed + epoch)
    indices = list(range(len(examples)))

    if shuffle:
        rng.shuffle(indices)

    batches = [
        indices[start:start + effective_batch_size]
        for start in range(0, len(indices), effective_batch_size)
        if len(indices[start:start + effective_batch_size]) == effective_batch_size
    ]

    logger.info(f"epoch={epoch + 1}: built {len(batches)} plain batches")
    return batches


def get_cosine_lr(step, total_steps, base_lr, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)

    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_loss(model, val_examples, args, device):
    if not val_examples or args.eval_batches <= 0:
        return None, {}

    model.eval()

    total_loss_sum = 0.0
    total_weight_sum = 0.0
    category_metrics = {}
    max_examples = min(len(val_examples), args.eval_batches * args.batch_size)

    for start in range(0, max_examples, args.batch_size):
        batch = val_examples[start:start + args.batch_size]
        input_ids, labels, weights, attention_mask = make_batch_tensors(batch, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

            per_token_ce = model._cached_per_token_ce
            weighted_loss = per_token_ce * weights

            total_loss_sum += weighted_loss.sum().item()
            total_weight_sum += weights.sum().item()

        collect_category_metrics(
            per_token_ce=per_token_ce.detach(),
            weights=weights.detach(),
            micro=batch,
            metrics=category_metrics,
        )

        del input_ids, labels, weights, attention_mask
        del per_token_ce, weighted_loss

    model.train()

    if total_weight_sum <= 0:
        return None, {}

    eval_loss = total_loss_sum / total_weight_sum
    eval_metrics = finalize_category_metrics(category_metrics, prefix="eval")

    return eval_loss, eval_metrics


def collect_category_metrics(per_token_ce, weights, micro, metrics):
    for row_idx, ex in enumerate(micro):
        category = safe_metric_name(ex["category"])
        mask = weights[row_idx] > 0

        if not mask.any():
            continue

        row_ce = per_token_ce[row_idx][mask]
        loss_sum = float(row_ce.sum().item())
        token_count = int(mask.sum().item())
        min_logprob = float((-row_ce).min().item())

        loss_key = f"loss_sum/{category}"
        token_key = f"tokens/{category}"
        min_logprob_key = f"min_logprob/{category}"

        metrics[loss_key] = metrics.get(loss_key, 0.0) + loss_sum
        metrics[token_key] = metrics.get(token_key, 0) + token_count

        if min_logprob_key not in metrics:
            metrics[min_logprob_key] = min_logprob
        else:
            metrics[min_logprob_key] = min(metrics[min_logprob_key], min_logprob)


def finalize_category_metrics(metrics, prefix: str):
    output = {}

    for key, loss_sum in metrics.items():
        if not key.startswith("loss_sum/"):
            continue

        category = key.split("/", 1)[1]
        token_count = metrics.get(f"tokens/{category}", 0)

        if token_count > 0:
            output[f"{prefix}/category_loss/{category}"] = loss_sum / token_count
            output[f"{prefix}/category_tokens/{category}"] = token_count

    for key, value in metrics.items():
        if key.startswith("min_logprob/"):
            category = key.split("/", 1)[1]
            output[f"{prefix}/category_min_logprob/{category}"] = value

    return output


def get_checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip("/"))
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if not match:
        return -1
    return int(match.group(1))


def find_last_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None

    checkpoints = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"checkpoint-\d+", name):
            checkpoints.append(path)

    if not checkpoints:
        return None

    checkpoints.sort(key=get_checkpoint_step)
    return checkpoints[-1]


def cleanup_old_checkpoints(output_dir: str, save_total_limit: int):
    if save_total_limit <= 0:
        return

    checkpoints = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"checkpoint-\d+", name):
            checkpoints.append(path)

    checkpoints.sort(key=get_checkpoint_step)

    while len(checkpoints) > save_total_limit:
        old = checkpoints.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        logger.info(f"Removed old checkpoint: {old}")


def save_training_checkpoint(model, tokenizer, optimizer, args, global_step: int, epoch: int):
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    state = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "optimizer": optimizer.state_dict(),
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
    cleanup_old_checkpoints(args.output_dir, args.save_total_limit)
    logger.info(f"Checkpoint saved: {checkpoint_dir}")


def load_adapter_weights_into_model(model, checkpoint_dir: str):
    adapter_weights = load_peft_weights(checkpoint_dir)
    model_state = model.state_dict()

    new_state = {}
    loaded = 0

    for key, value in adapter_weights.items():
        candidates = [
            key,
            key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
            key.replace(".backbone.lm_head.", ".lm_head."),
            key.replace(".backbone.lm_head.", ".lm_head.")
               .replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"),
        ]

        for candidate in candidates:
            if candidate in model_state:
                new_state[candidate] = value
                loaded += 1
                break

    model.load_state_dict(new_state, strict=False)

    if loaded == 0:
        raise RuntimeError(f"No adapter weights loaded from checkpoint: {checkpoint_dir}")

    logger.info(f"Loaded adapter weights from {checkpoint_dir}: {loaded}/{len(adapter_weights)} tensors")


def load_training_state(checkpoint_dir: str):
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"training_state.pt not found: {state_path}")
    return torch.load(state_path, map_location="cpu")


def init_wandb(args):
    if args.no_wandb:
        return False

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

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "huggingface"),
        name=f"{args.wandb_run_basename}-UnslothCCE-{args.data}-ep{args.epochs}-lr{args.lr}",
        config=vars(args),
        resume="allow",
    )

    return True


def train():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    use_wandb = init_wandb(args)

    resume_checkpoint = None
    if args.resume_from_checkpoint == "auto":
        resume_checkpoint = find_last_checkpoint(args.output_dir)
    elif args.resume_from_checkpoint and args.resume_from_checkpoint.lower() != "none":
        resume_checkpoint = args.resume_from_checkpoint

    if resume_checkpoint:
        logger.info(f"Will resume from checkpoint: {resume_checkpoint}")
    else:
        logger.info("No checkpoint found. Training from scratch.")

    sanity_check_gpu_and_kernels()

    logger.info(f"Загрузка токенизатора для {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Подготовка тренировочного датасета...")
    train_examples = load_examples(args.train_path, tokenizer, args, "train")
    logger.info(f"train dataset final len: {len(train_examples)}")

    logger.info("Подготовка валидационного датасета...")
    val_examples = load_examples(args.val_path, tokenizer, args, "val")
    logger.info(f"val dataset final len: {len(val_examples)}")

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Загрузка модели через Unsloth в bfloat16...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    logger.info(f"Adding LoRA target_modules={target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    FastLanguageModel.for_training(model)
    patch_mamba_fast_path()
    patch_forward_with_cce(model)

    resume_state = None
    if resume_checkpoint:
        load_adapter_weights_into_model(model, resume_checkpoint)
        resume_state = load_training_state(resume_checkpoint)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {trainable_params:,} trainable / {total_params:,} total parameters")

    device = next(model.parameters()).device
    effective_batch_size = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_examples) // effective_batch_size
    total_steps = args.epochs * steps_per_epoch

    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    if total_steps <= 0:
        raise RuntimeError(
            f"Not enough examples: len(train_examples)={len(train_examples)}, "
            f"effective_batch_size={effective_batch_size}"
        )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "paged_adamw_8bit":
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )

    global_step = 0
    resume_epoch = 0

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        global_step = int(resume_state.get("global_step", 0))
        resume_epoch = int(resume_state.get("epoch", 0))

        if "random_state" in resume_state:
            random.setstate(resume_state["random_state"])
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])

        logger.info(f"Resumed training state: global_step={global_step}, epoch={resume_epoch + 1}")

    logger.info(
        f"Training: epochs={args.epochs}, total_steps={total_steps}, "
        f"effective_batch_size={effective_batch_size}, micro_batch_size={args.batch_size}, "
        f"grad_accum={args.grad_accum}, lr={args.lr}, optimizer={args.optimizer}"
    )

    start_time = time.time()
    max_seconds = args.max_hours * 3600
    model.train()

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(resume_epoch, args.epochs):
        if global_step >= total_steps:
            break

        if args.no_stratified_batches:
            batches = make_plain_batches(
                examples=train_examples,
                effective_batch_size=effective_batch_size,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
            )
        else:
            batches = make_stratified_batches(
                examples=train_examples,
                effective_batch_size=effective_batch_size,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
            )

        already_seen_in_this_epoch = 0
        if global_step > 0:
            epoch_start_step = epoch * steps_per_epoch
            already_seen_in_this_epoch = max(0, global_step - epoch_start_step)

        if already_seen_in_this_epoch > 0:
            logger.info(f"epoch={epoch + 1}: skipping {already_seen_in_this_epoch} already trained batches")

        for batch_index, batch_indices in enumerate(batches):
            if batch_index < already_seen_in_this_epoch:
                continue

            if global_step >= total_steps:
                break

            if time.time() - start_time > max_seconds:
                logger.info(f"\n[ВНИМАНИЕ] Достигнут лимит времени ({(time.time() - start_time) / 3600:.2f} часов).")
                save_training_checkpoint(model, tokenizer, optimizer, args, global_step, epoch)
                if use_wandb:
                    wandb.finish()
                return

            batch = [train_examples[i] for i in batch_indices]

            if not args.no_sort_within_batch_by_length:
                batch = sorted(batch, key=lambda e: e["token_count"])

            total_loss_sum = 0.0
            total_weight_sum = 0.0
            step_category_metrics = {}

            n_accum = math.ceil(len(batch) / args.batch_size)
            step_t0 = time.time()

            for mb_start in range(0, len(batch), args.batch_size):
                micro = batch[mb_start:mb_start + args.batch_size]
                input_ids, labels, weights, attention_mask = make_batch_tensors(micro, device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )

                    per_token_ce = model._cached_per_token_ce
                    weighted_loss = per_token_ce * weights
                    weight_sum = weights.sum()
                    loss_sum = weighted_loss.sum()
                    loss = loss_sum / weight_sum if weight_sum > 0 else loss_sum * 0.0

                collect_category_metrics(
                    per_token_ce=per_token_ce.detach(),
                    weights=weights.detach(),
                    micro=micro,
                    metrics=step_category_metrics,
                )

                (loss / n_accum).backward()

                total_loss_sum += float(loss_sum.item())
                total_weight_sum += float(weight_sum.item())

                del input_ids, labels, weights, attention_mask
                del per_token_ce, weighted_loss, weight_sum, loss_sum, loss

            lr = get_cosine_lr(global_step, total_steps, args.lr, args.warmup_ratio)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=args.max_grad_norm,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0.0
            step_time = time.time() - step_t0
            elapsed_hours = (time.time() - start_time) / 3600
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9

            global_step += 1

            metrics = {
                "train/loss": loss_mean,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
                "train/step_time_sec": step_time,
                "train/elapsed_hours": elapsed_hours,
                "train/peak_vram_gb": peak_gb,
                "train/current_vram_gb": mem_gb,
                "epoch": epoch + 1,
            }

            metrics.update(finalize_category_metrics(step_category_metrics, prefix="train"))

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_loss, eval_category_metrics = evaluate_loss(model, val_examples, args, device)
                if eval_loss is not None:
                    metrics["eval/loss"] = eval_loss
                    metrics.update(eval_category_metrics)

            if use_wandb:
                wandb.log(metrics, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_training_checkpoint(model, tokenizer, optimizer, args, global_step, epoch)

            if global_step == 1 or global_step % args.logging_steps == 0:
                logger.info(
                    f"step {global_step}/{total_steps} | epoch={epoch + 1}/{args.epochs} | "
                    f"loss={loss_mean:.6f} | grad_norm={float(grad_norm):.4f} | "
                    f"lr={lr:.2e} | step_time={step_time:.1f}s | "
                    f"peak={peak_gb:.1f}GB | mem={mem_gb:.1f}GB"
                )

    logger.info(f"Training complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    save_training_checkpoint(model, tokenizer, optimizer, args, global_step, args.epochs - 1)
   

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
