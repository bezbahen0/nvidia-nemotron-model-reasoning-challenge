import re
import argparse
import hashlib
import pandas as pd

from transformers import AutoTokenizer

from src.augmentation.equations.cryptarithm_augment import CryptarithmAugmentGenerator
from src.augmentation.equations.cryptarithm_task_generator import (
    CryptarithmReplayTaskGenerator,
    CryptarithmTaskGeneratorConfig,
)
from src.augmentation.equations.numeral_equations_augment import NumeralEquationAugmentGenerator
from src.augmentation.bit_manipulation import BitMatchingAugmentGenerator
from src.augmentation.encryption import EncryptionTaskGenerator
from src.metric import verify
from src.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_generated_cot_tokens", type=int, default=7800)

    # Number of new validated full cryptarithm prompts to synthesize.
    # These rows get source=generated. Their extracted subtasks also get source=generated.
    parser.add_argument("--cryptarithm_generated_n", type=int, default=300)
    parser.add_argument("--cryptarithm_generated_timeout", type=float, default=10.0)
    parser.add_argument("--cryptarithm_generated_max_attempts", type=int, default=100)
    parser.add_argument("--cryptarithm_generated_max_cot_chars", type=int, default=0)
    parser.add_argument("--cryptarithm_generated_workers", type=int, default=24)

    parser.add_argument("--cryptarithm_generated_no_parallel", action="store_true")
    parser.add_argument("--cryptarithm_generated_progress_bar", action="store_true")
    
    return parser.parse_args()

def make_generated_id(row: pd.Series) -> str:
    raw = "\n".join([
        str(row["source"]),
        str(row["label"]),
        str(row["prompt"]),
        str(row["answer"]),
        str(row["computed_answer"]),
        str(row["generated_cot"]),
    ])

    digest = hashlib.blake2b(
        raw.encode("utf-8"),
        digest_size=12,
    ).hexdigest()

    return f"gen_{digest}"


def with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    if len(out):
        out["source"] = source
    elif "source" not in out.columns:
        out["source"] = pd.Series(dtype="object")
    return out


def normalize_generated_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["prompt", "answer", "label", "generated_cot", "computed_answer", "source"]
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = None
    return out[required]


def main():
    args = parse_args()

    logger.info(f"Load data from {args.data_path}...")
    data = pd.read_csv(args.data_path)

    # Remove all nan
    data = data[~data.computed_answer.isna()]
    data["source"] = len(data) * ["solver"]

    # skip all answer that not verified
    data = data[data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)]

    logger.info(f"Datset countes: \n{data.label.value_counts()}")

    # bit manipulation
    bit_matching_generator = BitMatchingAugmentGenerator(seed=args.seed)

    bit_mp_gen_dataset = bit_matching_generator.generate_dataset(
        source_data=data[data.label == "bit manipulation"].copy(),
        sample_frac=1.0,
        only_solver_correct=False,
    )
    bit_mp_gen_dataset = with_source(bit_mp_gen_dataset, "solver")

    logger.info("\nBit matching augmenter:")
    logger.info(bit_mp_gen_dataset.columns.tolist())
    logger.info(bit_mp_gen_dataset.task_mode.value_counts(normalize=True))
    logger.info(f"Generated rows: {len(bit_mp_gen_dataset)}")
    logger.info(
        f"Source solver correct rate: "
        f"{bit_mp_gen_dataset['source_solver_correct'].mean() if len(bit_mp_gen_dataset) else 0.0}"
    )
    
    multipliers = {
        "encryption": 1.0,
    }

    # Cryptarithm
    # Instead of generating random full cryptarithm tasks, derive local Alice-style
    # subtasks from existing replay CoTs: rule filtering, candidate checks,
    # domain propagation, rule verification, and target application.
    cryptarithm_augment_generator = CryptarithmAugmentGenerator(seed=args.seed)

    cryptarithm_aug_dataset = cryptarithm_augment_generator.generate_dataset(
        source_data=data[data.label == "cryptarithm"].copy(),
        sample_frac=0.5,
        only_solver_correct=False,
    )
    cryptarithm_aug_dataset = with_source(cryptarithm_aug_dataset, "solver")
    logger.info("\nCryptarithm augmenter:")
    logger.info(cryptarithm_aug_dataset.columns.tolist())
    if len(cryptarithm_aug_dataset) and "task_mode" in cryptarithm_aug_dataset.columns:
        logger.info(cryptarithm_aug_dataset.task_mode.value_counts(normalize=True))
    logger.info(f"Generated rows: {len(cryptarithm_aug_dataset)}")
    logger.info(
        f"Source solver correct rate: "
        f"{cryptarithm_aug_dataset['source_solver_correct'].mean() if len(cryptarithm_aug_dataset) and 'source_solver_correct' in cryptarithm_aug_dataset.columns else 0.0}"
    )
    logger.info(
        f'Accuracy: '
        f'{cryptarithm_aug_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean() if len(cryptarithm_aug_dataset) else 0.0}'
    )

    # New full cryptarithm tasks synthesized from a profile fitted to real solver rows.
    # The full generated tasks have source=generated.  Their extracted subtasks also
    # have source=generated.  The generator itself never creates subtasks; subtasks
    # are always produced here by CryptarithmAugmentGenerator.
    cryptarithm_generated_full_dataset = pd.DataFrame()
    cryptarithm_generated_aug_dataset = pd.DataFrame()
    if args.cryptarithm_generated_n > 0:
        crypt_config = CryptarithmTaskGeneratorConfig(
            solver_timeout_seconds=args.cryptarithm_generated_timeout,
            max_attempts_per_row=args.cryptarithm_generated_max_attempts,
            require_solver_success=True,
            max_cot_chars=(args.cryptarithm_generated_max_cot_chars or None),
        )
        cryptarithm_task_generator = CryptarithmReplayTaskGenerator.from_source_data(
            source_data=data[data.label == "cryptarithm"].copy(),
            seed=args.seed,
            config=crypt_config,
        )
        logger.info(
            f"Generating {args.cryptarithm_generated_n} full cryptarithm tasks "
            f"with nb_workers={args.cryptarithm_generated_workers}, "
            f"use_parallel={not args.cryptarithm_generated_no_parallel}."
        )
        cryptarithm_generated_full_dataset = cryptarithm_task_generator.generate_dataset(
            num_samples=args.cryptarithm_generated_n,
            progress_bar=args.cryptarithm_generated_progress_bar,
            nb_workers=args.cryptarithm_generated_workers,
            use_parallel=not args.cryptarithm_generated_no_parallel,
        )
        cryptarithm_generated_full_dataset = with_source(cryptarithm_generated_full_dataset, "generated")
        logger.info("\nGenerated full cryptarithm tasks:")
        logger.info(cryptarithm_generated_full_dataset.columns.tolist())
        if len(cryptarithm_generated_full_dataset) and "task_mode" in cryptarithm_generated_full_dataset.columns:
            logger.info(cryptarithm_generated_full_dataset.task_mode.value_counts(normalize=True))
        logger.info(f"Generated rows: {len(cryptarithm_generated_full_dataset)}")
        logger.info(
            f'Accuracy: '
            f'{cryptarithm_generated_full_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean() if len(cryptarithm_generated_full_dataset) else 0.0}'
        )

        cryptarithm_generated_aug_dataset = cryptarithm_augment_generator.generate_dataset(
            source_data=cryptarithm_generated_full_dataset.copy(),
            sample_frac=1.0,
            only_solver_correct=False,
        )
        cryptarithm_generated_aug_dataset = with_source(cryptarithm_generated_aug_dataset, "generated")
        logger.info("\nCryptarithm augmenter on generated tasks:")
        logger.info(cryptarithm_generated_aug_dataset.columns.tolist())
        if len(cryptarithm_generated_aug_dataset) and "task_mode" in cryptarithm_generated_aug_dataset.columns:
            logger.info(cryptarithm_generated_aug_dataset.task_mode.value_counts(normalize=True))
        logger.info(f"Generated rows: {len(cryptarithm_generated_aug_dataset)}")
        logger.info(
            f'Accuracy: '
            f'{cryptarithm_generated_aug_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean() if len(cryptarithm_generated_aug_dataset) else 0.0}'
        )

    # Numeral equations
    # Instead of generating new full AST brute-force tasks, derive small Alice-style
    # subtasks from existing solver CoTs: family matching and rule application.
    numeral_equations_augment_generator = NumeralEquationAugmentGenerator(seed=args.seed)

    numeral_equations_aug_dataset = numeral_equations_augment_generator.generate_dataset(
        source_data=data[data.label == "numeral equations"].copy(),
        sample_frac=1.0,
        only_solver_correct=False,
    )
    numeral_equations_aug_dataset = with_source(numeral_equations_aug_dataset, "solver")
    logger.info("\nNumeral equations augmenter:")
    logger.info(numeral_equations_aug_dataset.columns.tolist())
    logger.info(numeral_equations_aug_dataset.task_mode.value_counts(normalize=True))
    logger.info(f"Generated rows: {len(numeral_equations_aug_dataset)}")
    logger.info(
        f"Source solver correct rate: "
        f"{numeral_equations_aug_dataset['source_solver_correct'].mean() if len(numeral_equations_aug_dataset) else 0.0}"
    )
    logger.info(
        f'Accuracy: '
        f'{numeral_equations_aug_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean() if len(numeral_equations_aug_dataset) else 0.0}'
    )


    # Encryption

    global_vocab = set()
    for prompt in data[data.label == "encryption"]['prompt']:
        lines = [l.strip() for l in prompt.lower().splitlines() if "->" in l]
        for line in lines:
            plain = line.split("->", 1)[1]
            words = re.sub(r"[^a-z\s]", "", plain).split()
            global_vocab.update(words)

    for ans in data[data.label == "encryption"]['answer']:
        if isinstance(ans, str):
             global_vocab.update(re.sub(r"[^a-z\s]", "", ans.lower()).split())

    encryption_generator = EncryptionTaskGenerator(vocabulary=global_vocab, seed=args.seed)

    encryption_gen_dataset = encryption_generator.generate_dataset(
        int(len(data[data.label == "encryption"]) *  multipliers["encryption"])
    )
    encryption_gen_dataset = with_source(encryption_gen_dataset, "generated")


    generated_parts = [
        cryptarithm_aug_dataset,                 # subtasks from real solver cryptarithms
        cryptarithm_generated_full_dataset,      # new synthetic full cryptarithms
        cryptarithm_generated_aug_dataset,       # subtasks from synthetic full cryptarithms
        numeral_equations_aug_dataset,           # subtasks from real solver numeral equations
        encryption_gen_dataset,                  # synthetic encryption tasks
        bit_mp_gen_dataset,                      # subtasks from real solver bit-manipulation tasks
    ]
    generated_parts = [normalize_generated_columns(df) for df in generated_parts if df is not None and len(df)]
    data_gen = pd.concat(generated_parts, ignore_index=True) if generated_parts else pd.DataFrame(
        columns=["prompt", "answer", "label", "generated_cot", "computed_answer", "source"]
    )
    data_gen["id"] = data_gen.apply(make_generated_id, axis=1) if len(data_gen) else []

    data = pd.concat([data_gen, data])
     
    data = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    

    data = data[data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)]
    data["is_correct"] = data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)
    data["is_correct_rounded"] = data["is_correct"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    data["token_len"] = data["generated_cot"].fillna("").astype(str).apply(lambda text: len(tokenizer.encode(text)))

    before_token_filter = len(data)
    too_long_mask = data["token_len"] > args.max_generated_cot_tokens
    removed_by_token_filter = int(too_long_mask.sum())
    if removed_by_token_filter:
        logger.info(
            f"Token length filter: removing {removed_by_token_filter} rows with "
            f"generated_cot token_len > {args.max_generated_cot_tokens}; "
            f"keeping {before_token_filter - removed_by_token_filter} of {before_token_filter}."
        )
        logger.info(
            "Removed rows by label/source:\n"
            f"{data.loc[too_long_mask].groupby(['label', 'source']).size().sort_values(ascending=False)}"
        )
    else:
        logger.info(
            f"Token length filter: removed 0 rows with generated_cot token_len > "
            f"{args.max_generated_cot_tokens}; keeping {before_token_filter}."
        )

    data = data.loc[~too_long_mask].copy()

    logger.info(f"Final dataset len after token filter: \n{data.label.value_counts()}")
    logger.info(f"Final dataset source counts after token filter: \n{data.source.value_counts() if 'source' in data.columns else 'no source column'}")
    logger.info(f"\nDataset token len after token filter:\n{data.groupby('label').token_len.describe()}")

    data["computed"] = data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)

    logger.info(f'Compilted accuracy: \n{data.groupby("label").computed.value_counts()}')

    data.to_csv(args.output_path, index=False)

    
if __name__ == "__main__":
    main()