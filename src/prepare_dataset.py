from __future__ import annotations

import re
import argparse
import hashlib


import pandas as pd

from transformers import AutoTokenizer

from src.augmentation.equations.numeral_equations_augment import NumeralEquationAugmentGenerator
from src.augmentation.equations.cryptarithm_task_generator import (
    CryptarithmMappingSubtaskGenerator,
    MappingSubtaskConfig,
)

from src.augmentation.bit_manipulation import BitMatchingAugmentGenerator, BitMatchingAugmentConfig
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

    # Remove rows where the base solver did not emit any answer at all.
    data = data[~data.computed_answer.isna()].copy()
    data["source"] = len(data) * ["solver"]


    # skip all answer that not verified for the main solver-generated rows
    data = data[data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)].copy()

    logger.info(f"Datset countes: \n{data.label.value_counts()}")

    

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
        int(len(data[data.label == "encryption"]) *  1.0)
    )
    encryption_gen_dataset = with_source(encryption_gen_dataset, "generated")

    # Cryptarithm mapping subtasks
    # Derive focused subtasks from the v3 cryptarithm training CoTs.  These are
    # not new random cryptarithms: they slice a solved trace into rule-selection,
    # projection/domain-update, build-mapping, concat-direct, and target-application
    # tasks.  Fixed-concat target traces intentionally skip digit-map construction.
    cryptarithm_mapping_augment_generator = CryptarithmMappingSubtaskGenerator(
        config=MappingSubtaskConfig(
            include_build_mapping=True,
            include_rule_selection=True,
            include_projection_step=True,
            include_domain_update=True,
            include_target_application=True,
            include_concat_direct=True,
            include_rule_legend=True,
            skip_fixed_concat_build_mapping=True,
        )
    )
    cryptarithm_mapping_aug_dataset = cryptarithm_mapping_augment_generator.generate_dataset(
        source_data=data[data.label == "cryptarithm"].copy()
    )
    cryptarithm_mapping_aug_dataset = with_source(cryptarithm_mapping_aug_dataset, "solver")

    logger.info("\nCryptarithm mapping subtasks augmenter:")
    logger.info(cryptarithm_mapping_aug_dataset.columns.tolist())
    if len(cryptarithm_mapping_aug_dataset) and "task_mode" in cryptarithm_mapping_aug_dataset.columns:
        logger.info(cryptarithm_mapping_aug_dataset.task_mode.value_counts(dropna=False))
        logger.info(cryptarithm_mapping_aug_dataset.label.value_counts(dropna=False))
    logger.info(f"Generated rows: {len(cryptarithm_mapping_aug_dataset)}")
    logger.info(
        f'Accuracy: '
        f'{cryptarithm_mapping_aug_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean() if len(cryptarithm_mapping_aug_dataset) else 0.0}'
    )

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


    generated_parts = [
        numeral_equations_aug_dataset,           # subtasks from real solver numeral equations
        encryption_gen_dataset,                  # synthetic encryption tasks
        #encryption_cot_aug_dataset,
        bit_mp_gen_dataset,                      # subtasks from real solver bit-manipulation tasks
        cryptarithm_mapping_aug_dataset,         # mapping-focused subtasks from real cryptarithm CoTs
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
