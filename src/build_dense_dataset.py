import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--train_telemetry_dir", type=str, default=None)
    parser.add_argument("--eval_telemetry_dir", type=str, default=None)

    parser.add_argument("--label_weights", type=str, default="{}")
    parser.add_argument("--source_weights", type=str, default="{}")
    parser.add_argument("--source_label_weights", type=str, default="{}")

    parser.add_argument("--base_weight", type=float, default=1.0)
    parser.add_argument("--max_repeat", type=int, default=8)
    parser.add_argument("--min_repeat", type=int, default=0)

    parser.add_argument("--train_hard_multiplier", type=float, default=1.0)
    parser.add_argument("--train_very_hard_multiplier", type=float, default=1.0)
    parser.add_argument("--train_easy_multiplier", type=float, default=1.0)

    parser.add_argument("--eval_hard_multiplier", type=float, default=1.0)
    parser.add_argument("--eval_very_hard_multiplier", type=float, default=1.0)
    parser.add_argument("--eval_easy_multiplier", type=float, default=1.0)

    parser.add_argument("--hard_mean_nll", type=float, default=0.08)
    parser.add_argument("--hard_p05_logprob", type=float, default=-0.50)
    parser.add_argument("--hard_token_ratio", type=float, default=0.02)
    parser.add_argument("--hard_boxed_answer_mean_nll", type=float, default=0.08)
    parser.add_argument("--hard_boxed_answer_total_nll", type=float, default=0.80)

    parser.add_argument("--very_hard_mean_nll", type=float, default=0.18)
    parser.add_argument("--very_hard_p05_logprob", type=float, default=-1.00)
    parser.add_argument("--very_hard_token_ratio", type=float, default=0.08)
    parser.add_argument("--very_hard_boxed_answer_mean_nll", type=float, default=0.15)
    parser.add_argument("--very_hard_boxed_answer_total_nll", type=float, default=1.60)

    parser.add_argument("--easy_mean_nll", type=float, default=0.02)
    parser.add_argument("--easy_near_zero_token_ratio", type=float, default=0.97)
    parser.add_argument("--easy_hard_token_ratio", type=float, default=0.01)
    parser.add_argument("--easy_boxed_answer_mean_nll", type=float, default=0.02)
    parser.add_argument("--easy_boxed_answer_total_nll", type=float, default=0.20)

    parser.add_argument("--drop_incorrect", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--examples_report_path", type=str, default=None)

    return parser.parse_args()


def parse_json_dict(value, name):
    if value is None or str(value).strip() == "":
        return {}

    value = str(value).strip()

    if value.endswith(".json"):
        with open(value, "r", encoding="utf-8") as file:
            parsed = json.load(file)
    else:
        parsed = json.loads(value)

    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be json object")

    return parsed


def read_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad json in {path}, line {line_num}: {exc}") from exc

    return records


def load_telemetry(telemetry_dir, prefix):
    if telemetry_dir is None:
        return None

    telemetry_path = Path(telemetry_dir)

    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry dir does not exist: {telemetry_dir}")

    files = sorted(telemetry_path.rglob("*.jsonl"))

    if len(files) == 0:
        raise FileNotFoundError(f"No jsonl files found in {telemetry_dir}")

    records = []

    for file_path in files:
        file_records = read_jsonl(file_path)

        for record in file_records:
            record["telemetry_file"] = str(file_path)

        records.extend(file_records)

    if len(records) == 0:
        raise ValueError(f"No telemetry records found in {telemetry_dir}")

    df = pd.DataFrame(records)

    if "id" not in df.columns:
        raise ValueError(f"Telemetry records in {telemetry_dir} do not contain id")

    df["id"] = df["id"].astype(str)

    numeric_columns = [
        "checkpoint_bucket",
        "step",
        "epoch",
        "mean_nll",
        "p05_logprob",
        "hard_token_ratio",
        "near_zero_token_ratio",
        "boxed_answer_num_tokens",
        "boxed_answer_mean_nll",
        "boxed_answer_min_logprob",
        "boxed_answer_p05_logprob",
        "boxed_answer_hard_token_ratio",
        "boxed_answer_near_zero_token_ratio",
        "cot_mean_nll",
        "cot_hard_token_ratio",
        "format_mean_nll",
        "format_hard_token_ratio",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

    sort_columns = [
        column
        for column in ["checkpoint_bucket", "step", "epoch"]
        if column in df.columns
    ]

    if len(sort_columns) > 0:
        df = df.sort_values(sort_columns, ascending=True)

    df = df.drop_duplicates(subset=["id"], keep="last").copy()

    df["boxed_answer_total_nll"] = (
        df["boxed_answer_mean_nll"].fillna(0)
        * df["boxed_answer_num_tokens"].fillna(0)
    )

    df["boxed_answer_total_nll_norm"] = (df["boxed_answer_total_nll"] / 8.0).clip(lower=0, upper=1)

    df["boxed_answer_sequence_prob_est"] = np.exp(
        -df["boxed_answer_total_nll"].clip(lower=0, upper=50)
    )

    # mean_nll is auxiliary here. Answer/sequence-level signals matter more for exact-match tasks.
    df[f"{prefix}_difficulty_score"] = (
        0.20 * df["mean_nll"].fillna(0)
        + 0.20 * (-df["p05_logprob"]).clip(lower=0).fillna(0)
        + 0.20 * df["hard_token_ratio"].fillna(0)
        + 0.25 * df["boxed_answer_mean_nll"].fillna(0)
        + 0.15 * df["boxed_answer_total_nll_norm"].fillna(0)
    )

    df[f"{prefix}_answer_score"] = (
        df["boxed_answer_mean_nll"].fillna(0)
        + df["boxed_answer_hard_token_ratio"].fillna(0)
        + df["boxed_answer_total_nll_norm"].fillna(0)
    )

    df[f"{prefix}_cot_score"] = (
        df["cot_mean_nll"].fillna(0)
        + df["cot_hard_token_ratio"].fillna(0)
    )

    df[f"{prefix}_format_score"] = (
        df["format_mean_nll"].fillna(0)
        + df["format_hard_token_ratio"].fillna(0)
    )

    rename_columns = {
        "checkpoint_bucket": f"{prefix}_checkpoint_bucket",
        "step": f"{prefix}_step",
        "mean_nll": f"{prefix}_mean_nll",
        "p05_logprob": f"{prefix}_p05_logprob",
        "hard_token_ratio": f"{prefix}_hard_token_ratio",
        "near_zero_token_ratio": f"{prefix}_near_zero_token_ratio",
        "boxed_answer_num_tokens": f"{prefix}_boxed_answer_num_tokens",
        "boxed_answer_mean_nll": f"{prefix}_boxed_answer_mean_nll",
        "boxed_answer_min_logprob": f"{prefix}_boxed_answer_min_logprob",
        "boxed_answer_p05_logprob": f"{prefix}_boxed_answer_p05_logprob",
        "boxed_answer_hard_token_ratio": f"{prefix}_boxed_answer_hard_token_ratio",
        "boxed_answer_total_nll": f"{prefix}_boxed_answer_total_nll",
        "boxed_answer_total_nll_norm": f"{prefix}_boxed_answer_total_nll_norm",
        "boxed_answer_sequence_prob_est": f"{prefix}_boxed_answer_sequence_prob_est",
        "cot_mean_nll": f"{prefix}_cot_mean_nll",
        "format_mean_nll": f"{prefix}_format_mean_nll",
    }

    columns = ["id"] + list(rename_columns.keys()) + [
        f"{prefix}_difficulty_score",
        f"{prefix}_answer_score",
        f"{prefix}_cot_score",
        f"{prefix}_format_score",
    ]

    df = df[columns].rename(columns=rename_columns)

    return df


def add_hard_flags(df, prefix, args):
    mean_nll = df[f"{prefix}_mean_nll"]
    p05_logprob = df[f"{prefix}_p05_logprob"]
    hard_token_ratio = df[f"{prefix}_hard_token_ratio"]
    near_zero_token_ratio = df[f"{prefix}_near_zero_token_ratio"]
    boxed_answer_mean_nll = df[f"{prefix}_boxed_answer_mean_nll"]
    boxed_answer_total_nll = df[f"{prefix}_boxed_answer_total_nll"]

    answer_hard = (
        (boxed_answer_mean_nll >= args.hard_boxed_answer_mean_nll)
        | (boxed_answer_total_nll >= args.hard_boxed_answer_total_nll)
    )

    answer_very_hard = (
        (boxed_answer_mean_nll >= args.very_hard_boxed_answer_mean_nll)
        | (boxed_answer_total_nll >= args.very_hard_boxed_answer_total_nll)
    )

    general_hard = (
        (mean_nll >= args.hard_mean_nll)
        | (p05_logprob < args.hard_p05_logprob)
        | (hard_token_ratio >= args.hard_token_ratio)
    )

    general_very_hard = (
        (mean_nll >= args.very_hard_mean_nll)
        | (p05_logprob < args.very_hard_p05_logprob)
        | (hard_token_ratio >= args.very_hard_token_ratio)
    )

    df[f"{prefix}_is_easy"] = (
        (mean_nll < args.easy_mean_nll)
        & (near_zero_token_ratio > args.easy_near_zero_token_ratio)
        & (hard_token_ratio < args.easy_hard_token_ratio)
        & (boxed_answer_mean_nll.isna() | (boxed_answer_mean_nll < args.easy_boxed_answer_mean_nll))
        & (boxed_answer_total_nll.isna() | (boxed_answer_total_nll < args.easy_boxed_answer_total_nll))
    )

    # mean_nll contributes via general_hard, but answer_hard can independently trigger repeats.
    df[f"{prefix}_is_hard"] = general_hard | answer_hard
    df[f"{prefix}_is_very_hard"] = general_very_hard | answer_very_hard
    df[f"{prefix}_is_answer_hard"] = answer_hard
    df[f"{prefix}_is_answer_very_hard"] = answer_very_hard

    return df


def label_weight(row, label_weights, source_weights, source_label_weights):
    label = str(row.get("label", ""))
    source = str(row.get("source", ""))

    weight = 1.0
    weight *= float(label_weights.get(label, 1.0))
    weight *= float(source_weights.get(source, 1.0))
    weight *= float(source_label_weights.get(f"{source}|{label}", 1.0))

    return weight


def apply_telemetry_multiplier(row, prefix, hard_multiplier, very_hard_multiplier, easy_multiplier):
    if f"{prefix}_is_answer_very_hard" in row and bool(row[f"{prefix}_is_answer_very_hard"]):
        return very_hard_multiplier

    if f"{prefix}_is_very_hard" in row and bool(row[f"{prefix}_is_very_hard"]):
        return very_hard_multiplier

    if f"{prefix}_is_answer_hard" in row and bool(row[f"{prefix}_is_answer_hard"]):
        return hard_multiplier

    if f"{prefix}_is_hard" in row and bool(row[f"{prefix}_is_hard"]):
        return hard_multiplier

    if f"{prefix}_is_easy" in row and bool(row[f"{prefix}_is_easy"]):
        return easy_multiplier

    return 1.0


def stochastic_round(value, rng):
    if value <= 0:
        return 0

    base = int(np.floor(value))
    frac = value - base

    if rng.random() < frac:
        return base + 1

    return base


def safe_bool(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in ("true", "1", "yes"):
        return True

    if text in ("false", "0", "no"):
        return False

    return None


def build_report(df, dense_df, args):
    report = {
        "input_rows": int(len(df)),
        "output_rows": int(len(dense_df)),
        "base_weight": args.base_weight,
        "max_repeat": args.max_repeat,
        "min_repeat": args.min_repeat,
        "label_counts_before": df["label"].value_counts(dropna=False).to_dict() if "label" in df.columns else {},
        "label_counts_after": dense_df["label"].value_counts(dropna=False).to_dict() if "label" in dense_df.columns else {},
        "source_counts_before": df["source"].value_counts(dropna=False).to_dict() if "source" in df.columns else {},
        "source_counts_after": dense_df["source"].value_counts(dropna=False).to_dict() if "source" in dense_df.columns else {},
        "repeat_counts": df["dense_repeat"].value_counts(dropna=False).sort_index().to_dict(),
        "total_repeat_sum": int(df["dense_repeat"].sum()),
    }

    if "train_is_hard" in df.columns:
        report["train_hard_rows"] = int(df["train_is_hard"].fillna(False).sum())
        report["train_very_hard_rows"] = int(df["train_is_very_hard"].fillna(False).sum())
        report["train_easy_rows"] = int(df["train_is_easy"].fillna(False).sum())
        report["train_answer_hard_rows"] = int(df["train_is_answer_hard"].fillna(False).sum())

    if "eval_is_hard" in df.columns:
        report["eval_hard_rows"] = int(df["eval_is_hard"].fillna(False).sum())
        report["eval_very_hard_rows"] = int(df["eval_is_very_hard"].fillna(False).sum())
        report["eval_easy_rows"] = int(df["eval_is_easy"].fillna(False).sum())
        report["eval_answer_hard_rows"] = int(df["eval_is_answer_hard"].fillna(False).sum())

    return report


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    label_weights = parse_json_dict(args.label_weights, "label_weights")
    source_weights = parse_json_dict(args.source_weights, "source_weights")
    source_label_weights = parse_json_dict(args.source_label_weights, "source_label_weights")

    df = pd.read_csv(args.input_path)

    if "id" not in df.columns:
        raise ValueError("input_path must contain id column")

    df["id"] = df["id"].astype(str)

    if args.drop_incorrect and "is_correct" in df.columns:
        is_correct_values = df["is_correct"].apply(safe_bool)
        df = df[(is_correct_values != False)].copy()

    train_telemetry = load_telemetry(args.train_telemetry_dir, "train")
    eval_telemetry = load_telemetry(args.eval_telemetry_dir, "eval")

    if train_telemetry is not None:
        df = df.merge(train_telemetry, on="id", how="left")
        df = add_hard_flags(df, "train", args)

    if eval_telemetry is not None:
        df = df.merge(eval_telemetry, on="id", how="left")
        df = add_hard_flags(df, "eval", args)

    df["manual_weight"] = df.apply(
        lambda row: label_weight(row, label_weights, source_weights, source_label_weights),
        axis=1,
    )

    df["dense_weight"] = args.base_weight * df["manual_weight"]

    if train_telemetry is not None:
        df["dense_weight"] *= df.apply(
            lambda row: apply_telemetry_multiplier(
                row=row,
                prefix="train",
                hard_multiplier=args.train_hard_multiplier,
                very_hard_multiplier=args.train_very_hard_multiplier,
                easy_multiplier=args.train_easy_multiplier,
            ),
            axis=1,
        )

    if eval_telemetry is not None:
        df["dense_weight"] *= df.apply(
            lambda row: apply_telemetry_multiplier(
                row=row,
                prefix="eval",
                hard_multiplier=args.eval_hard_multiplier,
                very_hard_multiplier=args.eval_very_hard_multiplier,
                easy_multiplier=args.eval_easy_multiplier,
            ),
            axis=1,
        )

    repeats = []

    for value in df["dense_weight"].fillna(0).tolist():
        repeat = stochastic_round(float(value), rng)
        repeat = max(args.min_repeat, repeat)
        repeat = min(args.max_repeat, repeat)
        repeats.append(int(repeat))

    df["dense_repeat"] = repeats

    output_indexes = []

    for row_index, repeat in enumerate(df["dense_repeat"].tolist()):
        output_indexes.extend([row_index] * int(repeat))

    dense_df = df.iloc[output_indexes].copy()

    helper_columns = [
        column
        for column in dense_df.columns
        if column.startswith("train_")
        or column.startswith("eval_")
        or column in (
            "manual_weight",
            "dense_weight",
            "dense_repeat",
        )
    ]

    examples_report = df.copy()

    dense_df = dense_df.drop(columns=helper_columns, errors="ignore")

    if args.shuffle and len(dense_df) > 0:
        dense_df = dense_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    dense_df.to_csv(args.output_path, index=False)

    report_path = args.report_path

    if report_path is None:
        root, _ = os.path.splitext(args.output_path)
        report_path = f"{root}_report.json"

    examples_report_path = args.examples_report_path

    if examples_report_path is None:
        root, _ = os.path.splitext(args.output_path)
        examples_report_path = f"{root}_examples_report.csv"

    report = build_report(examples_report, dense_df, args)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    examples_report.to_csv(examples_report_path, index=False)

    print(f"Saved dense dataset: {args.output_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved examples report: {examples_report_path}")


if __name__ == "__main__":
    main()
