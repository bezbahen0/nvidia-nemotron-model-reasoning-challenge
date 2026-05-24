import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.log import logger


SECTION_NAMES = [
    "cot",
    "final_answer",
    "boxed_answer",
    "format",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--telemetry_dir", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--use_all_records", action="store_true")
    parser.add_argument("--top_hard_examples", type=int, default=1000)

    return parser.parse_args()


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


def load_telemetry(telemetry_dir):
    telemetry_path = Path(telemetry_dir)
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

    return pd.DataFrame(records), files


def load_train(train_path):
    df = pd.read_csv(train_path)
    df["_source_row_index"] = np.arange(len(df))

    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    return df


def to_numeric(df, columns):
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def normalize_telemetry(df):
    if "id" not in df.columns:
        df["id"] = df["source_row_index"].astype(str)

    if "source" not in df.columns:
        df["source"] = "unknown"

    if "label" not in df.columns:
        df["label"] = "unknown"

    df["id"] = df["id"].astype(str)
    df["source"] = df["source"].astype(str)
    df["label"] = df["label"].astype(str)

    numeric_columns = [
        "step",
        "epoch",
        "checkpoint_bucket",
        "source_row_index",
        "num_loss_tokens",
        "total_loss",
        "mean_nll",
        "min_logprob",
        "p05_logprob",
        "hard_token_ratio",
        "near_zero_token_ratio",
    ]

    for section_name in SECTION_NAMES:
        numeric_columns.extend([
            f"{section_name}_num_tokens",
            f"{section_name}_mean_nll",
            f"{section_name}_min_logprob",
            f"{section_name}_p05_logprob",
            f"{section_name}_hard_token_ratio",
            f"{section_name}_near_zero_token_ratio",
        ])

    df = to_numeric(df, numeric_columns)

    if "top_bad_tokens" not in df.columns:
        df["top_bad_tokens"] = [[] for _ in range(len(df))]

    df["top_bad_tokens"] = df["top_bad_tokens"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return df


def keep_latest_per_id(df):
    sort_columns = []

    for column in ["checkpoint_bucket", "step", "epoch"]:
        if column in df.columns:
            sort_columns.append(column)

    if len(sort_columns) == 0:
        return df.drop_duplicates(subset=["id"], keep="last").copy()

    latest_df = df.sort_values(sort_columns, ascending=True)
    latest_df = latest_df.drop_duplicates(subset=["id"], keep="last").copy()

    return latest_df


def merge_train_columns(df, train_df):
    columns_to_take = [
        "id",
        "prompt",
        "answer",
        "prompt_eda",
        "generated_cot",
        "computed_answer",
        "is_correct",
    ]

    existing_columns = [column for column in columns_to_take if column in train_df.columns]

    if "id" in train_df.columns:
        train_by_id = train_df[existing_columns].copy()
        train_by_id["id"] = train_by_id["id"].astype(str)

        df = df.merge(
            train_by_id,
            on="id",
            how="left",
            suffixes=("", "_train"),
        )

    fallback_columns = [
        "prompt",
        "answer",
        "prompt_eda",
        "generated_cot",
        "computed_answer",
        "is_correct",
    ]

    for column in fallback_columns:
        if column not in df.columns:
            df[column] = np.nan

        if column in train_df.columns and "source_row_index" in df.columns:
            value_by_row = train_df.set_index("_source_row_index")[column]
            fallback_values = df["source_row_index"].map(value_by_row)
            df[column] = df[column].where(df[column].notna(), fallback_values)

    return df


def add_scores(df):
    neg_p05 = (-df["p05_logprob"]).clip(lower=0).fillna(0)
    mean_nll = df["mean_nll"].fillna(0)
    hard_ratio = df["hard_token_ratio"].fillna(0)
    boxed_mean_nll = df["boxed_answer_mean_nll"].fillna(0)
    boxed_hard_ratio = df["boxed_answer_hard_token_ratio"].fillna(0)
    cot_mean_nll = df["cot_mean_nll"].fillna(0)
    cot_hard_ratio = df["cot_hard_token_ratio"].fillna(0)
    format_mean_nll = df["format_mean_nll"].fillna(0)
    format_hard_ratio = df["format_hard_token_ratio"].fillna(0)

    df["difficulty_score"] = (
        0.50 * mean_nll
        + 0.30 * neg_p05
        + 0.20 * hard_ratio
        + 0.20 * boxed_mean_nll
    )

    df["answer_score"] = boxed_mean_nll + boxed_hard_ratio
    df["cot_score"] = cot_mean_nll + cot_hard_ratio
    df["format_score"] = format_mean_nll + format_hard_ratio

    df["is_easy"] = (
        (df["mean_nll"] < 0.02)
        & (df["near_zero_token_ratio"] > 0.97)
        & (df["hard_token_ratio"] < 0.01)
        & ((df["boxed_answer_mean_nll"].isna()) | (df["boxed_answer_mean_nll"] < 0.02))
    )

    df["is_hard"] = (
        (df["mean_nll"] >= 0.08)
        | (df["p05_logprob"] < -0.50)
        | (df["hard_token_ratio"] >= 0.02)
        | (df["boxed_answer_mean_nll"] >= 0.08)
    )

    df["is_very_hard"] = (
        (df["mean_nll"] >= 0.18)
        | (df["p05_logprob"] < -1.00)
        | (df["hard_token_ratio"] >= 0.08)
        | (df["boxed_answer_mean_nll"] >= 0.15)
    )

    return df


def q95(series):
    return series.quantile(0.95)


def build_group_report(df, group_columns):
    report = df.groupby(group_columns, dropna=False).agg(
        count=("id", "count"),
        unique_ids=("id", "nunique"),

        mean_nll=("mean_nll", "mean"),
        p50_mean_nll=("mean_nll", "median"),
        p95_mean_nll=("mean_nll", q95),

        p05_logprob_mean=("p05_logprob", "mean"),
        hard_token_ratio=("hard_token_ratio", "mean"),
        near_zero_token_ratio=("near_zero_token_ratio", "mean"),

        cot_mean_nll=("cot_mean_nll", "mean"),
        final_answer_mean_nll=("final_answer_mean_nll", "mean"),
        boxed_answer_mean_nll=("boxed_answer_mean_nll", "mean"),
        format_mean_nll=("format_mean_nll", "mean"),

        difficulty_score=("difficulty_score", "mean"),
        answer_score=("answer_score", "mean"),
        cot_score=("cot_score", "mean"),
        format_score=("format_score", "mean"),

        hard_examples_ratio=("is_hard", "mean"),
        very_hard_examples_ratio=("is_very_hard", "mean"),
        easy_examples_ratio=("is_easy", "mean"),
    ).reset_index()

    report = report.sort_values(
        by=["difficulty_score", "hard_examples_ratio", "mean_nll"],
        ascending=[False, False, False],
    )

    return report


def short_bad_tokens(value):
    if not isinstance(value, list):
        return ""

    result = []

    for token in value[:5]:
        if not isinstance(token, dict):
            continue

        section = str(token.get("section", "unknown"))
        token_text = str(token.get("token_text", ""))
        logprob = token.get("logprob", None)
        context = str(token.get("context", ""))

        if logprob is None:
            result.append(f"{section}: {token_text} | {context}")
        else:
            result.append(f"{section}: {token_text} ({float(logprob):.4f}) | {context}")

    return " || ".join(result)


def build_hard_examples(df, top_n):
    df = df.copy()
    df["top_bad_tokens_short"] = df["top_bad_tokens"].apply(short_bad_tokens)

    columns = [
        "id",
        "source",
        "label",
        "source_row_index",
        "checkpoint_bucket",
        "step",

        "difficulty_score",
        "answer_score",
        "cot_score",
        "format_score",

        "mean_nll",
        "p05_logprob",
        "min_logprob",
        "hard_token_ratio",
        "near_zero_token_ratio",

        "cot_mean_nll",
        "final_answer_mean_nll",
        "boxed_answer_mean_nll",
        "format_mean_nll",

        "top_bad_tokens_short",

        "prompt",
        "answer",
        "computed_answer",
        "is_correct",
    ]

    existing_columns = [column for column in columns if column in df.columns]

    hard_df = df.sort_values(
        by=["difficulty_score", "mean_nll"],
        ascending=[False, False],
    )

    if top_n > 0:
        hard_df = hard_df.head(top_n)

    return hard_df[existing_columns]


def build_bad_tokens_report(df):
    rows = []

    for _, record in df.iterrows():
        top_bad_tokens = record.get("top_bad_tokens", [])

        if not isinstance(top_bad_tokens, list):
            continue

        for token in top_bad_tokens:
            if not isinstance(token, dict):
                continue

            rows.append({
                "id": record.get("id"),
                "source": record.get("source"),
                "label": record.get("label"),
                "source_row_index": record.get("source_row_index"),
                "checkpoint_bucket": record.get("checkpoint_bucket"),
                "step": record.get("step"),

                "section": token.get("section", "unknown"),
                "token_text": token.get("token_text", ""),
                "token_id": token.get("token_id", None),
                "logprob": token.get("logprob", None),
                "nll": token.get("nll", None),
                "context": token.get("context", ""),

                "mean_nll": record.get("mean_nll"),
                "boxed_answer_mean_nll": record.get("boxed_answer_mean_nll"),
                "cot_mean_nll": record.get("cot_mean_nll"),
                "format_mean_nll": record.get("format_mean_nll"),
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "source",
            "label",
            "section",
            "token_text",
            "count",
            "mean_logprob",
            "min_logprob",
            "mean_nll",
            "sample_context_1",
            "sample_context_2",
            "sample_context_3",
        ])

    token_df = pd.DataFrame(rows)
    token_df["logprob"] = pd.to_numeric(token_df["logprob"], errors="coerce")
    token_df["nll"] = pd.to_numeric(token_df["nll"], errors="coerce")

    def sample_context(series, index):
        values = [str(value) for value in series.dropna().tolist() if str(value).strip()]

        if index >= len(values):
            return ""

        return values[index]

    grouped_rows = []

    for keys, group in token_df.groupby(["source", "label", "section", "token_text"], dropna=False):
        source, label, section, token_text = keys

        grouped_rows.append({
            "source": source,
            "label": label,
            "section": section,
            "token_text": token_text,
            "count": int(len(group)),
            "mean_logprob": float(group["logprob"].mean()) if group["logprob"].notna().any() else np.nan,
            "min_logprob": float(group["logprob"].min()) if group["logprob"].notna().any() else np.nan,
            "mean_nll": float(group["nll"].mean()) if group["nll"].notna().any() else np.nan,
            "sample_context_1": sample_context(group["context"], 0),
            "sample_context_2": sample_context(group["context"], 1),
            "sample_context_3": sample_context(group["context"], 2),
        })

    bad_tokens_report = pd.DataFrame(grouped_rows)

    bad_tokens_report = bad_tokens_report.sort_values(
        by=["count", "mean_nll", "min_logprob"],
        ascending=[False, False, True],
    )

    return bad_tokens_report


def build_input_summary(raw_df, latest_df, files):
    summary = {
        "telemetry_files_count": int(len(files)),
        "telemetry_files": [str(file) for file in files],
        "raw_records_count": int(len(raw_df)),
        "latest_records_count": int(len(latest_df)),
        "unique_ids_raw": int(raw_df["id"].nunique()),
        "unique_ids_latest": int(latest_df["id"].nunique()),
        "unknown_source_count": int((latest_df["source"] == "unknown").sum()),
        "unknown_label_count": int((latest_df["label"] == "unknown").sum()),
        "zero_loss_tokens_count": int((latest_df["num_loss_tokens"].fillna(0) == 0).sum()),
        "checkpoint_buckets": sorted([
            int(value)
            for value in latest_df["checkpoint_bucket"].dropna().unique().tolist()
        ]),
    }

    return summary


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_df, files = load_telemetry(args.telemetry_dir)
    raw_df = normalize_telemetry(raw_df)

    if args.use_all_records:
        df = raw_df.copy()
    else:
        df = keep_latest_per_id(raw_df)

    train_df = load_train(args.train_path)
    df = merge_train_columns(df, train_df)
    df = add_scores(df)

    by_label = build_group_report(df, ["label"])
    by_source_label = build_group_report(df, ["source", "label"])
    hard_examples = build_hard_examples(df, args.top_hard_examples)
    bad_tokens = build_bad_tokens_report(df)
    summary = build_input_summary(raw_df, df, files)

    summary_path = os.path.join(args.output_dir, "00_input_summary.json")
    by_label_path = os.path.join(args.output_dir, "01_by_label.csv")
    by_source_label_path = os.path.join(args.output_dir, "02_by_source_label.csv")
    hard_examples_path = os.path.join(args.output_dir, "03_hard_examples.csv")
    bad_tokens_path = os.path.join(args.output_dir, "04_bad_tokens.csv")

    write_json(summary_path, summary)
    by_label.to_csv(by_label_path, index=False)
    by_source_label.to_csv(by_source_label_path, index=False)
    hard_examples.to_csv(hard_examples_path, index=False)
    bad_tokens.to_csv(bad_tokens_path, index=False)

    logger.info(f"Saved: {summary_path}")
    logger.info(f"Saved: {by_label_path}")
    logger.info(f"Saved: {by_source_label_path}")
    logger.info(f"Saved: {hard_examples_path}")
    logger.info(f"Saved: {bad_tokens_path}")


if __name__ == "__main__":
    main()
