import re
import argparse
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=False)

from src.solvers.bit_manipulation import BitManipulationSolver
from src.solvers.equations_base import BaseEquationSolver
from src.solvers.gravitational import GravitationalSolver
from src.solvers.numeral_system import NumeralSystemSolver
from src.solvers.unit_conversion import UnitConversionSolver
from src.solvers.encryption import EncryptionSolver
from src.metric import verify
from src.log import logger


def parse_args():
    parser = argparse.ArgumentParser(description="NVIDIA Nemotron alrgorithmic task solver")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()


def solver(task_df, solver_obj):
    task_df["generated_cot"] = task_df["prompt"].parallel_apply(solver_obj.generate_cot)

    task_df["computed_answer"] = task_df["generated_cot"].parallel_apply(solver_obj.extract_answer)

    task_df["is_correct"] = (
        task_df["computed_answer"].astype(str).str.strip()
        == task_df["answer"].astype(str).str.strip()
    )

    return task_df


def main():
    args = parse_args()

    data = pd.read_csv(args.train_path)

    data["prompt_eda"] = data.prompt.str.split(".").apply(lambda x: x[0])

    task_classes = {
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers": "bit manipulation",
        "In Alice's Wonderland, secret encryption rules are used on text": "encryption",
        "In Alice's Wonderland, numbers are secretly converted into a different numeral system": "conversion to diff numeral system",
        "In Alice's Wonderland, a secret unit conversion is applied to measurements": "unit conversion",
        "In Alice's Wonderland, the gravitational constant has been secretly changed": "gravitational",
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations": "equations transformation",
    }

    data["label"] = data.prompt_eda.map(task_classes)
    logger.info(f'\n{data["label"].value_counts()}')

    global_vocab = set()

    for prompt in data[data.label == "encryption"]["prompt"]:
        lines = [line.strip() for line in prompt.lower().splitlines() if "->" in line]

        for line in lines:
            plain = line.split("->", 1)[1]
            words = re.sub(r"[^a-z\s]", "", plain).split()
            global_vocab.update(words)

    task_solvers_map = {
        "bit manipulation": BitManipulationSolver(),
        "encryption": EncryptionSolver(vocabulary=global_vocab),
        "conversion to diff numeral system": NumeralSystemSolver(),
        "unit conversion": UnitConversionSolver(),
        "gravitational": GravitationalSolver(),
        "equations transformation": BaseEquationSolver(),
    }

    processed_dfs = []

    for task, task_solver in tqdm(task_solvers_map.items(), total=len(task_solvers_map)):
        task_df = data[data.label == task].copy()

        task_df = solver(task_df, task_solver)

        computed = task_df["computed_answer"].astype(str).str.lower().str.strip()
        true = task_df["answer"].astype(str).str.lower().str.strip()

        task_df["is_correct"] = computed == true

        if task == "equations transformation":
            task_df["label"] = task_df.prompt.apply(task_solver.classify)

        processed_dfs.append(task_df)

    data = pd.concat(processed_dfs, ignore_index=True)

    enc_mask = data["label"] == "encryption"
    enc_df = data[enc_mask].copy()
    failed_mask = enc_df["computed_answer"].isna()

    if failed_mask.sum() > 0:
        enc_solver = task_solvers_map["encryption"]

        def solve_with_fallback(row):
            return enc_solver.generate_cot(row["prompt"], answer_hint=row["answer"])

        enc_df.loc[failed_mask, "generated_cot"] = enc_df[failed_mask].apply(
            solve_with_fallback,
            axis=1,
        )

        enc_df.loc[failed_mask, "computed_answer"] = enc_df.loc[
            failed_mask,
            "generated_cot",
        ].apply(enc_solver.extract_answer)

        enc_df["is_correct"] = (
            enc_df["computed_answer"].astype(str).str.lower().str.strip()
            == enc_df["answer"].astype(str).str.lower().str.strip()
        )

        data.update(enc_df)

    data["is_correct"] = (
        data["computed_answer"].astype(str).str.lower().str.strip()
        == data["answer"].astype(str).str.lower().str.strip()
    )

    data["is_correct_rounded"] = data.apply(
        lambda x: verify(
            x["computed_answer"],
            x["answer"],
        ),
        axis=1,
    )

    logger.info(f"Save output in: {args.output_path}")
    data.to_csv(args.output_path, index=False)

    records = []
    result = []

    for task in tqdm(data.label.unique(), total=len(data.label.unique())):
        task_df = data[data.label == task].copy()

        exact_accuracy = task_df["is_correct"].mean() * 100
        round_accuracy = task_df["is_correct_rounded"].mean() * 100

        records.append({
            "Task Name": task,
            "Exact Match Accuracy (%)": round(exact_accuracy, 2),
            "Round Accuracy (%)": round(round_accuracy, 2),
            "Count": len(task_df),
        })

        result.append(round_accuracy)

    results_df = pd.DataFrame.from_records(records)

    logger.info("\n" + results_df.to_string(index=False, justify="center"))
    weighted_global_accuracy = data["is_correct_rounded"].mean() * 100
    logger.info(f"Weighted global accuracy: {weighted_global_accuracy:.2f}")


if __name__ == "__main__":
    main()