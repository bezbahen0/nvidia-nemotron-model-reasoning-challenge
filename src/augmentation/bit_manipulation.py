from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.solvers.bit_manipulation import (
    BitManipulationSolver,
    SECTION_ORDER,
    column_hash,
    compact_rule,
    lr_from_matches,
    normalize_bits,
)

SECTION_NAMES = list(SECTION_ORDER)

_BEST_PREFIX = {
    "Identity": "I",
    "NOT": "NOT",
    "Constant": "C",
    "AND": "AND",
    "OR": "OR",
    "XOR": "XOR",
    "AND-NOT": "AND-NOT",
    "OR-NOT": "OR-NOT",
    "XOR-NOT": "XOR-NOT",
}

BIT_PROMPT_MARKER = "secret bit manipulation rule transforms 8-bit binary numbers"


@dataclass
class BitMatchingAugmentConfig:
    sample_frac: float = 1.0
    sample_n: Optional[int] = None
    only_solver_correct: bool = False
    downsample_sparse: bool = True


class BitMatchingAugmentGenerator:
    """
    Tonghuikang-style matching augmenter.

    This intentionally follows augmenters/matching.py format:
      prompt = Alice prefix + x/y legend + input block
      completion = matching lines + Left + Right

    It does NOT create random full bit-manipulation tasks.
    It derives matching subtasks from existing prompts using the Tong-style solver analysis.
    """

    def __init__(self, seed: Optional[int] = None, config: Optional[BitMatchingAugmentConfig] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config or BitMatchingAugmentConfig()
        self.solver = BitManipulationSolver()

    @staticmethod
    def is_bit_prompt(prompt: Any) -> bool:
        return BIT_PROMPT_MARKER in ("" if prompt is None else str(prompt))

    @staticmethod
    def _stable_hash_int(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _make_id(source_id: str, section: str) -> str:
        return hashlib.sha256(f"matching_{source_id}_{section}".encode()).hexdigest()[:8]

    @staticmethod
    def _strip_best_prefix(section: str, best_line: str) -> str:
        prefix = _BEST_PREFIX.get(section, section)
        return re.sub(rf"^(Best: ){re.escape(prefix)}", r"\1", best_line)

    @staticmethod
    def _format_output_bit_columns(analysis: Any) -> List[str]:
        n_examples = len(analysis.outputs)
        return [
            f"{bit} {col} {column_hash(col, n_examples)}"
            for bit, col in enumerate(analysis.output_columns)
        ]

    @staticmethod
    def _format_section_data_lines(analysis: Any, section: str) -> List[str]:
        """Equivalent to matching.py data_lines: no section header, keep group blanks."""
        lines: List[str] = []
        prev_diff: Optional[int] = None

        for rec in analysis.records[section]:
            if len(rec.label) >= 2 and rec.label[0].isdigit() and rec.label[1].isdigit():
                diff = (int(rec.label[1]) - int(rec.label[0])) % 8
                if prev_diff is not None and diff != prev_diff:
                    lines.append("")
                prev_diff = diff

            line = f"{rec.label} {rec.col} {rec.hash_}"
            if rec.matches:
                line += " match " + " ".join(str(i) for i in rec.matches)
            lines.append(line)

        while lines and lines[-1].strip() == "":
            lines.pop()
        return lines

    @staticmethod
    def _format_matching_output_lines(analysis: Any, section: str) -> List[str]:
        """Equivalent to matching.py output_text: no 'Matching output' header."""
        per_bit = analysis.matches[section]

        mo_lines: List[str] = []
        for bit in range(8):
            cands = per_bit[bit]
            if cands:
                mo_lines.append(f"{bit} " + " ".join(compact_rule(c) for c in cands))
            else:
                mo_lines.append(f"{bit} absent")

        left_chain, best_left, right_chain, best_right = lr_from_matches(per_bit)
        best_left = BitMatchingAugmentGenerator._strip_best_prefix(section, f"Best: {best_left}")
        best_right = BitMatchingAugmentGenerator._strip_best_prefix(section, f"Best: {best_right}")

        return (
            mo_lines
            + ["", "Left"]
            + left_chain
            + [best_left]
            + ["", "Right"]
            + right_chain
            + [best_right]
        )

    @staticmethod
    def _section_flags(analysis: Any, section: str, output_lines: Sequence[str], data_lines: Sequence[str]) -> Dict[str, bool]:
        per_bit = analysis.matches[section]
        left_chain, _, right_chain, _ = lr_from_matches(per_bit)
        chain_text = " ".join(left_chain + right_chain)
        n_matches = sum(1 for line in data_lines if "match" in line)
        return {
            "has_x": bool(re.search(r"\dx", chain_text)),
            "has_y": bool(re.search(r"\dy", chain_text)),
            "all_absent": n_matches == 0,
            "both_none": left_chain == ["none"] and right_chain == ["none"],
            "few_matches": n_matches < 4,
        }

    def _keep_section(self, source_id: str, section: str, flags: Dict[str, bool]) -> bool:
        if not self.config.downsample_sparse:
            return True

        h = self._stable_hash_int(f"{source_id}_{section}")
        if flags["all_absent"]:
            return (h % 100) == 0
        if flags["both_none"]:
            return (h % 10) == 0
        if flags["few_matches"]:
            return (h % 5) < 1
        return True

    def _make_matching_problem(
        self,
        analysis: Any,
        source_id: str,
        section: str,
        source_solver_answer: str,
        source_gold_answer: str,
        source_solver_correct: bool,
    ) -> Optional[Dict[str, Any]]:
        obc_block = self._format_output_bit_columns(analysis)
        data_lines = self._format_section_data_lines(analysis, section)
        output_lines = self._format_matching_output_lines(analysis, section)

        flags = self._section_flags(analysis, section, output_lines, data_lines)
        if not self._keep_section(source_id, section, flags):
            return None

        input_text = "\n".join(obc_block) + "\n\n" + "\n".join(data_lines)
        output_text = "\n".join(output_lines)

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "x: not matched anywhere\n"
            "y: matched but wrong position\n\n"
            + input_text
        )

        pid = self._make_id(source_id, section)

        return {
            "id": pid,
            "prompt": prompt,
            "completion": output_text,
            "answer": output_text,
            "generated_cot": output_text,
            "computed_answer": output_text,
            "category": "matching",
            "label": "bit_matching",
            "task_mode": "matching",
            "section": section,
            "source_task_id": source_id,
            "source_solver_answer": source_solver_answer,
            "source_gold_answer": source_gold_answer,
            "source_solver_correct": source_solver_correct,
            **flags,
        }

    def make_tasks_for_row(self, row: pd.Series, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        source_id = source_id or str(row.get("id", row.name))
        prompt = row["prompt"]
        gold = normalize_bits(row["answer"]) if "answer" in row else ""

        analysis = self.solver.analyze(prompt, gold)
        if analysis is None:
            return []

        solver_answer = analysis.answer
        solver_correct = bool(gold and solver_answer == gold)
        if self.config.only_solver_correct and not solver_correct:
            return []

        rows: List[Dict[str, Any]] = []
        for section in SECTION_NAMES:
            item = self._make_matching_problem(
                analysis=analysis,
                source_id=source_id,
                section=section,
                source_solver_answer=solver_answer,
                source_gold_answer=gold,
                source_solver_correct=solver_correct,
            )
            if item is not None:
                rows.append(item)
        return rows

    def generate_dataset(
        self,
        source_data: pd.DataFrame,
        sample_frac: Optional[float] = None,
        sample_n: Optional[int] = None,
        only_solver_correct: Optional[bool] = None,
    ) -> pd.DataFrame:
        sample_frac = self.config.sample_frac if sample_frac is None else sample_frac
        sample_n = self.config.sample_n if sample_n is None else sample_n
        only_solver_correct = self.config.only_solver_correct if only_solver_correct is None else only_solver_correct

        work = source_data.copy()
        if "label" in work.columns:
            work = work[(work["label"].astype(str) == "bit manipulation") | work["prompt"].map(self.is_bit_prompt)].copy()
        else:
            work = work[work["prompt"].map(self.is_bit_prompt)].copy()

        if sample_n is not None:
            work = work.sample(n=min(sample_n, len(work)), random_state=self.seed)
        elif sample_frac < 1.0:
            work = work.sample(frac=sample_frac, random_state=self.seed)

        old_flag = self.config.only_solver_correct
        self.config.only_solver_correct = bool(only_solver_correct)
        try:
            rows: List[Dict[str, Any]] = []
            for _, row in work.iterrows():
                source_id = str(row["id"]) if "id" in row and pd.notna(row["id"]) else str(row.name)
                rows.extend(self.make_tasks_for_row(row, source_id=source_id))
        finally:
            self.config.only_solver_correct = old_flag

        if not rows:
            return pd.DataFrame(
                columns=["id", "prompt", "completion", "answer", "label", "generated_cot", "computed_answer", "task_mode"]
            )

        return pd.DataFrame.from_records(rows)