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
    "MAJ": "MAJ",
    "MIN": "MIN",
    "CHOICE": "CHOICE",
    "NCHOICE": "NCHOICE",
}

TERNARY_SECTIONS = frozenset({"MAJ", "MIN", "CHOICE", "NCHOICE"})
PAIR_GROUP_SECTIONS = frozenset({"AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT"})

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
    def _section_view(analysis: Any, section: str) -> Tuple[List[Any], List[List[Any]]]:
        """Return the same deterministic section view used by the solver trace.

        The solver evaluates every ternary candidate, but renders a compact view:
        the selected exact chain plus at most 16 deterministic partial matches that
        improve output-bit coverage. Matching augmentation must use that same view;
        otherwise MAJ/MIN/CHOICE/NCHOICE examples become much larger than the CoT
        they are intended to teach.
        """
        section_records = list(analysis.records.get(section, []))
        per_bit = [list(cands) for cands in analysis.matches.get(section, [[] for _ in range(8)])]

        if section not in TERNARY_SECTIONS:
            return section_records, per_bit

        chosen_labels: set[str] = set()
        for rule in getattr(analysis, "selected", []):
            if getattr(rule, "family", None) != section:
                continue
            operands = getattr(rule, "operands", ())
            if operands:
                chosen_labels.add(compact_rule(rule))

        covered: set[int] = set()
        for rec in section_records:
            if rec.label in chosen_labels:
                covered.update(rec.matches)

        for rec in sorted(section_records, key=lambda item: (-len(item.matches), item.label, item.col)):
            if len(chosen_labels) >= 16:
                break
            if rec.label in chosen_labels:
                continue
            if any(bit not in covered for bit in rec.matches):
                chosen_labels.add(rec.label)
                covered.update(rec.matches)

        filtered_records = [rec for rec in section_records if rec.label in chosen_labels]
        allowed_exprs = {f"{section}{label}" for label in chosen_labels}
        filtered_matches: List[List[Any]] = []
        for bit_cands in per_bit:
            filtered_matches.append([
                cand
                for cand in bit_cands
                if getattr(cand, "expr", "") in allowed_exprs
                or getattr(cand, "program_key", None) is not None
            ])
        return filtered_records, filtered_matches

    @staticmethod
    def _format_section_data_lines(section: str, section_records: Sequence[Any]) -> List[str]:
        """Equivalent to matching.py data_lines: no section header, keep legacy group blanks."""
        lines: List[str] = []
        prev_diff: Optional[int] = None

        for rec in section_records:
            if (
                section in PAIR_GROUP_SECTIONS
                and len(rec.label) >= 2
                and rec.label[0].isdigit()
                and rec.label[1].isdigit()
            ):
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
    def _format_matching_output_lines(section: str, per_bit: Sequence[Sequence[Any]]) -> List[str]:
        """Equivalent to matching.py output_text: no 'Matching output' header."""
        mo_lines: List[str] = []
        for bit in range(8):
            cands = per_bit[bit]
            if cands:
                seen: set[str] = set()
                labels: List[str] = []
                for cand in cands:
                    label = compact_rule(cand)
                    if label not in seen:
                        seen.add(label)
                        labels.append(label)
                mo_lines.append(f"{bit} " + " ".join(labels))
            else:
                mo_lines.append(f"{bit} absent")

        per_bit_lists = [list(cands) for cands in per_bit]
        left_chain, best_left, right_chain, best_right = lr_from_matches(per_bit_lists)
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
    def _section_flags(per_bit: Sequence[Sequence[Any]], data_lines: Sequence[str]) -> Dict[str, bool]:
        per_bit_lists = [list(cands) for cands in per_bit]
        left_chain, _, right_chain, _ = lr_from_matches(per_bit_lists)
        chain_text = " ".join(left_chain + right_chain)
        n_matches = sum(1 for line in data_lines if "match" in line)
        return {
            "has_x": bool(re.search(r"(?:^|\s)[^\s]*x(?:\s|$)", chain_text)),
            "has_y": bool(re.search(r"(?:^|\s)[^\s]*y(?:\s|$)", chain_text)),
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
        section_records, per_bit = self._section_view(analysis, section)
        data_lines = self._format_section_data_lines(section, section_records)
        output_lines = self._format_matching_output_lines(section, per_bit)

        flags = self._section_flags(per_bit, data_lines)
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