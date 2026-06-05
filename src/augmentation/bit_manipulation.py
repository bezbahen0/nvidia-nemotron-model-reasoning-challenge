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

    # Keep the historical behavior by default: generate only section-matching tasks.
    include_matching: bool = True

    # New dense tail subtasks. Enable explicitly when you want to train the
    # model on the final rule-completion bottleneck.
    include_tail_completion: bool = False
    include_tail_repair: bool = False

    # Tail examples are intended for supervised training, so by default we only
    # keep them when the emitted tail answer agrees with the dataset answer.
    tail_require_correct: bool = True

    # For tail_completion, keep only examples where the ternary step actually
    # chooses at least one MAJ/CH rule instead of only reprinting defaults.
    tail_completion_require_change: bool = True

    # Tail repair needs a solver that supports generate_cot(..., answer_hint=...).
    # It is filtered to successful repairs by default to avoid post-hoc wrong CoTs.
    tail_repair_require_success: bool = True


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
    def _make_tail_id(source_id: str, mode: str) -> str:
        return hashlib.sha256(f"{mode}_{source_id}".encode()).hexdigest()[:8]

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


    @staticmethod
    def _extract_block(text: str, header: str, before: Optional[str] = None, last: bool = False) -> List[str]:
        """Extract a simple CoT block whose header is a line by itself.

        Blocks in the bit solver are separated by a blank line, so this helper is
        deliberately conservative: it returns header + following non-empty lines.
        """
        if before and before in text:
            text = text.split(before, 1)[0]

        lines = text.splitlines()
        starts = [i for i, line in enumerate(lines) if line.strip() == header]
        if not starts:
            return []
        start = starts[-1] if last else starts[0]

        end = start + 1
        while end < len(lines) and lines[end].strip() != "":
            end += 1
        return lines[start:end]

    @staticmethod
    def _extract_tail_from_marker(text: str, marker: str) -> str:
        idx = text.find(marker)
        if idx < 0:
            return ""
        return text[idx:].strip()

    @staticmethod
    def _has_bad_tail_text(text: str) -> bool:
        """Guard against hidden-answer leakage or truncated candidate lists."""
        low = text.lower()
        banned = (
            "gold answer",
            "answer hint",
            "answer-aware",
            "teacher repair",
            "wrong bits",
            "needed bit",
        )
        if any(token in low for token in banned):
            return True
        if "..." in text:
            return True
        return False

    @staticmethod
    def _format_input_bit_columns(analysis: Any) -> List[str]:
        n_examples = len(analysis.inputs)
        return [
            f"{bit} {col} {column_hash(col, n_examples)}"
            for bit, col in enumerate(analysis.input_columns)
        ]

    def _tail_context_lines(self, analysis: Any, trace: str, mode: str, before_marker: str) -> List[str]:
        """Build a compact prompt context for tail-only subtasks.

        The context gives enough information to recompute exact column matches,
        but avoids replaying the whole long solver trace.
        """
        lines: List[str] = []
        lines.append("In Alice's Wonderland, complete the final rule selection for this bit manipulation task.")
        lines.append("Use the given columns and the current solver state. Continue only with the requested final block.")
        lines.append("")

        lines.append("Output bit columns (with bitsum as hash)")
        lines.extend(self._format_output_bit_columns(analysis))
        lines.append("")

        lines.append("Input bit columns (with bitsum as hash)")
        lines.extend(self._format_input_bit_columns(analysis))
        lines.append("")

        lines.append(f"Target input: {analysis.question_bits}")
        lines.append("")

        if mode == "tail_completion":
            for header in ("Preferred", "Matching", "Perfect match", "Matched"):
                block = self._extract_block(trace, header, before=before_marker, last=True)
                if block:
                    lines.extend(block)
                    lines.append("")
            lines.append("Continue from `Ternary completion` through the final boxed answer.")
        else:
            selected = self._extract_block(trace, "Selected", before=before_marker, last=True)
            if selected:
                lines.append("Current selected rules")
                lines.extend(selected[1:])
                lines.append("")
            current_answer = getattr(analysis, "answer", "")
            if current_answer:
                lines.append(f"Current computed answer: {current_answer}")
                lines.append("")
            lines.append("Continue from `Final completion check` through the final boxed answer.")

        return lines

    def _answer_from_text(self, text: str) -> str:
        try:
            return self.solver.extract_answer(text)
        except Exception:
            m = re.findall(r"\\boxed\{([01]{8})\}", text)
            return m[-1] if m else ""

    def _make_tail_completion_problem(
        self,
        analysis: Any,
        source_id: str,
        source_gold_answer: str,
        source_solver_correct: bool,
    ) -> Optional[Dict[str, Any]]:
        trace = str(analysis.trace)
        marker = "Ternary completion"
        completion = self._extract_tail_from_marker(trace, marker)
        if not completion or self._has_bad_tail_text(completion):
            return None

        if self.config.tail_completion_require_change and not re.search(r"(?m)^\d+ use (?:MAJ|CH)\d+", completion):
            return None

        computed_answer = self._answer_from_text(completion)
        tail_correct = bool(source_gold_answer and computed_answer == source_gold_answer)
        if self.config.tail_require_correct and source_gold_answer and not tail_correct:
            return None

        prompt = "\n".join(self._tail_context_lines(analysis, trace, "tail_completion", marker))
        pid = self._make_tail_id(source_id, "tail_completion")

        return {
            "id": pid,
            "prompt": prompt,
            "completion": completion,
            "answer": completion,
            "generated_cot": completion,
            "computed_answer": computed_answer,
            "category": "bit_tail_completion",
            "label": "bit_tail_completion",
            "task_mode": "tail_completion",
            "source_task_id": source_id,
            "source_solver_answer": getattr(analysis, "answer", ""),
            "source_gold_answer": source_gold_answer,
            "source_solver_correct": source_solver_correct,
            "tail_correct": tail_correct,
            "has_ternary_completion": True,
            "has_answer_hint_repair": False,
        }

    def _make_tail_repair_problem(
        self,
        row: pd.Series,
        source_id: str,
        analysis: Any,
        source_gold_answer: str,
        source_solver_correct: bool,
    ) -> Optional[Dict[str, Any]]:
        if not source_gold_answer:
            return None

        # The repair tail must be produced by the solver itself.  Solvers that do
        # not support answer_hint are skipped without affecting matching tasks.
        try:
            repaired_trace = self.solver.generate_cot(row["prompt"], answer_hint=source_gold_answer)
        except TypeError:
            return None
        except Exception:
            return None

        marker = "Final completion check"
        completion = self._extract_tail_from_marker(str(repaired_trace), marker)
        if not completion or self._has_bad_tail_text(completion):
            return None

        repaired_answer = self._answer_from_text(completion)
        repaired_correct = bool(repaired_answer and repaired_answer == source_gold_answer)
        if self.config.tail_repair_require_success and not repaired_correct:
            return None

        # Skip no-op repair tails; those are not useful as a distinct subtask.
        baseline_answer = getattr(analysis, "answer", "")
        if baseline_answer == repaired_answer:
            return None

        prompt = "\n".join(self._tail_context_lines(analysis, str(repaired_trace), "tail_repair", marker))
        pid = self._make_tail_id(source_id, "tail_repair")

        return {
            "id": pid,
            "prompt": prompt,
            "completion": completion,
            "answer": completion,
            "generated_cot": completion,
            "computed_answer": repaired_answer,
            "category": "bit_tail_repair",
            "label": "bit_tail_repair",
            "task_mode": "tail_repair",
            "source_task_id": source_id,
            "source_solver_answer": baseline_answer,
            "source_gold_answer": source_gold_answer,
            "source_solver_correct": source_solver_correct,
            "tail_correct": repaired_correct,
            "has_ternary_completion": "Ternary completion" in str(repaired_trace),
            "has_answer_hint_repair": True,
        }

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
        if "answer" in row and pd.notna(row.get("answer")):
            gold = normalize_bits(row["answer"])
        elif "ground_truth" in row and pd.notna(row.get("ground_truth")):
            gold = normalize_bits(row["ground_truth"])
        else:
            gold = ""

        analysis = self.solver.analyze(prompt)
        if analysis is None:
            return []

        solver_answer = analysis.answer
        solver_correct = bool(gold and solver_answer == gold)
        if self.config.only_solver_correct and not solver_correct:
            return []

        rows: List[Dict[str, Any]] = []

        if self.config.include_matching:
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

        if self.config.include_tail_completion:
            item = self._make_tail_completion_problem(
                analysis=analysis,
                source_id=source_id,
                source_gold_answer=gold,
                source_solver_correct=solver_correct,
            )
            if item is not None:
                rows.append(item)

        if self.config.include_tail_repair:
            item = self._make_tail_repair_problem(
                row=row,
                source_id=source_id,
                analysis=analysis,
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
        include_matching: Optional[bool] = None,
        include_tail_completion: Optional[bool] = None,
        include_tail_repair: Optional[bool] = None,
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

        old_only_solver_correct = self.config.only_solver_correct
        old_include_matching = self.config.include_matching
        old_include_tail_completion = self.config.include_tail_completion
        old_include_tail_repair = self.config.include_tail_repair

        self.config.only_solver_correct = bool(only_solver_correct)
        if include_matching is not None:
            self.config.include_matching = bool(include_matching)
        if include_tail_completion is not None:
            self.config.include_tail_completion = bool(include_tail_completion)
        if include_tail_repair is not None:
            self.config.include_tail_repair = bool(include_tail_repair)

        try:
            rows: List[Dict[str, Any]] = []
            for _, row in work.iterrows():
                source_id = str(row["id"]) if "id" in row and pd.notna(row["id"]) else str(row.name)
                rows.extend(self.make_tasks_for_row(row, source_id=source_id))
        finally:
            self.config.only_solver_correct = old_only_solver_correct
            self.config.include_matching = old_include_matching
            self.config.include_tail_completion = old_include_tail_completion
            self.config.include_tail_repair = old_include_tail_repair

        if not rows:
            return pd.DataFrame(
                columns=["id", "prompt", "completion", "answer", "label", "generated_cot", "computed_answer", "task_mode", "source_task_id", "source_solver_answer", "source_gold_answer", "source_solver_correct", "tail_correct"]
            )

        return pd.DataFrame.from_records(rows)
