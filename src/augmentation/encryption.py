from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

EN_DASH = "\u2013"
ENCRYPTION_PROMPT_MARKER = "secret encryption rules are used on text"


@dataclass
class EncryptionSpellingAugmentConfig:
    """Config for deriving spelling subtasks from existing encryption prompts."""

    sample_frac: float = 1.0
    sample_n: Optional[int] = None
    only_solver_correct: bool = False
    lines_per_problem: int = 100
    demo_lines: int = 3
    words_per_line: int = 3
    min_word_len: int = 1
    max_word_len: int = 24
    max_tasks_per_source: Optional[int] = None
    include_prompt_fixed_text: bool = False


class EncryptionSpellingAugmentGenerator:
    """
    Spelling-style augmenter for encryption prompts.

    This replaces random synthetic encryption generation. It derives small Alice-style
    processing subtasks from the *visible source encryption prompts* only.

    Task format follows the Tonghuikang spelling augmenter:
      input:  text rows
      output: row -> –c–h–a–r–s–, with spaces removed and en-dash separators

    No gold answer text is used to create the task, so this does not leak the target
    decrypted answer from the source encryption row.
    """

    def __init__(self, seed: Optional[int] = None, config: Optional[EncryptionSpellingAugmentConfig] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config or EncryptionSpellingAugmentConfig()

    @staticmethod
    def is_encryption_prompt(prompt: Any) -> bool:
        return ENCRYPTION_PROMPT_MARKER in ("" if prompt is None else str(prompt).lower())

    @staticmethod
    def _make_id(source_id: str, chunk_idx: int) -> str:
        return hashlib.sha256(f"encryption_spelling_{source_id}_{chunk_idx}".encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def spell_out(text: str) -> str:
        chars = [c for c in text.lower() if c != " "]
        return EN_DASH + EN_DASH.join(chars) + EN_DASH

    @staticmethod
    def _strip_fixed_prompt_text(prompt: str) -> str:
        """Keep task-specific examples/target, drop the fixed Alice boilerplate if desired."""
        text = prompt.lower()
        # Keep everything after the examples marker when present. This removes many repeated
        # fixed prompt words while preserving all visible cipher/plain examples and target cipher.
        marker = "here are some examples:"
        idx = text.find(marker)
        if idx >= 0:
            return text[idx + len(marker):]
        return text

    def _extract_visible_words(self, prompt: Any) -> List[str]:
        text = "" if prompt is None else str(prompt)
        if not self.config.include_prompt_fixed_text:
            text = self._strip_fixed_prompt_text(text)

        words = re.findall(r"[a-z]+", text.lower())
        words = [
            w
            for w in words
            if self.config.min_word_len <= len(w) <= self.config.max_word_len
        ]
        return words

    def _make_lines_from_words(self, words: Sequence[str]) -> List[str]:
        k = max(1, int(self.config.words_per_line))
        lines: List[str] = []
        for i in range(0, len(words), k):
            chunk = list(words[i:i + k])
            if len(chunk) == k:
                lines.append(" ".join(chunk))
        return lines

    def _make_problem(
        self,
        source_id: str,
        chunk_idx: int,
        source_lines: Sequence[str],
        demo_pool: Sequence[str],
        source_solver_answer: str,
        source_gold_answer: str,
        source_solver_correct: bool,
    ) -> Optional[Dict[str, Any]]:
        if not source_lines:
            return None

        demo_n = min(self.config.demo_lines, len(demo_pool))
        if demo_n <= 0:
            return None

        # Deterministic demos: seeded random sample from the source prompt's visible text.
        demos = list(demo_pool)
        local_rng = random.Random(f"{self.seed}_{source_id}_{chunk_idx}")
        local_rng.shuffle(demos)
        demo_inputs = demos[:demo_n]

        sample_input_lines = [f"{j:02d}\n{inp}" for j, inp in enumerate(demo_inputs)]
        sample_output_lines = [
            f"{j:02d}\n{inp} -> {self.spell_out(inp)}"
            for j, inp in enumerate(demo_inputs)
        ]

        test_inputs = [f"{row_num:02d}\n{inp}" for row_num, inp in enumerate(source_lines)]
        output_lines = [
            f"{row_num:02d}\n{inp} -> {self.spell_out(inp)}"
            for row_num, inp in enumerate(source_lines)
        ]

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )

        answer = "\n".join(output_lines)
        pid = self._make_id(source_id, chunk_idx)

        return {
            "id": pid,
            "prompt": prompt,
            "completion": answer,
            "answer": answer,
            "generated_cot": answer,
            "computed_answer": answer,
            "category": "spelling",
            "label": "encryption_spelling",
            "task_mode": "encryption_spelling",
            "source_task_id": source_id,
            "source_solver_answer": source_solver_answer,
            "source_gold_answer": source_gold_answer,
            "source_solver_correct": source_solver_correct,
            "n_lines": len(source_lines),
        }

    def make_tasks_for_row(self, row: pd.Series, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        source_id = source_id or str(row.get("id", row.name))
        prompt = row.get("prompt", "")
        if not self.is_encryption_prompt(prompt):
            return []

        source_gold_answer = str(row.get("answer", ""))
        source_solver_answer = str(row.get("computed_answer", ""))
        source_solver_correct = bool(source_gold_answer and source_solver_answer and source_gold_answer.strip() == source_solver_answer.strip())
        if self.config.only_solver_correct and not source_solver_correct:
            return []

        words = self._extract_visible_words(prompt)
        lines = self._make_lines_from_words(words)
        if not lines:
            return []

        # Keep demos from the same visible prompt. If there are very few lines, reuse the
        # available lines as demos; the task still remains deterministic and local.
        demo_pool = lines[:]
        per_problem = max(1, int(self.config.lines_per_problem))

        rows: List[Dict[str, Any]] = []
        for chunk_idx, start in enumerate(range(0, len(lines), per_problem)):
            if self.config.max_tasks_per_source is not None and chunk_idx >= self.config.max_tasks_per_source:
                break
            chunk = lines[start:start + per_problem]
            item = self._make_problem(
                source_id=source_id,
                chunk_idx=chunk_idx,
                source_lines=chunk,
                demo_pool=demo_pool,
                source_solver_answer=source_solver_answer,
                source_gold_answer=source_gold_answer,
                source_solver_correct=source_solver_correct,
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
            work = work[(work["label"].astype(str) == "encryption") | work["prompt"].map(self.is_encryption_prompt)].copy()
        else:
            work = work[work["prompt"].map(self.is_encryption_prompt)].copy()

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
