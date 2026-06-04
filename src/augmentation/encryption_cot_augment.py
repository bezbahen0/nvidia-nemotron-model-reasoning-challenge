from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ENCRYPTION_PROMPT_MARKER = "secret encryption rules are used on text"
DEFAULT_TASK_MODES: Tuple[str, ...] = (
    "word_pair_mapping",
    "apply_mapping_to_target_partial",
    "pattern_completion",
    "new_mapping_from_completed_word",
    "final_reconstruction",
)


@dataclass
class EncryptionCotAugmentConfig:
    sample_frac: float = 1.0
    sample_n: Optional[int] = None
    only_solver_correct: bool = False
    task_modes: Tuple[str, ...] = DEFAULT_TASK_MODES
    # Prevent word_pair_mapping from dominating the augmented dataset.
    # With None, every aligned example word becomes a separate task.
    # Default 1 keeps this label near the source encryption row count.
    max_word_pair_tasks_per_row: Optional[int] = 1
    # Optional global cap after all rows are generated. Use this when you want
    # a hard budget such as 1500 word_pair_mapping tasks total.
    max_word_pair_tasks_total: Optional[int] = None

    # Keep all apply_mapping tasks with unknown letters, but downsample fully
    # solved targets because they are much easier and do not train mask handling.
    keep_full_solved_apply_mapping_frac: float = 0.25

    # Pattern-completion tasks should contain real distractors; otherwise a
    # one-item vocabulary teaches copying rather than constraint checking.
    max_pattern_tasks_per_row: Optional[int] = None
    min_pattern_vocab_size: int = 3
    max_vocab_candidates: int = 12

    # Final reconstruction is most useful when multiple words are resolved.
    # Keep all 2+ resolved-word tasks, downsample single-word reconstruction.
    include_trivial_final_reconstruction: bool = False
    keep_single_resolved_final_reconstruction_frac: float = 0.25

    downsample_word_pair_identity_heavy: bool = False


@dataclass(frozen=True)
class EncryptionPromptParts:
    examples: Tuple[Tuple[str, str], ...]
    target: str


class EncryptionSubtaskAugmentGeneratorV4:
    """
    Balanced CoT-slicing augmenter for Alice/Wonderland monoalphabetic encryption tasks.

    IMPORTANT: each generated row uses a subtask-specific label:
      label = "encryption_" + task_mode
    Example: task_mode="word_pair_mapping" -> label="encryption_word_pair_mapping".

    It derives small local-training subtasks from existing full encryption prompts:
      1. word_pair_mapping: aligned encrypted/plain word -> character mappings
      2. apply_mapping_to_target_partial: known dictionary + encrypted target -> partial plaintext
      3. pattern_completion: partial word mask -> completed word
      4. new_mapping_from_completed_word: encrypted word + partial + completed word -> new mappings
      5. final_reconstruction: partial phrase + resolved words -> final plaintext

    Output rows follow the same broad shape as the other augmenters:
      prompt, completion, answer, generated_cot, computed_answer, label, task_mode, source_task_id
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[EncryptionCotAugmentConfig] = None,
        vocabulary: Optional[Iterable[str]] = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config or EncryptionCotAugmentConfig()
        self.vocabulary = set(vocabulary or [])

    @staticmethod
    def is_encryption_prompt(prompt: Any) -> bool:
        text = "" if prompt is None else str(prompt)
        return ENCRYPTION_PROMPT_MARKER in text

    @staticmethod
    def _stable_hash_int(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _make_id(source_id: str, task_mode: str, suffix: str) -> str:
        raw = f"encryption_cot_{source_id}_{task_mode}_{suffix}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _label_for_task_mode(task_mode: str) -> str:
        return f"encryption_{task_mode}"

    @staticmethod
    def _clean_answer(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        # Accept raw answers and boxed final answers.
        m = re.search(r"\\boxed\{([^{}]+)\}", text)
        if m:
            return m.group(1).strip()
        return text

    @staticmethod
    def _extract_words(text: str) -> List[str]:
        return re.findall(r"[A-Za-z]+", text.lower())

    @staticmethod
    def _parse_prompt(prompt: Any) -> Optional[EncryptionPromptParts]:
        text = "" if prompt is None else str(prompt)
        if not text.strip():
            return None

        examples: List[Tuple[str, str]] = []
        for line in text.splitlines():
            line = line.strip()
            if "->" not in line:
                continue
            left, right = line.split("->", 1)
            left = left.strip()
            right = right.strip().rstrip(".")
            if left and right:
                examples.append((left, right))

        target = ""
        patterns = [
            r"(?is)Now,\s*decrypt\s*the\s*following\s*text\s*:\s*(.+?)\s*$",
            r"(?is)decrypt\s*the\s*following\s*text\s*:\s*(.+?)\s*$",
            r"(?is)following\s*text\s*:\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                target = m.group(1).strip().rstrip(".")
                break

        if not examples or not target:
            return None
        return EncryptionPromptParts(examples=tuple(examples), target=target)

    @staticmethod
    def _aligned_word_pairs(examples: Sequence[Tuple[str, str]]) -> List[Tuple[int, int, str, str]]:
        pairs: List[Tuple[int, int, str, str]] = []
        for ex_idx, (cipher_phrase, plain_phrase) in enumerate(examples, start=1):
            c_words = cipher_phrase.split()
            p_words = plain_phrase.split()
            for word_idx, (cw, pw) in enumerate(zip(c_words, p_words), start=1):
                if len(cw) == len(pw) and cw.isalpha() and pw.isalpha():
                    pairs.append((ex_idx, word_idx, cw.lower(), pw.lower()))
        return pairs

    @staticmethod
    def _mapping_from_word_pair(cipher_word: str, plain_word: str) -> List[Tuple[str, str]]:
        seen: Dict[str, str] = {}
        ordered: List[Tuple[str, str]] = []
        for c, p in zip(cipher_word, plain_word):
            if c not in seen:
                seen[c] = p
                ordered.append((c, p))
        return ordered

    @staticmethod
    def _mapping_from_examples(examples: Sequence[Tuple[str, str]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _, _, cipher_word, plain_word in EncryptionSubtaskAugmentGeneratorV4._aligned_word_pairs(examples):
            for c, p in EncryptionSubtaskAugmentGeneratorV4._mapping_from_word_pair(cipher_word, plain_word):
                if c not in mapping:
                    mapping[c] = p
        return mapping

    @staticmethod
    def _format_equals_mapping(pairs: Sequence[Tuple[str, str]]) -> str:
        return ", ".join(f"'{c}'='{p}'" for c, p in pairs)

    @staticmethod
    def _format_arrow_mapping(pairs: Sequence[Tuple[str, str]]) -> str:
        return ", ".join(f"'{c}' -> '{p}'" for c, p in pairs)

    @staticmethod
    def _dictionary_lines(mapping: Dict[str, str]) -> str:
        return "\n".join(f"{c} -> {p}" for c, p in sorted(mapping.items()))

    @staticmethod
    def _apply_mapping(text: str, mapping: Dict[str, str]) -> str:
        out: List[str] = []
        for ch in text.lower():
            if ch == " ":
                out.append(" ")
            elif ch in mapping:
                out.append(mapping[ch])
            elif ch.isalpha():
                out.append("?")
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _word_matches_pattern(word: str, pattern: str) -> bool:
        if len(word) != len(pattern):
            return False
        return all(pc == "?" or pc == wc for wc, pc in zip(word, pattern))

    def _stable_sample(self, items: Sequence[str], k: int, key: str) -> List[str]:
        unique = sorted(set(items))
        if len(unique) <= k:
            return unique
        local_seed = self._stable_hash_int(f"{self.seed}_{key}") % (2**32)
        local_rng = random.Random(local_seed)
        return sorted(local_rng.sample(unique, k))

    def _stable_sample_records(self, items: Sequence[Any], k: int, key: str) -> List[Any]:
        items = list(items)
        if k is None or len(items) <= k:
            return items
        local_seed = self._stable_hash_int(f"{self.seed}_{key}") % (2**32)
        local_rng = random.Random(local_seed)
        indexed = list(enumerate(items))
        picked = local_rng.sample(indexed, k)
        # Keep original order after sampling so prompts remain natural/reproducible.
        return [item for _, item in sorted(picked, key=lambda x: x[0])]

    def _stable_keep(self, probability: float, key: str) -> bool:
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        bucket = self._stable_hash_int(f"{self.seed}_{key}") % 10_000
        return bucket < int(probability * 10_000)

    def _candidate_vocab(self, gold_word: str, pattern: str, row_vocab: Sequence[str], key: str) -> List[str]:
        vocab = sorted(set(self.vocabulary) | set(row_vocab) | {gold_word})
        same_len = [w for w in vocab if len(w) == len(pattern)]
        matching = [w for w in same_len if self._word_matches_pattern(w, pattern)]
        non_matching = [w for w in same_len if w not in matching]

        max_n = max(1, self.config.max_vocab_candidates)
        # Keep all exact matches if possible; fill the rest with same-length distractors.
        keep_matches = matching[:max_n]
        remaining = max_n - len(keep_matches)
        if remaining > 0:
            keep_distractors = self._stable_sample(non_matching, remaining, key)
        else:
            keep_distractors = []
        candidates = sorted(set(keep_matches + keep_distractors + [gold_word]))
        return candidates

    def _make_word_pair_mapping_task(
        self,
        source_id: str,
        ex_idx: int,
        word_idx: int,
        cipher_word: str,
        plain_word: str,
    ) -> Dict[str, Any]:
        pairs = self._mapping_from_word_pair(cipher_word, plain_word)
        answer = self._format_equals_mapping(pairs)
        prompt = (
            "In Alice's Wonderland, secret encryption mappings are used on text.\n\n"
            f"Encrypted word: {cipher_word}\n"
            f"Plain word: {plain_word}\n\n"
            "Extract the monoalphabetic substitution mappings by aligned positions."
        )
        solution = (
            f"Word '{cipher_word}' ({len(cipher_word)} letters) maps to '{plain_word}'. "
            f"Extracted: {answer}.\n\n"
            f"Answer: {answer}."
        )
        pid = self._make_id(source_id, "word_pair_mapping", f"{ex_idx}_{word_idx}_{cipher_word}_{plain_word}")
        return {
            "id": pid,
            "prompt": prompt,
            "completion": solution,
            "answer": answer,
            "generated_cot": solution,
            "computed_answer": answer,
            "category": "encryption",
            "label": self._label_for_task_mode("word_pair_mapping"),
            "task_mode": "word_pair_mapping",
            "source_task_id": source_id,
            "example_index": ex_idx,
            "word_index": word_idx,
        }

    def _make_apply_mapping_task(
        self,
        source_id: str,
        parts: EncryptionPromptParts,
        mapping: Dict[str, str],
        partial: str,
    ) -> Dict[str, Any]:
        prompt = (
            "In Alice's Wonderland, secret encryption mappings are used on text.\n\n"
            "Known substitution dictionary:\n"
            f"{self._dictionary_lines(mapping)}\n\n"
            f"Encrypted target:\n{parts.target}\n\n"
            "Decrypt using only the known mappings. Use ? for unknown letters. "
            "Preserve spaces and word lengths exactly."
        )
        solution = (
            f"Apply the known dictionary to the encrypted target '{parts.target}'. "
            "Every known encrypted letter is replaced by its plaintext letter, and every unknown letter is written as ?. "
            f"This gives the partial decryption: '{partial}'.\n\n"
            f"Answer: {partial}"
        )
        pid = self._make_id(source_id, "apply_mapping_to_target_partial", parts.target)
        return {
            "id": pid,
            "prompt": prompt,
            "completion": solution,
            "answer": partial,
            "generated_cot": solution,
            "computed_answer": partial,
            "category": "encryption",
            "label": self._label_for_task_mode("apply_mapping_to_target_partial"),
            "task_mode": "apply_mapping_to_target_partial",
            "source_task_id": source_id,
            "unknown_chars": partial.count("?"),
            "is_fully_solved_partial": "?" not in partial,
        }

    def _make_pattern_completion_task(
        self,
        source_id: str,
        word_idx: int,
        pattern: str,
        gold_word: str,
        row_vocab: Sequence[str],
    ) -> Dict[str, Any]:
        candidates = self._candidate_vocab(gold_word, pattern, row_vocab, f"{source_id}_{word_idx}_{pattern}")
        candidate_text = ", ".join(candidates)
        prompt = (
            "In Alice's Wonderland, secret encryption mappings are used on text.\n\n"
            f"Partial decrypted word: {pattern}\n"
            f"Candidate vocabulary: {candidate_text}\n\n"
            "Choose the word that matches the pattern exactly. "
            "A valid word must have the same length and preserve every known letter."
        )
        known_positions = [
            f"position {i + 1} is '{ch}'"
            for i, ch in enumerate(pattern)
            if ch != "?"
        ]
        constraints = "; ".join(known_positions) if known_positions else "only the word length is fixed"
        solution = (
            f"The pattern '{pattern}' has length {len(pattern)}. Its constraints are: {constraints}. "
            f"The word '{gold_word}' matches this pattern exactly.\n\n"
            f"Answer: {gold_word}"
        )
        pid = self._make_id(source_id, "pattern_completion", f"{word_idx}_{pattern}_{gold_word}")
        return {
            "id": pid,
            "prompt": prompt,
            "completion": solution,
            "answer": gold_word,
            "generated_cot": solution,
            "computed_answer": gold_word,
            "category": "encryption",
            "label": self._label_for_task_mode("pattern_completion"),
            "task_mode": "pattern_completion",
            "source_task_id": source_id,
            "word_index": word_idx,
            "pattern": pattern,
            "candidate_vocab_size": len(candidates),
        }

    def _make_new_mapping_task(
        self,
        source_id: str,
        word_idx: int,
        cipher_word: str,
        partial_word: str,
        gold_word: str,
    ) -> Optional[Dict[str, Any]]:
        new_pairs: List[Tuple[str, str]] = []
        seen = set()
        for c, partial_ch, gold_ch in zip(cipher_word, partial_word, gold_word):
            if partial_ch == "?" and c not in seen:
                seen.add(c)
                new_pairs.append((c, gold_ch))
        if not new_pairs:
            return None
        answer = self._format_equals_mapping(new_pairs)
        prompt = (
            "In Alice's Wonderland, secret encryption mappings are used on text.\n\n"
            f"Encrypted word: {cipher_word}\n"
            f"Partial decryption: {partial_word}\n"
            f"Completed word: {gold_word}\n\n"
            "Which new encrypted-to-plain letter mappings are deduced from the unknown positions?"
        )
        solution = (
            f"Compare the encrypted word '{cipher_word}' with the completed word '{gold_word}', "
            f"looking only at the ? positions in '{partial_word}'. "
            f"The newly deduced mappings are: {answer}.\n\n"
            f"Answer: {answer}."
        )
        pid = self._make_id(source_id, "new_mapping_from_completed_word", f"{word_idx}_{cipher_word}_{partial_word}_{gold_word}")
        return {
            "id": pid,
            "prompt": prompt,
            "completion": solution,
            "answer": answer,
            "generated_cot": solution,
            "computed_answer": answer,
            "category": "encryption",
            "label": self._label_for_task_mode("new_mapping_from_completed_word"),
            "task_mode": "new_mapping_from_completed_word",
            "source_task_id": source_id,
            "word_index": word_idx,
        }

    def _make_final_reconstruction_task(
        self,
        source_id: str,
        partial: str,
        gold_answer: str,
    ) -> Dict[str, Any]:
        partial_words = partial.split()
        gold_words = gold_answer.split()
        resolved = []
        for p, g in zip(partial_words, gold_words):
            if "?" in p:
                resolved.append(f"{p} -> {g}")
        resolved_text = "\n".join(resolved) if resolved else "No missing words; the partial phrase is already complete."
        prompt = (
            "In Alice's Wonderland, secret encryption mappings are used on text.\n\n"
            f"Partial decrypted phrase:\n{partial}\n\n"
            f"Resolved words:\n{resolved_text}\n\n"
            "Write the final decrypted phrase. Preserve word order and spaces."
        )
        solution = (
            "Replace each partial word with its resolved full word while preserving the original word order. "
            f"The final decrypted phrase is: '{gold_answer}'.\n\n"
            f"Answer: {gold_answer}"
        )
        pid = self._make_id(source_id, "final_reconstruction", gold_answer)
        return {
            "id": pid,
            "prompt": prompt,
            "completion": solution,
            "answer": gold_answer,
            "generated_cot": solution,
            "computed_answer": gold_answer,
            "category": "encryption",
            "label": self._label_for_task_mode("final_reconstruction"),
            "task_mode": "final_reconstruction",
            "source_task_id": source_id,
            "resolved_word_count": len(resolved),
        }

    def _row_vocab(self, parts: EncryptionPromptParts, gold_answer: str) -> List[str]:
        words: List[str] = []
        for _, plain in parts.examples:
            words.extend(self._extract_words(plain))
        words.extend(self._extract_words(gold_answer))
        return sorted(set(words))

    def make_tasks_for_row(self, row: pd.Series, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        source_id = source_id or str(row.get("id", row.name))
        prompt = row.get("prompt", "")
        parts = self._parse_prompt(prompt)
        if parts is None:
            return []

        gold_answer = self._clean_answer(row.get("answer", row.get("ground_truth", ""))).lower()
        if not gold_answer:
            return []

        if self.config.only_solver_correct:
            # Works with either training columns or eval columns.
            if "is_correct" in row and not bool(row.get("is_correct")):
                return []
            if "computed" in row and not bool(row.get("computed")):
                return []

        task_modes = set(self.config.task_modes)
        mapping = self._mapping_from_examples(parts.examples)
        partial = self._apply_mapping(parts.target, mapping)
        row_vocab = self._row_vocab(parts, gold_answer)
        target_cipher_words = parts.target.lower().split()
        partial_words = partial.split()
        gold_words = gold_answer.split()

        rows: List[Dict[str, Any]] = []

        if "word_pair_mapping" in task_modes:
            pairs = self._aligned_word_pairs(parts.examples)
            if self.config.max_word_pair_tasks_per_row is not None:
                pairs = self._stable_sample_records(
                    pairs,
                    self.config.max_word_pair_tasks_per_row,
                    key=f"{source_id}_word_pair_mapping",
                )
            for ex_idx, word_idx, cipher_word, plain_word in pairs:
                rows.append(self._make_word_pair_mapping_task(source_id, ex_idx, word_idx, cipher_word, plain_word))

        if "apply_mapping_to_target_partial" in task_modes:
            has_unknown = "?" in partial
            keep_full = self._stable_keep(
                self.config.keep_full_solved_apply_mapping_frac,
                key=f"{source_id}_apply_mapping_full_solved",
            )
            if has_unknown or keep_full:
                rows.append(self._make_apply_mapping_task(source_id, parts, mapping, partial))

        incomplete_items = [
            (i, cw, pw, gw)
            for i, (cw, pw, gw) in enumerate(zip(target_cipher_words, partial_words, gold_words), start=1)
            if "?" in pw and len(pw) == len(gw) and len(cw) == len(gw)
        ]
        if self.config.max_pattern_tasks_per_row is not None:
            incomplete_items = incomplete_items[: self.config.max_pattern_tasks_per_row]

        if "pattern_completion" in task_modes:
            for i, _cw, pw, gw in incomplete_items:
                candidates = self._candidate_vocab(gw, pw, row_vocab, f"{source_id}_{i}_{pw}")
                if len(candidates) < self.config.min_pattern_vocab_size:
                    continue
                rows.append(self._make_pattern_completion_task(source_id, i, pw, gw, row_vocab))

        if "new_mapping_from_completed_word" in task_modes:
            for i, cw, pw, gw in incomplete_items:
                item = self._make_new_mapping_task(source_id, i, cw, pw, gw)
                if item is not None:
                    rows.append(item)

        if "final_reconstruction" in task_modes:
            resolved_count = sum(1 for pw in partial_words if "?" in pw)
            if resolved_count == 0:
                if self.config.include_trivial_final_reconstruction:
                    rows.append(self._make_final_reconstruction_task(source_id, partial, gold_answer))
            elif resolved_count >= 2:
                rows.append(self._make_final_reconstruction_task(source_id, partial, gold_answer))
            else:
                keep_single = self._stable_keep(
                    self.config.keep_single_resolved_final_reconstruction_frac,
                    key=f"{source_id}_final_reconstruction_single",
                )
                if keep_single:
                    rows.append(self._make_final_reconstruction_task(source_id, partial, gold_answer))

        return rows

    def _apply_global_word_pair_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        cap = self.config.max_word_pair_tasks_total
        if cap is None or not len(df) or "task_mode" not in df.columns:
            return df

        word_pair_mask = df["task_mode"].astype(str) == "word_pair_mapping"
        n_word_pair = int(word_pair_mask.sum())
        if n_word_pair <= cap:
            return df

        word_pair_df = df.loc[word_pair_mask].copy()
        other_df = df.loc[~word_pair_mask].copy()

        # Stable global sample. random_state must be deterministic and independent
        # of row order as much as possible.
        random_state = self._stable_hash_int(f"{self.seed}_global_word_pair_cap_{cap}") % (2**32)
        word_pair_df = word_pair_df.sample(n=cap, random_state=random_state).sort_index()
        return pd.concat([other_df, word_pair_df], axis=0).sort_index().reset_index(drop=True)

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
            labels = work["label"].astype(str)
            work = work[(labels == "encryption") | labels.str.startswith("encryption_") | work["prompt"].map(self.is_encryption_prompt)].copy()
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

        columns = [
            "id", "prompt", "completion", "answer", "label", "generated_cot", "computed_answer",
            "category", "task_mode", "source_task_id",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        out = pd.DataFrame.from_records(rows)
        return self._apply_global_word_pair_cap(out)


# Backward-compatible alias for existing imports.
EncryptionSubtaskAugmentGenerator = EncryptionSubtaskAugmentGeneratorV4
EncryptionCotAugmentGenerator = EncryptionSubtaskAugmentGeneratorV4
