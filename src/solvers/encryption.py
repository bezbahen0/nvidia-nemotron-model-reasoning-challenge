from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable


_WONDERLAND_PATH = Path(__file__).with_name("wonderland.txt")
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_DASH = "–"

DEFAULT_WONDERLAND_WORDS: tuple[str, ...] = tuple(
    """
    above alice ancient around beyond bird book bright castle cat cave chases clever
    colorful creates crystal curious dark discovers door dragon draws dreams explores
    follows forest found garden golden hatter hidden imagines in inside island key king
    knight library magical map message mirror mountain mouse mysterious near ocean
    palace potion princess puzzle queen rabbit reads school secret sees silver story
    strange student studies teacher the through tower treasure turtle under valley
    village watches wise wizard wonderland writes
    """.split()
)


def _clean_phrase(text: str) -> str:
    """Keep lowercase latin words separated by a single space."""
    return " ".join(re.findall(r"[a-z]+", str(text).lower()))


@lru_cache(maxsize=1)
def _load_wonderland_words() -> tuple[str, ...]:
    """Load the task vocabulary, sorted deterministically."""
    if _WONDERLAND_PATH.exists():
        text = _WONDERLAND_PATH.read_text(encoding="utf-8")
        words = [_clean_phrase(w) for w in text.split()]
        return tuple(sorted(w for w in words if w))
    return tuple(sorted(DEFAULT_WONDERLAND_WORDS))


def _word_pattern(word: str) -> tuple[int, ...]:
    """Repeated-letter pattern: 'paper' -> (0, 1, 0, 2, 3)."""
    seen: dict[str, int] = {}
    pattern: list[int] = []
    for ch in word:
        if ch not in seen:
            seen[ch] = len(seen)
        pattern.append(seen[ch])
    return tuple(pattern)


def _pattern_str(word: str) -> str:
    return "-".join(str(x) for x in _word_pattern(word))


class EncryptionSolver:
    """Wonderland monoalphabetic substitution-cipher reasoning generator.

    This solver intentionally does not use answer_hint. It follows the same
    core constraints as the reference reasoner:
      * deterministic Wonderland vocabulary;
      * cipher->plain consistency;
      * plain->cipher consistency, so the mapping is bijective;
      * repeated-letter pattern equality between cipher and candidate words.

    By default, vocabulary passed by the caller is ignored because the benchmark
    task vocabulary is wonderland.txt. Set force_wonderland=False only if you
    intentionally want to solve against a custom vocabulary.
    """

    def __init__(self, vocabulary: Iterable[str] | None = None, *, force_wonderland: bool = True):
        raw_words = _load_wonderland_words() if force_wonderland or vocabulary is None else vocabulary
        words = {_clean_phrase(w) for w in raw_words}
        words.discard("")
        self.vocab: tuple[str, ...] = tuple(sorted(words))
        self.vocab_set: set[str] = set(self.vocab)
        self._by_len: dict[int, list[str]] = defaultdict(list)
        for word in self.vocab:
            self._by_len[len(word)].append(word)

    # ------------------------------------------------------------------
    # Prompt parsing
    # ------------------------------------------------------------------
    def _extract_target(self, prompt: str) -> str | None:
        lines = prompt.lower().splitlines()
        marker = re.compile(
            r"now[, ]*decrypt(?: the)?(?: following)?(?: text)?\s*:?\s*(.*)$",
            re.IGNORECASE,
        )
        for idx, line in enumerate(lines):
            m = marker.search(line)
            if not m:
                continue
            same_line = _clean_phrase(m.group(1))
            if same_line:
                return same_line
            for nxt in lines[idx + 1:]:
                cleaned = _clean_phrase(nxt)
                if cleaned:
                    return cleaned

        m = re.search(
            r"now[, ]*decrypt(?: the)?(?: following)?(?: text)?\s*:?\s*([a-z]+(?:[ \t]+[a-z]+)*)",
            prompt.lower(),
        )
        return _clean_phrase(m.group(1)) if m else None

    def _extract_examples(self, prompt: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for raw_line in prompt.lower().splitlines():
            if "->" not in raw_line:
                continue
            left, right = raw_line.split("->", 1)
            if ":" in left:
                left = left.rsplit(":", 1)[1]
            # Some prompts include comments/explanations after the pair.
            right = right.split("/", 1)[0]
            cipher = _clean_phrase(left)
            plain = _clean_phrase(right)
            if cipher and plain:
                pairs.append((cipher, plain))
        return pairs

    # ------------------------------------------------------------------
    # Mapping and candidate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_mapping(c2p: dict[str, str], p2c: dict[str, str], c: str, p: str) -> bool:
        if c in c2p:
            return c2p[c] == p
        if p in p2c:
            return p2c[p] == c
        c2p[c] = p
        p2c[p] = c
        return True

    def _decode_word(self, cipher_word: str, c2p: dict[str, str]) -> str:
        return "".join(c2p.get(ch, "?") for ch in cipher_word)

    def _candidate_words(self, cipher_word: str, c2p: dict[str, str], p2c: dict[str, str]) -> list[str]:
        candidates: list[str] = []
        cipher_pattern = _word_pattern(cipher_word)
        for word in self._by_len.get(len(cipher_word), []):
            if _word_pattern(word) != cipher_pattern:
                continue

            local_c2p: dict[str, str] = {}
            local_p2c: dict[str, str] = {}
            ok = True
            for cc, pc in zip(cipher_word, word):
                if cc in c2p and c2p[cc] != pc:
                    ok = False
                    break
                if pc in p2c and p2c[pc] != cc:
                    ok = False
                    break
                if cc in local_c2p and local_c2p[cc] != pc:
                    ok = False
                    break
                if pc in local_p2c and local_p2c[pc] != cc:
                    ok = False
                    break
                local_c2p[cc] = pc
                local_p2c[pc] = cc
            if ok:
                candidates.append(word)
        return candidates

    def _apply_word_mapping(
        self,
        cipher_word: str,
        plain_word: str,
        c2p: dict[str, str],
        p2c: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str]] | None:
        new_c2p = dict(c2p)
        new_p2c = dict(p2c)
        for c, p in zip(cipher_word, plain_word):
            if not self._add_mapping(new_c2p, new_p2c, c, p):
                return None
        return new_c2p, new_p2c

    def _initial_mappings(self, pairs: list[tuple[str, str]]) -> tuple[dict[str, str], dict[str, str], bool, list[str]]:
        c2p: dict[str, str] = {}
        p2c: dict[str, str] = {}
        ok = True
        trace: list[str] = []

        for cipher, plain in pairs:
            c_words = cipher.split()
            p_words = plain.split()
            if len(c_words) == len(p_words):
                word_pairs = zip(c_words, p_words)
            else:
                word_pairs = [(cipher.replace(" ", ""), plain.replace(" ", ""))]

            for cw, pw in word_pairs:
                if len(cw) != len(pw):
                    trace.append(f"Skipped {cw}->{pw}: different lengths.")
                    ok = False
                    continue
                for cc, pc in zip(cw, pw):
                    if not self._add_mapping(c2p, p2c, cc, pc):
                        trace.append(f"Conflict: {cc} cannot map to {pc}.")
                        ok = False
        return c2p, p2c, ok, trace

    def _solve_target(
        self,
        target_words: list[str],
        c2p: dict[str, str],
        p2c: dict[str, str],
    ) -> tuple[list[str], dict[str, str], list[dict[str, object]]] | None:
        """Backtracking with minimum-candidate-first word choice."""

        def recurse(
            cur_c2p: dict[str, str],
            cur_p2c: dict[str, str],
            cur_steps: list[dict[str, object]],
        ) -> tuple[list[str], dict[str, str], list[dict[str, object]]] | None:
            decoded = [self._decode_word(w, cur_c2p) for w in target_words]
            if all("?" not in w for w in decoded):
                if all(w in self.vocab_set for w in decoded):
                    return decoded, cur_c2p, cur_steps
                return None

            best_idx: int | None = None
            best_candidates: list[str] | None = None
            for idx, cipher_word in enumerate(target_words):
                if "?" not in decoded[idx]:
                    continue
                candidates = self._candidate_words(cipher_word, cur_c2p, cur_p2c)
                if not candidates:
                    return None
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_idx = idx
                    best_candidates = candidates
                    if len(candidates) == 1:
                        break

            assert best_idx is not None and best_candidates is not None
            cipher_word = target_words[best_idx]
            partial = decoded[best_idx]

            for candidate in best_candidates:
                applied = self._apply_word_mapping(cipher_word, candidate, cur_c2p, cur_p2c)
                if applied is None:
                    continue
                next_c2p, next_p2c = applied
                new_mappings = [
                    f"{cc}->{pc}"
                    for cc, pc in zip(cipher_word, candidate)
                    if cc not in cur_c2p
                ]
                step = {
                    "cipher_word": cipher_word,
                    "partial": partial,
                    "candidate_count": len(best_candidates),
                    "candidates": list(best_candidates),
                    "chosen": candidate,
                    "new_mappings": new_mappings,
                    "mapping_before": dict(cur_c2p),
                }
                solved = recurse(next_c2p, next_p2c, cur_steps + [step])
                if solved is not None:
                    return solved
            return None

        return recurse(dict(c2p), dict(p2c), [])

    # ------------------------------------------------------------------
    # CoT formatting
    # ------------------------------------------------------------------
    def _append_examples_cot(self, cot: list[str], pairs: list[tuple[str, str]], c2p: dict[str, str]) -> None:
        running: dict[str, str] = {}
        p_running: dict[str, str] = {}
        for cipher, plain in pairs:
            cot.append("")
            plain_quoted = " ".join(f"【{w}】" for w in plain.split())
            cot.append(f"【{cipher}】 -> 【{plain}】 / {plain_quoted}:")

            c_words = cipher.split()
            p_words = plain.split()
            if len(c_words) == len(p_words):
                word_pairs = zip(c_words, p_words)
            else:
                word_pairs = [(cipher.replace(" ", ""), plain.replace(" ", ""))]

            for wi, (cw, pw) in enumerate(word_pairs):
                if len(cw) != len(pw):
                    cot.append(f"Skipping 【{cw}】 -> 【{pw}】 because lengths differ.")
                    continue
                if wi > 0:
                    cot.append("")
                cot.append(f"【{cw}】->【{pw}】")
                cot.append(f"{_DASH.join(cw)}->{_DASH.join(pw)}")
                for cc, pc in zip(cw, pw):
                    status = "same" if cc in running else "new"
                    if cc not in running and pc not in p_running:
                        running[cc] = pc
                        p_running[pc] = cc
                    cot.append(f"{cc}->{pc} {status}")

        mapping_lines = "\n".join(f"{c}->{c2p.get(c, '?')}" for c in _ALPHABET)
        inv = {v: k for k, v in c2p.items()}
        inv_lines = "\n".join(f"{p}->{inv.get(p, '?')}" for p in _ALPHABET)
        unknown = "\n".join(c for c in _ALPHABET if c not in c2p)
        unmapped = "\n".join(p for p in _ALPHABET if p not in inv)
        cot.append("")
        cot.append(f"Mapping so far\n{mapping_lines}")
        cot.append(f"Inverse mapping\n{inv_lines}")
        cot.append(f"Unknown characters\n{unknown}")
        cot.append(f"Unmapped target letters\n{unmapped}")

    def _append_target_initial_cot(
        self,
        cot: list[str],
        target_cipher: str,
        target_words: list[str],
        c2p: dict[str, str],
    ) -> None:
        cot.append("")
        cot.append(f"Now decrypting 【 {target_cipher}】:")
        decoded_parts: list[str] = []
        all_unknown: set[str] = set()

        for i, cw in enumerate(target_words):
            if i > 0:
                cot.append("")
            display_chars: list[str] = []
            step_lines: list[str] = []
            for cc in cw:
                if cc in c2p:
                    display_chars.append(c2p[cc])
                    step_lines.append(f"{cc}->{c2p[cc]}")
                else:
                    display_chars.append(f"({cc})")
                    step_lines.append(f"{cc}->?")
                    all_unknown.add(cc)
            partial_display = _DASH.join(display_chars)
            plain_word = "".join(c2p.get(cc, "?") for cc in cw)
            decoded_parts.append(plain_word if "?" not in plain_word else partial_display)
            cot.append(f"【 {cw}】")
            cot.append(_DASH.join(cw))
            cot.extend(step_lines)
            cot.append(f"{partial_display}->【{plain_word if '?' not in plain_word else partial_display}】")

        cot.append("")
        cot.append("The sentence currently is")
        cot.append(" ".join(decoded_parts))
        cot.append("")
        if all_unknown:
            cot.append("Iterating over the unknown letters to see if they are in the question")
            cot.extend(f"{c} {'yes' if c in all_unknown else 'no'}" for c in _ALPHABET if c not in c2p)
            cot.append("")
            cot.append("The unknown letters")
            cot.extend(sorted(all_unknown))
            cot.append("")
            cot.append("Let me find the best matching wonderland words:")
        else:
            cot.append("There are no unknown letters in the question.")

    def _append_solution_steps_cot(self, cot: list[str], steps: list[dict[str, object]]) -> None:
        for step in steps:
            cipher_word = str(step["cipher_word"])
            partial = str(step["partial"])
            candidates = list(step["candidates"])
            chosen = str(step["chosen"])
            mapping_before = dict(step["mapping_before"])
            new_mappings = list(step["new_mappings"])

            display_dashed = _DASH.join(mapping_before.get(cc, f"({cc})") for cc in cipher_word)
            cot.append("")
            cot.append(f"【{display_dashed}】")
            cot.append(f"The length of the word is {len(cipher_word)}.")
            cot.append(f"The cipher pattern is {_pattern_str(cipher_word)}.")
            cot.append(f"The partial plaintext pattern is {partial}.")
            cot.append("Checking Wonderland words with the same length, same repeated-letter pattern, and bijective mapping:")

            if candidates:
                for cand in candidates:
                    cand_dashed = _DASH.join(cand)
                    marker = "best" if cand == chosen else "possible"
                    cot.append(f"{cand} {len(cand)} 【{cand_dashed}】 {marker}")
            else:
                cot.append("No candidates.")

            cot.append(f"Best match: 【{chosen}】")
            cot.append(f"【{display_dashed}】->【{_DASH.join(chosen)}】")
            for cc, pc in zip(cipher_word, chosen):
                if cc in mapping_before:
                    cot.append(f"【{mapping_before[cc]}】->【{pc}】same")
                else:
                    cot.append(f"【({cc})】->【{pc}】 new")
            if new_mappings:
                cot.append("Added mappings")
                cot.extend(str(x) for x in new_mappings)
            else:
                cot.append("Added mappings\nnone")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_cot(self, prompt: str, ) -> str:
        prompt = str(prompt).lower()
        target_cipher = self._extract_target(prompt)
        if not target_cipher:
            return "The target ciphertext could not be found in the prompt.\nFinal Answer: nan"

        pairs = self._extract_examples(prompt)
        c2p, p2c, ok, trace = self._initial_mappings(pairs)
        if not ok:
            details = "\n".join(trace)
            return f"Algorithmic Error: inconsistent examples were found.\n{details}\nFinal Answer: nan"

        target_words = target_cipher.split()

        cot: list[str] = []
        cot.append("We need to find the encryption mapping from the examples. It looks like a substitution cipher.")
        cot.append("I will put my final answer inside \\boxed{}.")
        
        cot.append("")
        cot.append("Listing the input words:")
        for cipher, _plain in pairs:
            cot.append("")
            cot.append(f"【{cipher}】")
            for word in cipher.split():
                cot.append(f" {word}")
        cot.append("")
        cot.append(f"【 {target_cipher}】")
        for word in target_words:
            cot.append(f" {word}")

        cot.append("")
        cot.append("Breaking down into characters:")
        for cipher, _plain in pairs:
            cot.append("")
            cot.append(f"【{cipher}】")
            for word in cipher.split():
                cot.append(_DASH.join(word))
        cot.append("")
        cot.append(f"【 {target_cipher}】")
        for word in target_words:
            cot.append(_DASH.join(word))

        self._append_examples_cot(cot, pairs, c2p)
        self._append_target_initial_cot(cot, target_cipher, target_words, c2p)

        solved = self._solve_target(target_words, c2p, p2c)
        if solved is None:
            cot.append("")
            cot.append("Algorithmic Error: no vocabulary-consistent bijective solution was found.")
            cot.append("Final Answer: nan")
            return "\n".join(cot)

        decoded_words, final_c2p, steps = solved
        self._append_solution_steps_cot(cot, steps)

        final_answer = " ".join(decoded_words)
        final_map = "\n".join(f"{c}->{final_c2p.get(c, '?')}" for c in _ALPHABET)
        cot.append("")
        cot.append(f"Final mapping\n{final_map}")
        cot.append("I will now return the answer in \\boxed{}")
        cot.append(f"The answer is \\boxed{{{final_answer}}}")
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text:
            return "nan"
        text = str(cot_text)
        lowered = text.lower()
        if "final answer: nan" in lowered or "algorithmic error" in lowered:
            return "nan"
        # The CoT intentionally contains an empty \boxed{} in the instruction line,
        # so use the last non-empty boxed value rather than the first one.
        boxed_values = re.findall(r"\\boxed\{([^}]*)\}", text)
        for value in reversed(boxed_values):
            cleaned = _clean_phrase(value)
            if cleaned:
                return cleaned
        match = re.search(r"final answer:\s*([a-z ]+)", lowered)
        return _clean_phrase(match.group(1)) if match else "nan"
