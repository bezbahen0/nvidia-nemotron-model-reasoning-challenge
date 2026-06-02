from __future__ import annotations

import ast
import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


CRYPTARITHM_PROMPT_MARKER = "secret set of transformation rules is applied to equations"


@dataclass
class CryptarithmAugmentConfig:
    sample_frac: float = 1.0
    sample_n: Optional[int] = None
    only_solver_correct: bool = False
    include_rule_filtering: bool = True
    include_candidate_check: bool = True
    include_domain_propagation: bool = True
    include_rule_verification: bool = True
    include_target_application: bool = True
    include_ambiguity_audit: bool = True
    # For final-answer training, skip target-application rows from sources whose
    # audit says the target is not unique. Local subtasks can still be used.
    skip_ambiguous_target_application: bool = True
    # Add a compact legend for std/rev/add1/mulm1/etc. so each subtask is
    # understandable without hidden solver state.
    include_rule_semantics_in_prompts: bool = True
    # Store lightweight QA metadata in each generated row. This is useful for
    # filtering training data without reparsing prompts/completions.
    include_quality_metadata: bool = True

    # Balanced extras: these do not change the source solution. They derive
    # extra local tasks from existing replay evidence so answer labels are less
    # degenerate inside each subtask type.
    include_candidate_keep_from_projections: bool = True
    include_domain_contradictions_from_rejects: bool = True
    include_contrastive_rule_verification_wrong: bool = True
    max_synthetic_candidate_keep_per_source: int = 2
    max_synthetic_domain_contradiction_per_source: int = 10
    max_contrastive_wrong_verification_per_source: int = 10

    # Optional downsampling after augmentation. It keeps every minority label and
    # caps majority labels inside binary/multiclass decision modes. This is safer
    # than fabricating unlimited negatives.
    balance_answer_labels: bool = False
    max_answer_label_ratio: float = 2.0


class CryptarithmAugmentGenerator:
    """
    Augmenter for cryptarithm replay CoTs produced by cryptarithm_replay_v2_3.

    Same philosophy as NumeralEquationAugmentGenerator:
      - do not create random full tasks;
      - derive focused subtasks from already-solved source prompts / CoTs;
      - keep prompts self-contained and in the "In Alice's Wonderland" style.

    Produced task modes:
      1. cryptarithm_rule_filtering
         Structural pre-check for one operator: length/sign family filtering.

      2. cryptarithm_candidate_check
         One rule-combo candidate check: test -> local check -> decision.

      3. cryptarithm_domain_propagation
         One state-changing local domain operation: no-leading-zero, projection,
         AllDifferent, or branch assignment.

      4. cryptarithm_rule_verification
         Verify one example using selected rule and digit map.

      5. cryptarithm_target_application
         Apply selected rule and digit map to the target expression.

      6. cryptarithm_ambiguity_audit
         Interpret the final target-invariance audit: unique, ambiguous, or incomplete.
    """

    def __init__(self, seed: Optional[int] = None, config: Optional[CryptarithmAugmentConfig] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config or CryptarithmAugmentConfig()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_cryptarithm_prompt(prompt: Any) -> bool:
        return CRYPTARITHM_PROMPT_MARKER in ("" if prompt is None else str(prompt))

    @staticmethod
    def _stable_hash_int(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _make_id(source_id: str, task_type: str, key: str) -> str:
        return hashlib.sha256(f"cryptarithm_{task_type}_{source_id}_{key}".encode()).hexdigest()[:12]

    @staticmethod
    def _first_present(row: pd.Series, names: Sequence[str], default: str = "") -> str:
        for name in names:
            if name in row and pd.notna(row[name]):
                value = str(row[name])
                if value.strip():
                    return value
        return default

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        # Prefer the explicit Final answer line because cryptarithm symbols can include braces.
        matches = re.findall(r"(?m)^Final answer:\s*(\S+)\s*$", text or "")
        return matches[-1] if matches else ""

    @staticmethod
    def _section(text: str, start_pat: str, end_pat: str) -> str:
        m = re.search(start_pat, text or "", flags=re.M)
        if not m:
            return ""
        tail = text[m.end():]
        e = re.search(end_pat, tail, flags=re.M)
        return tail[: e.start()].strip("\n") if e else tail.strip("\n")

    @staticmethod
    def _indent(text: str, prefix: str = "  ") -> str:
        return "\n".join(prefix + line if line else line for line in str(text).splitlines())

    @staticmethod
    def _compact_lines(text: str) -> List[str]:
        return [ln.rstrip() for ln in str(text or "").splitlines() if ln.strip()]

    # ------------------------------------------------------------------
    # Source dump parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_text_dump(text: str) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        parts = re.split(r"(?=^=== Категория:)", text or "", flags=re.M)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            mid = re.search(r"ID:\s*([^\s=]+)", part)
            source_id = mid.group(1) if mid else hashlib.sha256(part.encode()).hexdigest()[:8]
            mp = re.search(r"--- Промпт \(начало\) ---\n(.*?)\n\n--- Решение ---", part, flags=re.S)
            ms = re.search(r"--- Решение ---\n(.*)$", part, flags=re.S)
            if not mp or not ms:
                continue
            prompt = mp.group(1).strip()
            cot = ms.group(1).strip()
            records.append(
                {
                    "id": source_id,
                    "prompt": prompt,
                    "generated_cot": cot,
                    "answer": CryptarithmAugmentGenerator._extract_final_answer(cot),
                    "label": "cryptarithm",
                }
            )
        return pd.DataFrame.from_records(records)

    # ------------------------------------------------------------------
    # Shared extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_digit_map(cot: str) -> str:
        m = re.search(r"(?ms)^Selected digit map\n(.*?)(?=\n\nVerify selected rules|\nVerify selected rules|\Z)", cot or "")
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_rule_combo_order(cot: str) -> str:
        m = re.search(r"(?m)^Rule-combo order:\s*(.+)$", cot or "")
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_examples_header(cot: str) -> str:
        return CryptarithmAugmentGenerator._section(cot, r"^Examples\s*$", r"^Parsed equations\s*$")

    @staticmethod
    def _extract_rule_families_text() -> str:
        return "\n".join(
            [
                "1. concat: std/cat, rev/cat",
                "2. add_offset: std/add, std/add1, std/addm1, rev/add, rev/add1, rev/addm1",
                "3. abs_sub: std/abs, std/sub, rev/abs, rev/sub",
                "4. mul_offset: std/mul, std/mul1, std/mulm1, rev/mul, rev/mul1, rev/mulm1",
            ]
        )

    @staticmethod
    def _rule_semantics_text() -> str:
        return "\n".join(
            [
                "std: read each visible number left-to-right; rev: read each visible number right-to-left.",
                "cat: concatenate the two decoded operands.",
                "add/add1/addm1: compute A+B, A+B+1, or A+B-1.",
                "abs/sub: compute abs(A-B) or signed A-B.",
                "mul/mul1/mulm1: compute A*B, A*B+1, or A*B-1.",
                "Encode the numeric result back with the same injective digit map; signed negative outputs use the operator symbol as the minus sign when present in the CoT.",
            ]
        )

    def _add_rule_semantics(self, prompt_lines: List[str]) -> None:
        """Optionally insert the shared rule legend before output instructions.

        The legend is prompt context, not an output requirement, so place it
        before the first "Return ..." instruction when that instruction already
        exists; otherwise append it to the current context block.
        """
        if not self.config.include_rule_semantics_in_prompts:
            return
        block = ["", "Rule notation:", self._rule_semantics_text()]
        insert_at: Optional[int] = None
        for idx, line in enumerate(prompt_lines):
            if str(line).startswith("Return "):
                insert_at = idx
                break
        if insert_at is None:
            prompt_lines.extend(block)
        else:
            prompt_lines[insert_at:insert_at] = block

    @staticmethod
    def _answer_from_survivors(block_text: str) -> str:
        m = re.search(r"(?m)^surviving after structural filtering:\s*(.+)$", block_text or "")
        return m.group(1).strip() if m else ""

    @staticmethod
    def _answer_from_decision(block_text: str) -> str:
        # For candidate checks.
        md = re.search(r"(?m)^\s*decision:\s*(.+)$", block_text or "")
        if md:
            val = md.group(1).strip().lower()
            if val.startswith("reject"):
                return "reject"
            if val.startswith("keep") or "remains possible" in val:
                return "keep"
        # For domain propagation.
        md = re.search(r"(?m)^\s*decision:\s*(.+)$", block_text or "")
        if md:
            val = md.group(1).strip().lower()
            if "contradiction" in val or "empty" in val or val.startswith("reject"):
                return "contradiction"
            return "keep"
        return ""

    @staticmethod
    def _extract_apply_update_line(block_text: str) -> str:
        m = re.search(r"(?m)^\s*apply update:\s*(.+)$", block_text or "")
        return m.group(1).strip() if m else ""

    @staticmethod
    def _domain_update_pattern() -> re.Pattern[str]:
        # Domain tokens can contain symbols such as ] or a quote, for example:
        #   D[']']: {1,2,3} -> {1,2}
        #   D["'" ] is not produced, but D["'"] is.
        # A simple D\[[^\]]+\] regex breaks on D[']'], so parse quoted
        # symbol literals inside D[...] explicitly.
        return re.compile(
            r"(D\[(?:'[^']*'|\"[^\"]*\"|[^\]]+)\])"
            r"\s*:\s*"
            r"(\{[^{}]*\})"
            r"\s*->\s*"
            r"(\{[^{}]*\})"
        )

    @staticmethod
    def _extract_updates(update_text: str) -> List[str]:
        update_text = (update_text or "").strip()
        if not update_text or update_text.lower() == "none":
            return []
        pattern = CryptarithmAugmentGenerator._domain_update_pattern()
        return [
            f"{m.group(1)}: {m.group(2)} -> {m.group(3)}"
            for m in pattern.finditer(update_text)
        ]

    @staticmethod
    def _current_domains_from_update(update_text: str) -> List[str]:
        rows: List[str] = []
        pattern = CryptarithmAugmentGenerator._domain_update_pattern()
        for m in pattern.finditer(update_text or ""):
            # D['x']: {before} -> {after}  ==>  D['x']={before}
            rows.append(f"{m.group(1)}={m.group(2).strip()}")
        return rows


    @staticmethod
    def _domain_values_to_set(text: str) -> set[int]:
        text = (text or "").strip()
        if text == "{0..9}":
            return set(range(10))
        if not (text.startswith("{") and text.endswith("}")):
            return set()
        body = text[1:-1].strip()
        if not body:
            return set()
        out: set[int] = set()
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if ".." in part:
                lo, hi = part.split("..", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        return out

    @staticmethod
    def _set_to_domain_values(values: Iterable[int]) -> str:
        vals = sorted(set(int(v) for v in values))
        if vals == list(range(10)):
            return "{0..9}"
        return "{" + ",".join(str(v) for v in vals) + "}"

    @staticmethod
    def _removed_digits_from_update(update_text: str) -> set[int]:
        """Return digits removed by an apply-update line.

        This is primarily used to make AllDifferent subtasks self-contained:
        if an AllDifferent block updates D[x]: {1,2,9} -> {1,2}, then 9 is
        the fixed digit that must be removed from non-fixed domains.
        """
        removed: set[int] = set()
        pattern = CryptarithmAugmentGenerator._domain_update_pattern()
        for m in pattern.finditer(update_text or ""):
            before = CryptarithmAugmentGenerator._domain_values_to_set(m.group(2))
            after = CryptarithmAugmentGenerator._domain_values_to_set(m.group(3))
            removed.update(before - after)
        return removed

    @staticmethod
    def _remove_lines(block_text: str, prefixes: Sequence[str]) -> str:
        out: List[str] = []
        for ln in str(block_text or "").splitlines():
            stripped = ln.strip()
            if any(stripped.startswith(p) for p in prefixes):
                continue
            out.append(ln.rstrip())
        return "\n".join(out).strip()

    # ------------------------------------------------------------------
    # 1. Rule filtering extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rule_filtering_blocks(cot: str) -> List[Dict[str, Any]]:
        rule_section = CryptarithmAugmentGenerator._section(cot, r"^Rule matching\s*$", r"^Local replay steps\s*$")
        blocks: List[Dict[str, Any]] = []
        if not rule_section:
            return blocks
        pattern = re.compile(r"(?ms)^Operator\s+(.+?)\n(.*?)(?=^Operator\s+.+?\n|\Z)")
        for m in pattern.finditer(rule_section):
            op_text = m.group(1).strip()
            body = m.group(2).strip()
            full = f"Operator {op_text}\n{body}".strip()
            examples = ""
            mm = re.search(r"(?m)^examples:\s*(.+)$", body)
            if mm:
                examples = mm.group(1).strip()
            signature = ""
            msig = re.search(r"(?m)^structural signature:\s*(.+)$", body)
            if msig:
                signature = msig.group(1).strip()
            answer = CryptarithmAugmentGenerator._answer_from_survivors(full)
            if answer:
                blocks.append(
                    {
                        "operator": op_text,
                        "examples": examples,
                        "signature": signature,
                        "block_text": full,
                        "answer": answer,
                    }
                )
        return blocks

    def _make_rule_filtering_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
        idx: int,
    ) -> Dict[str, Any]:
        prompt_lines = [
            "In Alice's Wonderland, a hidden arithmetic rule is assigned to one operator symbol.",
            "Your task is to perform only the structural rule filtering step for this operator.",
            "Do not solve the digit map. Use only structural checks: output length, signed-output information, concat positional equality, and the rule family definitions.",
            "",
            f"Operator: {item['operator']}",
            "",
            "Examples:",
            item.get("examples", ""),
            "",
            f"Structural signature: {item.get('signature', '')}",
            "",
            "Rule families:",
            self._extract_rule_families_text(),
            "",
            "Return the structural filtering blocks and finish with:",
            "Answer: <comma-separated surviving candidates>",
        ]
        self._add_rule_semantics(prompt_lines)
        completion = item["block_text"].rstrip() + "\nAnswer: " + item["answer"]
        return self._row(
            source_id,
            "rule_filtering",
            f"{item['operator']}_{idx}",
            "cryptarithm_rule_filtering",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"operator": item["operator"]},
        )

    # ------------------------------------------------------------------
    # 2. Candidate check extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_candidate_check_blocks(cot: str) -> List[Dict[str, Any]]:
        local = CryptarithmAugmentGenerator._section(cot, r"^Local replay steps\s*$", r"^Selected digit map\s*$")
        if not local:
            return []
        # Start only at causal test blocks. Full-CoT traces can contain either
        #   Combo block N: test operator '+' -> std/add
        # or
        #   Combo block N: test prefix assignment operator '+' -> std/add
        # The latter exposes downstream continuation rejects; older augmentation
        # silently missed those examples.
        pattern = re.compile(
            r"(?ms)^Combo block\s+(\d+):\s+test\s+(prefix assignment\s+)?operator\s+(.+?)\s*->\s*([^\n]+)\n"
            r"(.*?)(?=^(?:Combo block|No-leading-zero block|Target no-leading-zero block|Projection block|AllDifferent block|Branch block|Map block|Selected digit map|Verify selected rules|Target)\b|\Z)"
        )
        out: List[Dict[str, Any]] = []
        for m in pattern.finditer(local):
            number = int(m.group(1))
            is_prefix = bool(m.group(2))
            op = m.group(3).strip()
            rule = m.group(4).strip()
            body = m.group(5).rstrip()
            header = "test prefix assignment operator" if is_prefix else "test operator"
            full = f"Combo block {number}: {header} {op} -> {rule}\n{body}".rstrip()
            answer = CryptarithmAugmentGenerator._answer_from_decision(full) or "reject"
            check_input = CryptarithmAugmentGenerator._remove_lines(body, ["apply update:", "result:", "decision:"])
            out.append(
                {
                    "number": number,
                    "operator": op,
                    "rule": rule,
                    "block_text": full,
                    "check_input": check_input,
                    "answer": answer,
                    "candidate_check_kind": "prefix_assignment" if is_prefix else "single_candidate",
                }
            )
        return out

    def _make_candidate_check_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_lines = [
            "In Alice's Wonderland, a candidate rule or rule-prefix is being tested during rule-combo search.",
            "Your task is to run the shown local check and decide whether this candidate/prefix is rejected or remains possible.",
            "Use the formula and the supported projection. If the supported projection is empty, the candidate is rejected.",
            "",
            "Candidate:",
            f"operator {item['operator']} -> {item['rule']}",
            "",
            "Local check:",
            item.get("check_input", ""),
            "",
            "Return the local check and finish with:",
            "Answer: reject or Answer: keep",
        ]
        self._add_rule_semantics(prompt_lines)
        completion = item["block_text"].rstrip() + "\nAnswer: " + item["answer"]
        return self._row(
            source_id,
            "candidate_check",
            f"{item['number']}_{item['operator']}_{item['rule']}",
            "cryptarithm_candidate_check",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"operator": item["operator"], "rule": item["rule"], "block_number": item["number"], "candidate_check_kind": item.get("candidate_check_kind", "single_candidate")},
        )

    # ------------------------------------------------------------------
    # 3. Domain propagation extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_block_start_pattern() -> re.Pattern[str]:
        # Domain-propagation examples may start only at one of these local
        # state-transition blocks. A Branch block is included only when it is
        # an explicit assignment, not when it merely chooses a symbol/tuple.
        # Full-CoT traces additionally contain Target no-leading-zero and
        # tuple-branch assignments; older augmentation missed both.
        return re.compile(
            r"(?m)^(No-leading-zero block|Target no-leading-zero block|Projection block|AllDifferent block)\s+(\d+)"
            r"|^(Branch block)\s+(\d+):\s+try\s+(set|tuple)\s+(.+)$"
        )

    @staticmethod
    def _major_replay_header_pattern() -> re.Pattern[str]:
        # Hard boundary for one local domain-propagation task.  The previous
        # extractor stopped only at the next domain block, which allowed a
        # Projection/AllDifferent block to swallow following Combo/Map/Branch
        # search-flow blocks.  That breaks the locality of the subtask.
        return re.compile(
            r"(?m)^(?:"
            r"No-leading-zero block\s+\d+"
            r"|Target no-leading-zero block\s+\d+"
            r"|Projection block\s+\d+"
            r"|AllDifferent block\s+\d+"
            r"|Branch block\s+\d+:"
            r"|Combo block\s+\d+:"
            r"|Map block\s+\d+:"
            r"|Selected digit map"
            r"|Verify selected rules"
            r"|Target"
            r")(?=\s|$)"
        )

    @staticmethod
    def _next_major_header_start(local: str, start: int) -> int:
        # Return the start index of the first major header after `start`.
        # Skip the header at `start` itself.
        pattern = CryptarithmAugmentGenerator._major_replay_header_pattern()
        for m in pattern.finditer(local, pos=start + 1):
            return m.start()
        return len(local)

    @staticmethod
    def _extract_hall_digits(block_text: str) -> List[int]:
        m = re.search(r"(?m)^\s*Hall set:\s+symbols\s+.+?;\s+union digits\s+(\{[^{}]*\})\s*$", block_text or "")
        if not m:
            return []
        return sorted(CryptarithmAugmentGenerator._domain_values_to_set(m.group(1)))

    @staticmethod
    def _extract_domain_blocks(cot: str) -> List[Dict[str, Any]]:
        local = CryptarithmAugmentGenerator._section(cot, r"^Local replay steps\s*$", r"^Selected digit map\s*$")
        if not local:
            return []
        starts = list(CryptarithmAugmentGenerator._domain_block_start_pattern().finditer(local))
        blocks: List[Dict[str, Any]] = []
        for m in starts:
            start = m.start()
            end = CryptarithmAugmentGenerator._next_major_header_start(local, start)
            raw = local[start:end].rstrip()
            first = raw.splitlines()[0].strip() if raw.splitlines() else ""
            if first.startswith("Target no-leading-zero"):
                kind = "target_no_leading_zero"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("No-leading-zero"):
                kind = "no_leading_zero"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("Projection"):
                kind = "projection"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("AllDifferent"):
                kind = "alldifferent"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("Branch") and "try tuple" in first:
                kind = "branch_tuple_assignment"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("Branch") and "try set" in first:
                kind = "branch_assignment"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            else:
                continue
            # Skip blocks without an apply update line; they are not domain propagation examples.
            update = CryptarithmAugmentGenerator._extract_apply_update_line(raw)
            if not update:
                continue
            answer = CryptarithmAugmentGenerator._answer_from_decision(raw) or "keep"
            # For prompt, remove apply update and decision so the model has to produce them.
            prompt_block = CryptarithmAugmentGenerator._remove_lines(raw, ["apply update:", "decision:"])
            current_domains = CryptarithmAugmentGenerator._current_domains_from_update(update)
            alldiff_digits_to_remove: List[int] = []
            alldiff_reason_kind = ""
            if kind == "alldifferent":
                hall_digits = CryptarithmAugmentGenerator._extract_hall_digits(raw)
                if hall_digits:
                    alldiff_digits_to_remove = hall_digits
                    alldiff_reason_kind = "hall_reserved_digits"
                else:
                    alldiff_digits_to_remove = sorted(CryptarithmAugmentGenerator._removed_digits_from_update(update))
                    alldiff_reason_kind = "fixed_digits"
                # If we cannot reconstruct which digits caused the update,
                # the task would not be self-contained. Do not emit it.
                if not alldiff_digits_to_remove:
                    continue
            blocks.append(
                {
                    "number": num,
                    "kind": kind,
                    "block_text": raw,
                    "prompt_block": prompt_block,
                    "current_domains": current_domains,
                    "fixed_digits_to_remove": alldiff_digits_to_remove,  # backward-compatible column name
                    "alldiff_digits_to_remove": alldiff_digits_to_remove,
                    "alldiff_reason_kind": alldiff_reason_kind,
                    "answer": "contradiction" if answer == "contradiction" else "keep",
                }
            )
        return blocks

    def _make_domain_propagation_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        kind_intro = {
            "projection": "digit domains are being narrowed by one local projection step.",
            "no_leading_zero": "digit domains are being narrowed by the no-leading-zero rule.",
            "target_no_leading_zero": "target operand domains are being narrowed by the target no-leading-zero rule.",
            "alldifferent": "digit domains are being narrowed by the AllDifferent rule.",
            "branch_assignment": "a branch assignment is being applied to the digit domains.",
            "branch_tuple_assignment": "a tuple branch assignment is being applied to multiple digit domains.",
            "projection_contradiction": "digit domains are being tested by a local projection step that may have no support.",
        }.get(item["kind"], "digit domains are being narrowed by one local step.")
        prompt_lines = [
            f"In Alice's Wonderland, {kind_intro}",
            "Your task is to apply this local domain operation and report the resulting domain updates.",
            "",
            "Local operation:",
            item.get("prompt_block", ""),
        ]
        self._add_rule_semantics(prompt_lines)
        if item.get("kind") == "alldifferent" and item.get("alldiff_digits_to_remove"):
            if item.get("alldiff_reason_kind") == "hall_reserved_digits":
                label = "Hall-reserved digits to remove from symbols outside the Hall set:"
            else:
                label = "Fixed digits to remove from non-fixed domains:"
            prompt_lines.extend(
                [
                    "",
                    label,
                    self._set_to_domain_values(item["alldiff_digits_to_remove"]),
                ]
            )
        if item.get("current_domains"):
            prompt_lines.extend(["", "Current domains needed for the update:"])
            prompt_lines.extend(item["current_domains"])
        prompt_lines.extend(
            [
                "",
                "Return the domain updates and finish with:",
                "Answer: keep or Answer: contradiction",
            ]
        )
        completion = item["block_text"].rstrip() + "\nAnswer: " + item["answer"]
        return self._row(
            source_id,
            "domain_propagation",
            f"{item['number']}_{item['kind']}",
            "cryptarithm_domain_propagation",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"block_number": item["number"], "domain_step_kind": item["kind"]},
        )

    # ------------------------------------------------------------------
    # 4. Rule verification extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rule_verification_tasks(cot: str) -> List[Dict[str, Any]]:
        # Verify-selected-rules is followed by Ambiguity audit in the final solver
        # and only then by Target.  Stop at the first of these headers; otherwise
        # the last verification example can accidentally absorb the audit block.
        section = CryptarithmAugmentGenerator._section(
            cot,
            r"^Verify selected rules\s*$",
            r"^(?:Ambiguity audit|Target)\s*$",
        )
        if not section:
            return []
        tasks: List[Dict[str, Any]] = []
        current_rule: Optional[Tuple[str, str]] = None
        lines = section.splitlines()
        hard_stop_headers = (
            "Ambiguity audit",
            "Target",
            "Selected digit map",
            "Local replay steps",
            "Rule matching",
        )
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if any(line.startswith(h) for h in hard_stop_headers):
                break
            mr = re.match(r"^Operator\s+(.+?)\s+uses\s+(.+)$", line)
            if mr:
                current_rule = (mr.group(1).strip(), mr.group(2).strip())
                i += 1
                continue
            if current_rule and line.startswith("- "):
                example = line[2:].strip()
                calc: List[str] = []
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].rstrip()
                    if (
                        nxt.startswith("Operator ")
                        or nxt.startswith("- ")
                        or any(nxt.startswith(h) for h in hard_stop_headers)
                    ):
                        break
                    calc.append(nxt)
                    j += 1
                text = "\n".join(calc).strip()
                ma = re.search(r"->\s*(MATCH|WRONG)\s*$", text, flags=re.M)
                answer = ma.group(1) if ma else "MATCH"
                # Keep only the local verification lines.  This guard prevents
                # unrelated source-level sections from entering the subtask
                # even if the source CoT format changes later.
                clean_calc = [
                    (ln[2:] if ln.startswith("  ") else ln)
                    for ln in calc
                    if ln.strip()
                    and not any(ln.lstrip().startswith(h) for h in hard_stop_headers)
                ]
                tasks.append(
                    {
                        "operator": current_rule[0],
                        "rule": current_rule[1],
                        "example": example,
                        "calc_lines": clean_calc,
                        "answer": answer,
                    }
                )
                i = j
                continue
            i += 1
        return tasks

    def _make_rule_verification_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
        idx: int,
        digit_map: str,
    ) -> Dict[str, Any]:
        prompt_lines = [
            "In Alice's Wonderland, a hidden operator rule and digit map have already been found.",
            "Your task is to verify one example.",
            "Decode the visible operands using the digit map, apply the selected rule, encode the numeric result back into symbols, and compare with the expected output.",
            "",
            "Selected rule:",
            f"operator {item['operator']} uses {item['rule']}",
            "",
            "Digit map:",
            digit_map,
            "",
            "Example:",
            item["example"],
            "",
            "Return the verification and finish with:",
            "Answer: MATCH or Answer: WRONG",
        ]
        self._add_rule_semantics(prompt_lines)
        completion = "\n".join(item["calc_lines"] + [f"Answer: {item['answer']}"])
        return self._row(
            source_id,
            "rule_verification",
            f"{idx}_{item['operator']}_{item['example']}",
            "cryptarithm_rule_verification",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"operator": item["operator"], "rule": item["rule"], "verification_index": idx},
        )

    # ------------------------------------------------------------------
    # 5. Target application extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_target_task(cot: str) -> Optional[Dict[str, Any]]:
        section = CryptarithmAugmentGenerator._section(cot, r"^Target\s*$", r"\Z")
        if not section:
            return None
        ms = re.search(r"(?m)^Use selected rule:\s*(.+)$", section)
        mr = re.search(r"(?m)^Replay the rule on target\s+(.+?):\s*$", section)
        ma = re.search(r"(?m)^Final answer:\s*(\S+)\s*$", section)
        if not ms or not mr or not ma:
            return None
        selected_rule = ms.group(1).strip()
        target_expr = mr.group(1).strip()
        answer = ma.group(1).strip()
        calc_lines: List[str] = []
        take = False
        for line in section.splitlines():
            if line.startswith("Target operator") or line.startswith("Use selected rule"):
                continue
            if line.startswith("Replay the rule on target"):
                take = True
                continue
            if line.startswith("Computed output") or line.startswith("Final answer") or line.startswith("\\boxed"):
                continue
            if take and line.strip():
                calc_lines.append(line[2:] if line.startswith("  ") else line)
        return {"selected_rule": selected_rule, "target_expr": target_expr, "calc_lines": calc_lines, "answer": answer}

    def _make_target_application_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
        digit_map: str,
    ) -> Dict[str, Any]:
        prompt_lines = [
            "In Alice's Wonderland, the hidden operator rules and digit map have already been found.",
            "Your task is to apply the selected rule to the target expression and compute the visible output.",
            "",
            "Selected rule:",
            item["selected_rule"],
            "",
            "Digit map:",
            digit_map,
            "",
            "Target:",
            item["target_expr"],
            "",
            "Return the calculation and finish with:",
            "Answer: <visible output>",
        ]
        self._add_rule_semantics(prompt_lines)
        completion = "\n".join(item["calc_lines"] + [f"Answer: {item['answer']}"])
        return self._row(
            source_id,
            "target_application",
            item["target_expr"],
            "cryptarithm_target_application",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"target_expression": item["target_expr"], "selected_rule": item["selected_rule"]},
        )

    # ------------------------------------------------------------------
    # 6. Ambiguity / target-invariance audit extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ambiguity_audit_task(cot: str) -> Optional[Dict[str, Any]]:
        section = CryptarithmAugmentGenerator._section(cot, r"^Ambiguity audit\s*$", r"^Target\s*$")
        if not section:
            return None
        md = re.search(r"(?m)^decision:\s*(.+)$", section)
        if not md:
            return None
        decision = md.group(1).strip()
        low = decision.lower()
        if "ambiguous item" in low or "multiple target outputs" in low:
            answer = "ambiguous"
        elif "audit incomplete" in low or "timeout" in low or "incomplete" in low:
            answer = "incomplete"
        elif "unique" in low:
            answer = "unique"
        else:
            answer = "incomplete"
        prompt_section = CryptarithmAugmentGenerator._remove_lines(section, ["decision:"])
        return {
            "audit_summary": section.strip(),
            "prompt_summary": prompt_section.strip(),
            "decision": decision,
            "answer": answer,
        }

    def _make_ambiguity_audit_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_lines = [
            "In Alice's Wonderland, the solver has run a target-invariance audit after finding a candidate solution.",
            "Your task is to interpret the audit summary and decide whether the target output is proven unique, ambiguous, or not exhaustively proven.",
            "Do not recompute the whole digit search; use the audit counts and target-output set shown below.",
            "",
            "Audit summary:",
            item.get("prompt_summary", ""),
            "",
            "Return the audit interpretation and finish with:",
            "Answer: unique, Answer: ambiguous, or Answer: incomplete",
        ]
        completion = "Ambiguity audit\n" + item["audit_summary"].rstrip() + "\nAnswer: " + item["answer"]
        return self._row(
            source_id,
            "ambiguity_audit",
            "ambiguity_audit",
            "cryptarithm_ambiguity_audit",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"audit_decision": item.get("decision", "")},
        )


    # ------------------------------------------------------------------
    # Balanced / contrastive extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_parsed_equation_operator_map(cot: str) -> Dict[str, str]:
        """Map rendered equation text to its operator symbol.

        Projection labels are rendered as:
            label: full match <display> via std/mul
        The <display> text is the same as Equation.display().  Reconstructing
        this map lets synthetic candidate-keep tasks name the tested operator
        without guessing from punctuation.
        """
        section = CryptarithmAugmentGenerator._section(cot, r"^Parsed equations\s*$", r"^Symbols\s*$")
        out: Dict[str, str] = {}
        if not section:
            return out
        pat = re.compile(
            r"(?m)^\d+\.\s+left=(.*?),\s+op=(.*?),\s+right=(.*?),\s+output=(.*?),\s+signed_output=(True|False)\s*$"
        )
        for m in pat.finditer(section):
            left = m.group(1)
            op_lit = m.group(2).strip()
            right = m.group(3)
            output = m.group(4)
            try:
                op = ast.literal_eval(op_lit)
            except Exception:
                op = op_lit.strip("'\"")
            display = f"{left}{op}{right}={output}"
            out[display] = op
        return out

    @staticmethod
    def _projection_check_from_projection_block(block_text: str) -> str:
        lines = str(block_text or "").splitlines()
        if lines and lines[0].startswith("Projection block"):
            lines = lines[1:]
        kept: List[str] = ["Projection check"]
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith(("apply update:", "decision:", "result:")):
                continue
            if not stripped:
                continue
            kept.append("    " + stripped)
        return "\n".join(kept).strip()

    @staticmethod
    def _completion_projection_check(
        block_number: int,
        operator_text: str,
        rule: str,
        projection_block_text: str,
        answer: str,
    ) -> str:
        check = CryptarithmAugmentGenerator._projection_check_from_projection_block(projection_block_text)
        apply_update = CryptarithmAugmentGenerator._extract_apply_update_line(projection_block_text) or "none"
        body_lines = []
        for ln in check.splitlines():
            if ln == "Projection check":
                body_lines.append("  Projection check")
            else:
                body_lines.append("  " + ln)
        if answer == "keep":
            body_lines.append(f"    apply update: {apply_update}")
            body_lines.append(f"    result: operator {operator_text} -> {rule}: supported local digit assignment exists")
            body_lines.append(f"  decision: keep operator {operator_text} -> {rule}")
        else:
            body_lines.append("    apply update: none")
            body_lines.append(f"    result: operator {operator_text} -> {rule}: no supported local digit assignment")
            body_lines.append(f"  decision: reject operator {operator_text} -> {rule}")
        return f"Combo block {block_number}: test operator {operator_text} -> {rule}\n" + "\n".join(body_lines)

    @staticmethod
    def _extract_candidate_keep_from_projection_blocks(cot: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Create real positive candidate-check examples from successful projections.

        These are not fabricated answers: a successful projection block already
        contains non-empty support and a keep decision.  We reframe at most
        `limit` of them as local candidate checks so candidate_check is not
        all-reject.
        """
        if limit <= 0:
            return []
        eq_to_op = CryptarithmAugmentGenerator._extract_parsed_equation_operator_map(cot)
        out: List[Dict[str, Any]] = []
        for block in CryptarithmAugmentGenerator._extract_domain_blocks(cot):
            if block.get("kind") != "projection":
                continue
            raw = block.get("block_text", "")
            if "decision: keep" not in raw:
                continue
            if re.search(r"supported projection\s*\(0 supports\)", raw):
                continue
            ml = re.search(r"(?m)^\s*label:\s*full match\s+(.+?)\s+via\s+([a-z]+/[a-z0-9]+)\s*$", raw)
            if not ml:
                continue
            display, rule = ml.group(1).strip(), ml.group(2).strip()
            op = eq_to_op.get(display)
            if op is None:
                continue
            operator_text = repr(op)
            block_number = int(block.get("number", len(out) + 1))
            check_input = CryptarithmAugmentGenerator._projection_check_from_projection_block(raw)
            full = CryptarithmAugmentGenerator._completion_projection_check(block_number, operator_text, rule, raw, "keep")
            out.append(
                {
                    "number": block_number,
                    "operator": operator_text,
                    "rule": rule,
                    "block_text": full,
                    "check_input": check_input,
                    "answer": "keep",
                    "candidate_check_kind": "projection_keep",
                    "synthetic_kind": "real_projection_keep",
                }
            )
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _extract_domain_contradictions_from_candidate_rejects(cot: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Reframe empty-support candidate rejects as domain contradictions.

        A candidate reject with supported projection (0 supports) is exactly a
        local projection that has no satisfying assignment.  This gives real
        contradiction examples for domain_propagation without inventing states.
        """
        if limit <= 0:
            return []
        out: List[Dict[str, Any]] = []
        for cand in CryptarithmAugmentGenerator._extract_candidate_check_blocks(cot):
            raw = cand.get("block_text", "")
            if cand.get("answer") != "reject":
                continue
            if not re.search(r"supported projection\s*\(0 supports\)", raw):
                continue
            check = cand.get("check_input", "")
            lines = [f"Projection block {cand.get('number', len(out) + 1)}"]
            for ln in check.splitlines():
                stripped = ln.strip()
                if not stripped or stripped == "Projection check":
                    continue
                lines.append("  " + stripped)
            prompt_block = "\n".join(lines)
            completion = prompt_block + "\n  apply update: none\n  decision: reject"
            out.append(
                {
                    "number": int(cand.get("number", len(out) + 1)),
                    "kind": "projection_contradiction",
                    "block_text": completion,
                    "prompt_block": prompt_block,
                    "current_domains": [],
                    "fixed_digits_to_remove": [],
                    "alldiff_digits_to_remove": [],
                    "alldiff_reason_kind": "",
                    "answer": "contradiction",
                    "synthetic_kind": "empty_support_projection_contradiction",
                }
            )
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _digit_symbols_from_map(digit_map: str) -> List[str]:
        symbols: List[str] = []
        for part in str(digit_map or "").split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            lhs = part.split("=", 1)[0].strip()
            try:
                sym = ast.literal_eval(lhs)
            except Exception:
                sym = lhs.strip("'\"")
            if isinstance(sym, str) and len(sym) == 1 and sym not in symbols:
                symbols.append(sym)
        return symbols

    @staticmethod
    def _corrupt_rhs(example: str, symbols: Sequence[str]) -> Optional[Tuple[str, str]]:
        if "=" not in example:
            return None
        lhs, rhs = example.rsplit("=", 1)
        if not rhs:
            return None
        pos = len(rhs) - 1
        old = rhs[pos]
        replacement = None
        for sym in symbols:
            if sym != old:
                replacement = sym
                break
        if replacement is None:
            replacement = old + old
        if len(replacement) == 1:
            new_rhs = rhs[:pos] + replacement + rhs[pos + 1:]
        else:
            new_rhs = rhs + old
        if new_rhs == rhs:
            return None
        return lhs + "=" + new_rhs, new_rhs

    @staticmethod
    def _make_wrong_verification_item(item: Dict[str, Any], digit_map: str) -> Optional[Dict[str, Any]]:
        symbols = CryptarithmAugmentGenerator._digit_symbols_from_map(digit_map)
        corrupted = CryptarithmAugmentGenerator._corrupt_rhs(item.get("example", ""), symbols)
        if corrupted is None:
            return None
        new_example, new_expected = corrupted
        calc_lines: List[str] = []
        replaced_compare = False
        produced = ""
        for ln in item.get("calc_lines", []):
            mprod = re.search(r"^encode result:\s*(\S+)\s*$", ln.strip())
            if mprod:
                produced = mprod.group(1)
            if re.search(r"compare visible output:.*->\s*MATCH\s*$", ln):
                if not produced:
                    mp = re.search(r"produced\s+(\S+)\s+vs", ln)
                    produced = mp.group(1) if mp else "<computed>"
                calc_lines.append(f"compare visible output: produced {produced} vs expected {new_expected} -> WRONG")
                replaced_compare = True
            else:
                calc_lines.append(ln)
        if not replaced_compare:
            if not produced:
                produced = "<computed>"
            calc_lines.append(f"compare visible output: produced {produced} vs expected {new_expected} -> WRONG")
        out = dict(item)
        out["example"] = new_example
        out["calc_lines"] = calc_lines
        out["answer"] = "WRONG"
        out["synthetic_kind"] = "contrastive_wrong_expected_output"
        return out

    def _make_domain_contradiction_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_lines = [
            "In Alice's Wonderland, digit domains are being tested by one local projection step.",
            "Your task is to apply this local domain operation and decide whether it keeps the state or creates a contradiction.",
            "If the supported projection is empty, no local digit assignment exists and the state is a contradiction.",
            "",
            "Local operation:",
            item.get("prompt_block", ""),
            "",
            "Return the domain updates and finish with:",
            "Answer: keep or Answer: contradiction",
        ]
        self._add_rule_semantics(prompt_lines)
        completion = item["block_text"].rstrip() + "\nAnswer: " + item["answer"]
        return self._row(
            source_id,
            "domain_propagation_contradiction",
            f"{item['number']}_{item['kind']}",
            "cryptarithm_domain_propagation",
            prompt_lines,
            completion,
            item["answer"],
            prompt,
            source_answer,
            source_solver_correct,
            extra={"block_number": item["number"], "domain_step_kind": item["kind"], "synthetic_kind": item.get("synthetic_kind", "")},
        )

    def _make_rule_verification_wrong_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        item: Dict[str, Any],
        idx: int,
        digit_map: str,
    ) -> Dict[str, Any]:
        row = self._make_rule_verification_problem(source_id, prompt, source_answer, source_solver_correct, item, idx, digit_map)
        row["id"] = self._make_id(source_id, "rule_verification_wrong", f"{idx}_{item['operator']}_{item['example']}")
        row["answer"] = "WRONG"
        row["computed_answer"] = "WRONG"
        row["synthetic_kind"] = item.get("synthetic_kind", "contrastive_wrong_expected_output")
        return row

    # ------------------------------------------------------------------
    # Row construction and public API
    # ------------------------------------------------------------------

    def _row(
        self,
        source_id: str,
        task_type: str,
        key: str,
        label: str,
        prompt_lines: Sequence[str],
        completion: str,
        answer: str,
        source_prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "id": self._make_id(source_id, task_type, key),
            "prompt": "\n".join(str(x) for x in prompt_lines).strip(),
            "completion": str(completion).strip(),
            "answer": str(answer).strip(),
            "generated_cot": str(completion).strip(),
            "computed_answer": str(answer).strip(),
            "category": label,
            "label": label,
            "task_mode": label,
            "source_task_id": source_id,
            "source_prompt": source_prompt,
            "source_answer": source_answer,
            "source_solver_correct": source_solver_correct,
        }
        if self.config.include_quality_metadata:
            row["subtask_type"] = task_type
            row["has_source_prompt"] = bool(source_prompt)
            row["answer_len"] = len(str(answer).strip())
        if extra:
            row.update(extra)
        return row

    def make_tasks_for_row(self, row: pd.Series, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        source_id = source_id or str(row.get("id", row.name))
        prompt = self._first_present(row, ["prompt"])
        cot = self._first_present(row, ["generated_cot", "cot", "solution", "completion"])
        if not prompt or not cot or not self.is_cryptarithm_prompt(prompt):
            return []

        source_answer = str(row.get("answer", "")) if "answer" in row else self._extract_final_answer(cot)
        source_solver_answer = self._extract_final_answer(cot)
        source_solver_correct = bool(source_answer and source_solver_answer and str(source_answer) == str(source_solver_answer))
        if self.config.only_solver_correct and not source_solver_correct:
            return []

        rows: List[Dict[str, Any]] = []
        digit_map = self._extract_digit_map(cot)
        audit_for_source = self._extract_ambiguity_audit_task(cot)
        source_audit_answer = audit_for_source.get("answer", "") if audit_for_source else ""
        source_audit_decision = audit_for_source.get("decision", "") if audit_for_source else ""

        if self.config.include_rule_filtering:
            for idx, item in enumerate(self._extract_rule_filtering_blocks(cot)):
                rows.append(self._make_rule_filtering_problem(source_id, prompt, source_answer, source_solver_correct, item, idx))

        if self.config.include_candidate_check:
            for item in self._extract_candidate_check_blocks(cot):
                rows.append(self._make_candidate_check_problem(source_id, prompt, source_answer, source_solver_correct, item))
            if self.config.include_candidate_keep_from_projections:
                for item in self._extract_candidate_keep_from_projection_blocks(
                    cot, limit=self.config.max_synthetic_candidate_keep_per_source
                ):
                    rows.append(self._make_candidate_check_problem(source_id, prompt, source_answer, source_solver_correct, item))

        if self.config.include_domain_propagation:
            for item in self._extract_domain_blocks(cot):
                rows.append(self._make_domain_propagation_problem(source_id, prompt, source_answer, source_solver_correct, item))
            if self.config.include_domain_contradictions_from_rejects:
                for item in self._extract_domain_contradictions_from_candidate_rejects(
                    cot, limit=self.config.max_synthetic_domain_contradiction_per_source
                ):
                    rows.append(self._make_domain_contradiction_problem(source_id, prompt, source_answer, source_solver_correct, item))

        if self.config.include_rule_verification:
            verification_items = self._extract_rule_verification_tasks(cot)
            for idx, item in enumerate(verification_items):
                rows.append(self._make_rule_verification_problem(source_id, prompt, source_answer, source_solver_correct, item, idx, digit_map))
            if self.config.include_contrastive_rule_verification_wrong:
                made_wrong = 0
                for idx, item in enumerate(verification_items):
                    wrong_item = self._make_wrong_verification_item(item, digit_map)
                    if wrong_item is None:
                        continue
                    rows.append(self._make_rule_verification_wrong_problem(source_id, prompt, source_answer, source_solver_correct, wrong_item, idx, digit_map))
                    made_wrong += 1
                    if made_wrong >= self.config.max_contrastive_wrong_verification_per_source:
                        break

        if self.config.include_ambiguity_audit:
            if audit_for_source is not None:
                rows.append(self._make_ambiguity_audit_problem(source_id, prompt, source_answer, source_solver_correct, audit_for_source))

        if self.config.include_target_application:
            target_application_allowed = True
            if self.config.skip_ambiguous_target_application and source_audit_answer and source_audit_answer != "unique":
                target_application_allowed = False
            if target_application_allowed:
                target = self._extract_target_task(cot)
                if target is not None:
                    rows.append(self._make_target_application_problem(source_id, prompt, source_answer, source_solver_correct, target, digit_map))

        for out_row in rows:
            out_row["source_audit_answer"] = source_audit_answer
            out_row["source_audit_decision"] = source_audit_decision
            out_row["source_target_is_unique"] = (source_audit_answer == "unique")
            out_row["target_application_policy"] = (
                "skip_nonunique" if self.config.skip_ambiguous_target_application else "allow_selected"
            )

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
        if "prompt" not in work.columns:
            return pd.DataFrame()
        if "label" in work.columns:
            work = work[(work["label"].astype(str) == "cryptarithm") | work["prompt"].map(self.is_cryptarithm_prompt)].copy()
        else:
            work = work[work["prompt"].map(self.is_cryptarithm_prompt)].copy()

        if sample_n is not None:
            work = work.sample(n=min(sample_n, len(work)), random_state=self.seed)
        elif sample_frac < 1.0:
            work = work.sample(frac=sample_frac, random_state=self.seed)

        old_flag = self.config.only_solver_correct
        self.config.only_solver_correct = bool(only_solver_correct)
        try:
            rows: List[Dict[str, Any]] = []
            for _, row in work.iterrows():
                sid = str(row["id"]) if "id" in row and pd.notna(row["id"]) else str(row.name)
                rows.extend(self.make_tasks_for_row(row, source_id=sid))
        finally:
            self.config.only_solver_correct = old_flag

        if not rows:
            return pd.DataFrame(
                columns=["id", "prompt", "completion", "answer", "label", "generated_cot", "computed_answer", "task_mode"]
            )
        df = pd.DataFrame.from_records(rows)
        if self.config.balance_answer_labels:
            df = self.balance_answer_labels(df, max_ratio=self.config.max_answer_label_ratio, seed=self.seed)
        return df

    @staticmethod
    def balance_answer_labels(
        df: pd.DataFrame,
        max_ratio: float = 2.0,
        seed: Optional[int] = None,
        task_modes: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Downsample majority answer labels inside decision-like modes.

        This intentionally does not balance target_application or
        rule_filtering by raw answer string: those answers are content labels,
        not binary decisions.  It balances the modes where the answer is a
        decision class such as keep/reject or MATCH/WRONG.
        """
        if df.empty or "task_mode" not in df.columns or "answer" not in df.columns:
            return df
        modes = set(task_modes or [
            "cryptarithm_candidate_check",
            "cryptarithm_domain_propagation",
            "cryptarithm_rule_verification",
            "cryptarithm_ambiguity_audit",
        ])
        rng_state = 0 if seed is None else seed
        kept_parts: List[pd.DataFrame] = []
        for mode, sub in df.groupby("task_mode", sort=False, dropna=False):
            if str(mode) not in modes:
                kept_parts.append(sub)
                continue
            counts = sub["answer"].astype(str).value_counts()
            if len(counts) <= 1:
                kept_parts.append(sub)
                continue
            min_count = int(counts.min())
            if min_count <= 0:
                kept_parts.append(sub)
                continue
            cap = max(min_count, int(round(min_count * float(max_ratio))))
            balanced_chunks: List[pd.DataFrame] = []
            for ans, ans_sub in sub.groupby(sub["answer"].astype(str), sort=False):
                if len(ans_sub) > cap:
                    balanced_chunks.append(ans_sub.sample(n=cap, random_state=rng_state))
                else:
                    balanced_chunks.append(ans_sub)
            kept_parts.append(pd.concat(balanced_chunks, axis=0))
        out = pd.concat(kept_parts, axis=0)
        # Preserve original row order as much as possible after sampling.
        return out.sort_index(kind="stable").reset_index(drop=True)

    def generate_dataset_from_text_dump(self, text: str) -> pd.DataFrame:
        return self.generate_dataset(self.parse_text_dump(text))

    # ------------------------------------------------------------------
    # Dataset QA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Return lightweight checks that catch common augmentation mistakes.

        The report is intentionally conservative: it flags likely leaks and
        distribution issues, but it does not attempt to re-solve cryptarithms.
        """
        report: Dict[str, Any] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "empty_prompt_rows": 0,
            "empty_completion_rows": 0,
            "empty_answer_rows": 0,
            "duplicate_id_rows": 0,
            "task_mode_counts": {},
            "answer_counts_by_task_mode": {},
            "candidate_prompts_with_output_markers": 0,
            "target_rows_from_nonunique_sources": 0,
            "candidate_check_has_keep": False,
            "domain_propagation_has_contradiction": False,
            "rule_verification_has_wrong": False,
        }
        if df.empty:
            return report

        def col(name: str) -> pd.Series:
            return df[name].astype(str) if name in df.columns else pd.Series([""] * len(df), index=df.index)

        report["empty_prompt_rows"] = int((col("prompt").str.strip() == "").sum())
        report["empty_completion_rows"] = int((col("completion").str.strip() == "").sum())
        report["empty_answer_rows"] = int((col("answer").str.strip() == "").sum())
        report["duplicate_id_rows"] = int(df.duplicated("id").sum()) if "id" in df.columns else 0
        if "task_mode" in df.columns:
            report["task_mode_counts"] = {str(k): int(v) for k, v in df["task_mode"].value_counts().items()}
            answer_counts: Dict[str, Dict[str, int]] = {}
            for mode, sub in df.groupby("task_mode", dropna=False):
                answer_counts[str(mode)] = {str(k): int(v) for k, v in sub["answer"].astype(str).value_counts().items()} if "answer" in sub.columns else {}
            report["answer_counts_by_task_mode"] = answer_counts

        if {"task_mode", "prompt"}.issubset(df.columns):
            cand = df[df["task_mode"].astype(str) == "cryptarithm_candidate_check"]
            if len(cand):
                # These markers are expected in completions, not in the local-check prompt.
                marker_re = r"(?m)^\s*(?:apply update|result|decision):"
                report["candidate_prompts_with_output_markers"] = int(cand["prompt"].astype(str).str.contains(marker_re, regex=True).sum())

        if {"task_mode", "source_audit_answer"}.issubset(df.columns):
            targ = df[df["task_mode"].astype(str) == "cryptarithm_target_application"]
            report["target_rows_from_nonunique_sources"] = int((targ["source_audit_answer"].astype(str) != "unique").sum())

        if {"task_mode", "answer"}.issubset(df.columns):
            cand = df[df["task_mode"].astype(str) == "cryptarithm_candidate_check"]
            report["candidate_check_has_keep"] = bool((cand["answer"].astype(str) == "keep").any())
            dom = df[df["task_mode"].astype(str) == "cryptarithm_domain_propagation"]
            report["domain_propagation_has_contradiction"] = bool((dom["answer"].astype(str) == "contradiction").any())
            ver = df[df["task_mode"].astype(str) == "cryptarithm_rule_verification"]
            report["rule_verification_has_wrong"] = bool((ver["answer"].astype(str) == "WRONG").any())
        return report

    @staticmethod
    def quality_issues(df: pd.DataFrame) -> List[str]:
        report = CryptarithmAugmentGenerator.quality_report(df)
        issues: List[str] = []
        for key in ["empty_prompt_rows", "empty_completion_rows", "empty_answer_rows", "duplicate_id_rows"]:
            if report.get(key, 0):
                issues.append(f"{key}: {report[key]}")
        if report.get("candidate_prompts_with_output_markers", 0):
            issues.append(f"candidate prompts contain output markers: {report['candidate_prompts_with_output_markers']}")
        if report.get("target_rows_from_nonunique_sources", 0):
            issues.append(f"target_application rows from non-unique sources: {report['target_rows_from_nonunique_sources']}")
        return issues

    @staticmethod
    def format_quality_report(report: Dict[str, Any]) -> str:
        lines = [f"rows: {report.get('rows', 0)}"]
        counts = report.get("task_mode_counts", {}) or {}
        if counts:
            lines.append("task_mode_counts:")
            lines.extend(f"  {k}: {v}" for k, v in counts.items())
        answer_counts = report.get("answer_counts_by_task_mode", {}) or {}
        if answer_counts:
            lines.append("answer_counts_by_task_mode:")
            for mode, counts2 in answer_counts.items():
                compact = ", ".join(f"{k}={v}" for k, v in counts2.items())
                lines.append(f"  {mode}: {compact}")
        checks = [
            "empty_prompt_rows",
            "empty_completion_rows",
            "empty_answer_rows",
            "duplicate_id_rows",
            "candidate_prompts_with_output_markers",
            "target_rows_from_nonunique_sources",
            "candidate_check_has_keep",
            "domain_propagation_has_contradiction",
            "rule_verification_has_wrong",
        ]
        lines.append("checks:")
        lines.extend(f"  {k}: {report.get(k)}" for k in checks)
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate cryptarithm local-subtask dataset from solved replay CoT text dumps.")
    parser.add_argument("input", help="Path to pasted solved cryptarithm text dump")
    parser.add_argument("--output", "-o", default="cryptarithm_subtasks.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--sample-n", type=int, default=None, help="Sample at most N source rows before augmentation")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Sample this fraction of source rows before augmentation")
    parser.add_argument("--only-solver-correct", action="store_true", help="Keep only rows whose source answer matches the replay final answer")
    parser.add_argument("--keep-nonunique-targets", action="store_true", help="Allow target_application rows from ambiguous/incomplete audits. Default skips them.")
    parser.add_argument("--no-rule-semantics", action="store_true", help="Do not add the compact rule-notation legend to prompts")
    parser.add_argument("--no-quality-metadata", action="store_true", help="Do not add lightweight metadata columns to output rows")
    parser.add_argument("--no-balanced-extras", action="store_true", help="Disable derived keep/contradiction/WRONG balancing examples")
    parser.add_argument("--max-balanced-extras-per-source", type=int, default=10, help="Cap each kind of balancing example per source row")
    parser.add_argument("--balance-answer-labels", action="store_true", help="Downsample majority answer labels inside decision-like modes")
    parser.add_argument("--max-answer-label-ratio", type=float, default=2.0, help="Max majority/minority ratio when --balance-answer-labels is used")
    parser.add_argument("--quality-report", action="store_true", help="Print a dataset QA report after generation")
    parser.add_argument("--quality-report-json", default="", help="Optional path to write the QA report as JSON")
    parser.add_argument("--fail-on-quality-issues", action="store_true", help="Exit nonzero if conservative QA checks find issues")
    args = parser.parse_args()

    config = CryptarithmAugmentConfig(
        sample_frac=args.sample_frac,
        sample_n=args.sample_n,
        only_solver_correct=args.only_solver_correct,
        skip_ambiguous_target_application=not args.keep_nonunique_targets,
        include_rule_semantics_in_prompts=not args.no_rule_semantics,
        include_quality_metadata=not args.no_quality_metadata,
        include_candidate_keep_from_projections=not args.no_balanced_extras,
        include_domain_contradictions_from_rejects=not args.no_balanced_extras,
        include_contrastive_rule_verification_wrong=not args.no_balanced_extras,
        max_synthetic_candidate_keep_per_source=args.max_balanced_extras_per_source,
        max_synthetic_domain_contradiction_per_source=args.max_balanced_extras_per_source,
        max_contrastive_wrong_verification_per_source=args.max_balanced_extras_per_source,
        balance_answer_labels=args.balance_answer_labels,
        max_answer_label_ratio=args.max_answer_label_ratio,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    gen = CryptarithmAugmentGenerator(seed=args.seed, config=config)
    df = gen.generate_dataset_from_text_dump(text)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    if len(df):
        print(df["task_mode"].value_counts().to_string())

    report = CryptarithmAugmentGenerator.quality_report(df)
    issues = CryptarithmAugmentGenerator.quality_issues(df)
    if args.quality_report:
        print("\nquality report")
        print(CryptarithmAugmentGenerator.format_quality_report(report))
    if args.quality_report_json:
        with open(args.quality_report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if issues:
        print("\nquality issues:")
        for issue in issues:
            print(f"- {issue}")
        if args.fail_on_quality_issues:
            raise SystemExit(1)
