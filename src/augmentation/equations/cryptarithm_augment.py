from __future__ import annotations

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
            "Do not solve the digit map. Use only output length, signed-output information, and the rule family definitions.",
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
        # Start only at causal test blocks.
        pattern = re.compile(
            r"(?ms)^Combo block\s+(\d+):\s+test operator\s+(.+?)\s*->\s*([^\n]+)\n"
            r"(.*?)(?=^(?:Combo block|No-leading-zero block|Projection block|AllDifferent block|Branch block|Map block|Selected digit map|Verify selected rules|Target)\b|\Z)"
        )
        out: List[Dict[str, Any]] = []
        for m in pattern.finditer(local):
            number = int(m.group(1))
            op = m.group(2).strip()
            rule = m.group(3).strip()
            body = m.group(4).rstrip()
            full = f"Combo block {number}: test operator {op} -> {rule}\n{body}".rstrip()
            answer = CryptarithmAugmentGenerator._answer_from_decision(full) or "reject"
            # Remove result/decision lines from the prompt-side local check.
            check_input = CryptarithmAugmentGenerator._remove_lines(body, ["result:", "decision:"])
            out.append(
                {
                    "number": number,
                    "operator": op,
                    "rule": rule,
                    "block_text": full,
                    "check_input": check_input,
                    "answer": answer,
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
            "In Alice's Wonderland, a candidate rule for one operator is being tested during rule-combo search.",
            "Your task is to run the shown local check and decide whether this candidate is rejected or remains possible.",
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
            extra={"operator": item["operator"], "rule": item["rule"], "block_number": item["number"]},
        )

    # ------------------------------------------------------------------
    # 3. Domain propagation extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_block_start_pattern() -> re.Pattern[str]:
        # Domain-propagation examples may start only at one of these local
        # state-transition blocks.  A Branch block is included only when it is
        # an explicit assignment, not when it merely chooses a symbol.
        return re.compile(
            r"(?m)^(No-leading-zero block|Projection block|AllDifferent block)\s+(\d+)"
            r"|^(Branch block)\s+(\d+):\s+try set\s+(.+)$"
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
            if first.startswith("No-leading-zero"):
                kind = "no_leading_zero"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("Projection"):
                kind = "projection"
                num = int(re.search(r"block\s+(\d+)", first).group(1))
            elif first.startswith("AllDifferent"):
                kind = "alldifferent"
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
            fixed_digits_to_remove: List[int] = []
            if kind == "alldifferent":
                fixed_digits_to_remove = sorted(CryptarithmAugmentGenerator._removed_digits_from_update(update))
                # If we cannot reconstruct which fixed digits caused the update,
                # the task would not be self-contained.  Do not emit it.
                if not fixed_digits_to_remove:
                    continue
            blocks.append(
                {
                    "number": num,
                    "kind": kind,
                    "block_text": raw,
                    "prompt_block": prompt_block,
                    "current_domains": current_domains,
                    "fixed_digits_to_remove": fixed_digits_to_remove,
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
            "alldifferent": "digit domains are being narrowed by the AllDifferent rule.",
            "branch_assignment": "a branch assignment is being applied to the digit domains.",
        }.get(item["kind"], "digit domains are being narrowed by one local step.")
        prompt_lines = [
            f"In Alice's Wonderland, {kind_intro}",
            "Your task is to apply this local domain operation and report the resulting domain updates.",
            "",
            "Local operation:",
            item.get("prompt_block", ""),
        ]
        if item.get("kind") == "alldifferent" and item.get("fixed_digits_to_remove"):
            prompt_lines.extend(
                [
                    "",
                    "Fixed digits to remove from non-fixed domains:",
                    self._set_to_domain_values(item["fixed_digits_to_remove"]),
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
        section = CryptarithmAugmentGenerator._section(cot, r"^Verify selected rules\s*$", r"^Target\s*$")
        if not section:
            return []
        tasks: List[Dict[str, Any]] = []
        current_rule: Optional[Tuple[str, str]] = None
        lines = section.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
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
                    nxt = lines[j]
                    if nxt.startswith("Operator ") or nxt.startswith("- "):
                        break
                    calc.append(nxt.rstrip())
                    j += 1
                text = "\n".join(calc).strip()
                ma = re.search(r"->\s*(MATCH|WRONG)\s*$", text, flags=re.M)
                answer = ma.group(1) if ma else "MATCH"
                tasks.append(
                    {
                        "operator": current_rule[0],
                        "rule": current_rule[1],
                        "example": example,
                        "calc_lines": [ln[2:] if ln.startswith("  ") else ln for ln in calc if ln.strip()],
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

        if self.config.include_rule_filtering:
            for idx, item in enumerate(self._extract_rule_filtering_blocks(cot)):
                rows.append(self._make_rule_filtering_problem(source_id, prompt, source_answer, source_solver_correct, item, idx))

        if self.config.include_candidate_check:
            for item in self._extract_candidate_check_blocks(cot):
                rows.append(self._make_candidate_check_problem(source_id, prompt, source_answer, source_solver_correct, item))

        if self.config.include_domain_propagation:
            for item in self._extract_domain_blocks(cot):
                rows.append(self._make_domain_propagation_problem(source_id, prompt, source_answer, source_solver_correct, item))

        if self.config.include_rule_verification:
            for idx, item in enumerate(self._extract_rule_verification_tasks(cot)):
                rows.append(self._make_rule_verification_problem(source_id, prompt, source_answer, source_solver_correct, item, idx, digit_map))

        if self.config.include_target_application:
            target = self._extract_target_task(cot)
            if target is not None:
                rows.append(self._make_target_application_problem(source_id, prompt, source_answer, source_solver_correct, target, digit_map))

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
        return pd.DataFrame.from_records(rows)

    def generate_dataset_from_text_dump(self, text: str) -> pd.DataFrame:
        return self.generate_dataset(self.parse_text_dump(text))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cryptarithm local-subtask dataset from solved CoT text dump.")
    parser.add_argument("input", help="Path to pasted solved cryptarithm text dump")
    parser.add_argument("--output", "-o", default="cryptarithm_subtasks.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    gen = CryptarithmAugmentGenerator()
    df = gen.generate_dataset_from_text_dump(text)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    if len(df):
        print(df["task_mode"].value_counts().to_string())
