from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


NUMERAL_PROMPT_MARKER = "secret set of transformation rules is applied to equations"


@dataclass
class NumeralEquationAugmentConfig:
    sample_frac: float = 1.0
    sample_n: Optional[int] = None
    only_solver_correct: bool = False
    include_family_matching: bool = True
    include_rule_application: bool = True
    # Family blocks are often negative examples. Keep all by default, but this
    # can be enabled if no-match blocks dominate too much.
    downsample_no_match_family: bool = False
    no_match_keep_mod: int = 4


class NumeralEquationAugmentGenerator:
    """
    Augmenter for numeral-equation reasoning traces.

    It follows the same philosophy as the bit-matching augmenter:
      - do NOT create random full tasks;
      - derive small subtasks from already-solved source prompts / CoTs;
      - split a large CoT into focused training examples.

    Produced task types:
      1. equation_family_matching
         Given relevant examples for one unknown operator and one candidate
         operation family, compute the family block and answer either
         `rev_ops=<...>/rev_result=<...>/<operation>` or `no`.

      2. equation_rule_application
         Given a known selected rule and one expression, apply the rule.  For
         verification examples the task also checks MATCH/WRONG against the
         expected output; for target examples it returns the computed output.
    """

    def __init__(self, seed: Optional[int] = None, config: Optional[NumeralEquationAugmentConfig] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config or NumeralEquationAugmentConfig()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_numeral_prompt(prompt: Any) -> bool:
        return NUMERAL_PROMPT_MARKER in ("" if prompt is None else str(prompt))

    @staticmethod
    def _stable_hash_int(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _make_id(source_id: str, task_type: str, key: str) -> str:
        return hashlib.sha256(f"numeral_{task_type}_{source_id}_{key}".encode()).hexdigest()[:10]

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
        matches = re.findall(r"(?m)^Final answer:\s*(\S+)\s*$", text or "")
        return matches[-1] if matches else ""

    @staticmethod
    def _strip_outer_blocks(text: str) -> str:
        return str(text or "").strip()

    # ------------------------------------------------------------------
    # Parsing source dumps / CoT sections
    # ------------------------------------------------------------------

    @staticmethod
    def parse_text_dump(text: str) -> pd.DataFrame:
        """Parse pasted audit dumps of the form used in the conversation."""
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
                    "answer": NumeralEquationAugmentGenerator._extract_final_answer(cot),
                    "label": "numeral equations",
                }
            )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _parse_examples_from_cot(cot: str) -> List[Dict[str, str]]:
        examples: List[Dict[str, str]] = []
        for line in str(cot or "").splitlines():
            m = re.match(r"^\s*\d+\.\s*(-?\d+)\s+([^\d\s]+)\s+(-?\d+)\s*=\s*(\S+)\s*$", line)
            if m:
                examples.append({"a": m.group(1), "op": m.group(2), "b": m.group(3), "out": m.group(4)})
        return examples

    @staticmethod
    def _parse_target_from_cot(cot: str) -> str:
        m = re.search(r"(?m)^Target:\s*(.+?)\s*$", str(cot or ""))
        return m.group(1).strip() if m else ""

    @staticmethod
    def _target_operator(target_expr: str) -> str:
        m = re.fullmatch(r"\s*-?\d+\s*([^\d\s]+)\s*-?\d+\s*", target_expr or "")
        return m.group(1).strip() if m else ""

    @staticmethod
    def _format_relevant_examples(examples: Sequence[Dict[str, str]], operator: str) -> List[str]:
        return [f"{e['a']} {e['op']} {e['b']} = {e['out']}" for e in examples if e["op"] == operator]

    @staticmethod
    def _extract_display_format(cot: str, operator: str) -> str:
        # Prefer the explicit normalization section.
        pattern = rf"Operator\s+{re.escape(repr(operator))}\s+detected display format:\s*(\S+)"
        m = re.search(pattern, cot)
        if m:
            return m.group(1)
        # Fallback: selected rule / uses rule.
        m = re.search(r"(?:selected rule|Use selected rule|uses)[:\s]+rev_ops=(?:True|False)/rev_result=(?:True|False)/[^/]+/format=(\S+)", cot)
        return m.group(1) if m else "num"

    # ------------------------------------------------------------------
    # Family-block extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_family_blocks(cot: str) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        pattern = re.compile(
            r"(?ms)^Family block\s+(\d+):\s*(?:FAIL:|PASS SELECTED:|PASS:)?\s*(.+?)\n"
            r"(.*?)(?=^Family block\s+\d+:|^selected rule:|^Stop search:|^Verify selected rules|^Target\b|\Z)"
        )
        for m in pattern.finditer(cot or ""):
            index = int(m.group(1))
            label = m.group(2).strip()
            body = m.group(3).rstrip()
            text = f"Family block {index:02d}: {label}\n{body}".rstrip()
            answer = "no"
            mm = re.search(
                r"family block contains at least one full match; choose first by transform order then operation order:\s*"
                r"rev_ops=(True|False),\s*rev_result=(True|False),\s*operation=([^\s]+)",
                body,
            )
            if mm:
                answer = f"rev_ops={mm.group(1)}/rev_result={mm.group(2)}/{mm.group(3)}"
            op_order = ""
            mo = re.search(r"operation order inside this family:\s*(.+)", body)
            if mo:
                op_order = mo.group(1).strip()
            formula = ""
            mf = re.search(r"family formula:\s*(.+)", body)
            if mf:
                formula = mf.group(1).strip()
            blocks.append(
                {
                    "index": index,
                    "family_label": label,
                    "text": text,
                    "body": body,
                    "answer": answer,
                    "is_match": answer != "no",
                    "operation_order": op_order,
                    "family_formula": formula,
                }
            )
        return blocks

    def _keep_family_block(self, source_id: str, block: Dict[str, Any]) -> bool:
        if block.get("is_match"):
            return True
        if not self.config.downsample_no_match_family:
            return True
        mod = max(1, int(self.config.no_match_keep_mod))
        return (self._stable_hash_int(f"{source_id}_family_{block['index']}") % mod) == 0

    def _make_family_matching_problem(
        self,
        source_id: str,
        prompt: str,
        cot: str,
        source_answer: str,
        source_solver_correct: bool,
        block: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._keep_family_block(source_id, block):
            return None

        examples = self._parse_examples_from_cot(cot)
        target = self._parse_target_from_cot(cot)
        op = self._target_operator(target)
        relevant_examples = self._format_relevant_examples(examples, op)
        display_format = self._extract_display_format(cot, op)

        family_prompt_lines = [
            "In Alice's Wonderland, an unknown numeric operator is tested against one candidate operation family.",
            "Your task is to decide whether this family can explain all examples for the operator.",
            "",
            f"Operator: {repr(op)}",
            f"Display format: {display_format}",
            "Examples:",
        ]
        family_prompt_lines.extend(relevant_examples or ["<no examples for this operator>"])
        family_prompt_lines.extend(
            [
                "",
                "Transform order:",
                "1. rev_ops=True, rev_result=True",
                "2. rev_ops=False, rev_result=False",
                "3. rev_ops=True, rev_result=False",
                "4. rev_ops=False, rev_result=True",
                "",
                f"Candidate family: {block['family_label']}",
                f"Operation order inside this family: {block.get('operation_order', '')}",
                f"Family formula: {block.get('family_formula', '')}",
                "",
                "Compute the family test for every transform pair.",
                "If at least one transform fully matches, return the first matching rule by transform order and operation order as:",
                "rev_ops=<True|False>/rev_result=<True|False>/<operation>",
                "If no transform fully matches, return exactly: no",
            ]
        )

        output_text = block["text"] + "\nAnswer: " + block["answer"]
        pid = self._make_id(source_id, "family", str(block["index"]))
        return {
            "id": pid,
            "prompt": "\n".join(family_prompt_lines),
            "completion": output_text,
            "answer": block["answer"],
            "generated_cot": output_text,
            "computed_answer": block["answer"],
            "category": "numeral_equations_family_matching",
            "label": "numeral equations_family_matching",
            "task_mode": "equation_family_matching",
            "family_index": block["index"],
            "family_label": block["family_label"],
            "is_match": block["is_match"],
            "source_task_id": source_id,
            "source_prompt": prompt,
            "source_answer": source_answer,
            "source_solver_correct": source_solver_correct,
        }

    # ------------------------------------------------------------------
    # Rule-application extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_verify_application_tasks(cot: str) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        lines = str(cot or "").splitlines()
        current_rule: Optional[Tuple[str, str]] = None
        i = 0
        while i < len(lines):
            line = lines[i]
            m_rule = re.match(r"^Operator\s+(.+?)\s+uses\s+(rev_ops=.*)$", line)
            if m_rule:
                current_rule = (m_rule.group(1).strip(), m_rule.group(2).strip())
                i += 1
                continue
            if current_rule and line.startswith("- "):
                header = line[2:].strip()
                calc_lines: List[str] = []
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if nxt.startswith("- ") or nxt.startswith("Operator ") or nxt.startswith("Target") or nxt.startswith("Rule matching"):
                        break
                    if nxt.startswith("  "):
                        calc_lines.append(nxt[2:])
                    elif nxt.strip():
                        # Keep unexpected non-empty continuation lines, but do not
                        # absorb a new section header.
                        break
                    j += 1

                computed = ""
                mm = re.search(r"display format\s+\S+:\s+visible output is\s+(\S+)", "\n".join(calc_lines))
                if mm:
                    computed = mm.group(1)
                match = ""
                mc = re.search(r"compare normalized output: .* ->\s*(MATCH|WRONG)", "\n".join(calc_lines))
                if mc:
                    match = mc.group(1)
                tasks.append(
                    {
                        "kind": "verify",
                        "operator": current_rule[0],
                        "rule": current_rule[1],
                        "expression": header,
                        "calc_lines": calc_lines,
                        "computed": computed,
                        "answer": match or computed,
                    }
                )
                i = j
                continue
            i += 1
        return tasks

    @staticmethod
    def _extract_target_application_task(cot: str) -> Optional[Dict[str, Any]]:
        m_rule = re.search(r"^Use selected rule:\s*(rev_ops=.*)$", cot or "", flags=re.M)
        m_replay = re.search(r"^Replay the rule on target\s+(.+?):\s*$", cot or "", flags=re.M)
        if not m_rule or not m_replay:
            return None

        start = m_replay.end()
        tail = cot[start:]
        # Stop at final answer; include calculation lines before it.
        m_final = re.search(r"(?m)^Final answer:\s*(\S+)\s*$", tail)
        if not m_final:
            return None
        calc_part = tail[: m_final.start()].strip("\n")
        calc_lines = [ln[2:] if ln.startswith("  ") else ln for ln in calc_part.splitlines() if ln.strip()]
        final = m_final.group(1)
        return {
            "kind": "target",
            "rule": m_rule.group(1).strip(),
            "expression": m_replay.group(1).strip(),
            "calc_lines": calc_lines,
            "computed": final,
            "answer": final,
        }

    def _make_rule_application_problem(
        self,
        source_id: str,
        prompt: str,
        source_answer: str,
        source_solver_correct: bool,
        task: Dict[str, Any],
        task_index: int,
    ) -> Dict[str, Any]:
        if task["kind"] == "verify":
            prompt_lines = [
                "In Alice's Wonderland, a rule for an unknown numeric operator has already been found.",
                "Apply the rule to the example and check whether it matches the expected output.",
                "",
                f"Known rule for operator {task['operator']}: {task['rule']}",
                f"Example to verify: {task['expression']}",
                "",
                "Return the calculation and finish with Answer: MATCH or Answer: WRONG.",
            ]
            completion_lines = list(task["calc_lines"])
            if task.get("computed"):
                completion_lines.append(f"Computed output: {task['computed']}")
            completion_lines.append(f"Answer: {task['answer']}")
        else:
            prompt_lines = [
                "In Alice's Wonderland, a rule for an unknown numeric operator has already been found.",
                "Apply the rule to the target expression.",
                "",
                f"Known rule: {task['rule']}",
                f"Target expression: {task['expression']}",
                "",
                "Return the calculation and finish with Answer: <visible output>.",
            ]
            completion_lines = list(task["calc_lines"])
            completion_lines.append(f"Answer: {task['answer']}")

        output_text = "\n".join(completion_lines)
        key = f"{task['kind']}_{task_index}_{task.get('expression', '')}"
        pid = self._make_id(source_id, "apply", key)
        return {
            "id": pid,
            "prompt": "\n".join(prompt_lines),
            "completion": output_text,
            "answer": task["answer"],
            "generated_cot": output_text,
            "computed_answer": task["answer"],
            "category": "numeral_equations_rule_application",
            "label": "numeral equations_rule_application",
            "task_mode": "equation_rule_application",
            "application_kind": task["kind"],
            "source_task_id": source_id,
            "source_prompt": prompt,
            "source_answer": source_answer,
            "source_solver_correct": source_solver_correct,
        }

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def make_tasks_for_row(self, row: pd.Series, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        source_id = source_id or str(row.get("id", row.name))
        prompt = self._first_present(row, ["prompt"])
        cot = self._first_present(row, ["generated_cot", "cot", "solution", "completion"])
        if not prompt or not cot or not self.is_numeral_prompt(prompt):
            return []

        source_answer = str(row.get("answer", "")) if "answer" in row else self._extract_final_answer(cot)
        source_solver_answer = self._extract_final_answer(cot)
        source_solver_correct = bool(source_answer and source_solver_answer and str(source_answer) == str(source_solver_answer))
        if self.config.only_solver_correct and not source_solver_correct:
            return []

        rows: List[Dict[str, Any]] = []
        if self.config.include_family_matching:
            for block in self._extract_family_blocks(cot):
                item = self._make_family_matching_problem(
                    source_id=source_id,
                    prompt=prompt,
                    cot=cot,
                    source_answer=source_answer,
                    source_solver_correct=source_solver_correct,
                    block=block,
                )
                if item is not None:
                    rows.append(item)

        if self.config.include_rule_application:
            application_tasks = self._extract_verify_application_tasks(cot)
            target_task = self._extract_target_application_task(cot)
            if target_task is not None:
                application_tasks.append(target_task)
            for idx, task in enumerate(application_tasks):
                rows.append(
                    self._make_rule_application_problem(
                        source_id=source_id,
                        prompt=prompt,
                        source_answer=source_answer,
                        source_solver_correct=source_solver_correct,
                        task=task,
                        task_index=idx,
                    )
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
        if "label" in work.columns:
            work = work[(work["label"].astype(str) == "numeral equations") | work["prompt"].map(self.is_numeral_prompt)].copy()
        else:
            work = work[work["prompt"].map(self.is_numeral_prompt)].copy()

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

    def generate_dataset_from_text_dump(self, text: str) -> pd.DataFrame:
        return self.generate_dataset(self.parse_text_dump(text))
