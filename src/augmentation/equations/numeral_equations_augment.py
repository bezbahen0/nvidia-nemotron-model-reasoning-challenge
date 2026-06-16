from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


NUMERAL_PROMPT_MARKER = "secret set of transformation rules is applied to equations"
STANDARD_OPS = set("+-*/")


@dataclass
class NumeralEquationAugmentConfig:
    sample_frac: float = 1.0
    sample_n: Optional[int] = None
    only_solver_correct: bool = False

    # Old/local subtasks from explicit family-search traces.
    include_family_matching: bool = True
    include_rule_application: bool = True
    downsample_no_match_family: bool = False
    no_match_keep_mod: int = 4

    # New/context subtasks for contextual_v2 solver.
    include_context_inference: bool = True
    include_missing_family_selection: bool = True
    include_contextual_fallback_application: bool = True

    # Synthetic training for hidden-like target-op absent cases.
    include_operator_dropout: bool = True
    dropout_only_solver_correct: bool = True
    dropout_nonstandard_only: bool = False

    # Avoid training wrong target answers when source solver is wrong.
    target_application_only_solver_correct: bool = True


class NumeralEquationAugmentGenerator:
    """
    Augmenter for numeral-equation CoTs.

    Supports both:
      - old target-first family-search traces;
      - new contextual fallback traces with visible-operator context and
        missing-family fallback.

    Main task families:
      1. equation_family_matching
      2. equation_rule_application
      3. equation_context_inference
      4. equation_missing_family_selection
      5. equation_contextual_fallback_application
      6. equation_operator_dropout_contextual
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
    def _extract_true_answer_from_part(part: str) -> str:
        m = re.search(r"--- Истинный ответ ---\s*\n\s*(\S+)", part or "", flags=re.S)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _numeric_expr_re() -> re.Pattern[str]:
        return re.compile(r"^\s*(-?\d+)\s*([^\d\s]+)\s*(-?\d+)\s*$")

    # ------------------------------------------------------------------
    # Parsing source dumps / prompts / CoTs
    # ------------------------------------------------------------------

    @staticmethod
    def parse_text_dump(text: str) -> pd.DataFrame:
        """Parse pasted audit dumps.

        Important fix vs the old generator: if `--- Истинный ответ ---` is
        present, use it as `answer`; store solver's answer separately as
        `solver_answer`. This makes `only_solver_correct=True` meaningful.
        """
        records: List[Dict[str, Any]] = []
        parts = re.split(r"(?=^=== Категория:)", text or "", flags=re.M)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            mid = re.search(r"ID:\s*([^\s=]+)", part)
            source_id = mid.group(1) if mid else hashlib.sha256(part.encode()).hexdigest()[:8]

            mp = re.search(r"--- Промпт \(начало\) ---\n(.*?)(?:\n\n--- Решение ---|\n\n\n--- Истинный ответ ---)", part, flags=re.S)
            ms = re.search(r"--- Решение ---\n(.*?)(?:\n\n--- Истинный ответ ---|\Z)", part, flags=re.S)
            if not mp:
                continue
            prompt = mp.group(1).strip()
            cot = ms.group(1).strip() if ms else ""
            solver_answer = NumeralEquationAugmentGenerator._extract_final_answer(cot)
            true_answer = NumeralEquationAugmentGenerator._extract_true_answer_from_part(part)
            records.append(
                {
                    "id": source_id,
                    "prompt": prompt,
                    "generated_cot": cot,
                    "answer": true_answer or solver_answer,
                    "solver_answer": solver_answer,
                    "source_solver_correct": bool(true_answer and solver_answer and true_answer == solver_answer),
                    "label": "numeral equations",
                }
            )
        return pd.DataFrame.from_records(records)

    @classmethod
    def _parse_examples_from_prompt(cls, prompt: str) -> List[Dict[str, str]]:
        examples: List[Dict[str, str]] = []
        rx = cls._numeric_expr_re()
        for line in str(prompt or "").splitlines():
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            m = rx.fullmatch(lhs.strip())
            if m:
                examples.append({"a": m.group(1), "op": m.group(2).strip(), "b": m.group(3), "out": rhs.strip().replace(" ", "")})
        return examples

    @classmethod
    def _parse_target_from_prompt(cls, prompt: str) -> str:
        patterns = [
            r"(?is)(?:now,\s*)?determine\s+the\s+(?:result|output)\s+for:\s*([^\n.]+)",
            r"(?is)(?:result|output)\s+for:\s*([^\n.]+)",
            r"(?is)target\s*:?\s*([^\n.]+)",
        ]
        rx = cls._numeric_expr_re()
        for pattern in patterns:
            m = re.search(pattern, str(prompt or ""))
            if m:
                candidate = m.group(1).strip().rstrip(".").replace(" ", "")
                if rx.fullmatch(candidate):
                    return candidate
        return ""

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
        if m:
            return m.group(1).strip()
        m = re.search(r"Replay (?:the rule|contextual fallback) on target\s+(.+?):\s*$", str(cot or ""), flags=re.M)
        return m.group(1).strip() if m else ""

    @classmethod
    def _target_operator(cls, target_expr: str) -> str:
        m = cls._numeric_expr_re().fullmatch(target_expr or "")
        return m.group(2).strip() if m else ""

    @staticmethod
    def _format_relevant_examples(examples: Sequence[Dict[str, str]], operator: str) -> List[str]:
        return [f"{e['a']} {e['op']} {e['b']} = {e['out']}" for e in examples if e["op"] == operator]

    @staticmethod
    def _extract_display_format(cot: str, operator: str) -> str:
        pattern = rf"Operator\s+{re.escape(repr(operator))}\s+detected display format:\s*(\S+)"
        m = re.search(pattern, cot)
        if m:
            return m.group(1)
        m = re.search(r"(?:selected rule|Use selected rule|uses|Chosen contextual fallback rule:)[:\s]+rev_ops=(?:True|False)/rev_result=(?:True|False)/[^/]+/format=(\S+)", cot)
        return m.group(1) if m else "num"

    # ------------------------------------------------------------------
    # Old family-block extraction
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
            blocks.append({"index": index, "family_label": label, "text": text, "body": body, "answer": answer, "is_match": answer != "no", "operation_order": op_order, "family_formula": formula})
        return blocks

    def _keep_family_block(self, source_id: str, block: Dict[str, Any]) -> bool:
        if block.get("is_match"):
            return True
        if not self.config.downsample_no_match_family:
            return True
        mod = max(1, int(self.config.no_match_keep_mod))
        return (self._stable_hash_int(f"{source_id}_family_{block['index']}") % mod) == 0

    def _make_family_matching_problem(self, source_id: str, prompt: str, cot: str, source_answer: str, source_solver_correct: bool, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._keep_family_block(source_id, block):
            return None
        examples = self._parse_examples_from_cot(cot) or self._parse_examples_from_prompt(prompt)
        target = self._parse_target_from_cot(cot) or self._parse_target_from_prompt(prompt)
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
        family_prompt_lines.extend([
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
        ])
        output_text = block["text"] + "\nAnswer: " + block["answer"]
        return self._row(source_id, "family", str(block["index"]), "equation_family_matching", "numeral_equations_family_matching", "\n".join(family_prompt_lines), output_text, block["answer"], source_answer, source_solver_correct, extra={"family_index": block["index"], "family_label": block["family_label"], "is_match": block["is_match"]})

    # ------------------------------------------------------------------
    # Old rule-application extraction
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
                tasks.append({"kind": "verify", "operator": current_rule[0], "rule": current_rule[1], "expression": header, "calc_lines": calc_lines, "computed": computed, "answer": match or computed})
                i = j
                continue
            i += 1
        return tasks

    @staticmethod
    def _extract_target_application_task(cot: str) -> Optional[Dict[str, Any]]:
        m_rule = re.search(r"^Use selected rule:\s*(rev_ops=.*)$", cot or "", flags=re.M)
        m_replay = re.search(r"^Replay the rule on target\s+(.+?):\s*$", cot or "", flags=re.M)
        kind = "target"
        if not m_rule or not m_replay:
            m_rule = re.search(r"^Chosen contextual fallback rule:\s*(rev_ops=.*)$", cot or "", flags=re.M)
            m_replay = re.search(r"^Replay contextual fallback on target\s+(.+?):\s*$", cot or "", flags=re.M)
            kind = "contextual_target"
        if not m_rule or not m_replay:
            return None
        start = m_replay.end()
        tail = cot[start:]
        m_final = re.search(r"(?m)^Final answer:\s*(\S+)\s*$", tail)
        if not m_final:
            return None
        calc_part = tail[: m_final.start()].strip("\n")
        calc_lines = [ln[2:] if ln.startswith("  ") else ln for ln in calc_part.splitlines() if ln.strip()]
        final = m_final.group(1)
        return {"kind": kind, "rule": m_rule.group(1).strip(), "expression": m_replay.group(1).strip(), "calc_lines": calc_lines, "computed": final, "answer": final}

    def _make_rule_application_problem(self, source_id: str, prompt: str, source_answer: str, source_solver_correct: bool, task: Dict[str, Any], task_index: int) -> Optional[Dict[str, Any]]:
        if task["kind"] in {"target", "contextual_target"} and self.config.target_application_only_solver_correct and not source_solver_correct:
            return None
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
        return self._row(source_id, "apply", f"{task['kind']}_{task_index}_{task.get('expression', '')}", "equation_rule_application", "numeral_equations_rule_application", "\n".join(prompt_lines), output_text, task["answer"], source_answer, source_solver_correct, extra={"application_kind": task["kind"]})

    # ------------------------------------------------------------------
    # Contextual-v2 extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_contextual_fallback(cot: str) -> Optional[Dict[str, Any]]:
        if "Contextual fallback" not in (cot or ""):
            return None
        visible = []
        for m in re.finditer(r"(?m)^- visible operator (.+?) with weight (\d+): (rev_ops=(?:True|False)/rev_result=(?:True|False)/[^/]+/format=\S+); family=(\S+)", cot):
            visible.append({"operator": m.group(1).strip(), "weight": int(m.group(2)), "rule": m.group(3).strip(), "family": m.group(4).strip()})
        m_reason = re.search(r"(?m)^Family choice reason:\s*(.+)$", cot)
        m_rule = re.search(r"(?m)^Chosen contextual fallback rule:\s*(rev_ops=(?:True|False)/rev_result=(?:True|False)/[^/]+/format=\S+)", cot)
        m_replay = re.search(r"(?m)^Replay contextual fallback on target\s+(.+?):\s*$", cot)
        m_final = re.search(r"(?m)^Final answer:\s*(\S+)\s*$", cot)
        if not m_rule or not m_replay or not m_final:
            return None
        tail = cot[m_replay.end():m_final.start()].strip("\n")
        calc_lines = [ln[2:] if ln.startswith("  ") else ln for ln in tail.splitlines() if ln.strip()]
        return {"visible": visible, "reason": m_reason.group(1).strip() if m_reason else "", "chosen_rule": m_rule.group(1).strip(), "target": m_replay.group(1).strip(), "calc_lines": calc_lines, "answer": m_final.group(1).strip()}

    @staticmethod
    def _rule_transform(rule: str) -> str:
        m = re.match(r"rev_ops=(True|False)/rev_result=(True|False)/", rule or "")
        return f"rev_ops={m.group(1)}/rev_result={m.group(2)}" if m else ""

    @staticmethod
    def _family_set(visible: Sequence[Dict[str, Any]]) -> List[str]:
        return sorted({v["family"] for v in visible})

    def _make_context_inference_problem(self, source_id: str, prompt: str, source_answer: str, source_solver_correct: bool, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        visible = ctx["visible"]
        if not visible:
            return None
        # Weighted majority transform from visible rules.
        weights: Dict[str, int] = {}
        for v in visible:
            tr = self._rule_transform(v["rule"])
            weights[tr] = weights.get(tr, 0) + int(v.get("weight", 1))
        dominant = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        families = ",".join(self._family_set(visible))
        prompt_text = "\n".join([
            "In Alice's Wonderland, several visible operators have already been solved.",
            "Infer the prompt-level context: visible rules, dominant operand/result transform, and visible operation families.",
            "",
            "Source prompt:",
            prompt,
            "",
            "Return the visible rules and finish with:",
            "Answer: dominant=<rev_ops=.../rev_result=...>; families=<comma-separated families>",
        ])
        lines = []
        for v in visible:
            lines.append(f"visible operator {v['operator']} with weight {v['weight']}: {v['rule']}; family={v['family']}")
        lines.append(f"Answer: dominant={dominant}; families={families}")
        output_text = "\n".join(lines)
        return self._row(source_id, "context", "infer", "equation_context_inference", "numeral_equations_context_inference", prompt_text, output_text, f"dominant={dominant}; families={families}", source_answer, source_solver_correct)

    def _make_missing_family_problem(self, source_id: str, prompt: str, source_answer: str, source_solver_correct: bool, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        q_op = self._target_operator(ctx["target"])
        if q_op in STANDARD_OPS:
            return None
        visible_families = ",".join(self._family_set(ctx["visible"])) or "<none>"
        prompt_text = "\n".join([
            "In Alice's Wonderland, the target operator is not shown in the examples and is non-standard punctuation.",
            "Choose the contextual fallback rule from visible operation-family context; do not memorize exact punctuation symbols.",
            "",
            f"Target expression: {ctx['target']}",
            f"Visible family set: {visible_families}",
            "Visible solved operators:",
            *[f"- {v['operator']}: {v['rule']}; family={v['family']}; weight={v['weight']}" for v in ctx["visible"]],
            "",
            "Return the reasoning and finish with Answer: <fallback rule>.",
        ])
        output_lines = []
        if ctx.get("reason"):
            output_lines.append(f"Family choice reason: {ctx['reason']}")
        output_lines.append(f"Chosen contextual fallback rule: {ctx['chosen_rule']}")
        output_lines.append(f"Answer: {ctx['chosen_rule']}")
        return self._row(source_id, "missing_family", q_op, "equation_missing_family_selection", "numeral_equations_missing_family_selection", prompt_text, "\n".join(output_lines), ctx["chosen_rule"], source_answer, source_solver_correct)

    def _make_contextual_fallback_application_problem(self, source_id: str, prompt: str, source_answer: str, source_solver_correct: bool, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.config.target_application_only_solver_correct and not source_solver_correct:
            return None
        prompt_text = "\n".join([
            "In Alice's Wonderland, a contextual fallback rule has been selected for an unseen target operator.",
            "Apply this rule to the target expression.",
            "",
            f"Known contextual fallback rule: {ctx['chosen_rule']}",
            f"Target expression: {ctx['target']}",
            "",
            "Return the calculation and finish with Answer: <visible output>.",
        ])
        output_lines = list(ctx["calc_lines"])
        output_lines.append(f"Answer: {ctx['answer']}")
        return self._row(source_id, "context_apply", ctx["target"], "equation_contextual_fallback_application", "numeral_equations_contextual_fallback_application", prompt_text, "\n".join(output_lines), ctx["answer"], source_answer, source_solver_correct)

    # ------------------------------------------------------------------
    # Operator dropout augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_target_examples_from_prompt(prompt: str, target_op: str) -> Tuple[str, int]:
        lines = str(prompt or "").splitlines()
        rx = NumeralEquationAugmentGenerator._numeric_expr_re()
        kept = []
        removed = 0
        for line in lines:
            if "=" in line:
                lhs = line.split("=", 1)[0].strip()
                m = rx.fullmatch(lhs)
                if m and m.group(2).strip() == target_op:
                    removed += 1
                    continue
            kept.append(line)
        return "\n".join(kept), removed

    @staticmethod
    def _extract_selected_rule(cot: str) -> str:
        m = re.search(r"(?m)^Use selected rule:\s*(rev_ops=.*)$", cot or "")
        return m.group(1).strip() if m else ""

    def _make_operator_dropout_problem(self, source_id: str, prompt: str, cot: str, source_answer: str, source_solver_correct: bool) -> Optional[Dict[str, Any]]:
        if self.config.dropout_only_solver_correct and not source_solver_correct:
            return None
        target = self._parse_target_from_prompt(prompt) or self._parse_target_from_cot(cot)
        q_op = self._target_operator(target)
        if not target or not q_op:
            return None
        examples = self._parse_examples_from_prompt(prompt)
        if not any(e["op"] == q_op for e in examples):
            return None
        if self.config.dropout_nonstandard_only and q_op in STANDARD_OPS:
            return None
        selected_rule = self._extract_selected_rule(cot)
        app_task = self._extract_target_application_task(cot)
        if not selected_rule or not app_task:
            return None
        dropped_prompt, removed = self._remove_target_examples_from_prompt(prompt, q_op)
        if removed <= 0:
            return None
        prompt_text = "\n".join([
            "In Alice's Wonderland, solve the target when its operator examples have been intentionally hidden.",
            "Use the remaining prompt context to infer a robust contextual fallback; do not rely on memorizing the exact punctuation symbol.",
            "",
            "Dropped-prompt version:",
            dropped_prompt,
            "",
            f"Hidden target operator: {repr(q_op)} ({removed} direct example(s) removed)",
            "Return the reasoning and finish with Answer: <visible output>.",
        ])
        output_lines = [
            f"The direct examples for operator {repr(q_op)} were removed, so treat the target as unseen.",
            f"Use the source-consistent target rule as the synthetic fallback label: {selected_rule}",
        ]
        output_lines.extend(app_task["calc_lines"])
        output_lines.append(f"Answer: {app_task['answer']}")
        return self._row(source_id, "dropout", q_op, "equation_operator_dropout_contextual", "numeral_equations_operator_dropout_contextual", prompt_text, "\n".join(output_lines), app_task["answer"], source_answer, source_solver_correct, extra={"dropped_operator": q_op, "removed_examples": removed, "dropout_target_type": "standard" if q_op in STANDARD_OPS else "nonstandard"})

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def _row(self, source_id: str, task_prefix: str, key: str, task_mode: str, label: str, prompt: str, completion: str, answer: str, source_answer: str, source_solver_correct: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row = {
            "id": self._make_id(source_id, task_prefix, key),
            "prompt": prompt,
            "completion": completion,
            "answer": answer,
            "generated_cot": completion,
            "computed_answer": answer,
            "category": label,
            "label": label,
            "task_mode": task_mode,
            "source_task_id": source_id,
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
        if not prompt or not self.is_numeral_prompt(prompt):
            return []

        source_answer = str(row.get("answer", "")) if "answer" in row and pd.notna(row.get("answer")) else self._extract_final_answer(cot)
        source_solver_answer = str(row.get("solver_answer", "")) if "solver_answer" in row and pd.notna(row.get("solver_answer")) else self._extract_final_answer(cot)
        if "source_solver_correct" in row and pd.notna(row.get("source_solver_correct")):
            source_solver_correct = bool(row.get("source_solver_correct"))
        else:
            source_solver_correct = bool(source_answer and source_solver_answer and str(source_answer) == str(source_solver_answer))
        if self.config.only_solver_correct and not source_solver_correct:
            return []

        rows: List[Dict[str, Any]] = []

        if cot and self.config.include_family_matching:
            for block in self._extract_family_blocks(cot):
                item = self._make_family_matching_problem(source_id, prompt, cot, source_answer, source_solver_correct, block)
                if item is not None:
                    rows.append(item)

        if cot and self.config.include_rule_application:
            application_tasks = self._extract_verify_application_tasks(cot)
            target_task = self._extract_target_application_task(cot)
            if target_task is not None:
                application_tasks.append(target_task)
            for idx, task in enumerate(application_tasks):
                item = self._make_rule_application_problem(source_id, prompt, source_answer, source_solver_correct, task, idx)
                if item is not None:
                    rows.append(item)

        ctx = self._extract_contextual_fallback(cot) if cot else None
        if ctx:
            if self.config.include_context_inference:
                item = self._make_context_inference_problem(source_id, prompt, source_answer, source_solver_correct, ctx)
                if item is not None:
                    rows.append(item)
            if self.config.include_missing_family_selection:
                item = self._make_missing_family_problem(source_id, prompt, source_answer, source_solver_correct, ctx)
                if item is not None:
                    rows.append(item)
            if self.config.include_contextual_fallback_application:
                item = self._make_contextual_fallback_application_problem(source_id, prompt, source_answer, source_solver_correct, ctx)
                if item is not None:
                    rows.append(item)

        if cot and self.config.include_operator_dropout:
            item = self._make_operator_dropout_problem(source_id, prompt, cot, source_answer, source_solver_correct)
            if item is not None:
                rows.append(item)

        return rows

    def generate_dataset(self, source_data: pd.DataFrame, sample_frac: Optional[float] = None, sample_n: Optional[int] = None, only_solver_correct: Optional[bool] = None) -> pd.DataFrame:
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
            for _, r in work.iterrows():
                sid = str(r["id"]) if "id" in r and pd.notna(r["id"]) else str(r.name)
                rows.extend(self.make_tasks_for_row(r, source_id=sid))
        finally:
            self.config.only_solver_correct = old_flag

        if not rows:
            return pd.DataFrame(columns=["id", "prompt", "completion", "answer", "label", "generated_cot", "computed_answer", "task_mode"])
        return pd.DataFrame.from_records(rows)

    def generate_dataset_from_text_dump(self, text: str) -> pd.DataFrame:
        return self.generate_dataset(self.parse_text_dump(text))
