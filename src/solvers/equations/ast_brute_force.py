from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


Hypothesis = Tuple[str, bool, bool, str]
Attempt = Dict[str, Any]


@dataclass
class SolverConfig:
    # Trace over the reduced search space:
    #   2 operand transforms × 2 result transforms × (8 common + 11 tier-2 rare operations)
    # Output prefix/suffix for negatives is normalized before matching, so it is NOT a search axis.
    #
    # Default policy is ordered first-match search:
    #   try candidates in deterministic priority order;
    #   select the first candidate that matches all examples;
    #   stop searching that operator.
    # This keeps the trace replay-complete without logging lower-priority candidates that
    # the algorithm never uses. Set stop_after_first_match=False to recover exhaustive
    # full-space logging.
    include_search_space: bool = True
    include_output_normalization: bool = True
    include_all_candidate_attempts: bool = True
    include_failed_candidate_steps: bool = True
    failed_candidate_mode: str = "until_first_failure"  # "until_first_failure" or "all_examples"
    include_verification: bool = True
    stop_after_first_match: bool = True
    include_untried_candidate_summary: bool = True
    max_candidate_attempts_per_operator: Optional[int] = None


@dataclass
class Example:
    a: str
    op: str
    b: str
    raw_out: str
    norm_out: str


@dataclass
class EvalStep:
    input_a: str
    input_b: str
    transformed_a: str
    transformed_b: str
    op_name: str
    raw_result: str
    after_result_transform: str
    final_output: str
    expected: Optional[str]
    match: Optional[bool]
    lines: List[str]


class ASTBruteForceSolver:
    """
    Equation solver with reduced search space and replay-complete trace.

    Main idea:
      1. Detect/normalize output sign notation per operator.
         Example: 18 } 50 = }32 is normalized to expected numeric output -32.
      2. Search only:
           reverse operands? yes/no
           reverse result? yes/no
           operation in common-first, tier-2-rare-second candidate list
      3. Use ordered first-match search by default: keep the complete trace up to the
         first candidate that matches all examples, then stop searching that operator.

    This intentionally removes the old axis:
      operand_config × operation × output_format
    and replaces it with:
      sign-normalization + (rev_ops × rev_result × operation)
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")

        self.common_ops_order = [
            "cat",
            "rev_cat",
            "add",
            "abs_diff",
            "neg_abs_diff",
            "sub",
            "rev_sub",
            "mul",
        ]
        # Tier 2 only. Tier 3 digit/cross/determinant feature-engineering operations
        # are intentionally excluded from the main training CoT search space.
        self.rare_ops_order = [
            "mul1",
            "mulm1",
            "add1",
            "addm1",
            "sub1",
            "subm1",
            "max_mod_min",
            "div",
            "mod",
            "rev_div",
            "rev_mod",
        ]
        self.transform_order = [
            (True, True),
            (False, False),
            (True, False),
            (False, True),
        ]

        self.op_desc = {
            "cat": "concatenation: write A followed by B",
            "rev_cat": "reverse concatenation: write B followed by A",
            "add": "addition: A + B",
            "abs_diff": "absolute difference: abs(A - B)",
            "neg_abs_diff": "negative absolute difference: -abs(A - B)",
            "sub": "subtraction: A - B",
            "rev_sub": "reverse subtraction: B - A",
            "mul": "multiplication: A * B",
            "mul1": "multiplication plus one: A * B + 1",
            "mulm1": "multiplication minus one: A * B - 1",
            "add1": "addition plus one: A + B + 1",
            "addm1": "addition minus one: A + B - 1",
            "sub1": "subtraction plus one: A - B + 1",
            "subm1": "subtraction minus one: A - B - 1",
            "max_mod_min": "larger operand modulo smaller operand: max(A,B) % min(A,B)",
            "div": "integer division: A // B",
            "mod": "modulo: A % B",
            "rev_div": "reverse integer division: B // A",
            "rev_mod": "reverse modulo: B % A",
            "digit_abs_diff": "digit absolute difference: |a1-b1| followed by |a2-b2|",
            "digit_add_mod10": "digit add mod 10: (a1+b1)%10 followed by (a2+b2)%10",
            "digit_sub_mod10": "digit sub mod 10: (a1-b1)%10 followed by (a2-b2)%10",
            "cross_mul": "cross multiply: a1*b1 + a2*b2",
            "cross_mul_rev": "reverse cross multiply: a1*b2 + a2*b1",
            "digit_mul": "digit multiply: a1*b1 followed by a2*b2",
            "digit_mul_rev": "reverse digit multiply: a1*b2 followed by a2*b1",
            "digit_sum_diff": "digit sum difference: (a1+a2) - (b1+b2)",
            "digit_sum_sum": "digit sum sum: (a1+a2) + (b1+b2)",
            "digit_product_diff": "digit product difference: a1*a2 - b1*b2",
            "digit_product_sum": "digit product sum: a1*a2 + b1*b2",
            "determinant": "determinant: a1*b2 - a2*b1",
            "abs_determinant": "absolute determinant: abs(a1*b2 - a2*b1)",
        }

    # Public API -------------------------------------------------------------

    def generate_cot(self, prompt: Any) -> str:
        examples_text, target_text = self._split_prompt(prompt)
        result = self.solve(examples_text, target_text)
        return result.get("solution") or "\n".join(result.get("debug", []))

    @staticmethod
    def extract_answer(cot_text: Any) -> str:
        text = "" if cot_text is None else str(cot_text)
        patterns = [
            r"(?im)^\s*Final answer\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Computed output\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Answer\s*:\s*(\S+)\s*$",
            r"\\boxed\{([^{}\s]+)\}",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                ans = m.group(1).strip()
                return "nan" if ans.lower() in {"none", "nan"} else ans
        return "nan"

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        qm = self._numeric_re.fullmatch(str(target_text).strip())
        if not qm:
            return self._failure("target expression must look like '<number><operator><number>'")

        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)
        raw_examples = self._parse_examples(examples_text)
        if not raw_examples:
            return self._failure("no valid examples found")

        fmt_by_op, examples = self._normalize_outputs(raw_examples)
        groups: Dict[str, List[Example]] = defaultdict(list)
        for ex in examples:
            groups[ex.op].append(ex)

        found_by_op: Dict[str, Hypothesis] = {}
        attempts_by_op: Dict[str, List[Attempt]] = {}
        for op in sorted(groups.keys()):
            found, attempts = self._match_operator(op, groups[op], fmt_by_op[op])
            attempts_by_op[op] = attempts
            if found is not None:
                found_by_op[op] = found

        lines: List[str] = []
        lines.extend(self._render_header(examples, f"{q_a}{q_op}{q_b}"))
        if self.config.include_search_space:
            lines.extend(self._render_search_space())
        if self.config.include_output_normalization:
            lines.extend(self._render_output_normalization(examples, fmt_by_op))
        lines.extend(self._render_rule_matching(groups, attempts_by_op, found_by_op))
        if self.config.include_verification:
            lines.extend(self._render_verification(groups, found_by_op))

        if q_op in found_by_op:
            final, target_lines = self._render_target_direct(q_a, q_op, q_b, found_by_op[q_op])
            lines.extend(target_lines)
            return self._success(final, lines, "direct_operator_rule", q_op, found_by_op[q_op], False)

        final, target_lines = self._render_fallback(q_a, q_op, q_b)
        lines.extend(target_lines)
        return self._success(final, lines, "fallback_abs_diff_for_unseen_operator", q_op, None, True)

    # Parsing / normalization ----------------------------------------------

    def _split_prompt(self, prompt: Any) -> Tuple[str, str]:
        text = "" if prompt is None else str(prompt)
        example_lines: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if "=" not in stripped:
                continue
            lhs = stripped.split("=", 1)[0].strip()
            if self._numeric_re.fullmatch(lhs):
                example_lines.append(stripped)

        target_patterns = [
            r"(?is)(?:now,\s*)?determine\s+the\s+(?:result|output)\s+for:\s*([^\n.]+)",
            r"(?is)(?:result|output)\s+for:\s*([^\n.]+)",
            r"(?is)target\s*:?\s*([^\n.]+)",
        ]
        target = ""
        for pattern in target_patterns:
            m = re.search(pattern, text)
            if m:
                candidate = m.group(1).strip().rstrip(".").replace(" ", "")
                if self._numeric_re.fullmatch(candidate):
                    target = candidate
                    break

        if not target:
            for line in reversed(text.splitlines()):
                candidate = line.strip().rstrip(".").replace(" ", "")
                if "=" not in candidate and self._numeric_re.fullmatch(candidate):
                    target = candidate
                    break

        return "\n".join(example_lines), target

    def _parse_examples(self, examples_text: str) -> List[Example]:
        parsed: List[Example] = []
        for line in str(examples_text).splitlines():
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if not m:
                continue
            a, op, b = m.group(1), m.group(2).strip(), m.group(3)
            out = rhs.replace(" ", "").strip()
            parsed.append(Example(a=a, op=op, b=b, raw_out=out, norm_out=out))
        return parsed

    def _normalize_outputs(self, raw_examples: List[Example]) -> Tuple[Dict[str, str], List[Example]]:
        groups: Dict[str, List[Example]] = defaultdict(list)
        for ex in raw_examples:
            groups[ex.op].append(ex)

        fmt_by_op: Dict[str, str] = {}
        normalized: List[Example] = []

        for op, group in groups.items():
            fmt = self._detect_format(op, group)
            fmt_by_op[op] = fmt
            for ex in group:
                normalized.append(
                    Example(
                        a=ex.a,
                        op=ex.op,
                        b=ex.b,
                        raw_out=ex.raw_out,
                        norm_out=self._normalize_output_for_format(ex.raw_out, op, fmt),
                    )
                )

        # Preserve original example order.
        by_key: Dict[Tuple[str, str, str, str], Example] = {
            (e.a, e.op, e.b, e.raw_out): e for e in normalized
        }
        ordered = [by_key[(e.a, e.op, e.b, e.raw_out)] for e in raw_examples]
        return fmt_by_op, ordered

    def _detect_format(self, op: str, group: List[Example]) -> str:
        if op != "-" and any(ex.raw_out.startswith(op) and len(ex.raw_out) > len(op) for ex in group):
            return "neg_prefix"
        if op != "-" and any(ex.raw_out.endswith(op) and len(ex.raw_out) > len(op) for ex in group):
            return "neg_suffix"
        if any(ex.raw_out.endswith("-") and len(ex.raw_out) > 1 for ex in group):
            return "neg_suffix_dash"
        # Do not treat a normal leading '-' as a separate output format. It is just a negative number.
        return "num"

    def _normalize_output_for_format(self, out: str, op: str, fmt: str) -> str:
        if fmt == "neg_prefix" and op and out.startswith(op) and len(out) > len(op):
            return "-" + out[len(op):]
        if fmt == "neg_suffix" and op and out.endswith(op) and len(out) > len(op):
            return "-" + out[:-len(op)]
        if fmt == "neg_suffix_dash" and out.endswith("-") and len(out) > 1:
            return "-" + out[:-1]
        return out

    def _denormalize_output_for_format(self, normalized: str, op: str, fmt: str) -> str:
        if fmt == "neg_prefix" and normalized.startswith("-"):
            return op + normalized[1:]
        if fmt == "neg_suffix" and normalized.startswith("-"):
            return normalized[1:] + op
        if fmt == "neg_suffix_dash" and normalized.startswith("-"):
            return normalized[1:] + "-"
        return normalized

    # Search ----------------------------------------------------------------

    def _match_operator(self, op: str, group: List[Example], fmt: str) -> Tuple[Optional[Hypothesis], List[Attempt]]:
        attempts: List[Attempt] = []
        selected: Optional[Hypothesis] = None
        attempt_index = 0

        for tier, op_names in (("common", self.common_ops_order), ("rare", self.rare_ops_order)):
            for rev_ops, rev_res in self.transform_order:
                for op_name in op_names:
                    attempt_index += 1
                    if (
                        self.config.max_candidate_attempts_per_operator is not None
                        and attempt_index > self.config.max_candidate_attempts_per_operator
                    ):
                        return selected, attempts

                    attempt = self._evaluate_candidate(
                        op_char=op,
                        group=group,
                        op_name=op_name,
                        rev_ops=rev_ops,
                        rev_res=rev_res,
                        fmt=fmt,
                        tier=tier,
                        attempt_index=attempt_index,
                    )
                    attempts.append(attempt)
                    if attempt["passes"] and selected is None:
                        selected = (op_name, rev_ops, rev_res, fmt)
                        if self.config.stop_after_first_match:
                            return selected, attempts

        return selected, attempts

    def _evaluate_candidate(
        self,
        op_char: str,
        group: List[Example],
        op_name: str,
        rev_ops: bool,
        rev_res: bool,
        fmt: str,
        tier: str,
        attempt_index: int,
    ) -> Attempt:
        example_steps: List[EvalStep] = []
        passes = True
        stop_reason = "all examples matched"

        for ex in group:
            step = self._eval_example(ex.a, ex.b, op_char, op_name, rev_ops, rev_res, fmt, expected=ex.norm_out)
            example_steps.append(step)
            if step.match is False:
                passes = False
                stop_reason = f"first mismatch on {ex.a}{op_char}{ex.b}: produced {step.after_result_transform}, expected {ex.norm_out}"
                if self.config.failed_candidate_mode != "all_examples":
                    break

        return {
            "index": attempt_index,
            "tier": tier,
            "op_name": op_name,
            "rev_ops": rev_ops,
            "rev_res": rev_res,
            "fmt": fmt,
            "passes": passes,
            "stop_reason": stop_reason,
            "steps": example_steps,
        }

    # Operation evaluation --------------------------------------------------

    def _rev_digits(self, s: str) -> str:
        return "-" + s[1:][::-1] if s.startswith("-") else s[::-1]

    def _transform_operands(self, a: str, b: str, rev_ops: bool) -> Tuple[str, str]:
        if rev_ops:
            return self._rev_digits(a), self._rev_digits(b)
        return a, b

    def _eval_example(
        self,
        a: str,
        b: str,
        op_char: str,
        op_name: str,
        rev_ops: bool,
        rev_res: bool,
        fmt: str,
        expected: Optional[str],
    ) -> EvalStep:
        ta, tb = self._transform_operands(a, b, rev_ops)
        lines: List[str] = []
        lines.append(self._operand_step_text(a, b, ta, tb, rev_ops))

        raw_result, op_lines = self._apply_operation(op_name, ta, tb)
        lines.extend(op_lines)

        if raw_result is None:
            normalized = "<invalid>"
            final = "<invalid>"
            lines.append(f"result transform: operation is invalid, so no result can be transformed")
        else:
            normalized = self._rev_digits(raw_result) if rev_res else raw_result
            if rev_res:
                lines.append(f"result transform: reverse digits of {raw_result} -> {normalized}")
            else:
                lines.append(f"result transform: keep result as {normalized}")
            final = self._denormalize_output_for_format(normalized, op_char, fmt)
            if fmt != "num":
                lines.append(f"display format {fmt}: normalized {normalized} -> visible {final}")
            else:
                lines.append(f"display format num: visible output is {final}")

        match: Optional[bool] = None
        if expected is not None:
            match = normalized == expected
            lines.append(f"compare normalized output: produced {normalized} vs expected {expected} -> {'MATCH' if match else 'WRONG'}")

        return EvalStep(
            input_a=a,
            input_b=b,
            transformed_a=ta,
            transformed_b=tb,
            op_name=op_name,
            raw_result=raw_result if raw_result is not None else "<invalid>",
            after_result_transform=normalized,
            final_output=final,
            expected=expected,
            match=match,
            lines=lines,
        )

    def _apply_operation(self, op_name: str, sa: str, sb: str) -> Tuple[Optional[str], List[str]]:
        lines: List[str] = []
        try:
            a, b = int(sa), int(sb)
        except ValueError:
            return None, [f"operation {op_name}: invalid integer operands A={sa}, B={sb}"]

        def emit(expr: str, value: Any) -> Tuple[str, List[str]]:
            return str(value), [f"operation {op_name}: {expr} = {value}"]

        if op_name == "cat":
            return sa + sb, [f"operation cat: concat({sa}, {sb}) = {sa + sb}"]
        if op_name == "rev_cat":
            return sb + sa, [f"operation rev_cat: concat({sb}, {sa}) = {sb + sa}"]
        if op_name == "add":
            return emit(f"{a} + {b}", a + b)
        if op_name == "abs_diff":
            return emit(f"abs({a} - {b})", abs(a - b))
        if op_name == "neg_abs_diff":
            return emit(f"-abs({a} - {b})", -abs(a - b))
        if op_name == "sub":
            return emit(f"{a} - {b}", a - b)
        if op_name == "rev_sub":
            return emit(f"{b} - {a}", b - a)
        if op_name == "mul":
            return emit(f"{a} * {b}", a * b)
        if op_name == "mul1":
            return emit(f"{a} * {b} + 1", a * b + 1)
        if op_name == "mulm1":
            return emit(f"{a} * {b} - 1", a * b - 1)
        if op_name == "add1":
            return emit(f"{a} + {b} + 1", a + b + 1)
        if op_name == "addm1":
            return emit(f"{a} + {b} - 1", a + b - 1)
        if op_name == "sub1":
            return emit(f"{a} - {b} + 1", a - b + 1)
        if op_name == "subm1":
            return emit(f"{a} - {b} - 1", a - b - 1)
        if op_name == "max_mod_min":
            if a == 0 or b == 0:
                return None, ["operation max_mod_min: invalid because min(A,B) is zero"]
            big, small = max(a, b), min(a, b)
            return emit(f"{big} % {small}", big % small)
        if op_name == "div":
            if b == 0:
                return None, ["operation div: invalid because B is zero"]
            return emit(f"{a} // {b}", a // b)
        if op_name == "mod":
            if b == 0:
                return None, ["operation mod: invalid because B is zero"]
            return emit(f"{a} % {b}", a % b)
        if op_name == "rev_div":
            if a == 0:
                return None, ["operation rev_div: invalid because A is zero"]
            return emit(f"{b} // {a}", b // a)
        if op_name == "rev_mod":
            if a == 0:
                return None, ["operation rev_mod: invalid because A is zero"]
            return emit(f"{b} % {a}", b % a)

        digs = self._two_digits(sa, sb)
        if digs is None:
            return None, [f"operation {op_name}: invalid because both transformed operands must be two unsigned digits"]
        d1, d2, d3, d4 = digs

        if op_name == "digit_abs_diff":
            value = f"{abs(d1 - d3)}{abs(d2 - d4)}"
            return value, [f"operation digit_abs_diff: |{d1}-{d3}| || |{d2}-{d4}| = {value}"]
        if op_name == "digit_add_mod10":
            value = f"{(d1 + d3) % 10}{(d2 + d4) % 10}"
            return value, [f"operation digit_add_mod10: ({d1}+{d3})%10 || ({d2}+{d4})%10 = {value}"]
        if op_name == "digit_sub_mod10":
            value = f"{(d1 - d3) % 10}{(d2 - d4) % 10}"
            return value, [f"operation digit_sub_mod10: ({d1}-{d3})%10 || ({d2}-{d4})%10 = {value}"]
        if op_name == "cross_mul":
            return emit(f"{d1}*{d3} + {d2}*{d4}", d1 * d3 + d2 * d4)
        if op_name == "cross_mul_rev":
            return emit(f"{d1}*{d4} + {d2}*{d3}", d1 * d4 + d2 * d3)
        if op_name == "digit_mul":
            value = f"{d1 * d3}{d2 * d4}"
            return value, [f"operation digit_mul: {d1}*{d3} || {d2}*{d4} = {value}"]
        if op_name == "digit_mul_rev":
            value = f"{d1 * d4}{d2 * d3}"
            return value, [f"operation digit_mul_rev: {d1}*{d4} || {d2}*{d3} = {value}"]
        if op_name == "digit_sum_diff":
            return emit(f"({d1}+{d2}) - ({d3}+{d4})", (d1 + d2) - (d3 + d4))
        if op_name == "digit_sum_sum":
            return emit(f"({d1}+{d2}) + ({d3}+{d4})", (d1 + d2) + (d3 + d4))
        if op_name == "digit_product_diff":
            return emit(f"{d1}*{d2} - {d3}*{d4}", d1 * d2 - d3 * d4)
        if op_name == "digit_product_sum":
            return emit(f"{d1}*{d2} + {d3}*{d4}", d1 * d2 + d3 * d4)
        if op_name == "determinant":
            return emit(f"{d1}*{d4} - {d2}*{d3}", d1 * d4 - d2 * d3)
        if op_name == "abs_determinant":
            return emit(f"abs({d1}*{d4} - {d2}*{d3})", abs(d1 * d4 - d2 * d3))

        return None, [f"operation {op_name}: unknown operation"]

    @staticmethod
    def _two_digits(sa: str, sb: str) -> Optional[Tuple[int, int, int, int]]:
        if sa.startswith("-") or sb.startswith("-"):
            return None
        if len(sa) != 2 or len(sb) != 2 or not sa.isdigit() or not sb.isdigit():
            return None
        return int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])

    # Rendering --------------------------------------------------------------

    @staticmethod
    def _literal(text: str) -> str:
        return repr(text)

    def _hypothesis_name(self, h: Hypothesis) -> str:
        op_name, rev_ops, rev_res, fmt = h
        return f"rev_ops={rev_ops}/rev_result={rev_res}/{op_name}/format={fmt}"

    def _render_header(self, examples: List[Example], target_expr: str) -> List[str]:
        policy = "ordered first-match search" if self.config.stop_after_first_match else "exhaustive matching"
        lines = [
            f"We need to infer the hidden equation transformation by {policy} over a reduced search space.",
            "The output is judged only by the final answer, but the trace below is written so the solution can be replayed from the prompt and this text.",
            "",
            "Examples",
        ]
        for i, ex in enumerate(examples, 1):
            lines.append(f"{i}. {ex.a} {ex.op} {ex.b} = {ex.raw_out}")
        lines.append(f"Target: {target_expr}")
        lines.append("")
        return lines

    def _render_search_space(self) -> List[str]:
        lines = [
            "Search space",
            "Search policy: ordered first-match search. The first candidate that matches all examples is selected; lower-priority candidates are not tested.",
            "We do not search arbitrary output formats. First we normalize operator-prefix/operator-suffix negative outputs, then we search:",
            "- operand transform: normal operands or reversed digits of both operands",
            "- result transform: normal result digits or reversed result digits",
            "- operations: common operations first, rare operations second",
            "Transform order:",
        ]
        for rev_ops, rev_res in self.transform_order:
            lines.append(f"- rev_ops={rev_ops}, rev_result={rev_res}")
        lines.append("Common operation order: " + ", ".join(self.common_ops_order))
        lines.append("Rare operation order: " + ", ".join(self.rare_ops_order))
        lines.append(f"Maximum possible candidates per operator: 4 * ({len(self.common_ops_order)} + {len(self.rare_ops_order)}) = {4 * (len(self.common_ops_order) + len(self.rare_ops_order))}")
        if not self.config.stop_after_first_match:
            lines.append("Configured mode: exhaustive; continue after a match and log all candidates.")
        lines.append("")
        return lines

    def _render_output_normalization(self, examples: List[Example], fmt_by_op: Dict[str, str]) -> List[str]:
        lines = ["Normalize outputs"]
        for op in sorted(fmt_by_op):
            fmt = fmt_by_op[op]
            lines.append(f"Operator {self._literal(op)} detected display format: {fmt}")
            op_examples = [e for e in examples if e.op == op]
            for ex in op_examples:
                if ex.raw_out == ex.norm_out:
                    lines.append(f"- {ex.raw_out} stays {ex.norm_out}")
                else:
                    lines.append(f"- {ex.raw_out} normalizes to {ex.norm_out}")
        lines.append("")
        return lines

    def _render_rule_matching(
        self,
        groups: Dict[str, List[Example]],
        attempts_by_op: Dict[str, List[Attempt]],
        found_by_op: Dict[str, Hypothesis],
    ) -> List[str]:
        lines: List[str] = ["Rule matching"]
        for op in sorted(groups.keys()):
            group = groups[op]
            attempts = attempts_by_op[op]
            selected = found_by_op.get(op)
            lines.append(f"Operator {self._literal(op)}")
            lines.append("examples: " + "; ".join(f"{ex.a} {op} {ex.b} -> raw {ex.raw_out}, normalized {ex.norm_out}" for ex in group))
            max_possible = 4 * (len(self.common_ops_order) + len(self.rare_ops_order))
            lines.append(f"attempted candidates: {len(attempts)} out of {max_possible} possible")
            if selected:
                lines.append(f"selected first passing candidate: {self._hypothesis_name(selected)}")
                if self.config.stop_after_first_match:
                    skipped = max_possible - len(attempts)
                    lines.append(
                        f"search stopped after the first full match; lower-priority candidates not tried: {skipped}"
                    )
            else:
                lines.append("selected first passing candidate: none")

            if self.config.include_all_candidate_attempts:
                for attempt in attempts:
                    mark = " SELECTED" if selected and self._attempt_to_hypothesis(attempt) == selected else ""
                    status = "PASS" if attempt["passes"] else "FAIL"
                    lines.append(
                        f"[{attempt['index']:03d}] {status}{mark}: "
                        f"tier={attempt['tier']}, rev_ops={attempt['rev_ops']}, rev_result={attempt['rev_res']}, operation={attempt['op_name']}"
                    )
                    if attempt["passes"]:
                        lines.append("  reason: all examples matched")
                    else:
                        lines.append(f"  reason: {attempt['stop_reason']}")

                    if attempt["passes"] or self.config.include_failed_candidate_steps:
                        for step_i, step in enumerate(attempt["steps"], 1):
                            lines.append(f"  example {step_i}: {step.input_a} {op} {step.input_b}")
                            for detail in step.lines:
                                lines.append(f"    {detail}")
                    elif attempt["steps"]:
                        step = attempt["steps"][-1]
                        lines.append(
                            f"  first checked example: produced {step.after_result_transform}, expected {step.expected}"
                        )
            lines.append("")
        return lines

    def _attempt_to_hypothesis(self, attempt: Attempt) -> Hypothesis:
        return (attempt["op_name"], attempt["rev_ops"], attempt["rev_res"], attempt["fmt"])

    def _render_verification(self, groups: Dict[str, List[Example]], found_by_op: Dict[str, Hypothesis]) -> List[str]:
        lines: List[str] = ["Verify selected rules"]
        for op in sorted(groups.keys()):
            if op not in found_by_op:
                lines.append(f"Operator {self._literal(op)} has no selected rule to verify")
                continue
            hyp = found_by_op[op]
            op_name, rev_ops, rev_res, fmt = hyp
            lines.append(f"Operator {self._literal(op)} uses {self._hypothesis_name(hyp)}")
            for ex in groups[op]:
                step = self._eval_example(ex.a, ex.b, op, op_name, rev_ops, rev_res, fmt, expected=ex.norm_out)
                lines.append(f"- {ex.a} {op} {ex.b} = {ex.raw_out}")
                for detail in step.lines:
                    lines.append(f"  {detail}")
        lines.append("")
        return lines

    def _render_target_direct(self, q_a: str, q_op: str, q_b: str, hyp: Hypothesis) -> Tuple[str, List[str]]:
        op_name, rev_ops, rev_res, fmt = hyp
        step = self._eval_example(q_a, q_b, q_op, op_name, rev_ops, rev_res, fmt, expected=None)
        lines = [
            "Target",
            f"Target operator {self._literal(q_op)} was found in the examples.",
            f"Use selected rule: {self._hypothesis_name(hyp)}",
            f"Replay the rule on target {q_a} {q_op} {q_b}:",
        ]
        for detail in step.lines:
            lines.append(f"  {detail}")
        lines.append(f"Computed output: {step.final_output}")
        lines.append(f"Final answer: {step.final_output}")
        lines.append(self._boxed(step.final_output))
        return step.final_output, lines

    def _render_fallback(self, q_a: str, q_op: str, q_b: str) -> Tuple[str, List[str]]:
        hyp: Hypothesis = ("abs_diff", False, False, "num")
        step = self._eval_example(q_a, q_b, q_op, "abs_diff", False, False, "num", expected=None)
        lines = [
            "Target",
            f"Target operator {self._literal(q_op)} was not found in the examples.",
            "Fallback rule: use absolute difference on the original operands.",
            "This fallback is explicit: no operand reversal, no result reversal, no learned operation from another symbol.",
            f"Replay fallback on target {q_a} {q_op} {q_b}:",
        ]
        for detail in step.lines:
            lines.append(f"  {detail}")
        lines.append(f"Computed output: {step.final_output}")
        lines.append(f"Final answer: {step.final_output}")
        lines.append(self._boxed(step.final_output))
        return step.final_output, lines

    def _operand_step_text(self, a: str, b: str, ta: str, tb: str, rev_ops: bool) -> str:
        if rev_ops:
            return f"operand transform: reverse both operands: {a}->{ta}, {b}->{tb}; A={int(ta)}, B={int(tb)}"
        return f"operand transform: keep operands: {a}->{ta}, {b}->{tb}; A={int(ta)}, B={int(tb)}"

    @staticmethod
    def _boxed(answer: str) -> str:
        # If the answer itself contains braces, do not try to make a malformed LaTeX box.
        if "{" in answer or "}" in answer:
            return f"Boxed answer text: {answer}"
        return f"\\boxed{{{answer}}}"

    def _success(
        self,
        answer: str,
        lines: List[str],
        rule_source: str,
        target_operator: str,
        selected: Optional[Hypothesis],
        fallback: bool,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "debug": lines,
            "trace": lines,
            "solution": "\n".join(lines),
            "rule_source": rule_source,
            "training_category": "equations_transformation.full_trace",
            "metadata": {
                "target_operator_seen_in_examples": not fallback,
                "uses_fallback_inference": fallback,
                "target_operator": target_operator,
                "selected_rule": self._hypothesis_name(selected) if selected else None,
                "search_space": "sign_normalization + rev_ops/rev_result/operation",
                "search_policy": "ordered_first_match" if self.config.stop_after_first_match else "exhaustive_all_candidates",
            },
        }

    @staticmethod
    def _failure(message: str) -> Dict[str, Any]:
        lines = ["Failed to solve", message, "Final answer: nan"]
        return {"answer": "nan", "debug": lines, "trace": lines, "solution": "\n".join(lines)}
