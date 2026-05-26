from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


Hypothesis = Tuple[str, bool, bool, str]
Attempt = Dict[str, Any]


@dataclass
class Example:
    a: str
    op: str
    b: str
    raw_out: str
    norm_out: str


@dataclass
class RequiredRow:
    input_a: str
    input_b: str
    transformed_a: str
    transformed_b: str
    raw_out: str
    norm_out: str
    required_raw_result: str
    rev_ops: bool
    rev_res: bool
    lines: List[str]




@dataclass
class FamilyResult:
    family: str
    family_label: str
    candidate_ops: List[str]
    op_passes: Dict[str, bool]
    op_reasons: Dict[str, str]
    lines: List[str]


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
    Equation solver with reduced search space and replay-complete family-search trace.

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

    def __init__(self):
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
        """Target-first solver.

        This version intentionally does NOT infer rules for every operator in the
        prompt. The metric uses only the final answer, so the search is focused on
        the operator that appears in the target expression.

        Policy:
          1. Parse and normalize all examples cheaply.
          2. If the target operator appears in examples, use only those examples.
          3. Run ordered family search for that operator.
          4. Verify the selected target-operator rule and apply it to the target.
          5. If the target operator never appears, use the explicit abs-diff fallback.

        This preserves the same per-operator rule search as the previous
        family-search solver, but avoids solving irrelevant operators.
        """
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

        lines: List[str] = []
        lines.extend(self._render_header(examples, f"{q_a}{q_op}{q_b}"))
        lines.extend(self._render_search_space())

        lines.append("Target-first filtering")
        lines.append(f"The target expression uses operator {self._literal(q_op)}.")
        if q_op in groups:
            ignored_ops = [op for op in sorted(groups) if op != q_op]
            lines.append(
                f"Use only examples with operator {self._literal(q_op)} first, because only this operator is needed to compute the final answer."
            )
            if ignored_ops:
                lines.append(
                    "Ignored for rule search: "
                    + ", ".join(f"operator {self._literal(op)} ({len(groups[op])} example{'s' if len(groups[op]) != 1 else ''})" for op in ignored_ops)
                    + "."
                )
            else:
                lines.append("No other operators are present.")
            lines.append("")

            lines.extend(self._render_output_normalization_for_ops(examples, fmt_by_op, [q_op]))

            found, attempts = self._match_operator(q_op, groups[q_op], fmt_by_op[q_op])
            attempts_by_op: Dict[str, List[Attempt]] = {q_op: attempts}
            found_by_op: Dict[str, Hypothesis] = {q_op: found} if found is not None else {}

            lines.extend(self._render_rule_matching({q_op: groups[q_op]}, attempts_by_op, found_by_op))
            lines.extend(self._render_verification({q_op: groups[q_op]}, found_by_op))

            if found is not None:
                final, target_lines = self._render_target_direct(q_a, q_op, q_b, found)
                lines.extend(target_lines)
                return self._success(final, lines, "target_first_direct_operator_rule", q_op, found, False)

            # Target operator was present, but no rule in the reduced family space
            # explained its examples. Use the same explicit fallback rather than
            # spending tokens solving unrelated operators that cannot define this
            # target operator under the per-operator policy.
            final, target_lines = self._render_fallback_no_rule(q_a, q_op, q_b)
            lines.extend(target_lines)
            return self._success(final, lines, "fallback_abs_diff_for_unresolved_target_operator", q_op, None, True)

        # Target operator is absent. Unseen operators
        # use the explicit absolute-difference fallback. There is no need to infer
        # unrelated operators first.
        lines.append(f"No example uses target operator {self._literal(q_op)}.")
        lines.append("Skip rule search and use the explicit unseen-operator fallback.")
        lines.append("")
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
        # operator prefix/suffix means a negative number displayed with the operator sign.
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

    def _family_search_order(self) -> List[Tuple[str, str, List[str]]]:
        """Deterministic family order used by the family-search policy.

        The old candidate order was operation-position based. In this policy the
        solver searches by mathematical family first. This is a deliberate policy
        change: fewer blocks are tested and the trace is shorter, but ambiguous
        cases can select a different rule than pure candidate-position search.
        """
        return [
            ("concat", "cat/rev_cat", ["cat", "rev_cat"]),
            ("add_offset", "add/add1/addm1", ["add", "add1", "addm1"]),
            ("abs_diff_sign", "abs_diff/neg_abs_diff", ["abs_diff", "neg_abs_diff"]),
            ("sub_offset", "sub/sub1/subm1", ["sub", "sub1", "subm1"]),
            ("single_rev_sub", "rev_sub", ["rev_sub"]),
            ("mul_offset", "mul/mul1/mulm1", ["mul", "mul1", "mulm1"]),
            ("mod_family", "max_mod_min/mod/rev_mod", ["max_mod_min", "mod", "rev_mod"]),
            ("div_family", "div/rev_div", ["div", "rev_div"]),
        ]

    def _match_operator(self, op: str, group: List[Example], fmt: str) -> Tuple[Optional[Hypothesis], List[Attempt]]:
        """Family-first search.

        Search unit:
          family block × all transform pairs

        Selection policy:
          1. Evaluate families in deterministic family order.
          2. Each family block computes decisions for every rev_ops/rev_result pair.
          3. If a family block contains at least one passing operation, select the
             first passing transform/op combination by transform order, then the
             operation order inside that family.
          4. Stop searching this operator after the first passing family block.

        This keeps the trace replayable while avoiding redundant candidate-level
        FAIL lines for operations that a family proof has already ruled out.
        """
        attempts: List[Attempt] = []
        selected: Optional[Hypothesis] = None
        required_cache: Dict[Tuple[bool, bool], List[RequiredRow]] = {}

        def get_required_rows(rv_ops: bool, rv_res: bool) -> List[RequiredRow]:
            k = (rv_ops, rv_res)
            if k not in required_cache:
                required_cache[k] = self._required_rows_for_transform(
                    op_char=op,
                    group=group,
                    rev_ops=rv_ops,
                    rev_res=rv_res,
                )
            return required_cache[k]

        family_order = self._family_search_order()
        for family_index, (family, family_label, op_order) in enumerate(family_order, 1):
            batch_results: Dict[Tuple[bool, bool], FamilyResult] = {}
            batch_required_rows: Dict[Tuple[bool, bool], List[RequiredRow]] = {}
            passing_hypotheses: List[Hypothesis] = []

            for rev_ops, rev_res in self.transform_order:
                rows = get_required_rows(rev_ops, rev_res)
                result = self._compute_family_result(family, rows)
                batch_required_rows[(rev_ops, rev_res)] = rows
                batch_results[(rev_ops, rev_res)] = result

                # Select within the family by transform order first, then by the
                # explicit operation order for this family. This is printed in the
                # CoT and is therefore replayable.
                for op_name in op_order:
                    if result.op_passes.get(op_name, False):
                        passing_hypotheses.append((op_name, rev_ops, rev_res, fmt))

            selected_in_block = passing_hypotheses[0] if passing_hypotheses else None
            attempt: Attempt = {
                "index": family_index,
                "family": family,
                "family_label": family_label,
                "op_order": op_order,
                "fmt": fmt,
                "passes": selected_in_block is not None,
                "selected_hypothesis": selected_in_block,
                "passing_hypotheses": passing_hypotheses,
                "batch_family_results": batch_results,
                "batch_required_rows": batch_required_rows,
                "family_blocks_reached": family_index,
                "total_family_blocks": len(family_order),
            }
            attempts.append(attempt)

            if selected_in_block is not None and selected is None:
                selected = selected_in_block
                break

        for a in attempts:
            a["family_blocks_reached"] = len(attempts)
            a["total_family_blocks"] = len(family_order)
        return selected, attempts

    def _required_rows_for_transform(
        self,
        op_char: str,
        group: List[Example],
        rev_ops: bool,
        rev_res: bool,
    ) -> List[RequiredRow]:
        rows: List[RequiredRow] = []
        for ex in group:
            ta, tb = self._transform_operands(ex.a, ex.b, rev_ops)
            lines: List[str] = []
            lines.append(self._operand_step_text(ex.a, ex.b, ta, tb, rev_ops))
            if rev_res:
                required = self._rev_digits(ex.norm_out)
                lines.append(
                    f"inverse result transform: visible normalized output {ex.norm_out} must have come from raw result {required}, because reversing {required} gives {ex.norm_out}"
                )
            else:
                required = ex.norm_out
                lines.append(f"inverse result transform: result is not reversed, so required raw result is {required}")
            rows.append(
                RequiredRow(
                    input_a=ex.a,
                    input_b=ex.b,
                    transformed_a=ta,
                    transformed_b=tb,
                    raw_out=ex.raw_out,
                    norm_out=ex.norm_out,
                    required_raw_result=required,
                    rev_ops=rev_ops,
                    rev_res=rev_res,
                    lines=lines,
                )
            )
        return rows

    def _family_for_operation(self, op_name: str) -> str:
        if op_name in {"cat", "rev_cat"}:
            return "concat"
        if op_name in {"add", "add1", "addm1"}:
            return "add_offset"
        if op_name in {"sub", "sub1", "subm1"}:
            return "sub_offset"
        if op_name in {"mul", "mul1", "mulm1"}:
            return "mul_offset"
        if op_name in {"abs_diff", "neg_abs_diff"}:
            return "abs_diff_sign"
        if op_name in {"div", "rev_div"}:
            return "div_family"
        if op_name in {"max_mod_min", "mod", "rev_mod"}:
            return "mod_family"
        return f"single_{op_name}"

    def _compute_family_result(self, family: str, rows: List[RequiredRow]) -> FamilyResult:
        if family == "concat":
            return self._compute_concat_family(rows)
        if family == "add_offset":
            return self._compute_offset_family(
                family=family,
                family_label="add/add1/addm1",
                rows=rows,
                base_name="A+B",
                base_fn=lambda a, b: a + b,
                op_by_offset={0: "add", 1: "add1", -1: "addm1"},
            )
        if family == "sub_offset":
            return self._compute_offset_family(
                family=family,
                family_label="sub/sub1/subm1",
                rows=rows,
                base_name="A-B",
                base_fn=lambda a, b: a - b,
                op_by_offset={0: "sub", 1: "sub1", -1: "subm1"},
            )
        if family == "mul_offset":
            return self._compute_offset_family(
                family=family,
                family_label="mul/mul1/mulm1",
                rows=rows,
                base_name="A*B",
                base_fn=lambda a, b: a * b,
                op_by_offset={0: "mul", 1: "mul1", -1: "mulm1"},
            )
        if family == "abs_diff_sign":
            return self._compute_abs_diff_family(rows)
        if family == "div_family":
            return self._compute_exact_operation_family(
                family=family,
                family_label="div/rev_div",
                rows=rows,
                op_names=["div", "rev_div"],
            )
        if family == "mod_family":
            return self._compute_mod_family(rows)
        if family.startswith("single_"):
            op_name = family[len("single_"):]
            return self._compute_exact_operation_family(
                family=family,
                family_label=op_name,
                rows=rows,
                op_names=[op_name],
            )
        return self._compute_exact_operation_family(family, family, rows, [])

    @staticmethod
    def _canonical_int(text: str) -> Optional[int]:
        try:
            value = int(text)
        except ValueError:
            return None
        return value if str(value) == text else None

    def _row_ints(self, row: RequiredRow) -> Tuple[int, int]:
        return int(row.transformed_a), int(row.transformed_b)

    def _canonical_required_values(self, rows: List[RequiredRow], lines: List[str]) -> Optional[List[int]]:
        values: List[int] = []
        for i, row in enumerate(rows, 1):
            y = self._canonical_int(row.required_raw_result)
            if y is None:
                lines.append(
                    f"  row {i}: required raw result {row.required_raw_result} is not a canonical integer string. "
                    f"Arithmetic operations output canonical integers like 2, -2, 0, so this family cannot match this row."
                )
                return None
            values.append(y)
        return values

    def _compute_offset_family(
        self,
        family: str,
        family_label: str,
        rows: List[RequiredRow],
        base_name: str,
        base_fn: Callable[[int, int], int],
        op_by_offset: Dict[int, str],
    ) -> FamilyResult:
        offset_desc = ", ".join(
            f"{op} needs k={'+' if off > 0 else ''}{off}" for off, op in op_by_offset.items()
        )
        lines: List[str] = [
            f"Family test {family_label}: compute k = required_raw_Y - ({base_name}). {offset_desc}.",
        ]
        required_values = self._canonical_required_values(rows, lines)
        op_passes = {op: False for op in op_by_offset.values()}
        op_reasons: Dict[str, str] = {}
        if required_values is None:
            for op in op_passes:
                op_reasons[op] = "required raw Y is not a canonical integer for at least one row"
            return FamilyResult(family, family_label, list(op_by_offset.values()), op_passes, op_reasons, lines)

        residuals: List[int] = []
        residual_terms: List[str] = []
        for row, y in zip(rows, required_values):
            a, b = self._row_ints(row)
            base = base_fn(a, b)
            k = y - base
            residuals.append(k)
            residual_terms.append(f"{y}-{base}={k}")
        lines.append(f"  residuals k: [{'; '.join(residual_terms)}] -> {residuals}")

        decisions: List[str] = []
        for offset, op in op_by_offset.items():
            passes = all(k == offset for k in residuals)
            op_passes[op] = passes
            offset_label = f"+{offset}" if offset > 0 else str(offset)
            op_reasons[op] = (
                f"all residuals are {offset_label}" if passes else f"residuals {residuals} are not all {offset_label}"
            )
            decisions.append(f"{op}={'MATCH' if passes else 'no'}")
        lines.append("  decisions: " + ", ".join(decisions))
        if not any(op_passes.values()):
            lines.append("  conclusion: no allowed offset 0, +1, or -1 fits all examples in this family")
        return FamilyResult(family, family_label, list(op_by_offset.values()), op_passes, op_reasons, lines)


    def _compute_concat_family(self, rows: List[RequiredRow]) -> FamilyResult:
        family = "concat"
        family_label = "cat/rev_cat"
        op_names = ["cat", "rev_cat"]
        op_passes = {op: True for op in op_names}
        op_reasons: Dict[str, str] = {}
        lines: List[str] = ["Family test cat/rev_cat: compare strings, because concatenation may keep leading zeros."]
        row_bits: List[str] = []
        for i, row in enumerate(rows, 1):
            cat = row.transformed_a + row.transformed_b
            rev_cat = row.transformed_b + row.transformed_a
            y = row.required_raw_result
            cat_match = cat == y
            rev_cat_match = rev_cat == y
            op_passes["cat"] = op_passes["cat"] and cat_match
            op_passes["rev_cat"] = op_passes["rev_cat"] and rev_cat_match
            row_bits.append(f"row{i}:Y={y}, cat={cat}({'ok' if cat_match else 'no'}), rev_cat={rev_cat}({'ok' if rev_cat_match else 'no'})")
        lines.append("  " + "; ".join(row_bits))
        lines.append("  decisions: " + ", ".join(f"{op}={'MATCH' if op_passes[op] else 'no'}" for op in op_names))
        for op in op_names:
            op_reasons[op] = f"{op} matches every row" if op_passes[op] else f"{op} fails at least one row"
        return FamilyResult(family, family_label, op_names, op_passes, op_reasons, lines)


    def _compute_abs_diff_family(self, rows: List[RequiredRow]) -> FamilyResult:
        family = "abs_diff_sign"
        family_label = "abs_diff/neg_abs_diff"
        op_names = ["abs_diff", "neg_abs_diff"]
        op_passes = {op: True for op in op_names}
        op_reasons: Dict[str, str] = {}
        lines: List[str] = ["Family test abs_diff/neg_abs_diff: D=abs(A-B); abs_diff needs Y=D, neg_abs_diff needs Y=-D."]
        required_values = self._canonical_required_values(rows, lines)
        if required_values is None:
            for op in op_names:
                op_passes[op] = False
                op_reasons[op] = "required raw Y is not a canonical integer for at least one row"
            return FamilyResult(family, family_label, op_names, op_passes, op_reasons, lines)
        row_bits: List[str] = []
        for i, (row, y) in enumerate(zip(rows, required_values), 1):
            a, b = self._row_ints(row)
            d = abs(a - b)
            abs_match = y == d
            neg_match = y == -d
            op_passes["abs_diff"] = op_passes["abs_diff"] and abs_match
            op_passes["neg_abs_diff"] = op_passes["neg_abs_diff"] and neg_match
            row_bits.append(f"row{i}:Y={y},D={d},Y=D:{'ok' if abs_match else 'no'},Y=-D:{'ok' if neg_match else 'no'}")
        lines.append("  " + "; ".join(row_bits))
        lines.append("  decisions: " + ", ".join(f"{op}={'MATCH' if op_passes[op] else 'no'}" for op in op_names))
        for op in op_names:
            op_reasons[op] = f"{op} matches every row" if op_passes[op] else f"{op} fails at least one row"
        return FamilyResult(family, family_label, op_names, op_passes, op_reasons, lines)


    def _compute_exact_operation_family(
        self,
        family: str,
        family_label: str,
        rows: List[RequiredRow],
        op_names: List[str],
    ) -> FamilyResult:
        op_passes = {op: True for op in op_names}
        op_reasons: Dict[str, str] = {}
        lines: List[str] = [f"Family test {family_label}: compute exact operation values and compare to required raw Y."]
        row_bits: List[str] = []
        for i, row in enumerate(rows, 1):
            pieces = [f"row{i}:Y={row.required_raw_result}"]
            for op in op_names:
                produced, _ = self._apply_operation(op, row.transformed_a, row.transformed_b)
                match = produced == row.required_raw_result
                op_passes[op] = op_passes[op] and bool(match)
                pieces.append(f"{op}={produced if produced is not None else 'invalid'}({'ok' if match else 'no'})")
            row_bits.append(",".join(pieces))
        if row_bits:
            lines.append("  " + "; ".join(row_bits))
        lines.append("  decisions: " + ", ".join(f"{op}={'MATCH' if op_passes[op] else 'no'}" for op in op_names))
        for op in op_names:
            op_reasons[op] = f"{op} matches every row" if op_passes[op] else f"{op} fails at least one row"
        return FamilyResult(family, family_label, op_names, op_passes, op_reasons, lines)


    def _compute_mod_family(self, rows: List[RequiredRow]) -> FamilyResult:
        family = "mod_family"
        family_label = "max_mod_min/mod/rev_mod"
        op_names = ["max_mod_min", "mod", "rev_mod"]
        op_passes = {op: True for op in op_names}
        op_reasons: Dict[str, str] = {}
        lines: List[str] = ["Family test max_mod_min/mod/rev_mod: first use modulo range 0..divisor-1, then exact values."]
        required_values = self._canonical_required_values(rows, lines)
        if required_values is None:
            for op in op_names:
                op_passes[op] = False
                op_reasons[op] = "required raw Y is not a canonical integer for at least one row"
            return FamilyResult(family, family_label, op_names, op_passes, op_reasons, lines)

        row_bits: List[str] = []
        for i, (row, y) in enumerate(zip(rows, required_values), 1):
            a, b = self._row_ints(row)
            pieces = [f"row{i}:Y={y}"]
            checks: List[Tuple[str, Optional[int], Optional[int]]] = []
            checks.append(("max_mod_min", None if (a == 0 or b == 0) else max(a, b) % min(a, b), None if (a == 0 or b == 0) else min(a, b)))
            checks.append(("mod", None if b == 0 else a % b, b if b != 0 else None))
            checks.append(("rev_mod", None if a == 0 else b % a, a if a != 0 else None))
            for op, produced, divisor in checks:
                if produced is None or divisor is None:
                    op_passes[op] = False
                    pieces.append(f"{op}=invalid(no)")
                    continue
                if y < 0 or y >= divisor:
                    op_passes[op] = False
                    pieces.append(f"{op}:Y outside 0..{divisor-1}(no)")
                    continue
                match = produced == y
                op_passes[op] = op_passes[op] and match
                pieces.append(f"{op}={produced}({'ok' if match else 'no'})")
            row_bits.append(",".join(pieces))
        lines.append("  " + "; ".join(row_bits))
        lines.append("  decisions: " + ", ".join(f"{op}={'MATCH' if op_passes[op] else 'no'}" for op in op_names))
        for op in op_names:
            op_reasons[op] = f"{op} matches every row" if op_passes[op] else f"{op} fails at least one row"
        return FamilyResult(family, family_label, op_names, op_passes, op_reasons, lines)


    def _eval_required_row(self, row: RequiredRow, op_char: str, op_name: str, fmt: str) -> EvalStep:
        # Kept for verification/debug compatibility. Main search uses family-pruning.
        lines: List[str] = []
        lines.append(
            f"use precomputed row: A={int(row.transformed_a)}, B={int(row.transformed_b)}, required raw result={row.required_raw_result}"
        )
        raw_result, op_lines = self._apply_operation(op_name, row.transformed_a, row.transformed_b)
        lines.extend(op_lines)
        if raw_result is None:
            produced = "<invalid>"
            final = "<invalid>"
            match = False
        else:
            produced = raw_result
            visible_norm = self._rev_digits(produced) if row.rev_res else produced
            final = self._denormalize_output_for_format(visible_norm, op_char, fmt)
            match = produced == row.required_raw_result
        lines.append(
            f"compare raw operation result: produced {produced} vs required {row.required_raw_result} -> {'MATCH' if match else 'WRONG'}"
        )
        return EvalStep(
            input_a=row.input_a,
            input_b=row.input_b,
            transformed_a=row.transformed_a,
            transformed_b=row.transformed_b,
            op_name=op_name,
            raw_result=produced,
            after_result_transform=self._rev_digits(produced) if produced != "<invalid>" and row.rev_res else produced,
            final_output=final,
            expected=row.required_raw_result,
            match=match,
            lines=lines,
        )

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
        lines = [
            "We need to infer only the rule needed for the target equation.",
            "The output is judged only by the final answer, so this solver uses target-first family search instead of solving every operator in the prompt.",
            "The trace below is written so the chosen rule and final answer can be replayed from the prompt and this text.",
            "",
            "Examples",
        ]
        for i, ex in enumerate(examples, 1):
            lines.append(f"{i}. {ex.a} {ex.op} {ex.b} = {ex.raw_out}")
        lines.append(f"Target: {target_expr}")
        lines.append("")
        return lines

    def _render_search_space(self) -> List[str]:
        family_order = self._family_search_order()
        lines = [
            "Search space",
            "Search policy: target-first ordered family search.",
            "Only the target operator is searched when it has examples; unrelated operators are not solved.",
            "A family block checks related operations together, for example add/add1/addm1 or mul/mul1/mulm1.",
            "Each family block is evaluated once for every operand/result transform pair.",
            "The first family block for the target operator that contains a full match is selected; lower-priority families are not tested.",
            "Inside a passing family block, selection is deterministic: transform order first, then the operation order listed for that family.",
            "We do not search arbitrary output formats. First we normalize operator-prefix/operator-suffix negative outputs for the relevant operator.",
            "Search axes for the target operator:",
            "- operand transform: normal operands or reversed digits of both operands",
            "- result transform: normal result digits or reversed result digits",
            "- family: related operations tested together",
            "Transform order:",
        ]
        for rev_ops, rev_res in self.transform_order:
            lines.append(f"- rev_ops={rev_ops}, rev_result={rev_res}")
        lines.append("Family order:")
        for i, (_, label, ops) in enumerate(family_order, 1):
            lines.append(f"{i}. {label}: operation order inside family = " + ", ".join(ops))
        lines.append(f"Maximum family blocks per operator: {len(family_order)}")
        lines.append(f"Equivalent candidate space still covered: 4 transforms * {sum(len(x[2]) for x in family_order)} operations = {4 * sum(len(x[2]) for x in family_order)} transform-operation candidates")
        lines.append("")
        return lines

    def _render_output_normalization_for_ops(
        self,
        examples: List[Example],
        fmt_by_op: Dict[str, str],
        ops: List[str],
    ) -> List[str]:
        lines = ["Normalize outputs for relevant examples"]
        for op in ops:
            if op not in fmt_by_op:
                continue
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
            total_family_blocks = len(self._family_search_order())
            lines.append(f"Operator {self._literal(op)}")
            lines.append("examples: " + "; ".join(f"{ex.a} {op} {ex.b} -> raw {ex.raw_out}, normalized {ex.norm_out}" for ex in group))
            lines.append(f"family blocks reached by ordered search: {len(attempts)} out of {total_family_blocks} possible")
            lines.append("selection policy for this operator: first passing family block wins; inside that block, transform order wins before operation order")
            if selected:
                lines.append(f"selected rule: {self._hypothesis_name(selected)}")
                lines.append(f"search stopped after this family block; lower-priority family blocks not tested: {total_family_blocks - len(attempts)}")
            else:
                lines.append("selected rule: none")

            for attempt in attempts:
                    status = "PASS" if attempt["passes"] else "FAIL"
                    selected_here = attempt.get("selected_hypothesis")
                    mark = " SELECTED" if selected_here and selected_here == selected else ""
                    lines.append(
                        f"Family block {attempt['index']:02d}: {status}{mark}: {attempt['family_label']}"
                    )
                    lines.append("  operation order inside this family: " + ", ".join(attempt.get("op_order", [])))
                    batch_results: Dict[Tuple[bool, bool], FamilyResult] = attempt["batch_family_results"]
                    batch_rows: Dict[Tuple[bool, bool], List[RequiredRow]] = attempt.get("batch_required_rows") or {}
                    # Print the family formula once. Do not repeat the same explanatory
                    # sentence for every transform pair; each transform line then shows
                    # only its rows, calculations, decisions, and passing operations.
                    first_result = batch_results[self.transform_order[0]]
                    if first_result.lines:
                        lines.append("  family formula: " + first_result.lines[0].strip())
                    lines.append("  test this family for every transform pair:")
                    for b_rev_ops, b_rev_res in self.transform_order:
                        b_key = (b_rev_ops, b_rev_res)
                        b_result: FamilyResult = batch_results[b_key]
                        rows = batch_rows.get(b_key, [])
                        row_summaries = []
                        for row_i, row in enumerate(rows, 1):
                            row_summaries.append(
                                f"r{row_i}:A={int(row.transformed_a)},B={int(row.transformed_b)},Y={row.required_raw_result}"
                            )
                        useful_details = [d.strip() for d in b_result.lines[1:]] if len(b_result.lines) > 1 else []
                        passing_ops = [op_name for op_name in attempt.get("op_order", []) if b_result.op_passes.get(op_name, False)]
                        pass_text = "pass=" + (", ".join(passing_ops) if passing_ops else "none")
                        details = " | ".join(useful_details) if useful_details else "no extra calculation details"
                        lines.append(
                            f"    rev_ops={b_rev_ops}, rev_result={b_rev_res}: "
                            + "; ".join(row_summaries)
                            + " | "
                            + details
                            + " | "
                            + pass_text
                        )

                    if selected_here:
                        op_name, rev_ops, rev_res, fmt = selected_here
                        lines.append(
                            f"  family block contains at least one full match; choose first by transform order then operation order: "
                            f"rev_ops={rev_ops}, rev_result={rev_res}, operation={op_name}"
                        )
                    else:
                        lines.append("  family block has no full match for any transform pair")
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
        return step.final_output, lines

    def _render_fallback_no_rule(self, q_a: str, q_op: str, q_b: str) -> Tuple[str, List[str]]:
        step = self._eval_example(q_a, q_b, q_op, "abs_diff", False, False, "num", expected=None)
        lines = [
            "Target",
            f"Target operator {self._literal(q_op)} was present in the examples, but no reduced-family rule matched all of its examples.",
            "Fallback rule: use absolute difference on the original operands.",
            "This fallback is explicit: no operand reversal, no result reversal, no learned operation from another symbol.",
            f"Replay fallback on target {q_a} {q_op} {q_b}:",
        ]
        for detail in step.lines:
            lines.append(f"  {detail}")
        lines.append(f"Computed output: {step.final_output}")
        lines.append(f"Final answer: {step.final_output}")
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
        return step.final_output, lines

    def _operand_step_text(self, a: str, b: str, ta: str, tb: str, rev_ops: bool) -> str:
        if rev_ops:
            return f"operand transform: reverse both operands: {a}->{ta}, {b}->{tb}; A={int(ta)}, B={int(tb)}"
        return f"operand transform: keep operands: {a}->{ta}, {b}->{tb}; A={int(ta)}, B={int(tb)}"

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
            "training_category": "equations_transformation.family_search",
            "metadata": {
                "target_operator_seen_in_examples": not fallback,
                "uses_fallback_inference": fallback,
                "target_operator": target_operator,
                "selected_rule": self._hypothesis_name(selected) if selected else None,
                "search_space": "sign_normalization + inverse_result_transform_required_table + rev_ops/rev_result/operation",
                "search_policy": "target_first_ordered_family_search",
            },
        }

    @staticmethod
    def _failure(message: str) -> Dict[str, Any]:
        lines = ["Failed to solve", message, "Final answer: nan"]
        return {"answer": "nan", "debug": lines, "trace": lines, "solution": "\n".join(lines)}
