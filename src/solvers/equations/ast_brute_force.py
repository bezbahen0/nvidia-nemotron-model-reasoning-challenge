from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


Hypothesis = Tuple[str, str, str]


@dataclass
class SolverConfig:
    max_candidates_shown: int = 10
    max_verification_examples_per_operator: int = 12
    include_candidate_counts: bool = True
    include_global_style: bool = True


class ASTBruteForceSolver:
    """
    Search is still brute-force over:
      operand_config × operation × output_format

    CoT format:
      Examples
      Shared style
      Rule matching
      Verify selected rules
      Target
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")

        self.config_desc = {
            "fwd": "read operands normally",
            "swap_ops": "swap the two operands",
            "rev_digits": "reverse the digits of both operands",
            "swap_rev": "reverse the digits of both operands and swap them",
        }

        self.op_desc = {
            "add": "addition",
            "sub": "subtraction",
            "mul": "multiplication",
            "abs_diff": "absolute difference",
            "div": "integer division",
            "mod": "modulo",
            "rev_div": "reverse integer division",
            "rev_mod": "reverse modulo",
            "rev_sub": "reverse subtraction",
            "add1": "addition plus one",
            "sub1": "subtraction plus one",
            "mul1": "multiplication plus one",
            "addm1": "addition minus one",
            "subm1": "subtraction minus one",
            "mulm1": "multiplication minus one",
            "neg_abs_diff": "negative absolute difference",
            "cat": "concatenation",
            "rev_cat": "reverse concatenation",
            "dsum_add": "sum of digit sums",
            "dsum_mul": "product of digit sums",
            "max_mod_min": "larger operand modulo smaller operand",
            "cross_sum": "cross digit sum",
            "cross_diff_abs": "absolute difference of digit sums",
            "cross_concat": "cross digit concatenation",
            "cross_rev_concat": "reverse cross digit concatenation",
        }

        self.fmt_desc = {
            "raw": "write the result directly",
            "abs": "write the absolute result",
            "zpad2": "write the result padded to 2 digits",
            "zpad3": "write the result padded to 3 digits",
            "rev": "reverse the result digits, preserving a leading minus sign if present",
            "abs_rev": "reverse the absolute result digits",
            "first_digit": "keep only the first result digit",
            "last_digit": "keep only the last result digit",
            "sign_pref_raw": "write the absolute result; if negative, put the operator symbol before it",
            "sign_suff_raw": "write the absolute result; if negative, put the operator symbol after it",
            "sign_pref_rev": "reverse the absolute result digits; if negative, put the operator symbol before them",
            "sign_suff_rev": "reverse the absolute result digits; if negative, put the operator symbol after them",
            "raw_pref": "prefix the operator to the raw result",
            "raw_suff": "suffix the operator to the raw result",
            "abs_pref": "prefix the operator to the absolute result",
            "abs_suff": "suffix the operator to the absolute result",
        }

    def generate_cot(self, prompt: Any) -> str:
        examples_text, target_text = self._split_prompt(prompt)
        result = self.solve(examples_text, target_text)
        return result.get("solution") or "\n".join(result.get("debug", []))

    @staticmethod
    def extract_answer(cot_text: Any) -> str:
        text = "" if cot_text is None else str(cot_text)
        patterns = [
            r"\\boxed\{([^{}\s]+)\}",
            r"(?im)^\s*Final answer\s*:\s*([^\s]+)\s*$",
            r"(?im)^\s*Computed output\s*:\s*([^\s]+)\s*$",
            r"(?im)^\s*Answer\s*:\s*([^\s]+)\s*$",
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
        parsed = self._parse_examples(examples_text)
        if not parsed:
            return self._failure("no valid examples found")

        ops_grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for ex in parsed:
            ops_grouped[ex["op"]].append(ex)

        op_hypotheses = self._find_operator_hypotheses(ops_grouped)
        global_config, global_fmt = self._infer_shared_style(op_hypotheses, ops_grouped)
        resolved_ops = self._select_operator_rules(op_hypotheses, ops_grouped, global_config, global_fmt)

        used_base_ops = {self._base_operation_name(h[1]) for h in resolved_ops.values()}

        lines: List[str] = []
        lines.extend(self._render_header(parsed, f"{q_a}{q_op}{q_b}"))

        if self.config.include_global_style:
            lines.append("Shared style")
            lines.append(f"operand style: {global_config} ({self.config_desc.get(global_config, global_config)})")
            lines.append(f"output format: {self._format_label(global_fmt)} ({self.fmt_desc.get(global_fmt, global_fmt)})")
            lines.append("")

        lines.extend(self._render_rule_matching(ops_grouped, op_hypotheses, resolved_ops, global_config, global_fmt))
        lines.extend(self._render_verification(ops_grouped, resolved_ops))

        if q_op in resolved_ops:
            selected = resolved_ops[q_op]
            try:
                a, b, sa, sb, value, final_ans = self._apply_hypothesis(q_a, q_b, q_op, selected)
            except Exception as exc:
                return self._failure(f"target calculation failed: {type(exc).__name__}: {exc}")

            lines.extend(self._render_target_direct(q_op, selected, a, b, sa, sb, value, final_ans))
            return {
                "answer": final_ans,
                "debug": lines,
                "trace": lines,
                "solution": "\n".join(lines),
                "rule_source": "direct_operator_rule",
                "training_category": "equations_transformation.tong_style_direct_operator",
                "metadata": {
                    "target_operator_seen_in_examples": True,
                    "uses_fallback_inference": False,
                    "target_operator": q_op,
                    "selected_rule": self._hypothesis_name(selected),
                },
            }

        fallback_result = self._fallback(q_a, q_op, q_b, global_config, global_fmt, used_base_ops)
        lines.extend(fallback_result["lines"])
        return {
            "answer": fallback_result["answer"],
            "debug": lines,
            "trace": lines,
            "solution": "\n".join(lines),
            "rule_source": fallback_result["rule_source"],
            "training_category": fallback_result["training_category"],
            "metadata": fallback_result["metadata"],
        }

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
                candidate = m.group(1).strip().rstrip(".")
                compact = candidate.replace(" ", "")
                if self._numeric_re.fullmatch(compact):
                    target = compact
                    break

        if not target:
            for line in reversed(text.splitlines()):
                candidate = line.strip().rstrip(".").replace(" ", "")
                if "=" not in candidate and self._numeric_re.fullmatch(candidate):
                    target = candidate
                    break

        return "\n".join(example_lines), target

    def _parse_examples(self, examples_text: str) -> List[Dict[str, str]]:
        parsed: List[Dict[str, str]] = []
        for line in str(examples_text).splitlines():
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if not m:
                continue
            parsed.append(
                {
                    "a": m.group(1),
                    "op": m.group(2).strip(),
                    "b": m.group(3),
                    "raw_out": rhs.replace(" ", "").strip(),
                }
            )
        return parsed

    @staticmethod
    def _literal(text: str) -> str:
        return repr(text)

    def _format_label(self, fmt: str) -> str:
        labels = {
            "raw": "result_direct",
            "abs": "absolute_result",
            "zpad2": "zero_padded_2_digits",
            "zpad3": "zero_padded_3_digits",
            "rev": "reverse_result_digits",
            "abs_rev": "reverse_absolute_result_digits",
            "first_digit": "first_result_digit",
            "last_digit": "last_result_digit",
            "sign_pref_raw": "result_with_operator_sign_if_negative",
            "sign_suff_raw": "result_with_operator_sign_suffix_if_negative",
            "sign_pref_rev": "reverse_result_with_operator_sign_if_negative",
            "sign_suff_rev": "reverse_result_with_operator_sign_suffix_if_negative",
            "raw_pref": "operator_prefixed_raw_result",
            "raw_suff": "operator_suffixed_raw_result",
            "abs_pref": "operator_prefixed_absolute_result",
            "abs_suff": "operator_suffixed_absolute_result",
        }
        return labels.get(fmt, fmt)

    def _rev(self, s: str) -> str:
        s = str(s)
        return "-" + s[1:][::-1] if s.startswith("-") else s[::-1]

    def _get_operand_configs(self, sa: str, sb: str) -> Dict[str, Tuple[int, int, str, str]]:
        return {
            "fwd": (int(sa), int(sb), sa, sb),
            "swap_ops": (int(sb), int(sa), sb, sa),
            "rev_digits": (int(self._rev(sa)), int(self._rev(sb)), self._rev(sa), self._rev(sb)),
            "swap_rev": (int(self._rev(sb)), int(self._rev(sa)), self._rev(sb), self._rev(sa)),
        }

    def _get_operations(self, a: int, b: int, sa: str, sb: str) -> Dict[str, int]:
        ops: Dict[str, int] = {
            "add": a + b,
            "sub": a - b,
            "mul": a * b,
            "abs_diff": abs(a - b),
            "rev_sub": b - a,
            "add1": a + b + 1,
            "sub1": a - b + 1,
            "mul1": a * b + 1,
            "addm1": a + b - 1,
            "subm1": a - b - 1,
            "mulm1": a * b - 1,
            "neg_abs_diff": -abs(a - b),
        }

        if b != 0:
            ops["div"] = a // b
            ops["mod"] = a % b
        if a != 0:
            ops["rev_div"] = b // a
            ops["rev_mod"] = b % a

        if len(sa + sb) < 15:
            try:
                ops["cat"] = int(sa + sb)
            except ValueError:
                pass

        if len(sb + sa) < 15:
            try:
                ops["rev_cat"] = int(sb + sa)
            except ValueError:
                pass

        ops["dsum_add"] = sum(int(d) for d in str(abs(a))) + sum(int(d) for d in str(abs(b)))
        ops["dsum_mul"] = sum(int(d) for d in str(abs(a))) * sum(int(d) for d in str(abs(b)))

        if a != 0 and b != 0:
            ops["max_mod_min"] = max(a, b) % min(a, b)

        if len(sa) == 2 and len(sb) == 2 and sa.lstrip("-").isdigit() and sb.lstrip("-").isdigit():
            d1, d2 = int(sa[-2]), int(sa[-1])
            d3, d4 = int(sb[-2]), int(sb[-1])
            ops["cross_sum"] = d1 * d3 + d2 * d4
            ops["cross_diff_abs"] = abs((d1 + d2) - (d3 + d4))
            ops["cross_concat"] = int(str(d1 * d3) + str(d2 * d4))
            ops["cross_rev_concat"] = int(str(d1 * d4) + str(d2 * d3))

        return ops

    def _get_formats(self, val: int, op_char: str) -> Dict[str, str]:
        sval = str(val)
        abs_val = abs(val)
        s_abs = str(abs_val)

        formats = {
            "raw": sval,
            "abs": s_abs,
            "zpad2": f"{val:02d}" if val >= 0 else f"-{abs_val:02d}",
            "zpad3": f"{val:03d}" if val >= 0 else f"-{abs_val:03d}",
            "rev": "-" + s_abs[::-1] if val < 0 else s_abs[::-1],
            "abs_rev": s_abs[::-1],
            "first_digit": sval[0] if val >= 0 else "-" + s_abs[0],
            "last_digit": sval[-1] if val >= 0 else "-" + s_abs[-1],
        }

        if val < 0 and op_char:
            formats["sign_pref_raw"] = op_char + s_abs
            formats["sign_suff_raw"] = s_abs + op_char
            formats["sign_pref_rev"] = op_char + s_abs[::-1]
            formats["sign_suff_rev"] = s_abs[::-1] + op_char
        elif val >= 0 and op_char:
            formats["sign_pref_raw"] = s_abs
            formats["sign_suff_raw"] = s_abs
            formats["sign_pref_rev"] = s_abs[::-1]
            formats["sign_suff_rev"] = s_abs[::-1]

        if op_char:
            formats["raw_pref"] = op_char + sval
            formats["raw_suff"] = sval + op_char
            formats["abs_pref"] = op_char + s_abs
            formats["abs_suff"] = s_abs + op_char

        return formats

    def _score_hypothesis(
        self,
        config: str,
        op_name: str,
        fmt: str,
        num_examples: int,
        global_config: Optional[str] = None,
        global_fmt: Optional[str] = None,
    ) -> int:
        score = 0

        score += {"fwd": 0, "swap_ops": 20, "rev_digits": 30, "swap_rev": 50}.get(config, 100)

        if fmt == "abs":
            score += 5
        elif fmt.startswith("sign_"):
            score += 7
        elif fmt.startswith("raw_") or fmt.startswith("abs_"):
            score += 9
        elif fmt in {"zpad2", "zpad3"}:
            score += 15
        elif fmt == "rev":
            score += 25
        elif fmt == "abs_rev":
            score += 27
        elif "digit" in fmt:
            score += 40

        op_penalties = {
            "add": 0,
            "sub": 1,
            "abs_diff": 2,
            "mul": 3,
            "cat": 4,
            "div": 10,
            "mod": 11,
            "rev_sub": 12,
            "rev_cat": 13,
            "add1": 20,
            "sub1": 21,
            "mul1": 22,
            "addm1": 23,
            "subm1": 24,
            "mulm1": 25,
            "neg_abs_diff": 26,
            "max_mod_min": 27,
            "dsum_add": 30,
            "dsum_mul": 31,
            "cross_sum": 40,
            "cross_diff_abs": 41,
            "cross_concat": 42,
            "cross_rev_concat": 43,
        }
        score += op_penalties.get(op_name, 50)

        if num_examples == 1 and op_name in {"mod", "div", "rev_mod", "rev_div"}:
            score += 100

        if global_config and global_fmt and config == global_config and fmt == global_fmt:
            score -= 1000

        return score

    def _find_operator_hypotheses(self, ops_grouped: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Hypothesis]]:
        configs_order = ["fwd", "rev_digits", "swap_ops", "swap_rev"]
        full_ops_keys = list(self._get_operations(12, 34, "12", "34").keys())
        op_hypotheses: Dict[str, List[Hypothesis]] = {}

        for op, group in ops_grouped.items():
            valid: List[Hypothesis] = []
            fmt_names = list(self._get_formats(1, op).keys())

            for op_config in configs_order:
                for op_name in full_ops_keys:
                    for out_fmt in fmt_names:
                        if out_fmt in {"first_digit", "last_digit"} and len(group) < 3:
                            continue

                        all_pass = True
                        for ex in group:
                            cfg = self._get_operand_configs(ex["a"], ex["b"])[op_config]
                            ops = self._get_operations(*cfg)

                            if op_name not in ops:
                                all_pass = False
                                break

                            formats = self._get_formats(ops[op_name], op)
                            if out_fmt not in formats or formats[out_fmt] != ex["raw_out"]:
                                all_pass = False
                                break

                        if all_pass:
                            valid.append((op_config, op_name, out_fmt))

            op_hypotheses[op] = valid

        return op_hypotheses

    def _infer_shared_style(
        self,
        op_hypotheses: Dict[str, List[Hypothesis]],
        ops_grouped: Dict[str, List[Dict[str, str]]],
    ) -> Tuple[str, str]:
        anomaly_config: Optional[str] = None
        anomaly_fmt: Optional[str] = None
        config_penalties = {"fwd": 0, "swap_ops": 20, "rev_digits": 30, "swap_rev": 50}

        for hyps in op_hypotheses.values():
            if not hyps:
                continue
            configs_used = {h[0] for h in hyps}
            if "fwd" not in configs_used:
                anomaly_config = min(configs_used, key=lambda c: config_penalties.get(c, 100))
                break

        for op, hyps in op_hypotheses.items():
            if not hyps:
                continue
            fmts_used = {h[2] for h in hyps}
            if "raw" not in fmts_used:
                best_h = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
                anomaly_fmt = best_h[2]
                break

        config_counts: Counter[str] = Counter()
        fmt_counts: Counter[str] = Counter()

        for op, hyps in op_hypotheses.items():
            if not hyps:
                continue
            best_base = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
            config_counts[best_base[0]] += 1
            fmt_counts[best_base[2]] += 1

        global_config = anomaly_config or (config_counts.most_common(1)[0][0] if config_counts else "fwd")
        global_fmt = anomaly_fmt or (fmt_counts.most_common(1)[0][0] if fmt_counts else "raw")

        return global_config, global_fmt

    def _select_operator_rules(
        self,
        op_hypotheses: Dict[str, List[Hypothesis]],
        ops_grouped: Dict[str, List[Dict[str, str]]],
        global_config: str,
        global_fmt: str,
    ) -> Dict[str, Hypothesis]:
        resolved: Dict[str, Hypothesis] = {}
        for op, hyps in op_hypotheses.items():
            if not hyps:
                continue
            resolved[op] = min(
                hyps,
                key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op]), global_config, global_fmt),
            )
        return resolved

    def _hypothesis_name(self, hyp: Hypothesis) -> str:
        return f"{hyp[0]}/{hyp[1]}/{self._format_label(hyp[2])}"

    def _rule_line(self, hyp: Hypothesis) -> str:
        config, op_name, fmt = hyp
        return (
            f"{config} ({self.config_desc.get(config, config)}); "
            f"{op_name} ({self.op_desc.get(op_name, op_name)}); "
            f"{self._format_label(fmt)} ({self.fmt_desc.get(fmt, fmt)})"
        )

    def _candidate_counts(self, hyps: List[Hypothesis]) -> str:
        cfg = Counter(h[0] for h in hyps)
        op = Counter(h[1] for h in hyps)
        fmt = Counter(h[2] for h in hyps)

        cfg_s = " ".join(f"{k}:{v}" for k, v in cfg.most_common())
        op_s = " ".join(f"{k}:{v}" for k, v in op.most_common(8))
        fmt_s = " ".join(f"{self._format_label(k)}:{v}" for k, v in fmt.most_common(8))

        return f"configs [{cfg_s}], operations [{op_s}], formats [{fmt_s}]"

    def _render_header(self, parsed: List[Dict[str, str]], target_expr: str) -> List[str]:
        lines = [
            "We need to infer the hidden equation transformation by matching examples.",
            "A rule has three parts: operand style, arithmetic operation, and output format.",
            "",
            "Examples",
        ]
        for i, ex in enumerate(parsed, 1):
            lines.append(f"{i}. {ex['a']} {ex['op']} {ex['b']} = {ex['raw_out']}")
        lines.append(f"Target: {target_expr}")
        lines.append("")
        return lines

    def _render_rule_matching(
        self,
        ops_grouped: Dict[str, List[Dict[str, str]]],
        op_hypotheses: Dict[str, List[Hypothesis]],
        resolved_ops: Dict[str, Hypothesis],
        global_config: str,
        global_fmt: str,
    ) -> List[str]:
        lines: List[str] = ["Rule matching"]

        for op in sorted(ops_grouped.keys()):
            group = ops_grouped[op]
            hyps = op_hypotheses.get(op, [])

            lines.append(f"Operator {self._literal(op)}")
            lines.append("examples: " + "; ".join(f"{ex['a']} {op} {ex['b']} -> {ex['raw_out']}" for ex in group))

            if not hyps:
                lines.append("matching candidates: none")
                lines.append("")
                continue

            if self.config.include_candidate_counts:
                lines.append(f"matching candidates: {len(hyps)}; {self._candidate_counts(hyps)}")

            ranked = sorted(
                hyps,
                key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(group), global_config, global_fmt),
            )

            lines.append("best candidates")
            for h in ranked[: self.config.max_candidates_shown]:
                score = self._score_hypothesis(h[0], h[1], h[2], len(group), global_config, global_fmt)
                mark = " <- selected" if h == resolved_ops.get(op) else ""
                lines.append(f"- {self._hypothesis_name(h)} score={score}{mark}")

            selected = resolved_ops[op]
            lines.append(f"Best: {self._hypothesis_name(selected)}")
            lines.append(f"meaning: {self._rule_line(selected)}")
            lines.append("")

        return lines

    def _render_verification(
        self,
        ops_grouped: Dict[str, List[Dict[str, str]]],
        resolved_ops: Dict[str, Hypothesis],
    ) -> List[str]:
        lines: List[str] = ["Verify selected rules"]

        for op in sorted(ops_grouped.keys()):
            if op not in resolved_ops:
                continue

            selected = resolved_ops[op]
            group = ops_grouped[op]
            lines.append(f"Operator {self._literal(op)} uses {self._hypothesis_name(selected)}")

            for ex in group[: self.config.max_verification_examples_per_operator]:
                a, b, sa, sb, value, formatted = self._apply_hypothesis(ex["a"], ex["b"], op, selected)
                op_text = self._operation_text(selected[1], a, b, sa, sb, value)
                status = "ok" if formatted == ex["raw_out"] else "fail"
                lines.append(
                    f"{ex['a']} {op} {ex['b']}: A={a}, B={b}; "
                    f"{op_text}; format -> {formatted}; expected={ex['raw_out']}; {status}"
                )

            if len(group) > self.config.max_verification_examples_per_operator:
                lines.append(f"... {len(group) - self.config.max_verification_examples_per_operator} more examples verified.")

        lines.append("")
        return lines

    def _render_target_direct(
        self,
        q_op: str,
        selected: Hypothesis,
        a: int,
        b: int,
        sa: str,
        sb: str,
        value: int,
        final_ans: str,
    ) -> List[str]:
        return [
            "Target",
            f"Use operator {self._literal(q_op)} rule: {self._hypothesis_name(selected)}",
            f"Decode operands: A={a}, B={b}",
            f"Apply operation: {self._operation_text(selected[1], a, b, sa, sb, value)}",
            f"Apply format: {self.fmt_desc.get(selected[2], selected[2])} -> {final_ans}",
            f"Computed output: {final_ans}",
            f"Final answer: {final_ans}",
            f"\\boxed{{{final_ans}}}",
        ]

    def _operation_text(self, op_name: str, a: int, b: int, sa: str, sb: str, value: int) -> str:
        if op_name == "add":
            return f"{a} + {b} = {value}"
        if op_name == "sub":
            return f"{a} - {b} = {value}"
        if op_name == "mul":
            return f"{a} * {b} = {value}"
        if op_name == "abs_diff":
            return f"abs({a} - {b}) = {value}"
        if op_name == "div":
            return f"{a} // {b} = {value}"
        if op_name == "mod":
            return f"{a} % {b} = {value}"
        if op_name == "rev_div":
            return f"{b} // {a} = {value}"
        if op_name == "rev_mod":
            return f"{b} % {a} = {value}"
        if op_name == "rev_sub":
            return f"{b} - {a} = {value}"
        if op_name == "add1":
            return f"{a} + {b} + 1 = {value}"
        if op_name == "sub1":
            return f"{a} - {b} + 1 = {value}"
        if op_name == "mul1":
            return f"{a} * {b} + 1 = {value}"
        if op_name == "addm1":
            return f"{a} + {b} - 1 = {value}"
        if op_name == "subm1":
            return f"{a} - {b} - 1 = {value}"
        if op_name == "mulm1":
            return f"{a} * {b} - 1 = {value}"
        if op_name == "neg_abs_diff":
            return f"-abs({a} - {b}) = {value}"
        if op_name == "cat":
            return f"concat({sa}, {sb}) = {value}"
        if op_name == "rev_cat":
            return f"concat({sb}, {sa}) = {value}"
        if op_name == "dsum_add":
            return f"digit_sum({a}) + digit_sum({b}) = {value}"
        if op_name == "dsum_mul":
            return f"digit_sum({a}) * digit_sum({b}) = {value}"
        if op_name == "max_mod_min":
            return f"max({a}, {b}) % min({a}, {b}) = {value}"
        if op_name == "cross_sum":
            return f"cross_sum({sa}, {sb}) = {value}"
        if op_name == "cross_diff_abs":
            return f"cross_diff_abs({sa}, {sb}) = {value}"
        if op_name == "cross_concat":
            return f"cross_concat({sa}, {sb}) = {value}"
        if op_name == "cross_rev_concat":
            return f"cross_rev_concat({sa}, {sb}) = {value}"
        return f"{op_name}({a}, {b}) = {value}"

    def _apply_hypothesis(self, left: str, right: str, op_char: str, hyp: Hypothesis) -> Tuple[int, int, str, str, int, str]:
        op_config, op_name, out_fmt = hyp
        a, b, sa, sb = self._get_operand_configs(left, right)[op_config]
        ops = self._get_operations(a, b, sa, sb)
        value = ops[op_name]
        formatted = self._get_formats(value, op_char)[out_fmt]
        return a, b, sa, sb, value, formatted

    def _base_operation_name(self, op_name: str) -> str:
        if op_name in {"add", "add1", "addm1", "dsum_add"}:
            return "add"
        if op_name in {"sub", "sub1", "subm1", "abs_diff", "neg_abs_diff", "rev_sub"}:
            return "sub"
        if op_name in {"mul", "mul1", "mulm1", "dsum_mul"}:
            return "mul"
        if op_name in {"cat", "rev_cat"}:
            return "cat"
        if op_name in {"div", "rev_div"}:
            return "div"
        if op_name in {"mod", "rev_mod", "max_mod_min"}:
            return "mod"
        return op_name

    def _fallback(
        self,
        q_a: str,
        q_op: str,
        q_b: str,
        global_config: str,
        global_fmt: str,
        used_base_ops: set[str],
    ) -> Dict[str, Any]:
        lines: List[str] = ["Target"]
        lines.append(f"Operator {self._literal(q_op)} does not have a directly verified rule.")

        strict_base_pool = ["add", "sub", "cat", "mul", "div"]
        avail_ops = [name for name in strict_base_pool if name not in used_base_ops]
        best_op = avail_ops[0] if avail_ops else strict_base_pool[0]
        fallback_hyp: Hypothesis = (global_config, best_op, global_fmt)

        lines.append("Fallback inference")
        lines.append(
            f"use unused core operation {best_op} with shared operand style {global_config} "
            f"and format {self._format_label(global_fmt)}"
        )
        lines.append("confidence: weaker than direct rule matching because the target operator was absent or unresolved")

        try:
            a, b, sa, sb, value, final_ans = self._apply_hypothesis(q_a, q_b, q_op, fallback_hyp)
            lines.append(f"Decode operands: A={a}, B={b}")
            lines.append(f"Apply operation: {self._operation_text(best_op, a, b, sa, sb, value)}")
            lines.append(f"Apply format: {self.fmt_desc.get(global_fmt, global_fmt)} -> {final_ans}")
            lines.append(f"Computed output: {final_ans}")
            lines.append(f"Final answer: {final_ans}")
            lines.append(f"\\boxed{{{final_ans}}}")
            return {
                "answer": final_ans,
                "lines": lines,
                "rule_source": "fallback_inference",
                "training_category": "equations_transformation.tong_style_fallback_operator_absent",
                "metadata": {
                    "target_operator_seen_in_examples": False,
                    "uses_fallback_inference": True,
                    "target_operator": q_op,
                    "fallback_operation": best_op,
                    "fallback_config": global_config,
                    "fallback_format": self._format_label(global_fmt),
                },
            }
        except Exception as exc:
            lines.append(f"Fallback failed: {type(exc).__name__}: {exc}")

        std = self._standard_arithmetic(q_a, q_op, q_b)
        if std is not None:
            lines.append(f"Standard arithmetic fallback gives {std}.")
            lines.append(f"Computed output: {std}")
            lines.append(f"Final answer: {std}")
            lines.append(f"\\boxed{{{std}}}")
            return {
                "answer": std,
                "lines": lines,
                "rule_source": "standard_arithmetic_fallback",
                "training_category": "equations_transformation.tong_style_standard_arithmetic_fallback",
                "metadata": {
                    "target_operator_seen_in_examples": False,
                    "uses_fallback_inference": True,
                    "target_operator": q_op,
                },
            }

        lines.append("No valid target output could be computed.")
        lines.append("Final answer: nan")
        return {
            "answer": None,
            "lines": lines,
            "rule_source": "failed",
            "training_category": "equations_transformation.failed",
            "metadata": {
                "target_operator_seen_in_examples": False,
                "uses_fallback_inference": True,
                "target_operator": q_op,
            },
        }

    @staticmethod
    def _standard_arithmetic(q_a: str, q_op: str, q_b: str) -> Optional[str]:
        try:
            a, b = int(q_a), int(q_b)
            if q_op == "+":
                return str(a + b)
            if q_op == "-":
                return str(a - b)
            if q_op == "*":
                return str(a * b)
            if q_op == "/" and b != 0:
                return str(a // b)
            if q_op == "%" and b != 0:
                return str(a % b)
            if q_op == "**":
                return str(a**b)
        except Exception:
            return None
        return None

    @staticmethod
    def _failure(reason: str) -> Dict[str, Any]:
        lines = [
            "We need to infer the hidden equation transformation by matching examples.",
            f"Solver failed: {reason}",
            "Final answer: nan",
        ]
        return {
            "answer": None,
            "debug": lines,
            "trace": lines,
            "solution": "\n".join(lines),
            "rule_source": "failed",
            "training_category": "equations_transformation.failed",
            "metadata": {},
        }