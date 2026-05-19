import re
from typing import Dict, Tuple, Any, Optional, List
from collections import defaultdict, Counter


class ASTBruteForceSolver:
    """Solves mathematical puzzles by deducing hidden operations between numbers based on examples.

    The returned debug trace is intentionally written as a clean training solution, not as a raw
    search/debug log. It explains the selected rule, why it was selected, verifies it on examples,
    and then applies it to the target.
    """

    def __init__(self):
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
        s_str = str(s)
        return "-" + s_str[1:][::-1] if s_str.startswith("-") else s_str[::-1]

    def _get_operand_configs(self, sa: str, sb: str) -> Dict[str, Tuple[int, int, str, str]]:
        return {
            "fwd": (int(sa), int(sb), sa, sb),
            "swap_ops": (int(sb), int(sa), sb, sa),
            "rev_digits": (int(self._rev(sa)), int(self._rev(sb)), self._rev(sa), self._rev(sb)),
            "swap_rev": (int(self._rev(sb)), int(self._rev(sa)), self._rev(sb), self._rev(sa)),
        }

    def _get_operations(self, a: int, b: int, sa: str, sb: str) -> Dict[str, int]:
        ops: Dict[str, int] = {}
        ops["add"] = a + b
        ops["sub"] = a - b
        ops["mul"] = a * b
        ops["abs_diff"] = abs(a - b)

        if b != 0:
            ops["div"] = a // b
            ops["mod"] = a % b
        if a != 0:
            ops["rev_div"] = b // a
            ops["rev_mod"] = b % a

        ops["rev_sub"] = b - a
        ops["add1"] = a + b + 1
        ops["sub1"] = a - b + 1
        ops["mul1"] = a * b + 1
        ops["addm1"] = a + b - 1
        ops["subm1"] = a - b - 1
        ops["mulm1"] = a * b - 1
        ops["neg_abs_diff"] = -abs(a - b)

        if sa + sb != "" and len(sa + sb) < 15:
            try:
                ops["cat"] = int(sa + sb)
            except ValueError:
                pass
        if sb + sa != "" and len(sb + sa) < 15:
            try:
                ops["rev_cat"] = int(sb + sa)
            except ValueError:
                pass

        ops["dsum_add"] = sum(int(d) for d in str(abs(a))) + sum(int(d) for d in str(abs(b)))
        ops["dsum_mul"] = sum(int(d) for d in str(abs(a))) * sum(int(d) for d in str(abs(b)))

        if a != 0 and b != 0:
            ops["max_mod_min"] = max(a, b) % min(a, b) if min(a, b) != 0 else 0

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
        if config == "swap_ops":
            score += 20
        elif config == "rev_digits":
            score += 30
        elif config == "swap_rev":
            score += 50

        if fmt == "abs":
            score += 5
        elif fmt.startswith("sign_"):
            score += 7
        elif fmt.startswith("raw_") or fmt.startswith("abs_"):
            score += 9
        elif fmt in ["zpad2", "zpad3"]:
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
        if num_examples == 1 and op_name in ["mod", "div", "rev_mod", "rev_div"]:
            score += 100
        if global_config and global_fmt and config == global_config and fmt == global_fmt:
            score -= 1000
        return score

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
        return f"{op_name}({a}, {b}) = {value}"

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

    def _apply_hypothesis(self, left: str, right: str, op_char: str, hyp: Tuple[str, str, str]) -> Tuple[int, int, str, str, int, str]:
        op_config, op_name, out_fmt = hyp
        a, b, sa, sb = self._get_operand_configs(left, right)[op_config]
        ops = self._get_operations(a, b, sa, sb)
        value = ops[op_name]
        formatted = self._get_formats(value, op_char)[out_fmt]
        return a, b, sa, sb, value, formatted

    def _rule_line(self, hyp: Tuple[str, str, str]) -> str:
        op_config, op_name, out_fmt = hyp
        return (
            f"config={op_config} ({self.config_desc.get(op_config, op_config)}), "
            f"operation={op_name} ({self.op_desc.get(op_name, op_name)}), "
            f"format={self._format_label(out_fmt)} ({self.fmt_desc.get(out_fmt, out_fmt)})"
        )

    def _selection_reason(self, op: str, hyps: List[Tuple[str, str, str]], selected: Tuple[str, str, str], global_config: str, global_fmt: str) -> str:
        if len(hyps) == 1:
            return "This is the only compact rule that matches every example for this operator."

        selected_config, _, selected_fmt = selected
        reason = (
            "Several compact rules match the examples. This rule is preferred because it uses a simple operation, "
            "verifies every example, and avoids rare transformations unless needed"
        )

        shared_parts: List[str] = []
        if selected_config == global_config:
            shared_parts.append(f"operand style config={global_config}")
        if selected_fmt == global_fmt:
            shared_parts.append(f"output style format={self._format_label(global_fmt)}")

        if shared_parts:
            return reason + "; it also follows the shared " + " and ".join(shared_parts) + "."

        return reason + "; the selected operator-specific format is kept because the shared output style does not match this operator."

    def _parse_examples(self, examples_text: str) -> List[Dict[str, str]]:
        parsed: List[Dict[str, str]] = []
        for line in examples_text.split("\n"):
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                parsed.append(
                    {
                        "a": m.group(1),
                        "op": m.group(2).strip(),
                        "b": m.group(3),
                        "raw_out": rhs.replace(" ", "").strip(),
                    }
                )
        return parsed

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        log: List[str] = []
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm:
            return {"answer": None, "debug": ["Parse error: target expression must look like '<number><operator><number>'."]}

        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)
        all_parsed = self._parse_examples(examples_text)
        if not all_parsed:
            return {"answer": None, "debug": ["No valid examples found."]}

        ops_grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for ex in all_parsed:
            ops_grouped[ex["op"]].append(ex)

        configs_order = ["fwd", "rev_digits", "swap_ops", "swap_rev"]
        full_ops_keys = list(self._get_operations(12, 34, "12", "34").keys())

        op_hypotheses: Dict[str, List[Tuple[str, str, str]]] = {}
        for op, group in ops_grouped.items():
            valid_hyps: List[Tuple[str, str, str]] = []
            fmt_names = list(self._get_formats(1, op).keys())

            for op_config in configs_order:
                for op_name in full_ops_keys:
                    for out_fmt in fmt_names:
                        if out_fmt in ["first_digit", "last_digit"] and len(group) < 3:
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
                            valid_hyps.append((op_config, op_name, out_fmt))
            op_hypotheses[op] = valid_hyps

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
            best_base_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
            config_counts[best_base_hyp[0]] += 1
            fmt_counts[best_base_hyp[2]] += 1

        global_config = anomaly_config or (config_counts.most_common(1)[0][0] if config_counts else "fwd")
        global_fmt = anomaly_fmt or (fmt_counts.most_common(1)[0][0] if fmt_counts else "raw")

        resolved_ops: Dict[str, Tuple[str, str, str]] = {}
        used_base_ops = set()
        for op, hyps in op_hypotheses.items():
            if not hyps:
                continue
            best_hyp = min(
                hyps,
                key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op]), global_config, global_fmt),
            )
            resolved_ops[op] = best_hyp
            used_base_ops.add(self._base_operation_name(best_hyp[1]))

        log.append("Task type: equations_transformation")
        log.append(f"Parsed examples: {', '.join(f'{ex['a']}{ex['op']}{ex['b']}={ex['raw_out']}' for ex in all_parsed)}")
        log.append(f"Target expression: {q_a}{q_op}{q_b}")
        log.append("")
        log.append("Rule search")

        for op in sorted(ops_grouped.keys()):
            group = ops_grouped[op]
            hyps = op_hypotheses.get(op, [])
            log.append(f"Operator {self._literal(op)}")
            log.append("Examples: " + ", ".join(f"{ex['a']} {op} {ex['b']} = {ex['raw_out']}" for ex in group))

            if not hyps:
                log.append("No compact rule from the supported rule set matches all examples for this operator.")
                log.append("")
                continue

            selected = resolved_ops[op]
            log.append(f"Selected rule: {self._rule_line(selected)}")
            log.append("Why this rule: " + self._selection_reason(op, hyps, selected, global_config, global_fmt))
            log.append("Verification:")
            for ex in group:
                a, b, sa, sb, value, formatted = self._apply_hypothesis(ex["a"], ex["b"], op, selected)
                op_text = self._operation_text(selected[1], a, b, sa, sb, value)
                status = "OK" if formatted == ex["raw_out"] else "MISMATCH"
                log.append(
                    f"  {ex['a']} {op} {ex['b']} -> A={a}, B={b}; "
                    f"{op_text}; format -> {formatted} [{status}]"
                )
            log.append("")

        log.append("Target calculation")
        log.append(f"Expression: {q_a} {q_op} {q_b}")

        if q_op in resolved_ops:
            selected = resolved_ops[q_op]
            a, b, sa, sb, value, final_ans = self._apply_hypothesis(q_a, q_b, q_op, selected)
            log.append(f"1. Use rule for operator {self._literal(q_op)}: {self._rule_line(selected)}")
            log.append(f"2. Apply config: A={a}, B={b}")
            log.append(f"3. Apply operation: {self._operation_text(selected[1], a, b, sa, sb, value)}")
            log.append(f"4. Apply format: {self.fmt_desc.get(selected[2], selected[2])} -> {final_ans}")
            log.append(f"Computed output: {final_ans}")
            return {
                "answer": final_ans,
                "debug": log,
                "rule_source": "direct_operator_rule",
                "training_category": "equations_transformation.direct_operator",
                "metadata": {
                    "target_operator_seen_in_examples": True,
                    "uses_fallback_inference": False,
                    "target_operator": q_op,
                },
            }

        log.append(f"Operator {self._literal(q_op)} was not shown in the examples.")
        strict_base_pool = ["add", "sub", "cat", "mul", "div"]
        avail_ops = [name for name in strict_base_pool if name not in used_base_ops]
        best_op = avail_ops[0] if avail_ops else strict_base_pool[0]
        fallback_hyp = (global_config, best_op, global_fmt)
        log.append("Rule source: fallback_inference.")
        log.append("Training category: equations_transformation.fallback_operator_absent.")
        log.append(
            f"Best-effort fallback: use the unused core operation {self._literal(best_op)} "
            f"with shared operand style config={global_config} and shared output style format={self._format_label(global_fmt)}."
        )
        log.append("Confidence note: this is weaker than a directly verified operator rule because the target operator was absent from the examples.")

        try:
            a, b, sa, sb, value, final_ans = self._apply_hypothesis(q_a, q_b, q_op, fallback_hyp)
            log.append(f"1. Apply config: A={a}, B={b}")
            log.append(f"2. Apply operation: {self._operation_text(best_op, a, b, sa, sb, value)}")
            log.append(f"3. Apply format: {self.fmt_desc.get(global_fmt, global_fmt)} -> {final_ans}")
            log.append(f"Computed output: {final_ans}")
            return {
                "answer": final_ans,
                "debug": log,
                "rule_source": "fallback_inference",
                "training_category": "equations_transformation.fallback_operator_absent",
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
            log.append(f"Fallback failed: {type(exc).__name__}: {exc}")

        if q_op in ["+", "-", "*", "/", "**", "%"]:
            try:
                res = str(int(eval(f"{int(q_a)}{q_op}{int(q_b)}")))
                log.append(f"As a final fallback, standard arithmetic gives {res}.")
                log.append(f"Computed output: {res}")
                return {
                    "answer": res,
                    "debug": log,
                    "rule_source": "standard_arithmetic_fallback",
                    "training_category": "equations_transformation.standard_arithmetic_fallback",
                    "metadata": {
                        "target_operator_seen_in_examples": False,
                        "uses_fallback_inference": True,
                        "target_operator": q_op,
                    },
                }
            except Exception:
                log.append("Standard arithmetic fallback failed.")

        log.append("No valid answer could be computed.")
        return {"answer": None, "debug": log}
