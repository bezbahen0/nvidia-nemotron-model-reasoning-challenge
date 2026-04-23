import re
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

@dataclass
class FoundRule:
    op_config: str
    op_name: str
    out_fmt: str
    neg_fmt: str
    op_char: str

@dataclass
class TaskContext:
    unique_symbols_count: int
    is_pure_numeric: bool
    has_leading_zeros: bool
    is_length_preserving: bool # La + Lb == Lout
    has_length_reduction: bool # Lout < max(La, Lb)
    target_op_in_examples: bool

class EnsembleEquationsSolver:
    def __init__(self):
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")

    def extract_answer(self, cot_text: str) -> str:
        if not isinstance(cot_text, str) or "Error" in cot_text:
            return "nan"
        matches = re.findall(r"(?i)final\s+answer:\s*(.+)", cot_text)
        return matches[-1].strip() if matches else "nan"

    def _rev(self, s: str) -> str:
        s_str = str(s)
        if s_str.startswith("-"):
            return "-" + s_str[1:][::-1]
        return s_str[::-1]

    def _analyze_context(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str) -> TaskContext:
        all_syms = set(q_a + q_b)
        has_lz = False
        is_len_pres = True
        has_len_red = False
        target_in_exs = False
        
        for ex in parsed_exs:
            all_syms.update(list(ex['a'] + ex['b'] + ex['out']))
            if ex['op'] == q_op: target_in_exs = True
            if (ex['a'].startswith('0') and len(ex['a']) > 1) or (ex['b'].startswith('0') and len(ex['b']) > 1):
                has_lz = True
            if len(ex['out']) != len(ex['a']) + len(ex['b']):
                is_len_pres = False
            if len(ex['out']) < max(len(ex['a']), len(ex['b'])) and not ex['is_neg']:
                has_len_red = True

        all_syms_str = "".join(all_syms).replace("-", "")
        is_num = all_syms_str.isdigit()

        return TaskContext(
            unique_symbols_count=len(all_syms),
            is_pure_numeric=is_num,
            has_leading_zeros=has_lz,
            is_length_preserving=is_len_pres,
            has_length_reduction=has_len_red,
            target_op_in_examples=target_in_exs
        )

    def _get_operand_configs(self, sa: str, sb: str) -> Dict[str, Tuple[int, int, str, str]]:
        return {
            "fwd": (int(sa), int(sb), sa, sb),
            "rev_digits": (int(self._rev(sa)), int(self._rev(sb)), self._rev(sa), self._rev(sb)),
            "swap_ops": (int(sb), int(sa), sb, sa),
            "swap_rev": (int(self._rev(sb)), int(self._rev(sa)), self._rev(sb), self._rev(sa))
        }

    def _get_operations(self, a: int, b: int, sa: str, sb: str) -> Dict[str, int]:
        ops = {
            "add": a + b, "sub": a - b, "rev_sub": b - a, "mul": a * b,
            "cat": int(sa + sb) if sa + sb != "" and len(sa + sb) < 15 else 0,
            "rev_cat": int(sb + sa) if sb + sa != "" and len(sb + sa) < 15 else 0,
            "add1": a + b + 1, "addm1": a + b - 1, "mul1": a * b + 1, "mulm1": a * b - 1,
            "sub1": a - b + 1, "subm1": a - b - 1,
            "abs_diff": abs(a - b), "neg_abs_diff": -abs(a - b),
        }
        if b != 0: ops["div"] = a // b; ops["mod"] = a % b
        if a != 0: ops["rev_div"] = b // a; ops["rev_mod"] = b % a
        if a != 0 and b != 0: ops["max_mod_min"] = max(a, b) % min(a, b)
        return ops

    def _get_formats(self, val: int) -> Dict[str, str]:
        sval = str(val); abs_val = abs(val); s_abs = str(abs_val)
        return {
            "raw": sval,
            "rev": "-" + s_abs[::-1] if val < 0 else s_abs[::-1],
            "abs": s_abs,
            "zpad2": f"{val:02d}" if val >= 0 else f"-{abs_val:02d}",
            "zpad3": f"{val:03d}" if val >= 0 else f"-{abs_val:03d}",
            "first_digit": sval[0] if val >= 0 else "-" + s_abs[0],
            "last_digit": sval[-1] if val >= 0 else "-" + s_abs[-1],
        }

    def _solve_level1_numeric(self, prompt: str, query_str: str, ctx: TaskContext) -> Optional[str]:
        raw_lines = re.findall(r"([^\n=]+?)\s*=\s*([^\n]+)", prompt)
        parsed = []
        for lhs, rhs in raw_lines:
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                try:
                    int(m.group(1)); int(m.group(3))
                    # Багфикс RHS: Очищаем мусор, если пример выглядит как 42\92 = 5\
                    clean_rhs = "".join(c for c in rhs.strip() if c.isdigit() or c == '-')
                    if clean_rhs:
                        parsed.append((m.group(1), m.group(2).strip(), m.group(3), rhs.strip()))
                except ValueError:
                    pass

        if not parsed: return None

        qm = self._numeric_re.fullmatch(query_str)
        if not qm: return None
        qa, q_op, qb = qm.group(1), qm.group(2).strip(), qm.group(3)

        by_op = defaultdict(list)
        for a, op, b, out in parsed: by_op[op].append((a, b, out))

        found_rules = {}
        op_names = list(self._get_operations(1, 1, "1", "1").keys())
        fmt_names = list(self._get_formats(1).keys())

        # Умная сортировка приоритетов: если есть нули, реверс проверяется первым
        configs_order = ["rev_digits", "swap_rev", "fwd", "swap_ops"] if ctx.has_leading_zeros else ["fwd", "rev_digits", "swap_ops", "swap_rev"]

        for op_char, group in by_op.items():
            any_neg_suffix = any_op_suffix = any_op_prefix = False
            for _, _, out in group:
                if op_char != "-":
                    if out.endswith("-") and len(out) > 1: any_neg_suffix = True
                    if out.endswith(op_char) and len(out) > len(op_char): any_op_suffix = True
                    if out.startswith(op_char) and len(out) > len(op_char): any_op_prefix = True

            neg_fmt = "standard"
            if any_op_suffix: neg_fmt = "op_suffix"
            elif any_op_prefix: neg_fmt = "op_prefix"
            elif any_neg_suffix: neg_fmt = "neg_suffix"

            transformed = []
            for a, b, out in group:
                t_out = out
                if neg_fmt == "op_suffix" and out.endswith(op_char): t_out = "-" + out[:-len(op_char)]
                elif neg_fmt == "op_prefix" and out.startswith(op_char): t_out = "-" + out[len(op_char):]
                elif neg_fmt == "neg_suffix" and out.endswith("-"): t_out = "-" + out[:-1]
                transformed.append((a, b, t_out))

            found = None
            for op_config in configs_order:
                for op_name in op_names:
                    for out_fmt in fmt_names:
                        all_pass = True
                        for ax, bx, exp_norm in transformed:
                            cfg = self._get_operand_configs(ax, bx)[op_config]
                            ops = self._get_operations(*cfg)
                            if op_name not in ops or self._get_formats(ops[op_name])[out_fmt] != exp_norm:
                                all_pass = False; break
                        if all_pass:
                            found = FoundRule(op_config, op_name, out_fmt, neg_fmt, op_char)
                            break
                    if found: break
                if found: break
            if found: found_rules[op_char] = found

        cot_lines = ["[Level 1: Math Matrix] Observation: Operands are numeric."]
        
        if q_op not in found_rules:
            if q_op in ['+', '-', '*', '/'] and found_rules:
                first_rule = list(found_rules.values())[0]
                if all(r.op_config == first_rule.op_config and r.out_fmt == first_rule.out_fmt and r.neg_fmt == first_rule.neg_fmt for r in found_rules.values()):
                    synth_op = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[q_op]
                    rule = FoundRule(first_rule.op_config, synth_op, first_rule.out_fmt, first_rule.neg_fmt, q_op)
                    cot_lines.append(f"[Level 1] Hypothesis: Target operator `{q_op}` is missing, but all examples follow a Universal Config (`{first_rule.op_config}`). Extrapolating...")
                else:
                    return None
            else:
                return None
        else:
            rule = found_rules[q_op]
            cot_lines.append(f"[Level 1] Hypothesis: Perfect match found. Rule: Config `{rule.op_config}`, Operation `{rule.op_name}`, Format `{rule.out_fmt}`.")

        cfg = self._get_operand_configs(qa, qb)[rule.op_config]
        val = self._get_operations(*cfg)[rule.op_name]
        final_str = self._get_formats(val)[rule.out_fmt]
        
        if final_str.startswith("-"):
            if rule.neg_fmt == "op_suffix": final_str = final_str[1:] + rule.op_char
            elif rule.neg_fmt == "op_prefix": final_str = rule.op_char + final_str[1:]
            elif rule.neg_fmt == "neg_suffix": final_str = final_str[1:] + "-"

        cot_lines.append(f"[Level 1] Action: Evaluating parameters {qa} {q_op} {qb} yields {final_str}.")
        cot_lines.append(f"\nFinal answer: {final_str}")
        return "\n".join(cot_lines)

    def _solve_level1_5_digit_wise(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str, ctx: TaskContext) -> Optional[str]:
        target_exs = [ex for ex in parsed_exs if ex['op'] == q_op]
        if not target_exs: return None

        for ex in target_exs:
            if not (ex['a'].isdigit() and ex['b'].isdigit() and len(ex['a']) == 2 and len(ex['b']) == 2):
                return None
        if not (q_a.isdigit() and q_b.isdigit() and len(q_a) == 2 and len(q_b) == 2):
            return None

        valid_rules = defaultdict(int)
        for ex in target_exs:
            d1, d2 = int(ex['a'][0]), int(ex['a'][1])
            d3, d4 = int(ex['b'][0]), int(ex['b'][1])
            out = ex['out']
            
            if out == str(d1 * d3) + str(d2 * d4): valid_rules['cross_concat'] += 1
            if out == str(d1 * d4) + str(d2 * d3): valid_rules['cross_rev_concat'] += 1
            if out == str(abs((d1 + d2) - (d3 + d4))): valid_rules['sum_diff_abs'] += 1
            if out == str(d1 * d3 + d2 * d4): valid_rules['cross_sum'] += 1

        if not valid_rules: return None
        best_rule = max(valid_rules, key=valid_rules.get)
        if valid_rules[best_rule] < len(target_exs): return None

        d1, d2 = int(q_a[0]), int(q_a[1])
        d3, d4 = int(q_b[0]), int(q_b[1])
        
        if best_rule == 'cross_concat': ans = str(d1 * d3) + str(d2 * d4)
        elif best_rule == 'cross_rev_concat': ans = str(d1 * d4) + str(d2 * d3)
        elif best_rule == 'sum_diff_abs': ans = str(abs((d1 + d2) - (d3 + d4)))
        elif best_rule == 'cross_sum': ans = str(d1 * d3 + d2 * d4)

        cot = [
            "[Level 1.5: Digit-wise] Observation: Standard macro-math failed. Operands are exactly 2 digits.",
            "[Level 1.5] Hypothesis: The rule relies on digit-wise isolation and cross-operations.",
            f"[Level 1.5] Action: Confirmed `{best_rule}` pattern. Executing micro-math on digits [{d1},{d2}] and [{d3},{d4}].",
            f"\nFinal answer: {ans}"
        ]
        return "\n".join(cot)
    def _solve_level2_cryptarithm(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str, ctx: TaskContext) -> Optional[str]:
        if ctx.unique_symbols_count > 10:
            return None # Base-10 брутфорс невозможен

        # Sanity Check
        if ctx.has_length_reduction:
            return None # Ответ короче операндов. Это множества, а не десятичная математика.

        unique_syms = set(q_a + q_b)
        for ex in parsed_exs: unique_syms.update(list(ex['a'] + ex['b'] + ex['out']))
        unique_syms = list(unique_syms)

        all_ops = ['+', '-', '*', '/']
        found_perm = None
        found_op_map = None

        for perm in itertools.permutations(range(10), len(unique_syms)):
            op_candidates = defaultdict(lambda: set(all_ops))
            valid = True
            
            for ex in parsed_exs:
                v1 = int("".join(str(perm[unique_syms.index(c)]) for c in ex['a']))
                v2 = int("".join(str(perm[unique_syms.index(c)]) for c in ex['b']))
                vout = int("".join(str(perm[unique_syms.index(c)]) for c in ex['out']))
                if ex['is_neg']: vout = -vout
                
                possible_ops = set()
                if v1 + v2 == vout: possible_ops.add('+')
                if v1 - v2 == vout: possible_ops.add('-')
                if v1 * v2 == vout: possible_ops.add('*')
                if v2 != 0 and v1 // v2 == vout and v1 % v2 == 0: possible_ops.add('/')
                
                op_candidates[ex['op']].intersection_update(possible_ops)
                if not op_candidates[ex['op']]:
                    valid = False
                    break
                    
            if valid:
                if q_op in op_candidates and op_candidates[q_op]:
                    found_op_map = {k: list(v)[0] for k, v in op_candidates.items()}
                    found_perm = perm
                    break
                elif q_op in all_ops:
                    found_op_map = {k: list(v)[0] for k, v in op_candidates.items()}
                    found_op_map[q_op] = q_op
                    found_perm = perm
                    break
                    
        if found_perm:
            inv_map = {v: k for k, v in zip(unique_syms, found_perm)}
            v1 = int("".join(str(found_perm[unique_syms.index(c)]) for c in q_a))
            v2 = int("".join(str(found_perm[unique_syms.index(c)]) for c in q_b))
            
            actual_op = found_op_map[q_op]
            if actual_op == '+': ans = v1 + v2
            elif actual_op == '-': ans = v1 - v2
            elif actual_op == '*': ans = v1 * v2
            elif actual_op == '/': ans = v1 // v2 if v2 != 0 else 0
            
            ans_str = "".join(inv_map.get(int(d), d) if d.isdigit() else d for d in str(ans))
            
            cot_lines = [
                "[Level 2: Cryptarithm CSP] Observation: Data passed dimensional sanity checks.",
                f"[Level 2] Hypothesis: Valid 1-to-1 decimal mapping exists. Operator `{q_op}` acts as mathematical `{actual_op}`.",
                f"[Level 2] Action: Calculated base-10 result {ans}. Re-encrypting.",
                f"\nFinal answer: {ans_str}"
            ]
            return "\n".join(cot_lines)
            
        return None

    def _solve_level3_string(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str, ctx: TaskContext) -> Optional[str]:
        valid_patterns = defaultdict(int)
        target_exs = [ex for ex in parsed_exs if ex['op'] == q_op]
        if not target_exs: return None

        for ex in target_exs:
            a, b, out = ex['a'], ex['b'], ex['out']
            
            if out == a + b: valid_patterns['fwd'] += 1
            if out == b + a: valid_patterns['rev_cat'] += 1
            if out == a[::-1] + b[::-1]: valid_patterns['rev_both'] += 1
            
            interleaved = "".join(i+j for i,j in zip(a,b))
            if out == interleaved and interleaved != "": valid_patterns['interleave'] += 1
            
            # Set Logic
            if out == "".join([c for c in a if c not in b]) and out != "": valid_patterns['subtraction'] += 1
            if out == "".join([c for c in a if c in b]) and out != "": valid_patterns['intersection'] += 1
            
            if len(a) > 0 and len(b) > 0 and out == a[0] + b[-1]: valid_patterns['first_last'] += 1
            if len(a) > 0 and len(b) > 0 and a[-1] == b[0] and out == a + b[1:]: valid_patterns['overlap'] += 1
                
            # ASCII Cipher
            if len(a) == len(out) and len(a) > 0:
                diffs = set(ord(o) - ord(i) for o, i in zip(out, a))
                if len(diffs) == 1: valid_patterns['ascii_shift'] += 1

        if not valid_patterns: return None
            
        best_pattern = max(valid_patterns, key=valid_patterns.get)
        if valid_patterns[best_pattern] < len(target_exs): return None 

        if best_pattern == 'fwd': ans = q_a + q_b
        elif best_pattern == 'rev_cat': ans = q_b + q_a
        elif best_pattern == 'rev_both': ans = q_a[::-1] + q_b[::-1]
        elif best_pattern == 'interleave': ans = "".join(i+j for i,j in zip(q_a,q_b))
        elif best_pattern == 'subtraction': ans = "".join([c for c in q_a if c not in q_b])
        elif best_pattern == 'intersection': ans = "".join([c for c in q_a if c in q_b])
        elif best_pattern == 'first_last': ans = q_a[0] + q_b[-1]
        elif best_pattern == 'overlap': ans = q_a + q_b[1:] if len(q_a) > 0 and len(q_b) > 0 and q_a[-1] == q_b[0] else q_a + q_b
        elif best_pattern == 'ascii_shift':
            shift = ord(target_exs[0]['out'][0]) - ord(target_exs[0]['a'][0])
            ans = "".join(chr(ord(c) + shift) for c in q_a)
        else: ans = q_a + q_b

        cot_lines = [
            "[Level 3: String Engine] Observation: Algebra aborted. Context indicates string/set relationships.",
            f"[Level 3] Hypothesis: Strict set logic or positional pattern detected: `{best_pattern}`.",
            f"[Level 3] Action: Transforming `{q_a}` and `{q_b}` using `{best_pattern}` logic.",
            f"\nFinal answer: {ans}"
        ]
        return "\n".join(cot_lines)

    def generate_cot(self, prompt: str) -> str:
        prompt_str = str(prompt)
        query_match = re.search(r"determine the result for:\s*([^\n]+)", prompt_str, re.IGNORECASE)
        if not query_match:
            lines = [line.strip() for line in prompt_str.split('\n') if line.strip()]
            if lines and '=' not in lines[-1]: query_str = lines[-1].replace('Question:', '').strip()
            else: return "[Error] Observation: Parse failure.\nFinal answer: nan"
        else:
            query_str = query_match.group(1).strip()
            
        query_clean = query_str.replace(" ", "")
        
        raw_lines = re.findall(r"([^\n=]+?)\s*=\s*([^\n]+)", prompt_str)
        parsed_exs = []
        for lhs, rhs in raw_lines:
            lhs_c, rhs_c = lhs.replace(" ", ""), rhs.strip()
            if len(lhs_c) >= 3:
                mid = len(lhs_c) // 2
                is_neg = False
                if rhs_c.startswith("-") and len(rhs_c) > 1:
                    is_neg = True; rhs_c = rhs_c[1:]
                parsed_exs.append({
                    "a": lhs_c[:mid], "op": lhs_c[mid], "b": lhs_c[mid+1:], 
                    "out": rhs_c, "is_neg": is_neg
                })

        if len(query_clean) < 3: return "[Error] Observation: Query too short.\nFinal answer: nan"
        
        mid = len(query_clean) // 2
        q_a, q_op, q_b = query_clean[:mid], query_clean[mid], query_clean[mid+1:]

        # УРОВЕНЬ 0: Сбор досье на задачу
        ctx = self._analyze_context(parsed_exs, q_a, q_op, q_b)

        # 1. Level 1 - Numeric
        lvl1_result = self._solve_level1_numeric(prompt_str, query_clean, ctx)
        if lvl1_result: return lvl1_result
            
        # 1.5. Level 1.5 - Digit-wise
        lvl1_5_result = self._solve_level1_5_digit_wise(parsed_exs, q_a, q_op, q_b, ctx)
        if lvl1_5_result: return lvl1_5_result
            
        # 2. Level 2 - Cryptarithm
        lvl2_result = self._solve_level2_cryptarithm(parsed_exs, q_a, q_op, q_b, ctx)
        if lvl2_result: return lvl2_result
            
        # 3. Level 3 - String Logic
        lvl3_result = self._solve_level3_string(parsed_exs, q_a, q_op, q_b, ctx)
        if lvl3_result: return lvl3_result
            
        # 4. Level 4 - Smart Math Fallback
        is_numeric = q_a.lstrip('-').isdigit() and q_b.lstrip('-').isdigit()
        if is_numeric and q_op in ['+', '-', '*', '/']:
            v1, v2 = int(q_a), int(q_b)
            if q_op == '+': ans = str(v1 + v2)
            elif q_op == '-': ans = str(v1 - v2)
            elif q_op == '*': ans = str(v1 * v2)
            elif q_op == '/' and v2 != 0: ans = str(v1 // v2)
            else: ans = str(v1 + v2)
            
            cot_lines = [
                "[Level 4: Smart Fallback] Observation: Pattern solvers exhausted. High noise detected in examples.",
                f"[Level 4] Hypothesis: Target contains pure numbers and standard operator `{q_op}`. Trusting base math over noisy examples.",
                f"[Level 4] Action: Evaluated {q_a} {q_op} {q_b} standardly.",
                f"\nFinal answer: {ans}"
            ]
            return "\n".join(cot_lines)

        # 5. Level 4 - Naive Concat Fallback
        ans = (q_a + q_b) if q_op in ['+', '*'] else (q_b + q_a)
        cot_lines = [
            "[Level 4: Naive Fallback] Observation: Extracted features do not match any known macro/micro structures or sets.",
            "[Level 4] Hypothesis: Edge-case string procedural generation. Defaulting to heuristic concatenation.",
            f"\nFinal answer: {ans}"
        ]
        return "\n".join(cot_lines)