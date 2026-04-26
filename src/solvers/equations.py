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
        
        # Mapping dictionaries to convert programmatic rules into natural language
        self.op_desc = {
            "add": "add the two numbers", "sub": "subtract the second number from the first",
            "rev_sub": "subtract the first number from the second", "mul": "multiply the numbers",
            "div": "divide the first number by the second", "mod": "find the remainder of the first number divided by the second",
            "rev_div": "divide the second number by the first", "rev_mod": "find the remainder of the second number divided by the first",
            "max_mod_min": "find the remainder of the larger number divided by the smaller number",
            "cat": "concatenate the numbers together", "rev_cat": "concatenate the numbers in reverse order",
            "add1": "add the two numbers and then add 1", "addm1": "add the two numbers and then subtract 1",
            "mul1": "multiply the numbers and then add 1", "mulm1": "multiply the numbers and then subtract 1",
            "sub1": "subtract the second number from the first and then add 1", "subm1": "subtract the second number from the first and then subtract 1",
            "abs_diff": "find the absolute difference between the numbers", "neg_abs_diff": "find the negative absolute difference between the numbers"
        }

        self.config_desc = {
            "fwd": "take the numbers exactly as they are provided",
            "rev_digits": "reverse the digits of each number",
            "swap_ops": "swap the positions of the first and second number",
            "swap_rev": "reverse the digits of each number and then swap their positions"
        }

        self.fmt_desc = {
            "raw": "leave the result exactly as computed",
            "rev": "reverse the digits of the final computed result",
            "abs": "take the absolute value, removing any negative signs",
            "zpad2": "pad the result with leading zeros to ensure it is exactly 2 digits long",
            "zpad3": "pad the result with leading zeros to ensure it is exactly 3 digits long",
            "first_digit": "extract only the very first digit of the result",
            "last_digit": "extract only the very last digit of the result"
        }

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

    def _solve_level1_numeric(self, prompt: str, query_str: str, ctx: TaskContext, answer_hint: Optional[str] = None) -> Optional[str]:
        raw_lines = re.findall(r"([^\n=]+?)\s*=\s*([^\n]+)", prompt)
        parsed = []
        for lhs, rhs in raw_lines:
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                try:
                    int(m.group(1)); int(m.group(3))
                    clean_rhs = "".join(c for c in rhs.strip() if c.isdigit() or c == '-')
                    if clean_rhs:
                        parsed.append((m.group(1), m.group(2).strip(), m.group(3), rhs.strip()))
                except ValueError:
                    pass

        if not parsed: return None

        qm = self._numeric_re.fullmatch(query_str)
        if not qm: return None
        qa, q_op, qb = qm.group(1), qm.group(2).strip(), qm.group(3)

        if answer_hint is not None:
            parsed.append((qa, q_op, qb, str(answer_hint).strip()))

        by_op = defaultdict(list)
        for a, op, b, out in parsed: by_op[op].append((a, b, out))

        found_rules = {}
        op_names = list(self._get_operations(1, 1, "1", "1").keys())
        fmt_names = list(self._get_formats(1).keys())

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

        if q_op not in found_rules:
            if q_op in ['+', '-', '*', '/'] and found_rules:
                first_rule = list(found_rules.values())[0]
                if all(r.op_config == first_rule.op_config and r.out_fmt == first_rule.out_fmt and r.neg_fmt == first_rule.neg_fmt for r in found_rules.values()):
                    synth_op = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[q_op]
                    rule = FoundRule(first_rule.op_config, synth_op, first_rule.out_fmt, first_rule.neg_fmt, q_op)
                else:
                    return None
            else:
                return None
        else:
            rule = found_rules[q_op]

        cfg = self._get_operand_configs(qa, qb)[rule.op_config]
        val = self._get_operations(*cfg)[rule.op_name]
        final_str = self._get_formats(val)[rule.out_fmt]
        
        if final_str.startswith("-"):
            if rule.neg_fmt == "op_suffix": final_str = final_str[1:] + rule.op_char
            elif rule.neg_fmt == "op_prefix": final_str = rule.op_char + final_str[1:]
            elif rule.neg_fmt == "neg_suffix": final_str = final_str[1:] + "-"

        cfg_text = self.config_desc.get(rule.op_config, f"apply a custom data transformation")
        op_text = self.op_desc.get(rule.op_name, f"perform a specialized math operation")
        fmt_text = self.fmt_desc.get(rule.out_fmt, f"format the result mathematically")

        # 1. Раскрываем промежуточные значения для Step 1 (подготовка операндов)
        step1_calc = ""
        if rule.op_config == "fwd": 
            step1_calc = f"Operands remain unchanged: {qa} and {qb}."
        elif rule.op_config == "rev_digits": 
            step1_calc = f"{qa} becomes {cfg[0]}, and {qb} becomes {cfg[1]}."
        elif rule.op_config == "swap_ops": 
            step1_calc = f"Operands are swapped: {cfg[0]} and {cfg[1]}."
        elif rule.op_config == "swap_rev": 
            step1_calc = f"Operands are reversed and swapped: {cfg[0]} and {cfg[1]}."

        # 2. Раскрываем математику для Step 2
        op_math_strings = {
            "add": f"{cfg[0]} + {cfg[1]} = {val}",
            "sub": f"{cfg[0]} - {cfg[1]} = {val}",
            "rev_sub": f"{cfg[1]} - {cfg[0]} = {val}",
            "mul": f"{cfg[0]} * {cfg[1]} = {val}",
            "div": f"{cfg[0]} / {cfg[1]} = {val}" if cfg[1] != 0 else f"{cfg[0]} / {cfg[1]} = 0",
            "mod": f"{cfg[0]} % {cfg[1]} = {val}" if cfg[1] != 0 else f"{cfg[0]} % {cfg[1]} = 0",
            "rev_div": f"{cfg[1]} / {cfg[0]} = {val}" if cfg[0] != 0 else f"{cfg[1]} / {cfg[0]} = 0",
            "rev_mod": f"{cfg[1]} % {cfg[0]} = {val}" if cfg[0] != 0 else f"{cfg[1]} % {cfg[0]} = 0",
            "max_mod_min": f"max({cfg[0]}, {cfg[1]}) % min({cfg[0]}, {cfg[1]}) = {val}",
            "cat": f"'{cfg[2]}' concatenated with '{cfg[3]}' = {val}",
            "rev_cat": f"'{cfg[3]}' concatenated with '{cfg[2]}' = {val}",
            "add1": f"{cfg[0]} + {cfg[1]} + 1 = {val}",
            "addm1": f"{cfg[0]} + {cfg[1]} - 1 = {val}",
            "mul1": f"{cfg[0]} * {cfg[1]} + 1 = {val}",
            "mulm1": f"{cfg[0]} * {cfg[1]} - 1 = {val}",
            "sub1": f"{cfg[0]} - {cfg[1]} + 1 = {val}",
            "subm1": f"{cfg[0]} - {cfg[1]} - 1 = {val}",
            "abs_diff": f"abs({cfg[0]} - {cfg[1]}) = {val}",
            "neg_abs_diff": f"-abs({cfg[0]} - {cfg[1]}) = {val}"
        }
        step2_calc = op_math_strings.get(rule.op_name, f"Result is {val}")
        
        # 3. Раскрываем форматирование для Step 3
        if rule.out_fmt == "raw" and rule.neg_fmt == "standard":
            step3_calc = f"Result remains {final_str}."
        else:
            step3_calc = f"Computed {val} becomes {final_str}."

        cot_lines = [
            "First, let's analyze the underlying pattern in the provided examples.",
            "The standard mathematical operators are being used as placeholders for a hidden, multi-step rule.",
            "By observing the relationship between the inputs and outputs, the consistent sequence of operations is:",
            f"Rule 1: We must {cfg_text}.",
            f"Rule 2: Next, we {op_text}.",
            f"Rule 3: Finally, we {fmt_text}.",
            "",
            f"Now, let's apply this exact sequence to the target query: {qa} {q_op} {qb}.",
            f"- Step 1 (Apply Rule 1): {step1_calc}",
            f"- Step 2 (Apply Rule 2): {step2_calc}",
            f"- Step 3 (Apply Rule 3): {step3_calc}",
            f"\nFinal answer: {final_str}"
        ]
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

        rule_explanations = {
            'cross_concat': "multiply the first digit of the first number by the first digit of the second number, then multiply the second digit of the first number by the second digit of the second number, and concatenate the results",
            'cross_rev_concat': "multiply the first digit of the first number by the second digit of the second number, then multiply the second digit of the first number by the first digit of the second number, and concatenate the results",
            'sum_diff_abs': "sum the digits of each number individually, then find the absolute difference between these two sums",
            'cross_sum': "multiply the first digits together, multiply the second digits together, and add those two products"
        }
        explanation = rule_explanations.get(best_rule, "apply a specific digit-cross operation")

        cot_lines = [
            "Looking closely at the numbers, standard arithmetic operations on the whole numbers do not match the examples.",
            "Since all operands are exactly two digits long, the pattern likely operates on the individual digits rather than the full integers.",
            "Let's split the numbers into their component digits.",
            f"The hidden rule is to {explanation}.",
            "",
            f"Let's apply this rule to our target operands: {q_a} and {q_b}.",
            f"- The digits of the first number are {d1} and {d2}.",
            f"- The digits of the second number are {d3} and {d4}.",
            f"- Applying the identified digit-wise math results in {ans}.",
            f"\nFinal answer: {ans}"
        ]
        return "\n".join(cot_lines)

    def _solve_level2_cryptarithm(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str, ctx: TaskContext) -> Optional[str]:
        if ctx.unique_symbols_count > 10:
            return None 

        if ctx.has_length_reduction:
            return None 

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
            char_to_digit = {c: str(found_perm[unique_syms.index(c)]) for c in unique_syms}
            digit_to_char = {str(v): k for k, v in zip(unique_syms, found_perm)}
            
            v1 = int("".join(char_to_digit[c] for c in q_a))
            v2 = int("".join(char_to_digit[c] for c in q_b))
            
            actual_op = found_op_map[q_op]
            if actual_op == '+': ans = v1 + v2
            elif actual_op == '-': ans = v1 - v2
            elif actual_op == '*': ans = v1 * v2
            elif actual_op == '/': ans = v1 // v2 if v2 != 0 else 0
            
            ans_str = "".join(digit_to_char.get(d, d) for d in str(ans))
            
            mapping_str = ", ".join(f"'{k}' -> {v}" for k, v in char_to_digit.items())
            
            cot_lines = [
                "Let's solve this step-by-step by deciphering the symbols.",
                f"1. Based on the examples, we deduce the exact character-to-digit mapping: {mapping_str}.",
                f"2. The operator '{q_op}' corresponds to the standard arithmetic operation '{actual_op}'.",
                "3. We translate the target operands into base-10 numbers using the mapping.",
                f"   - Operand 1: '{q_a}' translates character-by-character to {v1}.",
                f"   - Operand 2: '{q_b}' translates character-by-character to {v2}.",
                "4. Execute the mathematical operation in decimal format:",
                f"   {v1} {actual_op} {v2} = {ans}",
                "5. Encrypt the numerical result back into symbols.",
                f"   The number {ans} translates character-by-character to '{ans_str}'.",
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
            
            if out == "".join([c for c in a if c not in b]) and out != "": valid_patterns['subtraction'] += 1
            if out == "".join([c for c in a if c in b]) and out != "": valid_patterns['intersection'] += 1
            
            if len(a) > 0 and len(b) > 0 and out == a[0] + b[-1]: valid_patterns['first_last'] += 1
            if len(a) > 0 and len(b) > 0 and a[-1] == b[0] and out == a + b[1:]: valid_patterns['overlap'] += 1
                
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

        str_explanations = {
            'fwd': "concatenate the two strings together from left to right",
            'rev_cat': "concatenate the second string first, followed by the first string",
            'rev_both': "reverse the characters of both strings and then concatenate them",
            'interleave': "take one character from the first string, then one from the second, alternating until finished",
            'subtraction': "remove any characters from the first string that appear in the second string",
            'intersection': "keep only the characters in the first string that also exist in the second string",
            'first_last': "take only the very first character of the first string and the very last character of the second string",
            'overlap': "merge them by overlapping the shared character at the boundary",
            'ascii_shift': "shift the ASCII value of each character by a constant offset"
        }
        exp = str_explanations.get(best_pattern, "apply a strict string manipulation pattern")

        cot_lines = [
            "The examples provided do not behave like mathematical equations. The operator is acting as a string manipulation function.",
            "By tracking how the characters move from the left side of the equation to the right side, we can determine the exact operation.",
            f"The consistent pattern across all examples is to {exp}.",
            "",
            f"Now, let's process the target strings '{q_a}' and '{q_b}' using this exact logic.",
            f"- Following the rule, the new sequence of characters becomes '{ans}'.",
            f"\nFinal answer: {ans}"
        ]
        return "\n".join(cot_lines)
    
    def _solve_level4_string_operations(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str) -> Optional[str]:
        # Базовые строковые операции
        hypotheses = [
            ("concatenation (A+B)", lambda a, b: a + b),
            ("reverse concatenation (B+A)", lambda a, b: b + a),
            ("interleaving", lambda a, b: "".join(i + j for i, j in zip(a + " "*len(b), b + " "*len(a))).replace(" ", "")),
            ("string subtraction", lambda a, b: "".join(c for c in a if c not in b) or "-")
        ]

        for hyp_name, func in hypotheses:
            success = True
            for ex in parsed_exs:
                if "answer_hint" in ex: continue
                # Восстанавливаем сырую строку (с учетом знака минус)
                out_str = ("-" if ex["is_neg"] else "") + ex["out"]
                try:
                    if func(ex["a"], ex["b"]) != out_str:
                        success = False
                        break
                except Exception:
                    success = False
                    break
                    
            if success:
                ans_str = func(q_a, q_b)
                return "\n".join([
                    "Analyzing the transformation reveals it is not mathematical, but a structural string operation.",
                    f"The consistent rule across all examples is '{hyp_name}'.",
                    f"Applying this operation to '{q_a}' and '{q_b}':",
                    f"\nFinal answer: {ans_str}"
                ])
        return None
    
    def _solve_level5_robust_cryptarithm(self, parsed_exs: List[Dict], q_a: str, q_op: str, q_b: str) -> Optional[str]:
        import itertools
        
        all_chars = set()
        op_chars = set()
        for ex in parsed_exs:
            all_chars.update(ex["a"] + ex["b"] + ex["out"])
            op_chars.add(ex["op"])
        all_chars.update(q_a + q_b)
        op_chars.add(q_op)
        
        standard_ops = {'+', '-', '*'}
        encrypted_ops = [op for op in op_chars if op not in standard_ops]
        
        digit_chars = list(all_chars - set(encrypted_ops) - standard_ops)
        if len(digit_chars) > 10 or len(encrypted_ops) > 3: 
            return None
            
        op_assignments = []
        if not encrypted_ops:
            op_assignments.append({})
        else:
            for ops in itertools.product(['+', '-', '*'], repeat=len(encrypted_ops)):
                op_assignments.append(dict(zip(encrypted_ops, ops)))
                
        for perm in itertools.permutations("0123456789", len(digit_chars)):
            char_to_digit = dict(zip(digit_chars, perm))
            
            for op_map in op_assignments:
                success = True
                for ex in parsed_exs:
                    if "answer_hint" in ex: continue
                    
                    try:
                        # int() автоматически легализует ведущие нули (например, "05" -> 5)
                        a_val = int("".join(char_to_digit.get(c, c) for c in ex["a"]))
                        b_val = int("".join(char_to_digit.get(c, c) for c in ex["b"]))
                        out_val = int("".join(char_to_digit.get(c, c) for c in ex["out"]))
                        if ex["is_neg"]: out_val = -out_val
                        
                        actual_op = op_map.get(ex["op"], ex["op"])
                        
                        if actual_op == '+': res = a_val + b_val
                        elif actual_op == '-': res = a_val - b_val
                        elif actual_op == '*': res = a_val * b_val
                        else: success = False; break
                        
                        if res != out_val:
                            success = False; break
                    except Exception:
                        success = False; break
                
                if success:
                    try:
                        a_val = int("".join(char_to_digit.get(c, c) for c in q_a))
                        b_val = int("".join(char_to_digit.get(c, c) for c in q_b))
                        actual_q_op = op_map.get(q_op, q_op)
                        
                        if actual_q_op == '+': ans_val = a_val + b_val
                        elif actual_q_op == '-': ans_val = a_val - b_val
                        elif actual_q_op == '*': ans_val = a_val * b_val
                        else: return None
                        
                        digit_to_char = {v: k for k, v in char_to_digit.items()}
                        ans_str = ""
                        is_neg = ans_val < 0
                        for digit in str(abs(ans_val)):
                            if digit not in digit_to_char: raise ValueError
                            ans_str += digit_to_char[digit]
                            
                        if is_neg: ans_str = "-" + ans_str
                        
                        op_msg = "Some operators are also encrypted." if encrypted_ops else "Using standard arithmetic."
                        return "\n".join([
                            "By analyzing the patterns, this is an unrestricted cryptarithm where leading zeros are permitted.",
                            op_msg,
                            f"Mapping found: {', '.join([f'`{k}`->{v}' for k, v in char_to_digit.items()])}",
                            f"Calculating {a_val} {actual_q_op} {b_val} = {ans_val}.",
                            f"\nFinal answer: {ans_str}"
                        ])
                    except Exception:
                        return None
        return None

    def generate_cot(self, prompt: str, answer_hint: Optional[str] = None) -> Optional[str]:
        prompt_str = str(prompt)
        query_match = re.search(r"determine the result for:\s*([^\n]+)", prompt_str, re.IGNORECASE)
        
        # Фильтрация битых данных вместо генерации ответа с ошибкой
        if not query_match:
            lines = [line.strip() for line in prompt_str.split('\n') if line.strip()]
            if lines and '=' not in lines[-1]: 
                query_str = lines[-1].replace('Question:', '').strip()
            else: 
                return None 
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

        # Фильтрация коротких запросов
        if len(query_clean) < 3: 
            return None
        
        mid = len(query_clean) // 2
        q_a, q_op, q_b = query_clean[:mid], query_clean[mid], query_clean[mid+1:]

        ctx = self._analyze_context(parsed_exs, q_a, q_op, q_b)

        if answer_hint is not None:
            ans_clean = str(answer_hint).strip()
            is_neg = ans_clean.startswith("-") and len(ans_clean) > 1
            parsed_exs.append({
                "a": q_a, "op": q_op, "b": q_b, 
                "out": ans_clean[1:] if is_neg else ans_clean, 
                "is_neg": is_neg
            })
        
        # Инициализация истории рассуждений
        thought_process = [
            "Let's systematically analyze the examples to discover the hidden transformation rule."
        ]

        thought_process.append("The symbols represent standard base-10 mathematical operations.")
        lvl1_result = self._solve_level1_numeric(prompt_str, query_clean, ctx, answer_hint)
        if lvl1_result: 
            return "\n".join(thought_process + [lvl1_result])

        thought_process.append("Testing standard arithmetic reveals contradictions. For instance, the outputs in the examples do not match standard math.")
        thought_process.append("The operation might be applied digit-by-digit rather than on whole numbers.")
        lvl1_5_result = self._solve_level1_5_digit_wise(parsed_exs, q_a, q_op, q_b, ctx)
        if lvl1_5_result: 
            return "\n".join(thought_process + [lvl1_5_result])
            
        thought_process.append("Digit-wise operations do not yield a consistent rule across all examples.")
        thought_process.append("This could be a cryptarithm where symbols or letters map to specific base-10 digits.")
        lvl2_result = self._solve_level2_cryptarithm(parsed_exs, q_a, q_op, q_b, ctx)
        if lvl2_result: 
            return "\n".join(thought_process + [lvl2_result])
            
        thought_process.append("Analysis shows no consistent character-to-digit mapping satisfies all equations simultaneously. Let's pivot to a non-mathematical structure.")
        thought_process.append("The operators might be string manipulation functions (like concatenation, interleaving, or character subtraction).")
        
        lvl3_result = self._solve_level3_string(parsed_exs, q_a, q_op, q_b, ctx)
        if lvl3_result: 
            return "\n".join(thought_process + [lvl3_result])
        
        thought_process.append("Basic string concatenation or interleaving doesn't match the examples either. The transformation might be more structural.")

       # lvl4_result = self._solve_level4_string_operations(parsed_exs, q_a, q_op, q_b)
       # if lvl4_result: 
       #     return "\n".join(thought_process + [lvl4_result])
       # 
       # thought_process.append("String manipulations also fail to consistently explain the outputs.")
       # thought_process.append("Let's consider a robust cryptarithm where even operators are encrypted and leading zeros are allowed.")        
       # lvl5_result = self._solve_level5_robust_cryptarithm(parsed_exs, q_a, q_op, q_b)
       # if lvl5_result: 
       #     return "\n".join(thought_process + [lvl5_result])
            
        # Fallbacks (если ни одно правило не найдено)
        thought_process.append("All structural, mathematical, and encrypted hypotheses failed to find a reliable global rule for the examples.")
        
        is_numeric = q_a.lstrip('-').isdigit() and q_b.lstrip('-').isdigit()
        if is_numeric and q_op in ['+', '-', '*', '/']:
            v1, v2 = int(q_a), int(q_b)
            if q_op == '+': ans = str(v1 + v2)
            elif q_op == '-': ans = str(v1 - v2)
            elif q_op == '*': ans = str(v1 * v2)
            elif q_op == '/' and v2 != 0: ans = str(v1 // v2)
            else: ans = str(v1 + v2)
            
            fallback_numeric = "\n".join([
                f"However, the target query consists of pure numbers and a standard operator `{q_op}`.",
                "In the absence of a reliable hidden rule, the most logical fallback approach is to trust standard base-10 mathematics.",
                f"Evaluating {q_a} {q_op} {q_b} standardly.",
                f"\nFinal answer: {ans}"
            ])
            return "\n".join(thought_process + [fallback_numeric])

        ans = (q_a + q_b) if q_op in ['+', '*'] else (q_b + q_a)
        fallback_str = "\n".join([
            "Since no reliable rule can be extracted, we fall back to a basic heuristic combination.",
            "We will simply concatenate the two target strings.",
            f"\nFinal answer: {ans}"
        ])
        return "\n".join(thought_process + [fallback_str])