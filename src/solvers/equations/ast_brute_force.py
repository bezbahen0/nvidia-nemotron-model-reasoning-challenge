import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class ASTBruteForceSolver:
    """берет известные цифры и пытается понять, по какому правилу они взаимодействуют"""
    def __init__(self):
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")

    def _rev(self, s: str) -> str:
        s_str = str(s)
        return "-" + s_str[1:][::-1] if s_str.startswith("-") else s_str[::-1]

    def _get_operand_configs(self, sa: str, sb: str) -> Dict[str, Tuple[int, int, str, str]]:
        return {
            "fwd": (int(sa), int(sb), sa, sb),
            "swap_ops": (int(sb), int(sa), sb, sa),
            "rev_digits": (int(self._rev(sa)), int(self._rev(sb)), self._rev(sa), self._rev(sb)),
            "swap_rev": (int(self._rev(sb)), int(self._rev(sa)), self._rev(sb), self._rev(sa))
        }

    def _get_operations(self, a: int, b: int, sa: str, sb: str) -> Dict[str, int]:
        ops = {
            "add": a + b, 
            "abs_diff": abs(a - b),  # Поднято вверх: модуль разности чаще встречается в загадках
            "sub": a - b, 
            "rev_sub": b - a, 
            "mul": a * b,
            "cat": int(sa + sb) if sa + sb != "" and len(sa + sb) < 15 else 0,
            "rev_cat": int(sb + sa) if sb + sa != "" and len(sb + sa) < 15 else 0,
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
        if a != 0 and b != 0: 
            ops["max_mod_min"] = max(a, b) % min(a, b)
            
        if len(sa) == 2 and len(sb) == 2 and sa.isdigit() and sb.isdigit():
            d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
            ops["cross_sum"] = d1 * d3 + d2 * d4
            ops["cross_diff_abs"] = abs((d1 + d2) - (d3 + d4))
            ops["cross_concat"] = int(str(d1 * d3) + str(d2 * d4))
            ops["cross_rev_concat"] = int(str(d1 * d4) + str(d2 * d3))
            
        return ops

    def _get_formats(self, val: int) -> Dict[str, str]:
        sval = str(val)
        abs_val = abs(val)
        s_abs = str(abs_val)
        return {
            "raw": sval,
            "abs": s_abs,
            "rev": "-" + s_abs[::-1] if val < 0 else s_abs[::-1],
            "zpad2": f"{val:02d}" if val >= 0 else f"-{abs_val:02d}",
            "zpad3": f"{val:03d}" if val >= 0 else f"-{abs_val:03d}",
            "first_digit": sval[0] if val >= 0 else "-" + s_abs[0],
            "last_digit": sval[-1] if val >= 0 else "-" + s_abs[-1],
        }
        
    # Вспомогательные функции для валидации префиксов/суффиксов (Игнорируют математический минус)
    def _is_valid_prefix(self, raw: str, num: str, op: str) -> bool:
        if op == '-' and raw == num and num.startswith('-'): return False
        return raw.startswith(op) and len(raw) > len(op)

    def _is_valid_suffix(self, raw: str, num: str, op: str) -> bool:
        return raw.endswith(op) and len(raw) > len(op)

    def solve(self, examples_text: str, target_text: str) -> Optional[str]:
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm: return None
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)

        all_parsed = []
        for line in examples_text.split('\n'):
            if '=' not in line: continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                rhs_clean = rhs.strip()
                num_part = "".join(c for c in rhs_clean if c.isdigit() or c == '-')
                if num_part:
                    all_parsed.append({
                        'a': a, 'op': op, 'b': b, 
                        'num_out': num_part, 'raw_out': rhs_clean
                    })

        if not all_parsed: return None

        ops_grouped = defaultdict(list)
        for ex in all_parsed:
            ops_grouped[ex['op']].append(ex)

        configs_order = ["fwd", "rev_digits", "swap_ops", "swap_rev"]
        fmt_names = ["raw", "abs", "rev", "zpad2", "zpad3", "first_digit", "last_digit"]
        op_names = list(self._get_operations(1, 1, "1", "1").keys())

        global_best_rule = None
        global_op_mapping = {}

        # 1. Поиск "Единой теории"
        for op_config in configs_order:
            for out_fmt in fmt_names:
                valid_global = True
                temp_mapping = {}
                
                for op_char, group in ops_grouped.items():
                    matched_op_name = None
                    for op_name in op_names:
                        all_pass = True
                        for ex in group:
                            cfg = self._get_operand_configs(ex['a'], ex['b'])[op_config]
                            ops = self._get_operations(*cfg)
                            if op_name not in ops or self._get_formats(ops[op_name])[out_fmt] != ex['num_out']:
                                all_pass = False
                                break
                        if all_pass:
                            matched_op_name = op_name
                            break
                    
                    if matched_op_name:
                        temp_mapping[op_char] = matched_op_name
                    else:
                        valid_global = False
                        break
                        
                if valid_global:
                    global_best_rule = (op_config, out_fmt)
                    global_op_mapping = temp_mapping
                    break
            if global_best_rule: break

        # 2. Применение глобального правила
        if global_best_rule and q_op in global_op_mapping:
            op_config, out_fmt = global_best_rule
            op_name = global_op_mapping[q_op]
            
            target_exs = ops_grouped[q_op]
            uses_suffix = all([self._is_valid_suffix(ex['raw_out'], ex['num_out'], q_op) for ex in target_exs])
            uses_prefix = all([self._is_valid_prefix(ex['raw_out'], ex['num_out'], q_op) for ex in target_exs])
            
            cfg = self._get_operand_configs(q_a, q_b)[op_config]
            val = self._get_operations(*cfg)[op_name]
            final_ans = self._get_formats(val)[out_fmt]
            
            if uses_suffix: return final_ans + q_op
            if uses_prefix: return q_op + final_ans
            return final_ans

        # 3. Локальный поиск
        target_exs = ops_grouped.get(q_op, all_parsed)
        uses_suffix = all([self._is_valid_suffix(ex['raw_out'], ex['num_out'], q_op) for ex in target_exs])
        uses_prefix = all([self._is_valid_prefix(ex['raw_out'], ex['num_out'], q_op) for ex in target_exs])
        
        for op_config in configs_order:
            for out_fmt in fmt_names:
                if out_fmt in ["first_digit", "last_digit"] and len(target_exs) < 3:
                    continue
                    
                for op_name in op_names:
                    all_pass = True
                    for ex in target_exs:
                        cfg = self._get_operand_configs(ex['a'], ex['b'])[op_config]
                        ops = self._get_operations(*cfg)
                        
                        if op_name not in ops or self._get_formats(ops[op_name])[out_fmt] != ex['num_out']:
                            all_pass = False
                            break
                    
                    if all_pass:
                        cfg = self._get_operand_configs(q_a, q_b)[op_config]
                        val = self._get_operations(*cfg)[op_name]
                        final_ans = self._get_formats(val)[out_fmt]
                        
                        if uses_suffix: return final_ans + q_op
                        if uses_prefix: return q_op + final_ans
                        return final_ans

        if q_op in ['+', '-', '*', '/']:
            try:
                # Очистка ведущих нулей (например, "05" -> "5", "-02" -> "-2")
                clean_a = str(int(q_a))
                clean_b = str(int(q_b))
                return str(int(eval(f"{clean_a}{q_op}{clean_b}")))
            except (ZeroDivisionError, ValueError):
                pass

        return None