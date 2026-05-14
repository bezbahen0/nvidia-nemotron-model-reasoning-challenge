import re
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter

class ASTBruteForceSolver:
    """Solves mathematical puzzles by deducing hidden operations between numbers based on examples."""
    def __init__(self):
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")
        
        self.config_desc = {
            "fwd": "fwd (a, b)",
            "swap_ops": "swap (b, a)",
            "rev_digits": "rev_digits (rev(a), rev(b))",
            "swap_rev": "swap_rev (rev(b), rev(a))"
        }

        self.op_desc = {
            "add": "add", "sub": "sub", "mul": "mul", "abs_diff": "abs_diff",
            "div": "div", "mod": "mod", "rev_div": "rev_div", "rev_mod": "rev_mod",
            "rev_sub": "rev_sub", "add1": "add+1", "sub1": "sub+1", "mul1": "mul+1",
            "addm1": "add-1", "subm1": "sub-1", "mulm1": "mul-1", "neg_abs_diff": "-abs_diff",
            "cat": "concat", "rev_cat": "rev_concat", "dsum_add": "digit_sum_add",
            "dsum_mul": "digit_sum_mul", "max_mod_min": "max_mod_min",
            "cross_sum": "cross_sum", "cross_diff_abs": "cross_diff_abs",
            "cross_concat": "cross_concat", "cross_rev_concat": "cross_rev_concat"
        }

        self.fmt_desc = {
            "raw": "raw", "abs": "abs", "zpad2": "zpad2", "zpad3": "zpad3",
            "rev": "rev", "abs_rev": "abs_rev", "first_digit": "first_digit",
            "last_digit": "last_digit", "sign_pref_raw": "sign_pref_raw",
            "sign_suff_raw": "sign_suff_raw", "sign_pref_rev": "sign_pref_rev",
            "sign_suff_rev": "sign_suff_rev", "raw_pref": "raw_pref",
            "raw_suff": "raw_suff", "abs_pref": "abs_pref", "abs_suff": "abs_suff"
        }

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
        ops = {}
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
            try: ops["cat"] = int(sa + sb)
            except ValueError: pass
        if sb + sa != "" and len(sb + sa) < 15:
            try: ops["rev_cat"] = int(sb + sa)
            except ValueError: pass

        ops["dsum_add"] = sum(int(d) for d in str(abs(a))) + sum(int(d) for d in str(abs(b)))
        ops["dsum_mul"] = sum(int(d) for d in str(abs(a))) * sum(int(d) for d in str(abs(b)))

        if a != 0 and b != 0: 
            ops["max_mod_min"] = max(a, b) % min(a, b) if min(a, b) != 0 else 0
            
        if len(sa) == 2 and len(sb) == 2 and sa.lstrip('-').isdigit() and sb.lstrip('-').isdigit():
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

    def _score_hypothesis(self, config: str, op_name: str, fmt: str, num_examples: int, global_config: str = None, global_fmt: str = None) -> int:
        score = 0
        if config == "swap_ops": score += 20
        elif config == "rev_digits": score += 30
        elif config == "swap_rev": score += 50
        
        if fmt == "abs": score += 5
        elif fmt.startswith("sign_"): score += 7      
        elif fmt.startswith("raw_") or fmt.startswith("abs_"): score += 9
        elif fmt in ["zpad2", "zpad3"]: score += 15
        elif fmt == "rev": score += 25              
        elif fmt == "abs_rev": score += 27          
        elif "digit" in fmt: score += 40
        
        op_penalties = {
            "add": 0, "sub": 1, "abs_diff": 2, "mul": 3, "cat": 4,
            "div": 10, "mod": 11, "rev_sub": 12, "rev_cat": 13,
            "add1": 20, "sub1": 21, "mul1": 22, "addm1": 23, "subm1": 24, "mulm1": 25,
            "neg_abs_diff": 26, "max_mod_min": 27,
            "dsum_add": 30, "dsum_mul": 31,
            "cross_sum": 40, "cross_diff_abs": 41, "cross_concat": 42, "cross_rev_concat": 43
        }
        score += op_penalties.get(op_name, 50)
        if num_examples == 1 and op_name in ["mod", "div", "rev_mod", "rev_div"]: score += 100
        if global_config and global_fmt:
            if config == global_config and fmt == global_fmt: score -= 1000 
        return score

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        log = []
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm: 
            return {"answer": None, "debug": ["Parse Error."]}
            
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)

        all_parsed = []
        for line in examples_text.split('\n'):
            if '=' not in line: continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                all_parsed.append({'a': a, 'op': op, 'b': b, 'raw_out': rhs.replace(" ", "").strip()})

        if not all_parsed: 
            return {"answer": None, "debug": ["No valid examples found."]}

        ops_grouped = defaultdict(list)
        for ex in all_parsed:
            ops_grouped[ex['op']].append(ex)

        configs_order = ["fwd", "rev_digits", "swap_ops", "swap_rev"]
        full_ops_keys = list(self._get_operations(12, 34, "12", "34").keys())
        
        # --- ФАЗА 1: ОРАКУЛ (Определение лучших правил в фоне) ---
        op_hypotheses = {}
        for op, group in ops_grouped.items():
            valid_hyps = []
            fmt_names = list(self._get_formats(1, op).keys())
            
            for op_config in configs_order:
                for op_name in full_ops_keys:
                    for out_fmt in fmt_names:
                        if out_fmt in ["first_digit", "last_digit"] and len(group) < 3: continue
                        all_pass = True
                        for ex in group:
                            cfg = self._get_operand_configs(ex['a'], ex['b'])[op_config]
                            ops = self._get_operations(*cfg)
                            if op_name not in ops or out_fmt not in self._get_formats(ops[op_name], op) or self._get_formats(ops[op_name], op)[out_fmt] != ex['raw_out']:
                                all_pass = False; break
                        if all_pass:
                            valid_hyps.append((op_config, op_name, out_fmt))
            op_hypotheses[op] = valid_hyps

        # Вывод глобального стиля для Tie-Breaking
        anomaly_config, anomaly_fmt = None, None
        config_penalties = {"fwd": 0, "swap_ops": 20, "rev_digits": 30, "swap_rev": 50}
        
        for op, hyps in op_hypotheses.items():
            if not hyps: continue
            configs_used = set(h[0] for h in hyps)
            if "fwd" not in configs_used:
                anomaly_config = min(configs_used, key=lambda c: config_penalties.get(c, 100))
                break
                
        for op, hyps in op_hypotheses.items():
            if not hyps: continue
            fmts_used = set(h[2] for h in hyps)
            if "raw" not in fmts_used:
                best_h = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
                anomaly_fmt = best_h[2]
                break

        config_counts, fmt_counts = Counter(), Counter()
        for op, hyps in op_hypotheses.items():
            if hyps:
                best_base_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
                config_counts[best_base_hyp[0]] += 1
                fmt_counts[best_base_hyp[2]] += 1

        global_config = anomaly_config or (config_counts.most_common(1)[0][0] if config_counts else "fwd")
        global_fmt = anomaly_fmt or (fmt_counts.most_common(1)[0][0] if fmt_counts else "raw")

        # --- ФАЗА 2: ГЕНЕРАЦИЯ CoT (Визуализация процесса для LLM) ---
        resolved_ops = {}
        used_base_ops = set()
        
        for op, hyps in op_hypotheses.items():
            if not hyps: continue
            best_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op]), global_config, global_fmt))
            resolved_ops[op] = best_hyp
            
            for base_op in ["add", "sub", "mul", "div", "mod", "cat"]:
                if base_op in best_hyp[1]: used_base_ops.add(base_op)
            
            op_config, op_name, out_fmt = best_hyp
            ex = ops_grouped[op][0]
            
            log.append(f"Evaluating operator '{op}' using Example: {ex['a']} {op} {ex['b']} = {ex['raw_out']}")
            
            common_ops = ["add", "sub", "mul", "abs_diff", "neg_abs_diff", "cat", "rev_cat", "rev_sub"]
            
            # Формируем ярусы поиска
            if op_name in common_ops:
                ops_tiers = [("Base", common_ops)]
            else:
                rare_pool = [k for k in self.op_desc.keys() if k not in common_ops and k != op_name]
                rares_to_show = random.sample(rare_pool, min(6, len(rare_pool))) 
                rares_to_show.append(op_name)
                ops_tiers = [("Base", common_ops), ("Extended", rares_to_show)]
            
            configs_to_show = ["fwd", "swap_ops"]
            if "rev_digits" not in configs_to_show: configs_to_show.append("rev_digits")
            if op_config not in configs_to_show: configs_to_show.append(op_config)

            is_found = False

            for tier_name, current_ops in ops_tiers:
                if is_found: break
                log.append(f"Testing {tier_name} mathematical and structural combinations:")
                
                for cfg_k in configs_to_show:
                    if is_found: break
                    
                    cfg_val = self._get_operand_configs(ex['a'], ex['b'])[cfg_k]
                    ops_dict = self._get_operations(*cfg_val)
                    log.append(f" Config: {self.config_desc[cfg_k]} -> Input A: {cfg_val[0]}, Input B: {cfg_val[1]}")
                    
                    for test_op in current_ops: # Исправлено: теперь перебираем current_ops
                        if test_op in ops_dict:
                            test_val = ops_dict[test_op]
                            formats = self._get_formats(test_val, op)
                            
                            if cfg_k == op_config and test_op == op_name:
                                out_str = formats[out_fmt] if out_fmt in formats else test_val
                                log.append(f"  - {test_op} -> {test_val} | format ({self.fmt_desc.get(out_fmt, out_fmt)}) -> {out_str} [MATCH]")
                                is_found = True
                                break 
                            else:
                                log.append(f"  - {test_op} -> {test_val}")
            
            log.append(f"Rule identified for '{op}': {op_config} -> {op_name} -> {out_fmt}")
            
            if len(ops_grouped[op]) > 1:
                log.append("Verifying across remaining examples:")
                for o_ex in ops_grouped[op][1:]:
                    c_val = self._get_operand_configs(o_ex['a'], o_ex['b'])[op_config]
                    o_val = self._get_operations(*c_val)[op_name]
                    ans = self._get_formats(o_val, op)[out_fmt]
                    log.append(f"  {o_ex['a']} {op} {o_ex['b']} -> inputs {c_val[0]}, {c_val[1]} -> result {o_val} -> format {ans}")
            log.append("")

        # --- ФАЗА 3: ЦЕЛЕВОЕ ВЫЧИСЛЕНИЕ ---
        log.append(f"Target calculation: {q_a} {q_op} {q_b}")
        
        if q_op in resolved_ops:
            op_config, op_name, out_fmt = resolved_ops[q_op]
            cfg = self._get_operand_configs(q_a, q_b)[op_config]
            ops = self._get_operations(*cfg)
            val = ops[op_name]
            final_ans = self._get_formats(val, q_op)[out_fmt]
            
            log.append(f"1. Apply config '{op_config}': A = {cfg[0]}, B = {cfg[1]}")
            log.append(f"2. Apply operation '{op_name}': {val}")
            log.append(f"3. Apply format '{out_fmt}': {final_ans}")
            
            return {"answer": final_ans, "debug": log}
            
        else:
            # Логика резервного оператора (fallback)
            log.append(f"Operator '{q_op}' not found in examples. Identifying unused core operation.")
            strict_base_pool = ["add", "sub", "cat", "mul", "div"]
            avail_ops = [op for op in strict_base_pool if op not in used_base_ops]
            
            best_op = avail_ops[0] if avail_ops else strict_base_pool[0]
            log.append(f"Selected fallback operation: '{best_op}' with global style (config: {global_config}, format: {global_fmt})")
            
            cfg = self._get_operand_configs(q_a, q_b)[global_config]
            ops = self._get_operations(*cfg)
            
            if best_op in ops:
                val = ops[best_op]
                final_ans = self._get_formats(val, q_op)[global_fmt]
                
                log.append(f"1. Apply global config '{global_config}': A = {cfg[0]}, B = {cfg[1]}")
                log.append(f"2. Apply operation '{best_op}': {val}")
                log.append(f"3. Apply global format '{global_fmt}': {final_ans}")
                
                return {"answer": final_ans, "debug": log}

        if q_op in ['+', '-', '*', '/', '**', '%']:
            try:
                res = str(int(eval(f"{int(q_a)}{q_op}{int(q_b)}")))
                log.append(f"Evaluated as standard mathematical expression: {res}")
                return {"answer": res, "debug": log}
            except Exception:
                log.append("Standard evaluation failed.")

        return {"answer": None, "debug": log}