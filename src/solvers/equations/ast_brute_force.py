import re
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter

class ASTBruteForceSolver:
    """берет известные цифры и пытается понять, по какому правилу они взаимодействуют (Precise Tie-Breaker Edition)"""
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
        
        # 1. Sign Substitution (Символ заменяет минус, строго дешевле в штрафах)
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

        # 2. Dumb Append (Символ клеится всегда, получает больший штраф)
        if op_char:
            formats["raw_pref"] = op_char + sval
            formats["raw_suff"] = sval + op_char
            formats["abs_pref"] = op_char + s_abs
            formats["abs_suff"] = s_abs + op_char
            
        return formats

    def _score_hypothesis(self, config: str, op_name: str, fmt: str, num_examples: int, global_config: str = None, global_fmt: str = None) -> int:
        score = 0
        
        # 1. Штрафы конфигурации
        if config == "swap_ops": score += 20
        elif config == "rev_digits": score += 30
        elif config == "swap_rev": score += 50
        
        # 2. Точная лесенка штрафов форматов для предотвращения ничьих!
        if fmt == "abs": score += 5
        elif fmt.startswith("sign_"): score += 7      # Умный минус всегда бьет тупую склейку
        elif fmt.startswith("raw_") or fmt.startswith("abs_"): score += 9
        elif fmt in ["zpad2", "zpad3"]: score += 15
        elif fmt == "rev": score += 25              # Сохраняет знак, бьет abs_rev
        elif fmt == "abs_rev": score += 27          # Уничтожает знак, берется только от безысходности
        elif "digit" in fmt: score += 40
        
        # 3. Штрафы математики
        op_penalties = {
            "add": 0, "sub": 1, "abs_diff": 2, "mul": 3, "cat": 4,
            "div": 10, "mod": 11, "rev_sub": 12, "rev_cat": 13,
            "add1": 20, "sub1": 21, "mul1": 22, "addm1": 23, "subm1": 24, "mulm1": 25,
            "neg_abs_diff": 26, "max_mod_min": 27,
            "dsum_add": 30, "dsum_mul": 31,
            "cross_sum": 40, "cross_diff_abs": 41, "cross_concat": 42, "cross_rev_concat": 43
        }
        score += op_penalties.get(op_name, 50)
        
        # 4. Защита от оверфиттинга на 1 примере
        if num_examples == 1 and op_name in ["mod", "div", "rev_mod", "rev_div"]:
            score += 100
            
        # 5. Тай-брейкер: Бонус за следование глобальному стилю
        if global_config and global_fmt:
            if config == global_config and fmt == global_fmt:
                score -= 1000 
                
        return score

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        log = []
        log.append("--- AST Brute Force: Precise Tie-Breaker Edition ---")
        
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm: 
            log.append("Error: Failed to parse target text.")
            return {"answer": None, "debug": log}
            
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)
        log.append(f"Target: {q_a} '{q_op}' {q_b}")

        all_parsed = []
        for line in examples_text.split('\n'):
            if '=' not in line: continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                rhs_clean = rhs.replace(" ", "").strip()
                all_parsed.append({
                    'a': a, 'op': b, 'b': b, # Bug alert! Note: Intentionally kept standard parameter flow but wait...
                    'a': a, 'op': op, 'b': b, 
                    'raw_out': rhs_clean
                })

        if not all_parsed: 
            log.append("Error: No valid examples parsed.")
            return {"answer": None, "debug": log}

        ops_grouped = defaultdict(list)
        for ex in all_parsed:
            ops_grouped[ex['op']].append(ex)

        configs_order = ["fwd", "rev_digits", "swap_ops", "swap_rev"]
        full_ops_keys = list(self._get_operations(12, 34, "12", "34").keys())
        
        log.append("--- Phase 1: Strict Hypothesis Generation ---")
        op_hypotheses = {}
        for op, group in ops_grouped.items():
            valid_hyps = []
            fmt_names = list(self._get_formats(1, op).keys())
            
            for op_config in configs_order:
                for op_name in full_ops_keys:
                    for out_fmt in fmt_names:
                        if out_fmt in ["first_digit", "last_digit"] and len(group) < 3:
                            continue
                            
                        all_pass = True
                        for ex in group:
                            cfg = self._get_operand_configs(ex['a'], ex['b'])[op_config]
                            ops = self._get_operations(*cfg)
                            
                            if op_name not in ops:
                                all_pass = False; break
                                
                            fmts = self._get_formats(ops[op_name], op)
                            if out_fmt not in fmts:
                                all_pass = False; break
                                
                            generated_str = fmts[out_fmt]
                            
                            # Только точное строковое совпадение
                            if generated_str != ex['raw_out']:
                                all_pass = False
                                break
                                
                        if all_pass:
                            valid_hyps.append((op_config, op_name, out_fmt))
            op_hypotheses[op] = valid_hyps

        log.append("--- Phase 2: Anomaly-Based Global Style Extraction ---")
        anomaly_config = None
        anomaly_fmt = None
        
        config_penalties = {"fwd": 0, "swap_ops": 20, "rev_digits": 30, "swap_rev": 50}
        
        for op, hyps in op_hypotheses.items():
            if not hyps: continue
            configs_used = set(h[0] for h in hyps)
            if "fwd" not in configs_used:
                anomaly_config = min(configs_used, key=lambda c: config_penalties.get(c, 100))
                log.append(f"Anomaly detected! Operator '{op}' forces config: {anomaly_config}")
                break
                
        for op, hyps in op_hypotheses.items():
            if not hyps: continue
            fmts_used = set(h[2] for h in hyps)
            if "raw" not in fmts_used:
                best_h = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
                anomaly_fmt = best_h[2]
                log.append(f"Anomaly detected! Operator '{op}' forces format: {anomaly_fmt}")
                break

        config_counts = Counter()
        fmt_counts = Counter()
        for op, hyps in op_hypotheses.items():
            if hyps:
                best_base_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
                config_counts[best_base_hyp[0]] += 1
                fmt_counts[best_base_hyp[2]] += 1

        global_config = anomaly_config or (config_counts.most_common(1)[0][0] if config_counts else "fwd")
        global_fmt = anomaly_fmt or (fmt_counts.most_common(1)[0][0] if fmt_counts else "raw")
        log.append(f"Final Global Style -> config: '{global_config}', format: '{global_fmt}'")

        log.append("--- Phase 3: Final Resolution with Tie-Breaker ---")
        resolved_ops = {}
        used_base_ops = set()
        
        for op, hyps in op_hypotheses.items():
            if hyps:
                best_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op]), global_config, global_fmt))
                resolved_ops[op] = best_hyp
                
                for base_op in ["add", "sub", "mul", "div", "mod", "cat"]:
                    if base_op in best_hyp[1]:
                        used_base_ops.add(base_op)
                        
                log.append(f"Operator '{op}' resolved -> config: '{best_hyp[0]}', op: '{best_hyp[1]}', format: '{best_hyp[2]}'")

        log.append("--- Phase 4: Evaluating Target ---")
        if q_op in resolved_ops:
            op_config, op_name, out_fmt = resolved_ops[q_op]
            log.append(f"Target operator '{q_op}' is KNOWN. Applying local rule: {op_name}")
            
            cfg = self._get_operand_configs(q_a, q_b)[op_config]
            ops = self._get_operations(*cfg)
            val = ops[op_name]
            final_ans = self._get_formats(val, q_op)[out_fmt]
            
            log.append(f"Success! Result: {final_ans}")
            return {"answer": final_ans, "debug": log}
            
        else:
            log.append(f"Target operator '{q_op}' is UNKNOWN (Zero-Shot).")
            
            # Измененный порядок: cat теперь идет раньше mul, так как чаще используется для странных символов
            strict_base_pool = ["add", "sub", "cat", "mul", "div"]
            avail_ops = [op for op in strict_base_pool if op not in used_base_ops]
            
            if not avail_ops:
                log.append("All base ops used! Falling back to raw strict pool.")
                avail_ops = strict_base_pool 
                
            best_op = avail_ops[0]
            log.append(f"Applying Fuzzy Mutual Exclusivity. Selected free op: '{best_op}' with Global Style ({global_config}, {global_fmt})")
            
            cfg = self._get_operand_configs(q_a, q_b)[global_config]
            ops = self._get_operations(*cfg)
            
            if best_op in ops:
                val = ops[best_op]
                final_ans = self._get_formats(val, q_op)[global_fmt]
                
                log.append(f"Success (Zero-Shot)! Result: {final_ans}")
                return {"answer": final_ans, "debug": log}

        log.append("--- Phase 5: Fallback Eval ---")
        if q_op in ['+', '-', '*', '/', '**', '%']:
            try:
                res = str(int(eval(f"{int(q_a)}{q_op}{int(q_b)}")))
                log.append(f"Fallback eval succeeded: {res}")
                return {"answer": res, "debug": log}
            except Exception as e:
                log.append(f"Fallback eval failed: {e}")

        log.append("Failed to resolve target.")
        return {"answer": None, "debug": log}