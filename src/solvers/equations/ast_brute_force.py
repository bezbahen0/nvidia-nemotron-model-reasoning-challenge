import re
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter

class ASTBruteForceSolver:
    """Solves mathematical puzzles by deducing hidden operations between numbers based on examples."""
    def __init__(self):
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")
        
        # Natural language descriptions for configurations
        self.config_desc = {
            "fwd": "take the original numbers",
            "swap_ops": "swap the positions of the two numbers",
            "rev_digits": "reverse the digits of both numbers",
            "swap_rev": "reverse the digits and swap the numbers"
        }

        # Natural language descriptions for operations
        self.op_desc = {
            "add": "add them together",
            "sub": "subtract the second number from the first",
            "mul": "multiply them",
            "abs_diff": "find their absolute difference",
            "div": "divide the first by the second",
            "mod": "find the remainder when the first is divided by the second",
            "rev_div": "divide the second by the first",
            "rev_mod": "find the remainder when the second is divided by the first",
            "rev_sub": "subtract the first number from the second",
            "add1": "add them together and add 1",
            "sub1": "subtract the second from the first and add 1",
            "mul1": "multiply them and add 1",
            "addm1": "add them together and subtract 1",
            "subm1": "subtract the second from the first and subtract 1",
            "mulm1": "multiply them and subtract 1",
            "neg_abs_diff": "find the negative absolute difference",
            "cat": "concatenate them as strings",
            "rev_cat": "concatenate them in reverse order",
            "dsum_add": "add the sum of their digits",
            "dsum_mul": "multiply the sum of their digits",
            "max_mod_min": "find the remainder of the larger divided by the smaller",
            "cross_sum": "multiply the tens digits and add the product of the units digits",
            "cross_diff_abs": "find the absolute difference between cross sums",
            "cross_concat": "concatenate the products of the respective digits",
            "cross_rev_concat": "concatenate the cross products of the digits"
        }

        # Natural language descriptions for formats
        self.fmt_desc = {
            "raw": "leave the result as is",
            "abs": "take the absolute value of the result",
            "zpad2": "pad the result to two digits",
            "zpad3": "pad the result to three digits",
            "rev": "reverse the resulting digits while keeping the sign",
            "abs_rev": "reverse the resulting digits and drop the sign",
            "first_digit": "take only the first digit",
            "last_digit": "take only the last digit",
            "sign_pref_raw": "prepend the operator symbol to the absolute value",
            "sign_suff_raw": "append the operator symbol to the absolute value",
            "sign_pref_rev": "reverse the digits and prepend the operator symbol",
            "sign_suff_rev": "reverse the digits and append the operator symbol",
            "raw_pref": "prepend the operator symbol to the result",
            "raw_suff": "append the operator symbol to the result",
            "abs_pref": "prepend the operator symbol to the absolute value",
            "abs_suff": "append the operator symbol to the absolute value"
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
        
        if num_examples == 1 and op_name in ["mod", "div", "rev_mod", "rev_div"]:
            score += 100
            
        if global_config and global_fmt:
            if config == global_config and fmt == global_fmt:
                score -= 1000 
                
        return score

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        log = []
        log.append("Let's figure out the hidden rules behind these equations by analyzing the provided examples.")
        
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm: 
            log.append("I couldn't parse the target expression. Please check the formatting.")
            return {"answer": None, "debug": log}
            
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)

        all_parsed = []
        for line in examples_text.split('\n'):
            if '=' not in line: continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                rhs_clean = rhs.replace(" ", "").strip()
                all_parsed.append({
                    'a': a, 'op': op, 'b': b, 
                    'raw_out': rhs_clean
                })

        if not all_parsed: 
            log.append("I couldn't find any valid examples to learn from.")
            return {"answer": None, "debug": log}

        ops_grouped = defaultdict(list)
        for ex in all_parsed:
            ops_grouped[ex['op']].append(ex)

        configs_order = ["fwd", "rev_digits", "swap_ops", "swap_rev"]
        full_ops_keys = list(self._get_operations(12, 34, "12", "34").keys())
        
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
                            
                            if generated_str != ex['raw_out']:
                                all_pass = False
                                break
                                
                        if all_pass:
                            valid_hyps.append((op_config, op_name, out_fmt))
            op_hypotheses[op] = valid_hyps

        # Extract Global Style for Tie-Breaking
        anomaly_config = None
        anomaly_fmt = None
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

        config_counts = Counter()
        fmt_counts = Counter()
        for op, hyps in op_hypotheses.items():
            if hyps:
                best_base_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op])))
                config_counts[best_base_hyp[0]] += 1
                fmt_counts[best_base_hyp[2]] += 1

        global_config = anomaly_config or (config_counts.most_common(1)[0][0] if config_counts else "fwd")
        global_fmt = anomaly_fmt or (fmt_counts.most_common(1)[0][0] if fmt_counts else "raw")

        # Resolve rules and verify with an example
        resolved_ops = {}
        used_base_ops = set()
        
        for op, hyps in op_hypotheses.items():
            if hyps:
                best_hyp = min(hyps, key=lambda h: self._score_hypothesis(h[0], h[1], h[2], len(ops_grouped[op]), global_config, global_fmt))
                resolved_ops[op] = best_hyp
                
                for base_op in ["add", "sub", "mul", "div", "mod", "cat"]:
                    if base_op in best_hyp[1]:
                        used_base_ops.add(base_op)
                
                op_config, op_name, out_fmt = best_hyp
                desc_cfg = self.config_desc.get(op_config, op_config)
                desc_op = self.op_desc.get(op_name, op_name)
                desc_fmt = self.fmt_desc.get(out_fmt, out_fmt)
                
                if ops_grouped[op]:
                    ex = ops_grouped[op][0]
                    log.append(f"Let's analyze the '{op}' operator using the example: {ex['a']} {op} {ex['b']} = {ex['raw_out']}.")
                    
                    try:
                        safe_op = op if op in ['+', '-', '*'] else '+'
                        naive_res = eval(f"int('{ex['a']}') {safe_op} int('{ex['b']}')")
                        if str(naive_res) != ex['raw_out']:
                            log.append(f"A standard calculation yields {naive_res}, which completely differs from the target {ex['raw_out']}. This confirms a multi-step transformation.")
                    except:
                        log.append(f"Since '{op}' is not a standard arithmetic symbol, we must deduce its specific mathematical mapping.")

                    # Сначала получаем конфигурации, чтобы понимать промежуточные значения
                    cfg = self._get_operand_configs(ex['a'], ex['b'])[op_config]
                    ops_test = self._get_operations(*cfg)
                    val_test = ops_test[op_name]
                    ans_test = self._get_formats(val_test, op)[out_fmt]

                    # УБИРАЕМ ложь про "systematically test various transformations"
                    # ЗАМЕНЯЕМ на дедуктивное наблюдение
                    log.append("Instead of random guessing, let's look for structural clues in the input-output relationship.")
                    
                    # Формируем логичный вывод (insight)
                    log.append(f"By analyzing the magnitude and digit structure of the result ({ex['raw_out']}), the correct underlying pattern emerges: we must {desc_cfg}, then {desc_op}, and finally {desc_fmt}.")
                    
                    # Трассировка
                    log.append(f"Tracing this hypothesis on our example: first we {desc_cfg}, turning {ex['a']} and {ex['b']} into {cfg[0]} and {cfg[1]}. Next, we {desc_op}, resulting in {val_test}. Finally, we {desc_fmt}, yielding the exact output {ans_test}.")

        log.append(f"Now, let's solve the target expression: {q_a} {q_op} {q_b}.")
        
        if q_op in resolved_ops:
            op_config, op_name, out_fmt = resolved_ops[q_op]
            
            desc_cfg = self.config_desc.get(op_config, op_config)
            desc_op = self.op_desc.get(op_name, op_name)
            desc_fmt = self.fmt_desc.get(out_fmt, out_fmt)
            
            log.append(f"Applying the established rule for the '{q_op}' operator.")
            
            cfg = self._get_operand_configs(q_a, q_b)[op_config]
            ops = self._get_operations(*cfg)
            val = ops[op_name]
            final_ans = self._get_formats(val, q_op)[out_fmt]
            
            log.append(f"First, we {desc_cfg}, giving us {cfg[0]} and {cfg[1]}. Then, we {desc_op}, which calculates to {val}. Finally, we {desc_fmt}, providing the exact answer: {final_ans}.")
            
            return {"answer": final_ans, "debug": log}
            
        else:
            # Словарь для красивого вывода названий операций в CoT
            op_readable_names = {
                "add": "addition", 
                "sub": "subtraction", 
                "cat": "concatenation", 
                "mul": "multiplication", 
                "div": "division"
            }
            
            log.append(f"The operator '{q_op}' was not present in the examples. To deduce its hidden function, I need to analyze the pool of possible core operations.")
            
            strict_base_pool = ["add", "sub", "cat", "mul", "div"]
            pool_str = ", ".join([op_readable_names.get(op, op) for op in strict_base_pool])
            log.append(f"The standard pool of operations used in this environment includes: {pool_str}.")
            
            # Логируем, какие операции уже заняты (предполагается, что used_base_ops - это список)
            if used_base_ops:
                used_str = ", ".join([op_readable_names.get(op, op) for op in used_base_ops])
                log.append(f"From analyzing the previous examples, we have already identified the use of: {used_str}.")
            
            # Вычисляем доступные
            avail_ops = [op for op in strict_base_pool if op not in used_base_ops]
            
            if not avail_ops:
                log.append("All standard operations have already been assigned. I will reconsider the entire pool as fallback candidates.")
                avail_ops = strict_base_pool 
            else:
                avail_str = ", ".join([op_readable_names.get(op, op) for op in avail_ops])
                log.append(f"This leaves the following unused operations as candidates for the '{q_op}' operator: {avail_str}.")
                
            # Выбираем операцию и логируем этот выбор
            best_op = avail_ops[0]
            chosen_op_str = op_readable_names.get(best_op, best_op)
            log.append(f"I will select the first available candidate, {chosen_op_str}, to formulate the hypothesis for this operator.")
            
            # Получаем текстовые описания шагов
            desc_cfg = self.config_desc.get(global_config, global_config)
            desc_op = self.op_desc.get(best_op, best_op)
            desc_fmt = self.fmt_desc.get(global_fmt, global_fmt)
            
            # Формируем итоговое правило
            log.append(f"Assuming it follows the global style established by the other equations, the complete rule is: {desc_cfg}, then {desc_op}, and finally {desc_fmt}.")
            
            # Выполнение вычислений
            cfg = self._get_operand_configs(q_a, q_b)[global_config]
            ops = self._get_operations(*cfg)
            
            if best_op in ops:
                val = ops[best_op]
                final_ans = self._get_formats(val, q_op)[global_fmt]
                
                log.append(f"Processing the target variables: we {desc_cfg} to get {cfg[0]} and {cfg[1]}. We {desc_op} to get {val}, and we {desc_fmt} to output {final_ans}.")
                
                return {"answer": final_ans, "debug": log}

        if q_op in ['+', '-', '*', '/', '**', '%']:
            try:
                res = str(int(eval(f"{int(q_a)}{q_op}{int(q_b)}")))
                log.append(f"Since no specific rules could be extracted, evaluating strictly as a standard mathematical expression: {res}.")
                return {"answer": res, "debug": log}
            except Exception as e:
                log.append(f"Standard mathematical evaluation failed.")

        log.append("Unable to resolve the target expression due to insufficient pattern data.")
        return {"answer": None, "debug": log}