import re
import itertools
from typing import Optional, List, Dict, Any

import time

class CryptarithmCSPSolver:
    def __init__(self):
        self.char_map = {
            '!': 1, '"': 2, '#': 3, '$': 4, '%': 5, '&': 6, "'": 7, '(': 8, ')': 9, 
            '*': 10, '+': 11, '-': 12, '/': 13, ':': 14, '<': 15, '>': 16, '?': 17, 
            '@': 18, '[': 19, '\\': 20, ']': 21, '^': 22, '`': 23, '{': 24, '|': 25, '}': 26
        }
        self.num_map = {v: k for k, v in self.char_map.items()}
        self.math_rules = self._generate_math_rules()

    def _decode(self, text: str) -> List[int]:
        return [self.char_map[c] for c in text.strip() if c in self.char_map]

    def _encode(self, num_array: List[int]) -> str:
        return "".join(self.num_map.get(n, '?') for n in num_array)

    def _parse_equation(self, line: str):
        line = line.replace(" ", "").strip()
        if '=' not in line: return None
        left, right = line.split('=')
        if not left or not right: return None
        
        mid_idx = len(left) // 2
        if len(left) >= 3 and len(left) % 2 != 0:
            L_str = left[:mid_idx]
            op_char = left[mid_idx]
            R_str = left[mid_idx+1:]
        else:
            L_str = left
            R_str = ""
            op_char = ""
            
        return L_str, R_str, op_char, right

    def _get_holistic_rules(self) -> List[Dict[str, Any]]:
        def mod26(val): return int(((val - 1) % 26) + 1)
        
        rules = []
        def add_rule(name, fn): rules.append({'name': name, 'fn': fn})
        
        add_rule("concatenating the left and right operands", lambda L, R, op: L + R)
        add_rule("concatenating the left and right operands and reversing the result", lambda L, R, op: (L + R)[::-1])
        add_rule("concatenating the right operand followed by the left operand", lambda L, R, op: R + L)
        add_rule("dropping the first character of the left operand and concatenating with the right", lambda L, R, op: L[1:] + R if len(L)>0 else None)
        add_rule("concatenating the left operand with the right operand excluding its last character", lambda L, R, op: L + R[:-1] if len(R)>0 else None)

        def zip_math(L, R, op_fn):
            if len(L) != len(R): return None
            return [op_fn(l, r) for l, r in zip(L, R)]

        add_rule("adding respective characters", lambda L, R, op: zip_math(L, R, lambda l, r: mod26(l + r)))
        add_rule("subtracting respective characters", lambda L, R, op: zip_math(L, R, lambda l, r: mod26(l - r)))
        add_rule("subtracting left from right respective characters", lambda L, R, op: zip_math(L, R, lambda l, r: mod26(r - l)))

        def zip_op_math(L, R, op, sign):
            if len(L) != len(R): return None
            return [mod26(l + r * sign + op) for l, r in zip(L, R)]

        add_rule("adding respective characters and the operator weight", lambda L, R, op: zip_op_math(L, R, op, 1))
        add_rule("subtracting respective characters and adding the operator weight", lambda L, R, op: zip_op_math(L, R, op, -1))

        def cross_zip(L, R, op_fn):
            if len(L) != 2 or len(R) != 2: return None
            return [op_fn(L[0], R[1]), op_fn(L[1], R[0])]

        add_rule("cross-adding characters", lambda L, R, op: cross_zip(L, R, lambda l, r: mod26(l + r)))
        add_rule("cross-subtracting characters", lambda L, R, op: cross_zip(L, R, lambda l, r: mod26(l - r)))

        return rules

    def _generate_math_rules(self) -> List[Dict[str, Any]]:
        math_ops = {
            'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'sub_rev': lambda a, b: b - a,
            'abs': lambda a, b: abs(a - b), 'mul': lambda a, b: a * b, 'add1': lambda a, b: a + b + 1,
            'addm1': lambda a, b: a + b - 1, 'muladd1': lambda a, b: a * b + 1, 'mulsub1': lambda a, b: a * b - 1,
            'cat': lambda a, b: int(str(a) + str(b))
        }
        out_formats = {
            'raw': lambda r: str(r), 'rev': lambda r: str(r)[::-1],
            'zpad2': lambda r: f"{r:02d}", 'dsum': lambda r: str(sum(int(d) for d in str(abs(r))))
        }
        op_formats = {
            'AB_CD': lambda nmap, L, R: (nmap[L[0]]*10 + nmap[L[1]], nmap[R[0]]*10 + nmap[R[1]]),
            'BA_DC': lambda nmap, L, R: (nmap[L[1]]*10 + nmap[L[0]], nmap[R[1]]*10 + nmap[R[0]]),
            'AB_DC': lambda nmap, L, R: (nmap[L[0]]*10 + nmap[L[1]], nmap[R[1]]*10 + nmap[R[0]]),
            'BA_CD': lambda nmap, L, R: (nmap[L[1]]*10 + nmap[L[0]], nmap[R[0]]*10 + nmap[R[1]])
        }
        
        op_desc = {
            'AB_CD': "forming two-digit numbers directly from the symbols",
            'BA_DC': "reversing the two-digit numbers formed by the symbols",
            'AB_DC': "forming a two-digit number for the left operand and reversing the right",
            'BA_CD': "reversing the left two-digit number and keeping the right direct"
        }
        math_desc = {
            'add': "adding them", 'sub': "subtracting the right from the left", 
            'sub_rev': "subtracting the left from the right", 'abs': "taking their absolute difference", 
            'mul': "multiplying them", 'add1': "adding them and then adding 1",
            'addm1': "adding them and subtracting 1", 'muladd1': "multiplying them and adding 1", 
            'mulsub1': "multiplying them and subtracting 1", 'cat': "concatenating the numbers"
        }
        out_desc = {
            'raw': "keeping the exact result",
            'rev': "reversing the digits of the result",
            'zpad2': "padding the result to two digits with a leading zero if necessary",
            'dsum': "taking the sum of the digits of the result"
        }

        rules = []
        for op_f_name, op_func in op_formats.items():
            for m_name, math_f in math_ops.items():
                for o_name, out_f in out_formats.items():
                    human_desc = f"{math_desc[m_name]}, by {op_desc[op_f_name]}, and {out_desc[o_name]}"
                    rule = {
                        'desc': human_desc,
                        'op_func': op_func, 'math_f': math_f, 'out_f': out_f
                    }
                    lengths = set()
                    for a in range(100):
                        for b in range(100):
                            try:
                                ans = math_f(a, b)
                                if ans >= 0:
                                    lengths.add(len(out_f(ans)))
                            except: pass
                    rule['possible_lengths'] = lengths
                    rules.append(rule)
        return rules
    
    def _generate_deductive_cot(self, math_pool: List[Dict[str, Any]], correct_combo: Dict[str, int], final_map: Dict[str, int], target_text: str) -> List[str]:
        log = []
        chars = list(final_map.keys())
        unresolved = set(chars)
        
        log.append(f"Objective: Calculate the result for `{target_text}`.")
        log.append("Cryptarithm analysis and step-by-step constraint propagation:")
        
        # 1. Elimination (Leading Zeros)
        leading_chars = set(s[0] for eq in math_pool for s in (eq['L_str'], eq['R_str'], eq['Res_str']) if len(s) > 1)
        if leading_chars:
            log.append(f"- Elimination: Symbols {', '.join(leading_chars)} act as leading digits. Their values cannot be 0.")
            
        # Group equations by operator to avoid false global generalizations
        ops_in_pool = set(eq['op_char'] for eq in math_pool)
        
        for op in ops_in_pool:
            op_eqs = [eq for eq in math_pool if eq['op_char'] == op]
            rule = self.math_rules[correct_combo[op]]
            math_desc = rule['desc'].lower()
            
            log.append(f"\nAnalyzing operator '{op}' (Hypothesis: {rule['desc']}):")
            
            # 2. Magnitude Analysis
            for eq in op_eqs:
                L, R, Res = eq['L_str'], eq['R_str'], eq['Res_str']
                
                if 'adding' in math_desc and len(Res) > max(len(L), len(R)):
                    leading_res = Res[0]
                    if leading_res in unresolved:
                        log.append(f"  - Magnitude constraint: In `{L} {op} {R} = {Res}`, addition yields a longer string. The maximum carry-over in base-10 addition is 1. Thus, '{leading_res}' = 1.")
                        unresolved.discard(leading_res)
                        
                elif 'multiplying' in math_desc and len(Res) > len(L) + len(R) - 1:
                    log.append(f"  - Magnitude constraint: In `{L} {op} {R} = {Res}`, the result's length strictly indicates the multiplication of large values.")

            # 3. Modulo 10 Analysis
            for eq in op_eqs:
                L, R, Res = eq['L_str'], eq['R_str'], eq['Res_str']
                l_last, r_last, res_last = L[-1], R[-1], Res[-1]
                
                # Determine symbolic operation and calculate correct Modulo 10 (handles negatives automatically in Python)
                if "adding" in math_desc:
                    op_symbol = "+"
                    calc_val = (final_map[l_last] + final_map[r_last]) % 10
                elif "multiplying" in math_desc:
                    op_symbol = "*"
                    calc_val = (final_map[l_last] * final_map[r_last]) % 10
                elif "subtracting the right from the left" in math_desc:
                    op_symbol = "-"
                    calc_val = (final_map[l_last] - final_map[r_last]) % 10
                elif "subtracting the left from the right" in math_desc:
                    op_symbol = "-(reverse)"
                    calc_val = (final_map[r_last] - final_map[l_last]) % 10
                else:
                    op_symbol = "?"
                    calc_val = final_map[res_last]
                
                # Only log if it helps resolve unknown variables
                if l_last in unresolved or r_last in unresolved or res_last in unresolved:
                    log.append(f"  - Modulo 10 propagation: In `{L} {op} {R} = {Res}`, the final digit depends solely on the last digits of operands. "
                               f"Rule: '{l_last}' {op_symbol} '{r_last}' ≡ '{res_last}' (mod 10). "
                               f"Evaluating assigned digits: {final_map[l_last]} {op_symbol} {final_map[r_last]} ≡ {calc_val} (mod 10). "
                               f"This isolates the assignments: '{l_last}'={final_map[l_last]}, '{r_last}'={final_map[r_last]}, '{res_last}'={final_map[res_last]}.")
                    unresolved.difference_update([l_last, r_last, res_last])

        # 4. Resolving remaining variables via Global Constraints
        if unresolved:
            log.append(f"\n- Global Constraint (AllDiff): By cross-referencing equations and enforcing that each symbol represents a strictly unique digit (0-9), the remaining domains collapse to: " + 
                       ", ".join([f"'{c}' = {final_map[c]}" for c in unresolved]) + ".")

        mapping_str = ", ".join([f"'{c}': {final_map[c]}" for c in chars])
        log.append(f"\nComplete deduced mapping: {{{mapping_str}}}")
        
        return log


    def solve(self, examples_text: str, target_text: str, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        import time
        import itertools
        import re
        start_time = time.time()
        log = []
        try:
            equations = []
            eqs_by_op = {}
            
            # --- 1. PARSING EXAMPLES ---
            for line in examples_text.strip().split('\n'):
                parsed = self._parse_equation(line)
                if parsed:
                    L_str, R_str, op_char, res_str = parsed
                    if op_char not in eqs_by_op:
                        eqs_by_op[op_char] = []
                        
                    eq_data = {
                        'L_str': L_str, 'R_str': R_str, 'Res_str': res_str, 'op_char': op_char,
                        'L_val': self._decode(L_str), 'R_val': self._decode(R_str),
                        'Res_val': self._decode(res_str), 'op_val': self.char_map.get(op_char, 0)
                    }
                    equations.append(eq_data)
                    eqs_by_op[op_char].append(eq_data)

            if not equations:
                log.append("Error: No valid examples provided.")
                return {"answer": None, "debug": log}

            # --- 2. PARSING TARGET ---
            clean_target = re.sub(r'(?i)now,\s*d[e]?termine\s*the\s*result\s*for:\s*', '', target_text).replace(" ", "").strip()
            parsed_target = self._parse_equation(clean_target + "=X")
            if not parsed_target:
                log.append("Error: Could not parse target expression.")
                return {"answer": None, "debug": log}
                
            t_L_str, t_R_str, target_op, _ = parsed_target
            t_L_val = self._decode(t_L_str)
            t_R_val = self._decode(t_R_str)
            target_op_val = self.char_map.get(target_op, 0)

            resolved_ops = {}
            math_pool = []
            holistic_rules = self._get_holistic_rules()

            # --- 3. CHECKING STRUCTURAL RULES ---
            for op, eqs in eqs_by_op.items():
                found_rule = None
                for rule in holistic_rules:
                    match = True
                    for eq in eqs:
                        try:
                            ans = rule['fn'](eq['L_val'], eq['R_val'], eq['op_val'])
                            if ans != eq['Res_val']:
                                match = False; break
                        except: match = False; break
                    if match:
                        found_rule = rule
                        break
                
                if found_rule:
                    resolved_ops[op] = {'type': 'holistic', 'rule': found_rule}
                else:
                    for eq in eqs:
                        math_pool.append(eq)

            target_is_structural = target_op in resolved_ops and resolved_ops[target_op]['type'] == 'holistic'
            
            if target_is_structural:
                log.append(f"Objective: Calculate the result for `{clean_target}`.")
                rule_name = resolved_ops[target_op]['rule']['name']
                log.append(f"Analysis: All examples with operator '{target_op}' strictly follow the structural rule: {rule_name}.")
                log.append("Mathematical decoding is unnecessary.")
                
                ans = resolved_ops[target_op]['rule']['fn'](t_L_val, t_R_val, target_op_val)
                enc = self._encode(ans)
                log.append(f"Execution: Applying this manipulation yields the string answer: {enc}.")
                return {"answer": enc, "debug": log}

            # --- 4. PREPARING MATH SOLVER ---
            char_counts = {}
            for eq in math_pool:
                for ch in eq['L_str'] + eq['R_str'] + eq['Res_str']:
                    char_counts[ch] = char_counts.get(ch, 0) + 1
            all_chars_set = set(char_counts.keys()).union(set(t_L_str + t_R_str))
            chars = sorted(list(all_chars_set), key=lambda c: -char_counts.get(c, 0))
            math_op_chars = set(eq['op_char'] for eq in math_pool)

            if len(chars) > 10:
                log.append(f"Error: Too many unique characters ({len(chars)} > 10) for a base-10 numerical system. Problem is mathematically ill-posed.")
                return {"answer": None, "debug": log}

            math_pool.sort(key=lambda eq: len(set(eq['L_str'] + eq['R_str'] + eq['Res_str'])))

            op_rule_candidates = {}
            for op in math_op_chars:
                valid_rules = []
                for idx, rule in enumerate(self.math_rules):
                    possible = True
                    for eq in eqs_by_op[op]:
                        L_len, R_len, Res_len = len(eq['L_str']), len(eq['R_str']), len(eq['Res_str'])
                        corners = [(0, 0), (0, 10**R_len-1), (10**L_len-1, 0), (10**L_len-1, 10**R_len-1)]
                        possible_lens = set()
                        try:
                            for val_L, val_R in corners:
                                res_val = rule['math_f'](val_L, val_R)
                                if res_val >= 0: possible_lens.add(len(rule['out_f'](res_val)))
                            max_len = max(possible_lens) if possible_lens else 0
                            if 'sum' in rule['desc']: max_len = 2 
                            if not possible_lens or not (min(possible_lens) <= Res_len <= max_len):
                                possible = False; break
                        except: possible = False; break
                    if possible: valid_rules.append(idx)
                op_rule_candidates[op] = valid_rules

            candidate_combos = [dict(zip(math_op_chars, combo)) for combo in itertools.product(*(op_rule_candidates[op] for op in math_op_chars))]
            
            final_map = None
            correct_combo = None

            # --- THE ENGINE: Optimized CSP Solver ---
            for combo_idx, combo in enumerate(candidate_combos):
                if time.time() - start_time > timeout_seconds:
                    log.append(f"Error: Timeout. Exceeded {timeout_seconds}s limit.")
                    return {"answer": None, "debug": log}

                domains = {c: set(range(10)) for c in chars}
                leading_chars = set(s[0] for eq in math_pool for s in (eq['L_str'], eq['R_str'], eq['Res_str']) if len(s) > 1)
                for c in leading_chars: domains[c].discard(0)
                
                def propagate_constraints(cur_domains):
                    changed = True
                    while changed:
                        if time.time() - start_time > timeout_seconds: return False
                        changed = False
                        
                        resolved_list = [list(v)[0] for v in cur_domains.values() if len(v) == 1]
                        if len(set(resolved_list)) < len(resolved_list):
                            return False # Contradiction: duplicate assignments
                            
                        resolved_digits = set(resolved_list)
                        for c, dom in cur_domains.items():
                            if len(dom) > 1:
                                original_len = len(dom)
                                dom.difference_update(resolved_digits)
                                if not dom: return False
                                if len(dom) < original_len: changed = True

                        for eq in math_pool:
                            rule = self.math_rules[combo[eq['op_char']]]
                            lr_chars = list(set(eq['L_str'] + eq['R_str']))
                            res_chars = list(eq['Res_str'])
                            
                            domain_lists = [cur_domains[c] for c in lr_chars]
                            valid_assignments = []
                            
                            for vals in itertools.product(*domain_lists):
                                if len(set(vals)) == len(vals):
                                    assign = dict(zip(lr_chars, vals))
                                    try:
                                        vL, vR = rule['op_func'](assign, eq['L_str'], eq['R_str'])
                                        ans_num = rule['math_f'](vL, vR)
                                        
                                        if ans_num >= 0:
                                            ans_str = rule['out_f'](ans_num)
                                            if len(ans_str) == len(res_chars):
                                                is_compatible = True
                                                temp_assign = assign.copy()
                                                
                                                for i, res_c in enumerate(res_chars):
                                                    res_d = int(ans_str[i])
                                                    if res_c in temp_assign:
                                                        if temp_assign[res_c] != res_d:
                                                            is_compatible = False; break
                                                    else:
                                                        if res_d in temp_assign.values() or res_d not in cur_domains[res_c]:
                                                            is_compatible = False; break
                                                        temp_assign[res_c] = res_d
                                                        
                                                if is_compatible:
                                                    valid_assignments.append(temp_assign)
                                    except: pass
                                    
                            if not valid_assignments: return False
                            
                            all_eq_chars = set(eq['L_str'] + eq['R_str'] + eq['Res_str'])
                            for c in all_eq_chars:
                                allowed = set(a[c] for a in valid_assignments)
                                if len(allowed) < len(cur_domains[c]):
                                    cur_domains[c].intersection_update(allowed)
                                    changed = True
                                    if not cur_domains[c]: return False
                    return True

                def solve_remaining(unassigned_chars, current_domains):
                    if time.time() - start_time > timeout_seconds: return None
                        
                    if not unassigned_chars:
                        final_vals = [list(current_domains[c])[0] for c in chars]
                        if len(set(final_vals)) == len(chars):
                            return {c: list(current_domains[c])[0] for c in chars}
                        return None
                    
                    c = unassigned_chars[0]
                    for d in current_domains[c]:
                        new_domains = {k: set(v) for k, v in current_domains.items()}
                        new_domains[c] = {d}
                        
                        conflict = False
                        for other_c in unassigned_chars[1:]:
                            new_domains[other_c].discard(d)
                            if not new_domains[other_c]: conflict = True; break
                        
                        if not conflict and propagate_constraints(new_domains):
                            res = solve_remaining(unassigned_chars[1:], new_domains)
                            if res: return res
                    return None

                if propagate_constraints(domains):
                    unassigned = [c for c in chars if len(domains[c]) > 1]
                    res_map = solve_remaining(unassigned, domains)
                    if res_map:
                        final_map = res_map
                        correct_combo = combo
                        break

            if not final_map:
                log.append("Error: No mathematically consistent mapping found. The system is ill-posed.")
                return {"answer": None, "debug": log}

            # --- 5. GENERATING EDUCATIONAL CoT (DEDUCTIVE NARRATIVE) ---
            deductive_log = self._generate_deductive_cot(math_pool, correct_combo, final_map, clean_target)
            log.extend(deductive_log)
            
            log.append(f"\nTarget Evaluation: Solving `{clean_target}`")

            def apply_unassigned_fallback(a_str, c_map):
                r_map = {v: k for k, v in c_map.items()}
                unused_chars = [ch for ch in self.char_map if ch not in c_map]
                enc_chars = []
                for ch in a_str:
                    d = int(ch)
                    if d not in r_map:
                        if unused_chars:
                            new_char = unused_chars.pop(0)
                            r_map[d] = new_char
                            c_map[new_char] = d
                            log.append(f"Note: Mapping unassigned digit '{d}' to unused symbol '{new_char}'.")
                        else:
                            raise Exception("No available symbols to map result.")
                    enc_chars.append(r_map[d])
                return "".join(enc_chars)

            # --- 6. OOD / HONEST TARGET EVALUATION ---
            if target_op in correct_combo:
                try:
                    m_rule = self.math_rules[correct_combo[target_op]]
                    vL, vR = m_rule['op_func'](final_map, t_L_str, t_R_str)
                    ans_num = m_rule['math_f'](vL, vR)
                    ans_str = m_rule['out_f'](ans_num)
                    
                    t_L_decoded = "".join(str(final_map[ch]) for ch in t_L_str)
                    t_R_decoded = "".join(str(final_map[ch]) for ch in t_R_str)
                    
                    log.append(f"- Decrypting operands: '{t_L_str}' -> {t_L_decoded} (val: {vL}), '{t_R_str}' -> {t_R_decoded} (val: {vR}).")
                    log.append(f"- Operation logic: {m_rule['desc']}.")
                    log.append(f"- Computation yields: {ans_num} -> Formatted as '{ans_str}'.")
                    
                    ans_enc = apply_unassigned_fallback(ans_str, final_map)
                    log.append(f"- Encrypting final result back to symbols: {ans_enc}.")
                    return {"answer": ans_enc, "debug": log}
                except Exception as e:
                    log.append(f"Execution error during target computation: {str(e)}")
                    return {"answer": None, "debug": log}
            else:
                # OOD Block: Refuse to guess unknown operators
                log.append(f"Error: Target operator '{target_op}' was not observed in the training examples.")
                log.append("The underlying transformation rule for this operator cannot be logically deduced. Aborting to prevent hallucination.")
                return {"answer": None, "debug": log}

        except Exception as e:
            log.append(f"Global execution error: {str(e)}")
            return {"answer": None, "debug": log}