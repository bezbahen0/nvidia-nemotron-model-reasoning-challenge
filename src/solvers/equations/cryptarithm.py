import re
import itertools
from typing import Optional, List, Dict, Any

class CryptarithmCSPSolver:
    """
    Ультимативный контекстный солвер v17 (Deterministic High-Speed Hybrid).
    - Возвращено детальное логирование.
    - Жестко зафиксирован приоритет математических правил (убран баг с set).
    - Отключено слепое угадывание Holistic-правил для неизвестных операторов (Zero-Shot только через Math).
    """
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

    # ==========================================
    # ДВИЖОК 1: СТРУКТУРНЫЕ И ВЕКТОРНЫЕ ПРАВИЛА
    # ==========================================
    def _get_holistic_rules(self) -> List[Dict[str, Any]]:
        def mod26(val): return int(((val - 1) % 26) + 1)
        
        rules = []
        def add_rule(name, fn): rules.append({'name': name, 'fn': fn})
        
        add_rule("Concat L+R", lambda L, R, op: L + R)
        add_rule("Concat L+R (Rev)", lambda L, R, op: (L + R)[::-1])
        add_rule("Concat R+L", lambda L, R, op: R + L)
        add_rule("Slice: Drop L head", lambda L, R, op: L[1:] + R if len(L)>0 else None)
        add_rule("Slice: Drop R tail", lambda L, R, op: L + R[:-1] if len(R)>0 else None)

        def zip_math(L, R, op_fn):
            if len(L) != len(R): return None
            return [op_fn(l, r) for l, r in zip(L, R)]

        add_rule("Zip: L + R", lambda L, R, op: zip_math(L, R, lambda l, r: mod26(l + r)))
        add_rule("Zip: L - R", lambda L, R, op: zip_math(L, R, lambda l, r: mod26(l - r)))
        add_rule("Zip: R - L", lambda L, R, op: zip_math(L, R, lambda l, r: mod26(r - l)))

        def zip_op_math(L, R, op, sign):
            if len(L) != len(R): return None
            return [mod26(l + r * sign + op) for l, r in zip(L, R)]

        add_rule("Zip: L + R + Op_Weight", lambda L, R, op: zip_op_math(L, R, op, 1))
        add_rule("Zip: L - R + Op_Weight", lambda L, R, op: zip_op_math(L, R, op, -1))

        def cross_zip(L, R, op_fn):
            if len(L) != 2 or len(R) != 2: return None
            return [op_fn(L[0], R[1]), op_fn(L[1], R[0])]

        add_rule("Cross-Zip: L + R", lambda L, R, op: cross_zip(L, R, lambda l, r: mod26(l + r)))
        add_rule("Cross-Zip: L - R", lambda L, R, op: cross_zip(L, R, lambda l, r: mod26(l - r)))

        return rules

    # ==========================================
    # ДВИЖОК 2: ГЕНЕРАЦИЯ И КЭШИРОВАНИЕ ПРАВИЛ
    # ==========================================
    def _generate_math_rules(self) -> List[Dict[str, Any]]:
        # Порядок словарей важен! Он задает естественный приоритет правил.
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
        
        rules = []
        for op_f_name, op_func in op_formats.items():
            for m_name, math_f in math_ops.items():
                for o_name, out_f in out_formats.items():
                    rule = {
                        'desc': f"{op_f_name} | {m_name} | {o_name}",
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

    # ==========================================
    # ГЛАВНЫЙ МЕТОД: HYBRID PIPELINE
    # ==========================================
    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        log = []
        try:
            equations = []
            eqs_by_op = {}
            
            log.append("--- Parsing ---")
            for idx, line in enumerate(examples_text.strip().split('\n')):
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

            clean_target = re.sub(r'(?i)now,\s*d[e]?termine\s*the\s*result\s*for:\s*', '', target_text).replace(" ", "").strip()
            t_L_str, t_R_str, target_op, _ = self._parse_equation(clean_target + "=X")
            t_L_val = self._decode(t_L_str)
            t_R_val = self._decode(t_R_str)
            target_op_val = self.char_map.get(target_op, 0)
            
            log.append(f"Target: {clean_target} (Op: '{target_op}')")

            resolved_ops = {}
            math_pool = []
            holistic_rules = self._get_holistic_rules()

            # ФАЗА 1: Holistic Filter
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
                    log.append(f"[PHASE 1] Op '{op}' is Holistic: {found_rule['name']}")
                else:
                    # Фильтр для математики: пропускаем только строгие AB_CD форматы
                    for eq in eqs:
                        if len(eq['L_str']) == 2 and len(eq['R_str']) == 2:
                            math_pool.append(eq)

            # ФАЗА 2: Filtered Math CSP
            final_map = {}
            if math_pool:
                log.append(f"[PHASE 2] Starting Math CSP with {len(math_pool)} equations...")
                char_counts = {}
                for eq in math_pool:
                    for ch in eq['L_str'] + eq['R_str'] + eq['Res_str']:
                        char_counts[ch] = char_counts.get(ch, 0) + 1
                
                all_chars_set = set(char_counts.keys()).union(set(t_L_str + t_R_str))
                chars = sorted(list(all_chars_set), key=lambda c: -char_counts.get(c, 0))

                if len(chars) <= 10:
                    math_op_chars = set(eq['op_char'] for eq in math_pool)
                    # Используем LIST вместо SET для сохранения приоритета правил
                    initial_cands = {op: list(range(len(self.math_rules))) for op in math_op_chars}
                    
                    for eq in math_pool:
                        target_len = len(eq['Res_str'])
                        op = eq['op_char']
                        to_remove = set()
                        for r_idx in initial_cands[op]:
                            if target_len not in self.math_rules[r_idx]['possible_lengths']:
                                to_remove.add(r_idx)
                        initial_cands[op] = [r for r in initial_cands[op] if r not in to_remove]

                    def is_math_valid(eq, nmap, rule):
                        vL, vR = rule['op_func'](nmap, eq['L_str'], eq['R_str'])
                        ans_num = rule['math_f'](vL, vR)
                        if ans_num < 0: return False
                        ans_str = rule['out_f'](ans_num)
                        if len(ans_str) != len(eq['Res_str']): return False
                        expected_str = "".join(str(nmap[ch]) for ch in eq['Res_str'])
                        return ans_str == expected_str

                    def solve_csp(char_idx, current_map, used_digits, op_rule_candidates):
                        if char_idx == len(chars):
                            return current_map, op_rule_candidates

                        c = chars[char_idx]
                        for d in range(10):
                            if d in used_digits: continue
                            current_map[c] = d
                            
                            valid = True
                            # Сохраняем списки (list), а не множества
                            new_candidates = {k: list(v) for k, v in op_rule_candidates.items()}
                            
                            for eq in math_pool:
                                if all(ch in current_map for ch in eq['L_str'] + eq['R_str'] + eq['Res_str']):
                                    op = eq['op_char']
                                    possible = [r_idx for r_idx in new_candidates[op] if is_math_valid(eq, current_map, self.math_rules[r_idx])]
                                    if not possible:
                                        valid = False; break
                                    new_candidates[op] = possible
                                    
                            if valid:
                                used_digits.add(d)
                                res_map, final_cands = solve_csp(char_idx + 1, current_map, used_digits, new_candidates)
                                if res_map: return res_map, final_cands
                                used_digits.remove(d)
                                
                        del current_map[c]
                        return None, None

                    final_map, final_cands = solve_csp(0, {}, set(), initial_cands)
                    
                    if final_map:
                        log.append("[PHASE 2] Math CSP Success!")
                        for op, cands in final_cands.items():
                            if cands:
                                resolved_ops[op] = {'type': 'math', 'rule': self.math_rules[cands[0]]}
                    else:
                        log.append("[PHASE 2] Math CSP Failed: No valid 0-9 mapping found.")
                else:
                    log.append(f"[PHASE 2] Math CSP Skipped: Too many unique characters ({len(chars)} > 10).")

            # ФАЗА 3: Zero-Shot / Resolution
            if target_op in resolved_ops:
                r_data = resolved_ops[target_op]
                if r_data['type'] == 'holistic':
                    try:
                        ans = r_data['rule']['fn'](t_L_val, t_R_val, target_op_val)
                        enc = self._encode(ans)
                        if '?' not in enc: 
                            log.append(f"[PHASE 3] Evaluated using known Holistic rule: {r_data['rule']['name']}")
                            return {"answer": enc, "debug": log}
                    except: pass
                elif r_data['type'] == 'math' and final_map:
                    try:
                        m_rule = r_data['rule']
                        vL, vR = m_rule['op_func'](final_map, t_L_str, t_R_str)
                        ans_str = m_rule['out_f'](m_rule['math_f'](vL, vR))
                        rev_map = {v: k for k, v in final_map.items()}
                        if all(int(ch) in rev_map for ch in ans_str):
                            ans_enc = "".join(rev_map[int(ch)] for ch in ans_str)
                            log.append(f"[PHASE 3] Evaluated using known Math rule: {m_rule['desc']}")
                            return {"answer": ans_enc, "debug": log}
                    except: pass

            else:
                log.append("[PHASE 3] Zero-Shot Target. Applying Math Elimination...")
                if final_map:
                    used_m_descs = {r['rule']['desc'] for r in resolved_ops.values() if r['type'] == 'math'}
                    for m_rule in self.math_rules:
                        if m_rule['desc'] not in used_m_descs:
                            try:
                                vL, vR = m_rule['op_func'](final_map, t_L_str, t_R_str)
                                ans_str = m_rule['out_f'](m_rule['math_f'](vL, vR))
                                rev_map = {v: k for k, v in final_map.items()}
                                if all(int(ch) in rev_map for ch in ans_str):
                                    enc = "".join(rev_map[int(ch)] for ch in ans_str)
                                    log.append(f"  [ZERO-SHOT] Matched Math: {m_rule['desc']}")
                                    return {"answer": enc, "debug": log}
                            except: pass

            log.append("[FAIL] Target could not be resolved.")
            return {"answer": None, "debug": log}

        except Exception as e:
            log.append(f"CRITICAL ERROR: {str(e)}")
            return {"answer": None, "debug": log}