import re
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class FoundOp:
    op_name: str
    rev_ops: bool
    rev_res: bool
    fmt: str
    op_char: str

class UnifiedEquationsSolver:
    """Оркестратор и решатель для численных и криптарифметических SFT-задач с генерацией CoT."""

    def __init__(self):
        self._numeric_re = re.compile(r"^(-?\d+)\s*([^\d\s]+)\s*(-?\d+)$")

    def _rev(self, s: str) -> str:
        s_str = str(s)
        if s_str.startswith("-"):
            return "-" + s_str[1:][::-1]
        return s_str[::-1]

    def extract_answer(self, cot_text: str) -> str:
        if not isinstance(cot_text, str) or "Error" in cot_text:
            return "nan"
        matches = re.findall(r"\\boxed\{([^}]+)\}", cot_text)
        return matches[-1].strip() if matches else "nan"

    def generate_cot(self, prompt: str) -> str:
        prompt = str(prompt)
        query_match = re.search(r"determine the result for:\s*([^\n]+)", prompt, re.IGNORECASE)
        if query_match:
            query_str = query_match.group(1).strip()
        else:
            lines = [line.strip() for line in prompt.split('\n') if line.strip()]
            if lines and '=' not in lines[-1]:
                query_str = lines[-1].replace('Question:', '').strip()
            else:
                return "Parse Error: Target not found."
            
        if not query_str:
            return "Parse Error: Empty target."
        
        query_clean = query_str.replace(" ", "")
        if re.search(r'\d', query_clean) and self._numeric_re.fullmatch(query_clean):
            return self._solve_numeric(prompt, query_clean)
        else:
            return self._solve_symbolic(prompt, query_clean)

    # ==========================================
    # БЛОК ЧИСЛЕННОГО РЕШЕНИЯ (NUMERIC)
    # ==========================================
    
    def _common_candidates(self, a: int, b: int, sa: str, sb: str) -> List[Tuple[str, str]]:
        return [
            ("concatenation", sa + sb),
            ("reverse concatenation", sb + sa),
            ("addition", str(a + b)),
            ("absolute difference", str(abs(a - b))),
            ("negated absolute difference", str(-abs(a - b))),
            ("subtraction (a-b)", str(a - b)),
            ("reverse subtraction (b-a)", str(b - a)),
            ("multiplication", str(a * b)),
        ]

    def _rare_candidates(self, a: int, b: int, sa: str, sb: str) -> List[Tuple[str, str]]:
        out = [
            ("multiply+1", str(a * b + 1)),
            ("multiply-1", str(a * b - 1)),
            ("add+1", str(a + b + 1)),
            ("add-1", str(a + b - 1)),
            ("sub+1", str(a - b + 1)),
            ("sub-1", str(a - b - 1)),
        ]
        if a != 0 and b != 0:
            out.append(("max mod min", str(max(a, b) % min(a, b))))
        if b != 0:
            out.append(("integer division (a/b)", str(a // b)))
            out.append(("modulo (a mod b)", str(a % b)))
        if a != 0:
            out.append(("reverse division (b/a)", str(b // a)))
            out.append(("reverse modulo (b mod a)", str(b % a)))
            
        sa_c, sb_c = sa.replace("-", ""), sb.replace("-", "")
        if len(sa_c) == 2 and len(sb_c) == 2:
            try:
                d1, d2, d3, d4 = int(sa_c[0]), int(sa_c[1]), int(sb_c[0]), int(sb_c[1])
                out.extend([
                    ("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))),
                    ("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)),
                    ("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)),
                    ("cross multiply", str(d1 * d3 + d2 * d4)),
                    ("cross multiply rev", str(d1 * d4 + d2 * d3)),
                    ("digit multiply", str(d1 * d3) + str(d2 * d4)),
                    ("digit multiply rev", str(d1 * d4) + str(d2 * d3)),
                    ("digit sum diff", str((d1 + d2) - (d3 + d4))),
                    ("digit sum sum", str((d1 + d2) + (d3 + d4))),
                    ("digit product diff", str(d1 * d2 - d3 * d4)),
                    ("digit product sum", str(d1 * d2 + d3 * d4)),
                ])
                det_val = d1 * d4 - d2 * d3
                out.append(("determinant", str(det_val)))
                out.append(("abs determinant", str(abs(det_val))))
            except ValueError:
                pass
        return out

    def _all_candidates(self, a: int, b: int, sa: str, sb: str) -> List[Tuple[str, str]]:
        return self._common_candidates(a, b, sa, sb) + self._rare_candidates(a, b, sa, sb)

    def _solve_numeric(self, prompt: str, query_str: str) -> str:
        raw_lines = re.findall(r"([^\n=]+?)\s*=\s*([^\n]+)", prompt)
        parsed = []
        for lhs, rhs in raw_lines:
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                try:
                    int(m.group(1))
                    int(m.group(3))
                    parsed.append((m.group(1), m.group(2).strip(), m.group(3), rhs.strip()))
                except ValueError:
                    pass

        if not parsed:
            return "Parse Error: No valid numeric examples."

        by_op = defaultdict(list)
        for a, op, b, out in parsed:
            by_op[op].append((a, b, out))

        qm = self._numeric_re.fullmatch(query_str)
        if not qm: return f"Parse Error: Invalid numeric query format -> {query_str}"
        qa, q_op, qb = qm.group(1), qm.group(2).strip(), qm.group(3)

        lines = ["Let's solve this step-by-step by finding the hidden transformation rule.", "\n**Step 1: Analyze the given examples**"]
        for a, b, out in by_op.get(q_op, []):
            lines.append(f"For inputs {a} and {b}, the result is {out}.")
        
        lines.append("\n**Step 2: Hypothesize and test operations**")
        lines.append("I will evaluate combinations of arithmetic operations, string manipulations, and their reversals across all examples to find a consistent pattern.")

        found_ops = {}
        for op_char, group in by_op.items():
            fmt = "num"
            any_neg_suffix = False
            any_op_suffix = False
            any_op_prefix = False

            for _, _, out in group:
                if op_char != "-":
                    if out.endswith("-") and len(out) > 1: any_neg_suffix = True
                    if out.endswith(op_char) and len(out) > len(op_char): any_op_suffix = True
                    if out.startswith(op_char) and len(out) > len(op_char): any_op_prefix = True

            transformed = []
            for a, b, out in group:
                t_out = out
                if any_op_suffix and out.endswith(op_char):
                    fmt = "op_suffix"
                    t_out = "-" + out[:-len(op_char)]
                elif any_op_prefix and out.startswith(op_char):
                    fmt = "op_prefix"
                    t_out = "-" + out[len(op_char):]
                elif any_neg_suffix and out.endswith("-"):
                    fmt = "neg_suffix"
                    t_out = "-" + out[:-1]
                transformed.append((a, b, t_out))

            found = None
            for set_name, cand_fn in [("common", self._common_candidates), ("rare", self._rare_candidates)]:
                for rev_ops, rev_res in [(False, False), (True, False), (False, True), (True, True)]:
                    if found: break
                    
                    ca_str, cb_str = transformed[0][0], transformed[0][1]
                    cta = self._rev(ca_str) if rev_ops else ca_str
                    ctb = self._rev(cb_str) if rev_ops else cb_str
                    
                    try:
                        candidates = cand_fn(int(cta), int(ctb), cta, ctb)
                    except ValueError:
                        continue

                    for cand_name, _ in candidates:
                        all_pass = True
                        for ax, bx, exp_x in transformed:
                            rax = self._rev(ax) if rev_ops else ax
                            rbx = self._rev(bx) if rev_ops else bx
                            
                            try:
                                match_res = next((r for n, r in self._all_candidates(int(rax), int(rbx), rax, rbx) if n == cand_name), None)
                            except ValueError:
                                all_pass = False; break

                            if match_res is None:
                                all_pass = False; break
                                
                            fin = self._rev(match_res) if rev_res else match_res
                            if fin != exp_x:
                                all_pass = False; break
                                
                        if all_pass:
                            found = FoundOp(cand_name, rev_ops, rev_res, fmt, op_char)
                            break
            if found:
                found_ops[op_char] = found

        if q_op not in found_ops:
            if found_ops: q_op = list(found_ops.keys())[0]
            else: return "Algorithmic Error: Rule not found."

        found = found_ops[q_op]
        lines.append(f"A consistent rule was found: `{found.op_name}`.")
        lines.append(f"Rule details - Reverse inputs: {found.rev_ops}, Reverse output: {found.rev_res}, Formatting: {found.fmt}.")

        lines.append("\n**Step 3: Apply the rule to the target query**")
        lines.append(f"Target query: {qa} {q_op} {qb}")
        
        ta = self._rev(qa) if found.rev_ops else qa
        tb = self._rev(qb) if found.rev_ops else qb
        if found.rev_ops:
            lines.append(f"Reversing inputs gives {ta} and {tb}.")
            
        raw_result = ""
        for name, res in self._all_candidates(int(ta), int(tb), ta, tb):
            if name == found.op_name:
                raw_result = res
                lines.append(f"Applying operation `{found.op_name}` to {ta} and {tb} yields {raw_result}.")
                break

        final = self._rev(raw_result) if found.rev_res else raw_result
        if found.rev_res:
            lines.append(f"Reversing the raw result yields {final}.")
        
        if final.lstrip("-").isdigit():
            max_len = max(len(qa.lstrip("-")), len(qb.lstrip("-")))
            if max_len == 2 and len(final.lstrip("-")) == 1:
                if final.startswith("-"): final = "-0" + final[1:]
                else: final = "0" + final

        if final.startswith("-"):
            if found.fmt == "op_suffix": final = final[1:] + found.op_char
            elif found.fmt == "op_prefix": final = found.op_char + final[1:]
            elif found.fmt == "neg_suffix": final = final[1:] + "-"

        lines.append(f"\nThe final computed string is {final}.")
        lines.append(f"The answer is \\boxed{{{final}}}")
        return "\n".join(lines)

    # ==========================================
    # БЛОК СИМВОЛЬНОГО РЕШЕНИЯ (CRYPTARITHM + CONCAT)
    # ==========================================

    def _solve_symbolic(self, prompt: str, query_str: str) -> str:
        raw_lines = re.findall(r"([^\n=]+?)\s*=\s*([^\n]+)", prompt)
        parsed_exs = []
        
        for lhs, rhs in raw_lines:
            lhs_c = lhs.replace(" ", "")
            rhs_c = rhs.strip()
            
            if len(lhs_c) >= 3:
                mid = len(lhs_c) // 2
                a, op, b = lhs_c[:mid], lhs_c[mid], lhs_c[mid+1:]
                
                is_neg = False
                if rhs_c.startswith("-") and len(rhs_c) > 1:
                    is_neg = True
                    rhs_c = rhs_c[1:]
                    
                parsed_exs.append({"a": a, "op": op, "b": b, "out": rhs_c, "is_neg": is_neg})
            
        query_clean = query_str.replace(" ", "")
        if not parsed_exs or len(query_clean) < 3:
            return "Parse Error: Invalid symbolic format or query too short."

        mid = len(query_clean) // 2
        q_a, q_op, q_b = query_clean[:mid], query_clean[mid], query_clean[mid+1:]

        unique_syms = set(q_a + q_b)
        for eq in parsed_exs:
            unique_syms.update(list(eq['a'] + eq['b'] + eq['out']))
        
        unique_syms = list(unique_syms)
        lines = ["Let's solve this symbolic logic puzzle step-by-step.", "\n**Step 1: Extract equations and symbols**"]
        lines.append(f"Found {len(parsed_exs)} equations. Unique symbols involved: {', '.join(unique_syms)}.")
        
        if 0 < len(unique_syms) <= 10:
            lines.append("\n**Step 2: Solve as Cryptarithmetic (Substitution Cipher)**")
            lines.append("We will search for a valid mapping of symbols to digits (0-9) that satisfies all arithmetic properties simultaneously.")
            compiled = []
            valid_math_ops = {'+', '-', '*', '/'}
            for eq in parsed_exs:
                if eq['op'] not in valid_math_ops: continue
                try:
                    i1 = [unique_syms.index(c) for c in eq['a']]
                    i2 = [unique_syms.index(c) for c in eq['b']]
                    ir = [unique_syms.index(c) for c in eq['out']]
                    compiled.append((i1, eq['op'], i2, ir, eq['is_neg']))
                except ValueError:
                    pass
                
            if compiled:
                found_perm = None
                for perm in itertools.permutations((0,1,2,3,4,5,6,7,8,9), len(unique_syms)):
                    valid = True
                    for i1, op, i2, ir, is_neg in compiled:
                        if len(i1) > 1 and perm[i1[0]] == 0: valid = False; break
                        if len(i2) > 1 and perm[i2[0]] == 0: valid = False; break
                        if len(ir) > 1 and perm[ir[0]] == 0: valid = False; break
                        
                        v1 = sum(perm[idx] * (10**(len(i1)-1-j)) for j, idx in enumerate(i1))
                        v2 = sum(perm[idx] * (10**(len(i2)-1-j)) for j, idx in enumerate(i2))
                        vr = sum(perm[idx] * (10**(len(ir)-1-j)) for j, idx in enumerate(ir))
                        
                        if is_neg: vr = -vr
                            
                        if op == '+': valid = (v1 + v2 == vr)
                        elif op == '-': valid = (v1 - v2 == vr)
                        elif op == '*': valid = (v1 * v2 == vr)
                        elif op == '/': valid = (v2 != 0 and v1 // v2 == vr and v1 % v2 == 0)
                        
                        if not valid: break
                    if valid:
                        found_perm = perm
                        break
                        
                if found_perm and q_op in valid_math_ops:
                    inv_map = {v: k for k, v in zip(unique_syms, found_perm)}
                    mapping_str = ", ".join([f"{k}={v}" for k, v in zip(unique_syms, found_perm)])
                    lines.append(f"Success! Valid mapping found: {mapping_str}.")
                    
                    lines.append("\n**Step 3: Apply mapping to target query**")
                    lines.append(f"Target expression: {q_a} {q_op} {q_b}")
                    
                    v1 = sum(found_perm[unique_syms.index(c)] * (10**(len(q_a)-1-j)) for j, c in enumerate(q_a))
                    v2 = sum(found_perm[unique_syms.index(c)] * (10**(len(q_b)-1-j)) for j, c in enumerate(q_b))
                    
                    if q_op == '+': ans = v1 + v2
                    elif q_op == '-': ans = v1 - v2
                    elif q_op == '*': ans = v1 * v2
                    elif q_op == '/': ans = v1 // v2 if v2 != 0 else 0
                    
                    lines.append(f"Substituting digits yields: {v1} {q_op} {v2} = {ans}.")
                    
                    ans_str = str(abs(ans))
                    fs = "-" if ans < 0 else ""
                    fs += "".join(inv_map.get(int(d), "?") for d in ans_str)
                    
                    lines.append(f"Converting the numeric answer {ans} back to symbols results in {fs}.")
                    lines.append(f"The answer is \\boxed{{{fs}}}")
                    return "\n".join(lines)

        lines.append("\n**Step 2: Fallback to String Concatenation**")
        lines.append("Cryptarithmetic mapping failed or is not applicable. Analyzing concatenation patterns.")
        concat_types = {}
        for ex in parsed_exs:
            if ex["out"] == (ex["a"] + ex["b"]): concat_types[ex["op"]] = "fwd"
            elif ex["out"] == (ex["b"] + ex["a"]): concat_types[ex["op"]] = "rev"
        
        c_type = concat_types.get(q_op, "fwd")
        ans = (q_a + q_b) if c_type == "fwd" else (q_b + q_a)
        
        lines.append(f"Based on examples, the rule for '{q_op}' is {c_type} concatenation.")
        lines.append(f"Concatenating {q_a} and {q_b} accordingly yields {ans}.")
        lines.append(f"The answer is \\boxed{{{ans}}}")
        return "\n".join(lines)