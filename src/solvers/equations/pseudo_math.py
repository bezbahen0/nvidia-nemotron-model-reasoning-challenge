import re
from typing import Optional, Dict, Any

class PseudoMathStringSolver:
    """
    Solver for Pseudo-Math (String/Format Logic) tasks.
    Looks for string manipulation patterns (concatenation, interleaving,
    ASCII shifts, intersections), ignoring the mathematical properties of numbers.
    """
    def __init__(self):
        # Using \w instead of \d in case string operations 
        # are applied to alphabetical operands
        self._numeric_re = re.compile(r"^(-?\w+)\s*([^\w\s]+)\s*(-?\w+)$")

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        log = []
        
        # 1. Parsing the target expression
        log.append(f"Analyzing target expression: '{target_text.strip()}'")
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm: 
            log.append("Failed to parse target expression. Does not match expected operand-operator-operand format.")
            return {"answer": None, "debug": log}
            
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)
        log.append(f"Target parsed successfully: left='{q_a}', operator='{q_op}', right='{q_b}'.")

        # 2. Parsing and filtering examples strictly by the target operator
        parsed_exs = []
        log.append(f"Extracting examples matching the target operator '{q_op}'...")
        for line in examples_text.split('\n'):
            if '=' not in line:
                continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                if op == q_op:
                    # Allow letters and numbers in the answer (important for ascii_shift)
                    clean_rhs = "".join(c for c in rhs.strip() if c.isalnum() or c == '-')
                    if clean_rhs:
                        parsed_exs.append((a, b, clean_rhs))

        if not parsed_exs:
            log.append(f"No valid examples found for operator '{q_op}'. Cannot infer pattern.")
            return {"answer": None, "debug": log}
            
        log.append(f"Found {len(parsed_exs)} valid example(s) for operator '{q_op}'.")

        # Helper function to check ASCII shift
        def get_ascii_shift(a_str: str, out_str: str) -> Optional[int]:
            if len(a_str) != len(out_str) or len(a_str) == 0: 
                return None
            diffs = set(ord(o) - ord(i) for o, i in zip(out_str, a_str))
            return diffs.pop() if len(diffs) == 1 else None

        # 3. Iterating over string transformation hypotheses
        best_pattern = None
        global_shift_val = None
        
        patterns = [
            'fwd_concat', 'rev_concat', 'rev_both', 'interleave', 
            'subtraction', 'intersection', 'first_last', 'overlap', 'ascii_shift'
        ]

        log.append("Testing known string manipulation hypotheses against examples...")

        for pattern in patterns:
            all_match = True
            shift_val = None
            
            for a, b, expected_out in parsed_exs:
                match = False
                
                if pattern == 'fwd_concat' and expected_out == a + b: 
                    match = True
                elif pattern == 'rev_concat' and expected_out == b + a: 
                    match = True
                elif pattern == 'rev_both' and expected_out == a[::-1] + b[::-1]: 
                    match = True
                elif pattern == 'interleave':
                    # Improved interleaving (supports different string lengths)
                    interleaved = "".join(i+j for i,j in zip(a,b)) + a[len(b):] + b[len(a):]
                    if expected_out == interleaved and interleaved != "": 
                        match = True
                elif pattern == 'subtraction':
                    res = "".join([c for c in a if c not in b])
                    if expected_out == res and expected_out != "": 
                        match = True
                elif pattern == 'intersection':
                    res = "".join([c for c in a if c in b])
                    if expected_out == res and expected_out != "": 
                        match = True
                elif pattern == 'first_last':
                    if len(a) > 0 and len(b) > 0 and expected_out == a[0] + b[-1]: 
                        match = True
                elif pattern == 'overlap':
                    if len(a) > 0 and len(b) > 0 and a[-1] == b[0]:
                        if expected_out == a + b[1:]: 
                            match = True
                    elif expected_out == a + b: 
                        match = True
                elif pattern == 'ascii_shift':
                    shift = get_ascii_shift(a, expected_out)
                    if shift is not None:
                        if shift_val is None: 
                            shift_val = shift
                        if shift_val == shift: 
                            match = True
                
                if not match:
                    all_match = False
                    break
            
            if all_match:
                best_pattern = pattern
                global_shift_val = shift_val
                log.append(f"Success: Pattern '{best_pattern}' consistently matches all examples.")
                break

        # 4. Applying the found rule to the target query
        if best_pattern:
            log.append(f"Applying '{best_pattern}' logic to target operands '{q_a}' and '{q_b}'.")
            ans = None
            if best_pattern == 'fwd_concat': ans = q_a + q_b
            elif best_pattern == 'rev_concat': ans = q_b + q_a
            elif best_pattern == 'rev_both': ans = q_a[::-1] + q_b[::-1]
            elif best_pattern == 'interleave': 
                ans = "".join(i+j for i,j in zip(q_a, q_b)) + q_a[len(q_b):] + q_b[len(q_a):]
            elif best_pattern == 'subtraction': ans = "".join([c for c in q_a if c not in q_b])
            elif best_pattern == 'intersection': ans = "".join([c for c in q_a if c in q_b])
            elif best_pattern == 'first_last': ans = q_a[0] + q_b[-1] if (len(q_a)>0 and len(q_b)>0) else ""
            elif best_pattern == 'overlap': ans = q_a + q_b[1:] if (len(q_a)>0 and len(q_b)>0 and q_a[-1]==q_b[0]) else q_a + q_b
            elif best_pattern == 'ascii_shift' and global_shift_val is not None:
                ans = "".join(chr(ord(c) + global_shift_val) for c in q_a)
            
            log.append(f"Calculated result: '{ans}'")
            return {"answer": ans, "debug": log}

        log.append("Failed to resolve target. No known string manipulation pattern matched all examples.")
        return {"answer": None, "debug": log}