import re
from typing import Dict, Any

class PseudoMathStringSolver:
    def __init__(self):
        self._numeric_re = re.compile(r"^(-?\w+)\s*([^\w\s]+)\s*(-?\w+)$")

    def solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        cot = []
        
        # 1. Тихий парсинг
        target_text = target_text.strip()
        qm = self._numeric_re.fullmatch(target_text)
        if not qm: 
            return {"answer": None, "debug": ["Failed to parse the target expression format."]}
            
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)
        
        parsed_exs = []
        for line in examples_text.split('\n'):
            if '=' not in line: continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                if op == q_op:
                    clean_rhs = "".join(c for c in rhs.strip() if c.isalnum() or c == '-')
                    if clean_rhs:
                        parsed_exs.append((a, b, clean_rhs))

        if not parsed_exs:
            return {"answer": None, "debug": [f"Could not find any valid examples for the '{q_op}' operator."]}

        # 2. Проверка гипотез
        best_pattern = None
        ans = None
        patterns = ['fwd_concat', 'rev_concat', 'interleave', 'subtraction']

        for pattern in patterns:
            all_match = True
            for a, b, expected_out in parsed_exs:
                if pattern == 'fwd_concat' and expected_out != a + b: all_match = False
                elif pattern == 'rev_concat' and expected_out != b + a: all_match = False
                elif pattern == 'interleave':
                    interleaved = "".join(i+j for i,j in zip(a,b)) + a[len(b):] + b[len(a):]
                    if expected_out != interleaved or interleaved == "": all_match = False
                elif pattern == 'subtraction':
                    res = "".join([c for c in a if c not in b])
                    if expected_out != res or expected_out == "": all_match = False
                
                if not all_match: break
            
            if all_match:
                best_pattern = pattern
                break

        # 3. Рассуждения (С явным показом проверки остальных примеров)
        if best_pattern:
            ex1_a, ex1_b, ex1_out = parsed_exs[0]
            
            if best_pattern == 'fwd_concat':
                cot.append(f"Looking at the first example ({ex1_a} {q_op} {ex1_b} = {ex1_out}), the result appears to be formed by directly concatenating the two numbers.")
                if len(parsed_exs) > 1:
                    cot.append("Let's verify this hypothesis with the other examples:")
                    for a, b, out in parsed_exs[1:]:
                        cot.append(f"- '{a}' concatenated with '{b}' equals '{out}'. This matches.")
                cot.append("The rule is confirmed: direct string concatenation.")
                ans = q_a + q_b
                
            elif best_pattern == 'rev_concat':
                cot.append(f"Analyzing the first example ({ex1_a} {q_op} {ex1_b} = {ex1_out}), the result matches the second operand placed before the first one.")
                if len(parsed_exs) > 1:
                    cot.append("Let's verify if this reversed concatenation applies to the rest:")
                    for a, b, out in parsed_exs[1:]:
                        cot.append(f"- Placing '{b}' before '{a}' gives '{out}'. This is a match.")
                cot.append("The rule is confirmed: reversed string concatenation.")
                ans = q_b + q_a
                
            elif best_pattern == 'interleave':
                cot.append(f"In the first example ({ex1_a} {q_op} {ex1_b} = {ex1_out}), the result is formed by alternating characters from the left and right operands.")
                if len(parsed_exs) > 1:
                    cot.append("Let's test this interleaving pattern on the other examples:")
                    for a, b, out in parsed_exs[1:]:
                        interleaved = "".join(i+j for i,j in zip(a,b)) + a[len(b):] + b[len(a):]
                        cot.append(f"- Alternating characters of '{a}' and '{b}' gives '{interleaved}'. This is a match.")
                cot.append("The rule is confirmed: interleaving the characters.")
                ans = "".join(i+j for i,j in zip(q_a, q_b)) + q_a[len(q_b):] + q_b[len(q_a):]
                
            elif best_pattern == 'subtraction':
                cot.append(f"Looking at the first example ({ex1_a} {q_op} {ex1_b} = {ex1_out}), the result is the first number, but with any digits present in the second number removed.")
                if len(parsed_exs) > 1:
                    cot.append("Let's check if this character subtraction rule holds up:")
                    for a, b, out in parsed_exs[1:]:
                        res = "".join([c for c in a if c not in b])
                        cot.append(f"- Removing digits of '{b}' from '{a}' leaves '{res}'. This matches.")
                cot.append("The rule is confirmed: removing the right operand's characters from the left operand.")
                ans = "".join([c for c in q_a if c not in q_b])

            # 4. Применение к целевому запросу
            cot.append(f"Now, let's apply this rule to our target expression: {q_a} {q_op} {q_b}.")
            
            if best_pattern == 'fwd_concat':
                cot.append(f"Concatenating '{q_a}' and '{q_b}' directly gives '{ans}'.")
            elif best_pattern == 'rev_concat':
                cot.append(f"Placing '{q_b}' before '{q_a}' gives '{ans}'.")
            elif best_pattern == 'interleave':
                cot.append(f"Alternating the characters of '{q_a}' and '{q_b}' gives '{ans}'.")
            elif best_pattern == 'subtraction':
                cot.append(f"Removing the digits of '{q_b}' from '{q_a}' leaves '{ans}'.")

            return {"answer": ans, "debug": cot}

        cot.append("None of the standard string manipulation rules match all the provided examples.")
        return {"answer": None, "debug": cot}