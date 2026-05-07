import re
from typing import List, Tuple, Dict, Optional

from src.solvers.equations.ast_brute_force import ASTBruteForceSolver
from src.solvers.equations.pseudo_math import PseudoMathStringSolver
from src.solvers.equations.cryptarithm import CryptarithmCSPSolver

class BaseEquationSolver:
    """
    Base router class. 
    Responsible for parsing the raw prompt, heuristically classifying the task, 
    and delegating execution to the appropriate sub-solver.
    """
    def __init__(self):
        # Initialize the pool of available solvers
        self.ast_solver = ASTBruteForceSolver()
        self.csp_solver = CryptarithmCSPSolver()
        self.string_solver = PseudoMathStringSolver()

    def _extract_sections(self, text: str) -> Tuple[str, str]:
        pattern = r"few examples:?\s*(.*?)\s*now, d[e]?termine the result for:\s*(.*)"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", ""

    def _classify_task(self, examples: str, target: str) -> Tuple[str, List[str]]:
        if not examples.strip():
            return "Empty", []
        
        cot = ["First, we examine the operators in the equations."]
                
        # 2. Pseudo-Math (strictly by the target operator)
        # Find the first non-alphanumeric character, excluding spaces and '='
        target_op_match = re.search(r'[^\w\s=]', target)
        cot.append("Let's look at the target expression and its operator, which is usually located in the middle.")
        cot.append("Initially, we need to determine if this equation is not a standard mathematical problem, but rather simple string operations.")
        
        if target_op_match:
            target_op = target_op_match.group(0)
            cot.append(f"The target operator in the expression is '{target_op}'.")
            cot.append("Let's analyze the examples with this operator.")
            
            for line in examples.split('\n'):
                if '=' in line and target_op in line.split('=')[0]:
                    lhs, rhs = line.split('=', 1)
                    
                    # Split the string strictly by the operator and remove extra spaces
                    operands = [op.strip() for op in lhs.split(target_op)]
                    rhs_clean = rhs.strip()
                    
                    if len(operands) == 2:
                        left, right = operands[0], operands[1]
                        
                        # Direct concatenation (A + B = AB)
                        if rhs_clean == "".join(operands):
                            cot.append(f"Example '{line}' is a string concatenation: left '{left}' + right '{right}' = '{rhs_clean}'.")
                            return "Pseudo-Math (Format/String)", cot
                        
                        # Reverse concatenation of operands (A + B = BA)
                        if rhs_clean == "".join(reversed(operands)):
                            cot.append(f"Example '{line}' is a reversed string concatenation: right '{right}' + left '{left}' = '{rhs_clean}'.")
                            return "Pseudo-Math (Format/String)", cot
                        
                        # Full character-by-character reverse of direct concatenation (rare, but happens: AB -> BA)
                        if rhs_clean == "".join(operands)[::-1]:
                            cot.append(f"Example '{line}' is a fully reversed string concatenation: ('{left}' + '{right}') reversed = '{rhs_clean}'.")
                            return "Pseudo-Math (Format/String)", cot
        else:
            cot.append("It is not immediately obvious, but this is likely a mathematical problem rather than string manipulation.")
        

        # If there are no digits at all and it's not string concatenation, route to cryptarithm
        if not re.search(r'\d', examples):
            cot.append("There are no digits in the examples. This implies it is a cryptarithm; we need to decode the encrypted operations and values to calculate the result.")
            return "Cryptarithm (CSP)", cot
        
        cot.append("The examples contain digits, suggesting we simply need to deduce the hidden mathematical operations encoded by the operators.")
        
        # 3. Everything else
        return "AST Brute-force", cot

    def solve(self, prompt: str) -> Optional[str]:
        examples, target = self._extract_sections(prompt)
        if not examples or not target:
            return None
            
        task_type, _ = self._classify_task(examples, target)
        
        # Delegate to the specific solver depending on the task class
        if task_type == "AST Brute-force":
            return self.ast_solver.solve(examples, target)
        elif task_type == "Cryptarithm (CSP)":
            #return self.csp_solver.solve(examples, target)
            return "HUI"
        elif task_type == "Pseudo-Math (Format/String)":
            return self.string_solver.solve(examples, target)
            
        return None

    def extract_answer(self, cot_text: str) -> str:
        if not isinstance(cot_text, str) or "Error" in cot_text:
            return "nan"
        
        matches = re.findall(r"(?i)final\s+answer:\s*(.+)", cot_text)
        return matches[-1].strip() if matches else "nan"

    def generate_cot(self, prompt: str) -> str:
        examples, target = self._extract_sections(prompt)
        cot = [f"I see this is an equations problem. Based on the examples, we need to determine the result for: {target}."]

        if not examples or not target:
            return ""
            
        task_type, class_cot = self._classify_task(examples, target)
        cot.extend(class_cot)
        
        if task_type == "AST Brute-force":
            result = self.ast_solver.solve(examples, target)
        elif task_type == "Cryptarithm (CSP)":
            result = self.csp_solver.solve(examples, target)
        elif task_type == "Pseudo-Math (Format/String)":
            result = self.string_solver.solve(examples, target)
            
        cot.extend(result["debug"])
        cot.append(f"Final answer: {result['answer']}")

        return "\n".join(cot)