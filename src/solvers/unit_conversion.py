import re
from decimal import Decimal, getcontext, ROUND_HALF_EVEN

getcontext().prec = 50


class UnitConversionSolver:
    """unit conversion"""
    
    def generate_cot(self, prompt: str) -> str:
        target_match = re.search(r"convert the following measurement:\s*([\d.]+)", prompt, re.IGNORECASE)
        if not target_match:
            return "Parse Error: Target not found."
        
        target_str = target_match.group(1)
        target_dec = Decimal(target_str)
        
        examples = re.findall(r"([\d.]+)\s*[a-zA-Z]*\s*becomes\s*([\d.]+)", prompt)
        if not examples:
            return "Parse Error: Examples not found."

        cot = ["Let's determine the exact unit conversion ratio using infinite precision.\n"]
        
        min_possible_ratio = Decimal('0')
        max_possible_ratio = Decimal('Infinity')
        delta = Decimal('0.005')
        
        for a_str, b_str in examples:
            a_dec = Decimal(a_str)
            b_dec = Decimal(b_str)
            
            if a_dec > Decimal('0'):
                lower = (b_dec - delta) / a_dec
                upper = (b_dec + delta) / a_dec
                
                if lower > min_possible_ratio:
                    min_possible_ratio = lower
                if upper < max_possible_ratio:
                    max_possible_ratio = upper
                    
                cot.append(f"  {a_str} -> {b_str} implies ratio in [{lower:.8f}, {upper:.8f}]")

        if min_possible_ratio > max_possible_ratio:
            cot.append("\nMath Error: Bounds contradict. Falling back to least squares midpoint.")
            sum_x = sum(Decimal(a) for a, _ in examples)
            sum_y = sum(Decimal(b) for _, b in examples)
            avg_ratio = sum_y / sum_x if sum_x != Decimal('0') else Decimal('1')
            result = target_dec * avg_ratio
            final_answer = str(result.quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN))
            cot.append(f"The final answer is {final_answer}.")
            return "\n".join(cot)

        y_min = target_dec * min_possible_ratio
        y_max = target_dec * max_possible_ratio
        
        y_min_rounded = y_min.quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        y_max_rounded = y_max.quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        
        cot.append(f"\nTarget {target_str} boundaries: [{y_min:.6f}, {y_max:.6f}]")
        
        if y_min_rounded == y_max_rounded:
            final_answer = str(y_min_rounded)
            cot.append(f"Both bounds round to exactly {final_answer}. 100% certainty.")
        else:
            avg_ratio = (min_possible_ratio + max_possible_ratio) / Decimal('2')
            result = target_dec * avg_ratio
            final_answer = str(result.quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN))
            cot.append(f"Ambiguity detected (bounds round differently). Using midpoint ratio {avg_ratio:.8f}.")
            cot.append(f"Calculation yields {result:.6f}, rounding to {final_answer}.")
            
        cot.append(f"The final answer is {final_answer}.")
        
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return None
        match = re.search(r"The final answer is ([\d.]+)\.", cot_text)
        return match.group(1) if match else None
