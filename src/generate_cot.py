import re
from decimal import Decimal, getcontext, ROUND_HALF_EVEN

getcontext().prec = 50

class NumeralSystemSolver:
    """conversion to diff numeral system"""
    
    def __init__(self):
        self.roman_vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        self.roman_syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

    def generate_cot(self, prompt: str) -> str:
        """Chain-of-Thought"""
        target_match = re.search(r"write the number (\d+)", prompt, re.IGNORECASE)
        if not target_match:
            return "Parse Error: Target not found."
        
        target_num = int(target_match.group(1))
        examples = re.findall(r"(\d+)\s*->\s*([A-Z]+)", prompt)
        
        cot = ["Let's identify the secret numeral system used in Wonderland.\n"]
        cot.append("Looking at the examples provided:")
        
        for arab, rom in examples[:3]:
            cot.append(f"  {arab} -> {rom}")
            
        cot.append("\nThe output symbols (I, V, X, L, C, D, M) and their combinations clearly indicate standard Roman Numerals.")
        cot.append(f"\nWe need to convert the number {target_num} into Roman numerals using greedy decomposition:")
        
        remaining = target_num
        parts = []
        
        for v, s in zip(self.roman_vals, self.roman_syms):
            while remaining >= v:
                parts.append(s)
                remaining -= v
                cot.append(f"  - Subtract {v} ({s}): remainder is {remaining}.")
                
        final_roman = "".join(parts)
        cot.append(f"\nCombining the symbols gives us: {final_roman}.")
        cot.append(f"The final answer is {final_roman}.")
        
        return "\n".join(cot)

    # TODO: 
    # Добавить \\boxed в ответ?
    def extract_answer(self, cot_text: str) -> str:
        if "Parse Error" in cot_text:
            return None

        match = re.search(r"The final answer is ([A-Z]+)\.", cot_text)
        return match.group(1) if match else None
    

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


class GravitationalSolver:
    """gravitational"""
    
    def generate_cot(self, prompt: str) -> str:
        target_match = re.search(r"determine the falling distance for t\s*=\s*([\d.]+)s", prompt, re.IGNORECASE)
        if not target_match:
            return "Parse Error: Target time not found."
        
        t_query = float(target_match.group(1))

        examples = re.findall(r"t\s*=\s*([\d.]+)s[,\s]*distance\s*=\s*([\d.]+)\s*m", prompt, re.IGNORECASE)
        if not examples:
            return "Parse Error: Examples not found."

        cot = ["WARNING: This is Wonderland gravity, NOT Earth's 9.81 m/s^2!\n"]
        cot.append("Step 1: Calculate the gravitational constant (g).")
        cot.append("The formula is d = 0.5 * g * t^2. Therefore, g = d / (0.5 * t^2).")
        cot.append("To minimize rounding errors from individual examples, we will calculate g using the sum of all distances divided by the sum of all (0.5 * t^2) values:\n")
        
        sum_d = 0
        sum_half_t_sq = 0
        
        for i, (t_str, d_str) in enumerate(examples[:6], 1):
            t, d = float(t_str), float(d_str)
            if t > 0:
                half_t_sq = 0.5 * (t ** 2)
                sum_d += d
                sum_half_t_sq += half_t_sq
                cot.append(f"  Example {i}:")
                cot.append(f"    Given: t = {t}s, d = {d}m")
                cot.append(f"    0.5 * t^2 = 0.5 * {t**2:.4f} = {half_t_sq:.4f}")
        
        if sum_half_t_sq == 0:
            return "Math Error: Sum of t^2 is zero."
            
        g_avg = sum_d / sum_half_t_sq
        
        cot.append(f"\nStep 2: Average gravitational constant")
        cot.append(f"  sum(d) = {sum_d:.4f}")
        cot.append(f"  sum(0.5 * t^2) = {sum_half_t_sq:.4f}")
        cot.append(f"  g = {sum_d:.4f} / {sum_half_t_sq:.4f} = {g_avg:.6f} m/s^2\n")
        
        cot.append(f"Step 3: Apply to query (t = {t_query}s)")
        
        t_squared = t_query ** 2
        product = g_avg * t_squared
        d_result = 0.5 * product
        
        final_answer = f"{d_result:.2f}"
        
        cot.append(f"  Formula: d = 0.5 * g * t^2")
        cot.append(f"  Substitute: d = 0.5 * {g_avg:.6f} * ({t_query})^2")
        cot.append(f"  Calculate t^2: ({t_query})^2 = {t_squared:.4f}")
        cot.append(f"  Calculate g*t^2: {g_avg:.6f} * {t_squared:.4f} = {product:.4f}")
        cot.append(f"  Calculate 0.5*(g*t^2): 0.5 * {product:.4f} = {d_result:.6f}")
        cot.append(f"  Rounded to 2 decimals: {final_answer} m")
        cot.append(f"\nThe final answer is {final_answer}.")
        
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        """Извлекает ответ для проверки."""
        if "Error" in cot_text:
            return None
        match = re.search(r"The final answer is ([\d.]+)\.", cot_text)
        return match.group(1) if match else None


def main():
    pass


if __name__ == "__main__":
    main()