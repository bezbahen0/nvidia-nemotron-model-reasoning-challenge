import re


class GravitationalSolver:   
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