import re

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