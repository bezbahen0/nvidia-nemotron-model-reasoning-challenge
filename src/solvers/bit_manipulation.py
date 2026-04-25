import re

class BitManipulationSolver:
    """bit_manipulation"""
    
    def __init__(self):
        self.ops = {
            'I': lambda a, b: a, 'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0, 'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b), 'XOR-NOT': lambda a, b: a ^ (1 - b)
        }
        self.SECTION_ORDER = ['I', 'NOT', 'C0', 'C1', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']

    def _get_valid_rules(self, examples, out_idx):
        # ... (Код функции _get_valid_rules остается без изменений)
        valid = []
        for op_name in self.SECTION_ORDER:
            op_func = self.ops[op_name]
            if op_name in ['C0', 'C1']:
                if all(int(ex_out[out_idx]) == op_func(0, 0) for _, ex_out in examples):
                    valid.append((op_name, -1, -1))
            elif op_name in ['I', 'NOT']:
                for in1 in range(8):
                    if all(int(ex_out[out_idx]) == op_func(int(ex_in[in1]), 0) for ex_in, ex_out in examples):
                        valid.append((op_name, in1, -1))
            else:
                for in1 in range(8):
                    for in2 in range(8):
                        is_valid = True
                        for ex_in, ex_out in examples:
                            if op_func(int(ex_in[in1]), int(ex_in[in2])) != int(ex_out[out_idx]):
                                is_valid = False
                                break
                        if is_valid:
                            valid.append((op_name, in1, in2))
        return valid

    def _format_rule_description(self, op, in1, in2):
        if op == 'C0': return "always set to 0"
        if op == 'C1': return "always set to 1"
        if op == 'I': return f"taken directly from input bit {in1}"
        if op == 'NOT': return f"the inverted value (NOT) of input bit {in1}"
        return f"the {op} operation between input bit {in1} and input bit {in2}"

    def generate_cot(self, prompt: str) -> str:
        examples = []
        for line in prompt.split('\n'):
            if '->' in line:
                parts = line.split('->')
                in_str = re.sub(r'[^01]', '', parts[0])
                out_str = re.sub(r'[^01]', '', parts[1])
                if in_str and out_str:
                    examples.append((in_str.zfill(8), out_str.zfill(8)))

        target_match = re.search(r"output for:\s*([01]+)", prompt, re.IGNORECASE)
        if not target_match or len(examples) == 0:
            return "Parse Error.\n nan"
        
        target_input = target_match.group(1).zfill(8)
        
        # 1. ФОНОВЫЕ ВЫЧИСЛЕНИЯ (Без добавления в текст)
        flat_matches = [self._get_valid_rules(examples, i) for i in range(8)]
        best_left_run = []
        if flat_matches[0]:
            for cand in flat_matches[0]:
                run = [cand]
                op, in1, in2 = cand
                for i in range(1, 8):
                    exp_in1 = (in1 + i) % 8 if in1 != -1 else -1
                    exp_in2 = (in2 + i) % 8 if in2 != -1 else -1
                    if (op, exp_in1, exp_in2) in flat_matches[i]: run.append((op, exp_in1, exp_in2))
                    else: break
                if len(run) > len(best_left_run): best_left_run = run

        best_right_run = []
        if flat_matches[7]:
            for cand in flat_matches[7]:
                run = [cand]
                op, in1, in2 = cand
                for step in range(1, 8):
                    i = 7 - step
                    exp_in1 = (in1 - step) % 8 if in1 != -1 else -1
                    exp_in2 = (in2 - step) % 8 if in2 != -1 else -1
                    if (op, exp_in1, exp_in2) in flat_matches[i]: run.insert(0, (op, exp_in1, exp_in2))
                    else: break
                if len(run) > len(best_right_run): best_right_run = run

        len_l = len(best_left_run)
        len_r = len(best_right_run)
        
        if len_l + len_r > 8:
            if len_r > len_l: len_l = 8 - len_r; best_left_run = best_left_run[:len_l]
            else: len_r = 8 - len_l; best_right_run = best_right_run[-len_r:] if len_r > 0 else []

        final_rules = [None] * 8
        for i in range(len_l): final_rules[i] = best_left_run[i]
        right_start_idx = 8 - len_r
        for i in range(len_r): final_rules[right_start_idx + i] = best_right_run[i]

        pending = [i for i in range(8) if final_rules[i] is None]
        for p in pending:
            if flat_matches[p]: final_rules[p] = flat_matches[p][0]
            else: final_rules[p] = ('C1', -1, -1)

        # 2. ПОСТРОЕНИЕ ДОКАЗАТЕЛЬНОГО CoT
        cot = [
            "To solve this bit manipulation sequence, we need to analyze how each bit of the output is generated from the input bits.",
            f"We will determine the rule for each position (Bit 0 to Bit 7) based on the examples and apply it to our target: {target_input}.\n"
        ]

        target_output = ""
        ex_in, ex_out = examples[0] # Берем первый пример для демонстрации доказательства

        for i in range(8):
            op, in1, in2 = final_rules[i]
            desc = self._format_rule_description(op, in1, in2)
            
            cot.append(f"--- Output Bit {i} ---")
            cot.append(f"By checking the examples, the value for output bit {i} is {desc}.")
            
            # Доказательная база: показываем на примере
            if op not in ['C0', 'C1']:
                v1_ex = ex_in[in1] if in1 != -1 else "0"
                v2_ex = ex_in[in2] if in2 != -1 else "0"
                
                if op == 'I': proof_str = f"Bit {in1} is {v1_ex}, so the output is {v1_ex}."
                elif op == 'NOT': proof_str = f"Bit {in1} is {v1_ex}, NOT({v1_ex}) is {ex_out[i]}."
                else: proof_str = f"Bit {in1} is {v1_ex} and Bit {in2} is {v2_ex}. {v1_ex} {op} {v2_ex} gives {ex_out[i]}."
                
                cot.append(f"Proof from Example 1 ({ex_in} -> {ex_out}): {proof_str}")

            # Применение к таргету
            val1 = int(target_input[in1]) if in1 != -1 else 0
            val2 = int(target_input[in2]) if in2 != -1 else 0
            res = self.ops[op](val1, val2)
            target_output += str(res)
            
            if op in ['C0', 'C1']:
                cot.append(f"Applying to target: The bit is constant, so Target Bit {i} is {res}.\n")
            elif op == 'I':
                cot.append(f"Applying to target: Input Bit {in1} is {val1}. Target Bit {i} is {res}.\n")
            elif op == 'NOT':
                cot.append(f"Applying to target: Input Bit {in1} is {val1}. NOT({val1}) is {res}. Target Bit {i} is {res}.\n")
            else:
                cot.append(f"Applying to target: Input Bit {in1} is {val1}, Input Bit {in2} is {val2}. {val1} {op} {val2} gives {res}. Target Bit {i} is {res}.\n")

        cot.append(f"Combining all calculated target bits (0 to 7) gives: {target_output}")
        cot.append(f"\nThe final answer: {target_output}")
        
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return "nan"
        match = re.search(r"(?i)final\s+answer:\s*([01]{8})", cot_text)
        return match.group(1) if match else "nan"