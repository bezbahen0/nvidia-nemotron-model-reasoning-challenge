import re
from collections import defaultdict

class BitManipulationSolver:
    def __init__(self, max_proof_examples=3):
        self.ops = {
            'I': lambda a, b: a, 'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0, 'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b), 'XOR-NOT': lambda a, b: a ^ (1 - b)
        }
        self.SECTION_ORDER = ['I', 'NOT', 'C0', 'C1', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']
        self.max_proof_examples = max_proof_examples

    def _get_valid_rules(self, examples, out_idx):
        """Оригинальный, надежный фоновый поиск всех валидных правил."""
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

    def _detect_macro_pattern(self, examples):
        if all(ex_out == "".join('1' if b == '0' else '0' for b in ex_in) for ex_in, ex_out in examples): return "NOT_ALL", "invert all bits of the input string"
        for shift in range(1, 8):
            if all(ex_out == ex_in[shift:] + ex_in[:shift] for ex_in, ex_out in examples): return f"ROL_{shift}", f"rotate the input left by {shift} position(s)"
            if all(ex_out == ex_in[-shift:] + ex_in[:-shift] for ex_in, ex_out in examples): return f"ROR_{shift}", f"rotate the input right by {shift} position(s)"
            if all(ex_out == ex_in[shift:] + '0'*shift for ex_in, ex_out in examples): return f"SHL_{shift}", f"shift the input left by {shift} position(s), padding with 0s"
            if all(ex_out == '0'*shift + ex_in[:-shift] for ex_in, ex_out in examples): return f"SHR_{shift}", f"shift the input right by {shift} position(s), padding with 0s"
            if all(ex_out == ex_in[shift:] + '1'*shift for ex_in, ex_out in examples): return f"SHL_1_{shift}", f"shift the input left by {shift} position(s), padding with 1s"
            if all(ex_out == '1'*shift + ex_in[:-shift] for ex_in, ex_out in examples): return f"SHR_1_{shift}", f"shift the input right by {shift} position(s), padding with 1s"
        return None, None

    def _apply_macro_pattern(self, pattern_id, target_input):
        if pattern_id == "NOT_ALL": return "".join('1' if b == '0' else '0' for b in target_input)
        elif pattern_id.startswith("ROL_"): return target_input[int(pattern_id.split("_")[1]):] + target_input[:int(pattern_id.split("_")[1])]
        elif pattern_id.startswith("ROR_"): return target_input[-int(pattern_id.split("_")[1]):] + target_input[:-int(pattern_id.split("_")[1])]
        elif pattern_id.startswith("SHL_1_"): return target_input[int(pattern_id.split("_")[2]):] + '1'*int(pattern_id.split("_")[2])
        elif pattern_id.startswith("SHR_1_"): return '1'*int(pattern_id.split("_")[2]) + target_input[:-int(pattern_id.split("_")[2])]
        elif pattern_id.startswith("SHL_"): return target_input[int(pattern_id.split("_")[1]):] + '0'*int(pattern_id.split("_")[1])
        elif pattern_id.startswith("SHR_"): return '0'*int(pattern_id.split("_")[1]) + target_input[:-int(pattern_id.split("_")[1])]
        return ""

    def _get_matches_for_single_example(self, ex_in, ex_out):
        matches = []
        if ex_out == "".join('1' if b == '0' else '0' for b in ex_in): matches.append("Invert all bits")
        for shift in range(1, 8):
            if ex_out == ex_in[shift:] + ex_in[:shift]: matches.append(f"Rotate left by {shift}")
            if ex_out == ex_in[-shift:] + ex_in[:-shift]: matches.append(f"Rotate right by {shift}")
            if ex_out == ex_in[shift:] + '0'*shift: matches.append(f"Shift left by {shift} (pad 0)")
            if ex_out == '0'*shift + ex_in[:-shift]: matches.append(f"Shift right by {shift} (pad 0)")
            if ex_out == ex_in[shift:] + '1'*shift: matches.append(f"Shift left by {shift} (pad 1)")
            if ex_out == '1'*shift + ex_in[:-shift]: matches.append(f"Shift right by {shift} (pad 1)")
        return matches

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

        macro_id, macro_desc = self._detect_macro_pattern(examples)
        is_macro = macro_id is not None
        
        target_output_arr = ['?'] * 8
        final_rules = []

        if is_macro:
            target_output = self._apply_macro_pattern(macro_id, target_input)
        else:
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

            len_l, len_r = len(best_left_run), len(best_right_run)
            if len_l + len_r > 8:
                if len_r > len_l: len_l = 8 - len_r; best_left_run = best_left_run[:len_l]
                else: len_r = 8 - len_l; best_right_run = best_right_run[-len_r:] if len_r > 0 else []

            final_rules = [None] * 8
            for i in range(len_l): final_rules[i] = best_left_run[i]
            for i in range(len_r): final_rules[8 - len_r + i] = best_right_run[i]
            
            for p in range(8):
                if final_rules[p] is None:
                    # ЧЕСТНЫЙ ОТКАЗ: если правила нет, ставим статус UNKNOWN
                    final_rules[p] = flat_matches[p][0] if flat_matches[p] else ('UNKNOWN', -1, -1)

        cot = [
            "First, we define all possible operations that could transform the input into the output.",
            "The possible operations are:",
            "1. Macro operations: Invert all bits, Rotate Left/Right, Shift Left/Right.",
            "2. Micro operations: Bit-level logic (Constants, Copies, NOT, AND, OR, XOR) evaluated across all possible input indices.",
            ""
        ]

        for idx, (ex_in, ex_out) in enumerate(examples):
            matches = self._get_matches_for_single_example(ex_in, ex_out)
            if not is_macro: matches.append("Bit-level logical mapping")
            if not matches: matches.append("Complex structural mapping")
            cot.append(f"Look at Example {idx+1}: {ex_in} -> {ex_out}")
            cot.append(f"Matched: {', '.join(matches)}.")
            cot.append("")

        cot.append("---")
        
        if is_macro:
            cot.append("After reviewing all examples, we must find the operation that applies consistently across every case.")
            cot.append(f"The only operation that perfectly satisfies all examples is to {macro_desc}.")
            cot.append(f"Now, we apply this operation to the target input: {target_input}")
            
        else:
            cot.append("After reviewing all examples, no single macro-operation applies to the entire string. We must deduce the structural mapping for each individual output bit.")
            cot.append("To do this, we test all possible logical operations against all combinations of input indices. We select the rule that perfectly matches the target output bit across every single example.")
            cot.append(f"\nNow, we apply the mathematically verified operations to the target input: {target_input}")
            
            for bit_idx, rule in enumerate(final_rules):
                op, in1, in2 = rule
                cot.append(f"\nOutput Bit {bit_idx}:")
                
                if op == 'UNKNOWN':
                    cot.append(f"- Rule found: FAILED. No standard logical operation could consistently explain Output Bit {bit_idx} across all examples.")
                    target_output_arr[bit_idx] = '?'
                    cot.append(f"- Target calculation: Unknown -> ?")
                else:
                    if op in ['C0', 'C1']:
                        cot.append(f"- Rule found: In all examples, Output Bit {bit_idx} is constantly {op[1]}, regardless of the input.")
                    elif op == 'I':
                        cot.append(f"- Rule found: In all examples, Output Bit {bit_idx} perfectly matches the value of Input[{in1}].")
                    elif op == 'NOT':
                        cot.append(f"- Rule found: In all examples, Output Bit {bit_idx} perfectly matches the inverted value of Input[{in1}] (NOT).")
                    else:
                        cot.append(f"- Rule found: In all examples, Output Bit {bit_idx} perfectly matches the result of Input[{in1}] {op} Input[{in2}].")

                    # Вычисление результата для найденных правил
                    v1_t = int(target_input[in1]) if in1 != -1 else 0
                    v2_t = int(target_input[in2]) if in2 != -1 else 0
                    res = self.ops[op](v1_t, v2_t)
                    target_output_arr[bit_idx] = str(res)

                    if op == 'C0':
                        cot.append(f"- Target calculation: Constant 0 -> 0")
                    elif op == 'C1':
                        cot.append(f"- Target calculation: Constant 1 -> 1")
                    elif op == 'I':
                        cot.append(f"- Target calculation: Copy Input[{in1}] ({v1_t}) -> {res}")
                    elif op == 'NOT':
                        cot.append(f"- Target calculation: NOT Input[{in1}] ({v1_t}) -> {res}")
                    else:
                        cot.append(f"- Target calculation: Input[{in1}] ({v1_t}) {op} Input[{in2}] ({v2_t}) -> {res}")

            target_output = "".join(target_output_arr)

        cot.append(f"\nThe result of this transformation is {target_output}.")
        cot.append(f"The final answer is: {target_output}")
        
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return "nan"

        match = re.search(r"(?i)final\s+answer\s*(?:is)?:\s*([01\?]{8})", cot_text)
        return match.group(1) if match else "nan"

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return "nan"
        match = re.search(r"(?i)final\s+answer\s*(?:is)?:\s*([01]{8})", cot_text)
        return match.group(1) if match else "nan"