import re

class BitManipulationSolver:
    """bit_manipulation - SFT Explanatory Edition"""
    
    def __init__(self):
        self.ops = {
            'I': lambda a, b: a, 'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0, 'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b), 'XOR-NOT': lambda a, b: a ^ (1 - b)
        }
        self.SECTION_ORDER = ['I', 'NOT', 'C0', 'C1', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']

    def _get_valid_rules(self, examples, out_idx):
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

    def _format_op(self, rule):
        op, in1, in2 = rule
        if op in ['C0', 'C1']: return f"constant {op[-1]}"
        if op == 'I': return f"the value at index {in1}"
        if op == 'NOT': return f"the inverted value of index {in1}"
        return f"the {op} operation between index {in1} and index {in2}"

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
            return "Parse Error"
        
        target_input = target_match.group(1).zfill(8)
        
        cot = [
            "We need to deduce the 8-bit to 8-bit transformation rule from the provided examples.",
            "Instead of guessing the whole expression at once, let's analyze the transformation bit by bit, looking for continuous sequences (strides) where the operation remains the same but the input indices shift by +1.\n"
        ]

        flat_matches = [self._get_valid_rules(examples, i) for i in range(8)]

        # Left Run
        best_left_run = []
        if flat_matches[0]:
            for cand in flat_matches[0]:
                run = [cand]
                op, in1, in2 = cand
                for i in range(1, 8):
                    exp_in1 = (in1 + i) % 8 if in1 != -1 else -1
                    exp_in2 = (in2 + i) % 8 if in2 != -1 else -1
                    if (op, exp_in1, exp_in2) in flat_matches[i]:
                        run.append((op, exp_in1, exp_in2))
                    else:
                        break
                if len(run) > len(best_left_run):
                    best_left_run = run

        # Right Run
        best_right_run = []
        if flat_matches[7]:
            for cand in flat_matches[7]:
                run = [cand]
                op, in1, in2 = cand
                for step in range(1, 8):
                    i = 7 - step
                    exp_in1 = (in1 - step) % 8 if in1 != -1 else -1
                    exp_in2 = (in2 - step) % 8 if in2 != -1 else -1
                    if (op, exp_in1, exp_in2) in flat_matches[i]:
                        run.insert(0, (op, exp_in1, exp_in2))
                    else:
                        break
                if len(run) > len(best_right_run):
                    best_right_run = run

        len_l = len(best_left_run)
        len_r = len(best_right_run)
        
        cot.append(f"Analyzing from the left (bit 0), the longest consistent operation is `{best_left_run[0][0]}` which successfully covers {len_l} bits.")
        cot.append(f"Analyzing from the right (bit 7) backwards, the longest consistent operation is `{best_right_run[-1][0]}` which covers {len_r} bits.\n")

        # Truncation
        if len_l + len_r > 8:
            cot.append("The left and right sequences overlap. We must truncate the shorter sequence to resolve the conflict.")
            if len_r > len_l:
                len_l = 8 - len_r
                best_left_run = best_left_run[:len_l]
                cot.append("Since the right sequence is longer, we truncate the left sequence.")
            else:
                len_r = 8 - len_l
                best_right_run = best_right_run[-len_r:] if len_r > 0 else []
                cot.append("Since the left sequence is longer (or equal), we truncate the right sequence.")
            cot.append("")

        final_rules = [None] * 8
        for i in range(len_l): final_rules[i] = best_left_run[i]
        right_start_idx = 8 - len_r
        for i in range(len_r): final_rules[right_start_idx + i] = best_right_run[i]

        pending = [i for i in range(8) if final_rules[i] is None]
        
        # Filling Holes
        if pending:
            cot.append(f"Bits {pending} are currently unmatched. We will attempt to deduce their rules.")
            anchor_run = best_right_run if len_r > len_l else best_left_run
            anchor_idx = right_start_idx if len_r > len_l else 0
            base_op = anchor_run[0][0]
            
            can_extrapolate = True
            temp_rules = {}
            for p in pending:
                offset = p - anchor_idx
                exp_in1 = (anchor_run[0][1] + offset) % 8 if anchor_run[0][1] != -1 else -1
                exp_in2 = (anchor_run[0][2] + offset) % 8 if anchor_run[0][2] != -1 else -1
                if (base_op, exp_in1, exp_in2) in flat_matches[p]:
                    temp_rules[p] = (base_op, exp_in1, exp_in2)
                else:
                    can_extrapolate = False
                    break
            
            if can_extrapolate:
                cot.append(f"We can successfully extrapolate the `{base_op}` operation from the dominant sequence to fill all missing bits.")
                for p in pending: final_rules[p] = temp_rules[p]
                pending = []
            else:
                perfect_cat = None
                for cat in self.SECTION_ORDER:
                    if all(any(c[0] == cat for c in flat_matches[p]) for p in pending):
                        perfect_cat = cat
                        break
                if perfect_cat:
                    cot.append(f"Extrapolation failed, but we found that a `{perfect_cat}` operation perfectly fits all remaining bits independently.")
                    for p in pending:
                        final_rules[p] = next(c for c in flat_matches[p] if c[0] == perfect_cat)
                    pending = []

        if pending:
            cot.append("No unified pattern fits the remaining bits. We will apply the best local operation or default to 1 as a fallback.")
            for p in pending:
                if flat_matches[p]:
                    final_rules[p] = flat_matches[p][0]
                else:
                    final_rules[p] = ('C1', -1, -1)
        
        cot.append("\nNow, we apply the final derived rules mapping to the target input string: " + target_input)
        target_output = ""
        for i in range(8):
            op, in1, in2 = final_rules[i]
            val1 = int(target_input[in1]) if in1 != -1 else 0
            val2 = int(target_input[in2]) if in2 != -1 else 0
            res = self.ops[op](val1, val2)
            target_output += str(res)
            
            explanation = self._format_op(final_rules[i])
            cot.append(f"  Bit {i}: Use {explanation} -> {res}")

        cot.append(f"\nThe final answer is {target_output}.")
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return None
        match = re.search(r"The final answer is ([01]{8})\.", cot_text)
        return match.group(1) if match else None