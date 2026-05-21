import random
import pandas as pd

from src.solvers.bit_manipulation import BitManipulationSolver

bit_manipulation_prompt_template = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
{examples}

Now, determine the output for: {target_input}"""

class BitManipulationTaskGenerator:    
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.solver = BitManipulationSolver() # Возвращаем солвер

        self.ops = {
            'I': lambda a, b: a, 'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0, 'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b), 'XOR-NOT': lambda a, b: a ^ (1 - b)
        }
        self.op_names = list(self.ops.keys())
        self.macro_types = ['NOT_ALL', 'ROL', 'ROR', 'SHL', 'SHR', 'SHL_1', 'SHR_1']

    def _get_random_macro_id(self):
        macro_type = self.rng.choice(self.macro_types)
        if macro_type == 'NOT_ALL':
            return "NOT_ALL"
        shift = self.rng.randint(1, 7)
        return f"{macro_type}_{shift}"

    def _apply_macro_pattern(self, pattern_id: str, target_input: str) -> str:
        if pattern_id == "NOT_ALL": return "".join('1' if b == '0' else '0' for b in target_input)
        
        parts = pattern_id.split("_")
        if pattern_id.startswith("ROL_"): return target_input[int(parts[1]):] + target_input[:int(parts[1])]
        elif pattern_id.startswith("ROR_"): return target_input[-int(parts[1]):] + target_input[:-int(parts[1])]
        elif pattern_id.startswith("SHL_1_"): return target_input[int(parts[2]):] + '1'*int(parts[2])
        elif pattern_id.startswith("SHR_1_"): return '1'*int(parts[2]) + target_input[:-int(parts[2])]
        elif pattern_id.startswith("SHL_"): return target_input[int(parts[1]):] + '0'*int(parts[1])
        elif pattern_id.startswith("SHR_"): return '0'*int(parts[1]) + target_input[:-int(parts[1])]
        return ""

    def _create_rule(self, op: str, bit_idx: int, shift1: int, shift2: int):
        in1 = (bit_idx + shift1) % 8 if op not in ['C0', 'C1'] else -1
        in2 = (bit_idx + shift2) % 8 if op not in ['C0', 'C1', 'I', 'NOT'] else -1
        return (op, in1, in2)

    def _generate_pattern_rules(self):
        op = self.rng.choice(self.op_names)
        shift1 = self.rng.randint(-3, 3)
        shift2 = self.rng.randint(-3, 3)
        return [self._create_rule(op, i, shift1, shift2) for i in range(8)]

    def _generate_split_rules(self):
        split_idx = self.rng.randint(2, 6) 
        
        op1 = self.rng.choice(self.op_names)
        s1_1, s1_2 = self.rng.randint(-3, 3), self.rng.randint(-3, 3)
        
        op2 = self.rng.choice(self.op_names)
        s2_1, s2_2 = self.rng.randint(-3, 3), self.rng.randint(-3, 3)

        rules = []
        for i in range(8):
            if i < split_idx:
                rules.append(self._create_rule(op1, i, s1_1, s1_2))
            else:
                rules.append(self._create_rule(op2, i, s2_1, s2_2))
        return rules

    def _generate_split_gap_rules(self):
        left_len = self.rng.randint(2, 3)
        right_len = self.rng.randint(2, 3)
        
        op_l = self.rng.choice(self.op_names)
        sl_1, sl_2 = self.rng.randint(-3, 3), self.rng.randint(-3, 3)
        
        op_r = self.rng.choice(self.op_names)
        sr_1, sr_2 = self.rng.randint(-3, 3), self.rng.randint(-3, 3)

        rules = []
        for i in range(8):
            if i < left_len:
                rules.append(self._create_rule(op_l, i, sl_1, sl_2))
            elif i >= 8 - right_len:
                rules.append(self._create_rule(op_r, i, sr_1, sr_2))
            else:
                op_gap = self.rng.choice(self.op_names)
                g1, g2 = self.rng.randint(0, 7), self.rng.randint(0, 7)
                rules.append((op_gap, g1 if op_gap not in ['C0', 'C1'] else -1, g2 if op_gap not in ['C0', 'C1', 'I', 'NOT'] else -1))
        return rules

    def _apply_rules(self, input_str: str, rules: list) -> str:
        out = ""
        for i in range(8):
            op, in1, in2 = rules[i]
            val1 = int(input_str[in1]) if in1 != -1 else 0
            val2 = int(input_str[in2]) if in2 != -1 else 0
            out += str(self.ops[op](val1, val2))
        return out

    def generate_task(self, num_examples=8, mode='random'):
        valid_modes = ['macro', 'pattern', 'split', 'split_gap']
        if mode not in valid_modes + ['random']:
            raise ValueError(f"Mode must be one of {valid_modes + ['random']}")
            
        if mode == 'random':
            mode = self.rng.choice(valid_modes)

        inputs = set()
        while len(inputs) < num_examples + 1:
            inputs.add(format(self.rng.randint(0, 255), '08b'))
        
        inputs = list(inputs)
        examples_inputs = inputs[:-1]
        target_input = inputs[-1]

        prompt_lines = []
        rules_or_macro = None

        if mode == 'macro':
            macro_id = self._get_random_macro_id()
            rules_or_macro = macro_id
            for ex_in in examples_inputs:
                ex_out = self._apply_macro_pattern(macro_id, ex_in)
                prompt_lines.append(f"{ex_in} -> {ex_out}")
            target_out = self._apply_macro_pattern(macro_id, target_input)
            
        else:
            if mode == 'pattern': rules = self._generate_pattern_rules()
            elif mode == 'split': rules = self._generate_split_rules()
            elif mode == 'split_gap': rules = self._generate_split_gap_rules()
            
            rules_or_macro = rules
            for ex_in in examples_inputs:
                ex_out = self._apply_rules(ex_in, rules)
                prompt_lines.append(f"{ex_in} -> {ex_out}")
            target_out = self._apply_rules(target_input, rules)
        
        return "\n".join(prompt_lines), target_out, target_input, rules_or_macro, mode

    def generate_dataset(self, num_samples):
        results = []
        for i in range(num_samples):
            random_num_examples = self.rng.randint(8, 12)
            
            # Генерация задачи
            examples_line, answer, target_input, rules, task_mode = self.generate_task(
                num_examples=random_num_examples, mode='random'
            )
            prompt = bit_manipulation_prompt_template.format(examples=examples_line, target_input=target_input)
            
            # Восстановленный блок с солвером
            generated_cot = self.solver.generate_cot(prompt)
            computed_answer = self.solver.extract_answer(generated_cot)

            results.append({
                "prompt": prompt,
                "answer": answer,
                "generated_cot": generated_cot,
                "computed_answer": computed_answer,
                "task_mode": task_mode,
                "label": "bit manipulation"
            })
            
        return pd.DataFrame.from_records(results)

if __name__ == "__main__":
    generator = BitManipulationTaskGenerator(seed=42)
    
    # Тестовая генерация датасета из 5 примеров
    df = generator.generate_dataset(num_samples=5)
    
    # Выводим точность и примеры
    df['is_correct'] = df['answer'] == df['computed_answer']
    print(f"Accuracy: {df['is_correct'].mean() * 100}%\n")
    print(df[['task_mode', 'answer', 'computed_answer', 'is_correct']].head())