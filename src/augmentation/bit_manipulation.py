import random
import pandas as pd

from src.solvers.bit_manipulation import BitManipulationSolver

bit_manipulation_prompt_template = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
{examples}

Now, determine the output for: {target_input}"""


class BitManipulationTaskGenerator:    
    def __init__(self, seed: int = None):
        self.bit_manipulation_solver = BitManipulationSolver()
        self.rng = random.Random(seed)

        self.ops = {
            'I': lambda a, b: a, 'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0, 'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b), 'XOR-NOT': lambda a, b: a ^ (1 - b)
        }
        self.op_names = list(self.ops.keys())

    def _generate_pattern_rules(self):
        """Создает системное правило (например, каждый бит - это XOR текущего и сдвинутого на 1)."""
        op = self.rng.choice(self.op_names)
        shift1 = self.rng.randint(-3, 3)
        shift2 = self.rng.randint(-3, 3)
        
        rules = []
        for i in range(8):
            in1 = (i + shift1) % 8
            in2 = (i + shift2) % 8
            rules.append((op, in1, in2))
        return rules

    def _generate_mixed_rules(self):
        """Создает независимое случайное правило для каждого бита."""
        rules = []
        for i in range(8):
            op = self.rng.choice(self.op_names)
            in1 = self.rng.randint(0, 7)
            in2 = self.rng.randint(0, 7)
            rules.append((op, in1, in2))
        return rules

    def _apply_rules(self, input_str: str, rules: list) -> str:
        """Применяет правила к 8-битной строке для получения ответа."""
        out = ""
        for i in range(8):
            op, in1, in2 = rules[i]
            val1 = int(input_str[in1])
            val2 = int(input_str[in2])
            out += str(self.ops[op](val1, val2))
        return out

    def generate_task(self, num_examples=8, mode='pattern'):
        rules = self._generate_pattern_rules() if mode == 'pattern' else self._generate_mixed_rules()

        inputs = set()
        while len(inputs) < num_examples + 1:
            inputs.add(format(self.rng.randint(0, 255), '08b'))
        
        inputs = list(inputs)
        examples_inputs = inputs[:-1]
        target_input = inputs[-1]

        prompt_lines = []
        for ex_in in examples_inputs:
            ex_out = self._apply_rules(ex_in, rules)
            prompt_lines.append(f"{ex_in} -> {ex_out}")
        
        target_out = self._apply_rules(target_input, rules)
        
        return "\n".join(prompt_lines), target_out, target_input, rules

    def generate_dataset(self, num_samples):
        results = []
        for i in range(num_samples):
            # after EDA i see that example in train bit manipulation task is in range of [8, 12]
            random_num_examples = self.rng.randint(8, 12)
            examples_line, answer, target_input, rules = self.generate_task(num_examples=random_num_examples, mode='pattern')
            prompt = bit_manipulation_prompt_template.format(examples=examples_line, target_input=target_input)
            generated_cot = self.bit_manipulation_solver.generate_cot(prompt)
            computed_answer = self.bit_manipulation_solver.extract_answer(generated_cot)

            results.append({
                "prompt": prompt,
                "answer": answer,
                "generated_cot": generated_cot,
                "computed_answer": computed_answer,
                "label": "bit manipulation"
            })
        return pd.DataFrame.from_records(results)

if __name__ == "__main__":

    generator = BitManipulationTaskGenerator(seed=12)
    solver = BitManipulationSolver()
    
    results = []
    for i in range(2):
        examples_line, answer, target_input, rules = generator.generate_task(num_examples=8, mode='pattern')
        prompt = bit_manipulation_prompt_template.format(examples=examples_line, target_input=target_input)
        cot = solver.generate_cot(prompt)
        computed_answer = solver.extract_answer(cot)
        print(prompt)

        if answer == computed_answer:
            d = 1
        else:
            d = 0
        
        results.append(d)
    
    print(sum(results) / float(len(results)))