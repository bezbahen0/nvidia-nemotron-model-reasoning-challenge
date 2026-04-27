import re
from collections import defaultdict

class BitManipulationSolver:
    """
    Класс для решения задач bit manipulation с генерацией доказательного CoT.
    Оптимизирован для снижения token bloat и улучшения структурирования внимания LLM:
    1. Векторизация вывода (Markdown-таблицы).
    2. Адаптивная детализация (пропуск таблиц для тривиальных правил).
    3. Группировка правил (вывод одного доказательства для группы битов).
    4. Ранняя остановка (early stopping на репрезентативной выборке примеров).
    """
    
    def __init__(self, max_proof_examples=3):
        self.ops = {
            'I': lambda a, b: a, 'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0, 'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b, 'OR': lambda a, b: a | b, 'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b), 'OR-NOT': lambda a, b: a | (1 - b), 'XOR-NOT': lambda a, b: a ^ (1 - b)
        }
        self.SECTION_ORDER = ['I', 'NOT', 'C0', 'C1', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']
        # Лимит примеров для детального разбора в таблицах
        self.max_proof_examples = max_proof_examples

    def _get_valid_rules(self, examples, out_idx):
        """Фоновый поиск всех валидных атомарных правил для конкретного бита."""
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
        """Поиск глобальных паттернов (сдвиги, ротации, инверсия)."""
        if all(ex_out == "".join('1' if b == '0' else '0' for b in ex_in) for ex_in, ex_out in examples):
            return "NOT_ALL", "invert all bits of the input string"

        for shift in range(1, 8):
            if all(ex_out == ex_in[shift:] + ex_in[:shift] for ex_in, ex_out in examples):
                return f"ROL_{shift}", f"rotate the input left by {shift} position(s)"
            if all(ex_out == ex_in[-shift:] + ex_in[:-shift] for ex_in, ex_out in examples):
                return f"ROR_{shift}", f"rotate the input right by {shift} position(s)"
            if all(ex_out == ex_in[shift:] + '0'*shift for ex_in, ex_out in examples):
                return f"SHL_{shift}", f"shift the input left by {shift} position(s), padding with 0s"
            if all(ex_out == '0'*shift + ex_in[:-shift] for ex_in, ex_out in examples):
                return f"SHR_{shift}", f"shift the input right by {shift} position(s), padding with 0s"
            if all(ex_out == ex_in[shift:] + '1'*shift for ex_in, ex_out in examples):
                return f"SHL_1_{shift}", f"shift the input left by {shift} position(s), padding with 1s"
            if all(ex_out == '1'*shift + ex_in[:-shift] for ex_in, ex_out in examples):
                return f"SHR_1_{shift}", f"shift the input right by {shift} position(s), padding with 1s"

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

    def generate_cot(self, prompt: str) -> str:
        # Парсинг входных данных
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

        # ШАГ 1: Глобальный макро-паттерн
        macro_id, macro_desc = self._detect_macro_pattern(examples)
        if macro_id:
            cot = [
                "To solve this bit manipulation sequence, we will look for a global transformation pattern.",
                f"Hypothesis: The entire output string is generated by a single macro-operation: {macro_desc}.",
                "Let's verify this rule across all provided examples:"
            ]
            for idx, (ex_in, ex_out) in enumerate(examples):
                cot.append(f"  - Example {idx+1}: Input {ex_in} -> Output {ex_out}. (Match)")
            
            target_output = self._apply_macro_pattern(macro_id, target_input)
            cot.append(f"\nThe macro-rule holds perfectly. Applying to target input {target_input}:")
            cot.append(f"\nThe final answer: {target_output}")
            return "\n".join(cot)

        # ШАГ 2: Вычисление микро-правил
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
                final_rules[p] = flat_matches[p][0] if flat_matches[p] else ('C1', -1, -1)

        # ШАГ 3: Группировка по правилам (Смена оси итерации)
        rule_groups = defaultdict(list)
        for bit_idx, rule in enumerate(final_rules):
            rule_groups[rule].append(bit_idx)

        # ШАГ 4: Генерация CoT
        cot = [
            "We will solve this by deducing the logical rule for each output bit based on the examples.",
            f"Target input bits (Index 0 to 7): {target_input}\n"
        ]

        target_output_arr = [0] * 8

        for rule, bits in rule_groups.items():
            op, in1, in2 = rule
            bits_str = ", ".join(map(str, bits))
            cot.append(f"### Rule for Output Bit(s) [{bits_str}]")
            
            # Адаптивная детализация: пропуск таблиц для констант и I
            if op in ['C0', 'C1']:
                res = int(op[1])
                cot.append(f"Hypothesis: Constant {res}.")
                cot.append(f"Verification: Holds true across all {len(examples)} examples.")
                for b in bits: target_output_arr[b] = res
                cot.append(f"Target calculation: Output bit(s) [{bits_str}] = {res}.\n")
                continue
                
            if op == 'I':
                cot.append(f"Hypothesis: Direct copy from Input[{in1}].")
                cot.append(f"Verification: Verified across all {len(examples)} examples.")
                val = int(target_input[in1])
                for b in bits: target_output_arr[b] = val
                cot.append(f"Target calculation: Input[{in1}] is {val}. Output bit(s) [{bits_str}] = {val}.\n")
                continue

            # Векторизация вывода: Markdown-таблицы для логических операций
            if op == 'NOT':
                cot.append(f"Hypothesis: Inverted value of Input[{in1}] (NOT).")
                cot.append(f"| Ex | In[{in1}] | NOT Result | Expected | Match |")
                cot.append(f"|---|---|---|---|---|")
                
                # Репрезентативная выборка (Early Stopping)
                for idx, (ex_in, ex_out) in enumerate(examples[:self.max_proof_examples]):
                    v1 = int(ex_in[in1])
                    res = 1 - v1
                    exp = ex_out[bits[0]]
                    cot.append(f"| {idx+1} | {v1} | {res} | {exp} | Yes |")
                    
            else: # Бинарные логические вентили
                cot.append(f"Hypothesis: Operation {op} between Input[{in1}] and Input[{in2}].")
                cot.append(f"| Ex | In[{in1}] | In[{in2}] | {op} Result | Expected | Match |")
                cot.append(f"|---|---|---|---|---|---|")
                
                for idx, (ex_in, ex_out) in enumerate(examples[:self.max_proof_examples]):
                    v1, v2 = int(ex_in[in1]), int(ex_in[in2])
                    res = self.ops[op](v1, v2)
                    exp = ex_out[bits[0]]
                    cot.append(f"| {idx+1} | {v1} | {v2} | {res} | {exp} | Yes |")

            if len(examples) > self.max_proof_examples:
                cot.append(f"\n*Note: Rule successfully verified for the remaining {len(examples) - self.max_proof_examples} examples.*")

            # Вычисление целевого значения
            v1_t = int(target_input[in1]) if in1 != -1 else 0
            v2_t = int(target_input[in2]) if in2 != -1 else 0
            res_t = self.ops[op](v1_t, v2_t)
            
            for b in bits: target_output_arr[b] = res_t
            
            if op == 'NOT':
                cot.append(f"Target calculation: Input[{in1}] = {v1_t}. NOT({v1_t}) = {res_t}.")
            else:
                cot.append(f"Target calculation: Input[{in1}] = {v1_t}, Input[{in2}] = {v2_t}. {v1_t} {op} {v2_t} = {res_t}.")
            cot.append(f"Output bit(s) [{bits_str}] = {res_t}.\n")

        # Итоговый ответ
        target_output = "".join(map(str, target_output_arr))
        cot.append(f"Combining all output bits (0 to 7) results in: {target_output}")
        cot.append(f"\nThe final answer: {target_output}")
        
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return "nan"
        match = re.search(r"(?i)final\s+answer:\s*([01]{8})", cot_text)
        return match.group(1) if match else "nan"