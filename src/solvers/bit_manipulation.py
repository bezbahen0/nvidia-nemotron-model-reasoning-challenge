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
        self.bin_ops_keys = ['AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']
        self.symmetric_ops = ['AND', 'OR', 'XOR']
        self.SECTION_ORDER = ['I', 'NOT', 'C0', 'C1'] + self.bin_ops_keys
        self.max_proof_examples = max_proof_examples

    def _get_bitsum(self, bit_str):
        """Хэширование по методу Tong Hui Kang: 'a', если все биты одинаковые, иначе сумма."""
        if len(set(bit_str)) == 1:
            return 'a'
        return str(bit_str.count('1'))

    # === СОХРАНЯЕМ ВАШИ ОРИГИНАЛЬНЫЕ ФУНКЦИИ ПОИСКА ===
    def _get_valid_rules(self, examples, out_idx):
        # ... (Ваш оригинальный код без изменений) ...
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
        # ... (Ваш оригинальный код без изменений) ...
        if all(ex_out == "".join('1' if b == '0' else '0' for b in ex_in) for ex_in, ex_out in examples): return "NOT_ALL", "invert all bits"
        for shift in range(1, 8):
            if all(ex_out == ex_in[shift:] + ex_in[:shift] for ex_in, ex_out in examples): return f"ROL_{shift}", f"rotate left by {shift}"
            if all(ex_out == ex_in[-shift:] + ex_in[:-shift] for ex_in, ex_out in examples): return f"ROR_{shift}", f"rotate right by {shift}"
            if all(ex_out == ex_in[shift:] + '0'*shift for ex_in, ex_out in examples): return f"SHL_{shift}", f"shift left by {shift} (pad 0)"
            if all(ex_out == '0'*shift + ex_in[:-shift] for ex_in, ex_out in examples): return f"SHR_{shift}", f"shift right by {shift} (pad 0)"
            if all(ex_out == ex_in[shift:] + '1'*shift for ex_in, ex_out in examples): return f"SHL_1_{shift}", f"shift left by {shift} (pad 1)"
            if all(ex_out == '1'*shift + ex_in[:-shift] for ex_in, ex_out in examples): return f"SHR_1_{shift}", f"shift right by {shift} (pad 1)"
        return None, None

    def _apply_macro_pattern(self, pattern_id, target_input):
         # ... (Ваш оригинальный код без изменений) ...
        if pattern_id == "NOT_ALL": return "".join('1' if b == '0' else '0' for b in target_input)
        elif pattern_id.startswith("ROL_"): return target_input[int(pattern_id.split("_")[1]):] + target_input[:int(pattern_id.split("_")[1])]
        elif pattern_id.startswith("ROR_"): return target_input[-int(pattern_id.split("_")[1]):] + target_input[:-int(pattern_id.split("_")[1])]
        elif pattern_id.startswith("SHL_1_"): return target_input[int(pattern_id.split("_")[2]):] + '1'*int(pattern_id.split("_")[2])
        elif pattern_id.startswith("SHR_1_"): return '1'*int(pattern_id.split("_")[2]) + target_input[:-int(pattern_id.split("_")[2])]
        elif pattern_id.startswith("SHL_"): return target_input[int(pattern_id.split("_")[1]):] + '0'*int(pattern_id.split("_")[1])
        elif pattern_id.startswith("SHR_"): return '0'*int(pattern_id.split("_")[1]) + target_input[:-int(pattern_id.split("_")[1])]
        return ""

    # === ГЕНЕРАТОР CoT (ИЛЛЮЗИЯ ВЫЧИСЛЕНИЙ) ===
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
        cot = []

        # ШАГ 1: Внутреннее решение задачи скриптом (Оракул)
        macro_id, macro_desc = self._detect_macro_pattern(examples)
        is_macro = macro_id is not None
        
        flat_matches = [self._get_valid_rules(examples, i) for i in range(8)]
        best_left_run, best_right_run = [], []
        
        if not is_macro:
            # ОРИГИНАЛЬНАЯ ЛОГИКА ПОИСКА STRIDES
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

        # ШАГ 2: Генерация текста (Обучающий материал)
        if is_macro:
            # Для макросов матрица не нужна, генерируем структурный разбор
            cot.append("Macro pattern detected.")
            cot.append(f"Operation perfectly satisfying all examples: {macro_desc}.")
            target_output = self._apply_macro_pattern(macro_id, target_input)
        
        else:
            # Для микроопераций ГЕНЕРИРУЕМ МАТРИЦУ TONG HUI KANG
            out_cols = ["".join(ex[1][i] for ex in examples) for i in range(8)]
            matches_dict = defaultdict(list)
            
            for op_name in self.bin_ops_keys:
                cot.append(op_name)
                op_func = self.ops[op_name]
                is_sym = op_name in self.symmetric_ops
                
                for diff in range(1, 8):
                    for start in range(8):
                        i = start
                        j = (start + diff) % 8
                        if is_sym and i > j: continue
                        
                        res_str = "".join(str(op_func(int(ex_in[i]), int(ex_in[j]))) for ex_in, _ in examples)
                        bitsum = self._get_bitsum(res_str)
                        
                        matched_out_idx = [idx for idx, col in enumerate(out_cols) if res_str == col]
                        if matched_out_idx:
                            for out_idx in matched_out_idx:
                                matches_dict[out_idx].append((op_name, i, j))
                                
                        line = f"{i}{j} {j}{i} {res_str} {bitsum}"
                        if matched_out_idx: line += " match " + " ".join(map(str, matched_out_idx))
                        cot.append(line)
                    cot.append("")
            
            # Печать совпадений и Stride логики
            cot.append("Matching output")
            for out_idx in range(8):
                if flat_matches[out_idx]:
                    # Показываем все валидные операции для бита (вкл. константы и унарные)
                    valid_strs = []
                    for op, in1, in2 in flat_matches[out_idx]:
                        if in2 == -1: valid_strs.append(f"{op}{in1 if in1 != -1 else ''}")
                        else: valid_strs.append(f"{op}{in1}{in2}")
                    cot.append(f"{out_idx} " + " ".join(valid_strs))
                else:
                    cot.append(f"{out_idx} None")
            cot.append("")

            cot.append("Left")
            cot.append("Best left run: " + " ".join([f"{op}{in1}{in2 if in2!=-1 else ''}" for op, in1, in2 in best_left_run]))
            cot.append("Right")
            cot.append("Best right run: " + " ".join([f"{op}{in1}{in2 if in2!=-1 else ''}" for op, in1, in2 in best_right_run]))
            cot.append("")

            # Сборка финальных правил
            len_l, len_r = len(best_left_run), len(best_right_run)
            if len_l + len_r > 8:
                if len_r > len_l: len_l = 8 - len_r; best_left_run = best_left_run[:len_l]
                else: len_r = 8 - len_l; best_right_run = best_right_run[-len_r:] if len_r > 0 else []

            final_rules = [None] * 8
            for i in range(len_l): final_rules[i] = best_left_run[i]
            for i in range(len_r): final_rules[8 - len_r + i] = best_right_run[i]
            for p in range(8):
                if final_rules[p] is None:
                    final_rules[p] = flat_matches[p][0] if flat_matches[p] else ('UNKNOWN', -1, -1)

            # Вычисление результата с подробным логированием
            cot.append("Selected")
            for idx, rule in enumerate(final_rules):
                op, in1, in2 = rule
                if op == 'UNKNOWN': cot.append(f"{idx} UNKNOWN (Default to 1)")
                elif in2 == -1: cot.append(f"{idx} {op}{in1 if in1 != -1 else ''}")
                else: cot.append(f"{idx} {op}{in1}{in2}")

            cot.append(f"\nApplying to {target_input}")
            cot.append("Input")
            for idx, bit in enumerate(target_input): cot.append(f"{idx} {bit}")
            cot.append("Output")
            
            target_output_arr = []
            for bit_idx, rule in enumerate(final_rules):
                op, in1, in2 = rule
                if op == 'UNKNOWN':
                    target_output_arr.append('1')
                    cot.append(f"{bit_idx} UNKNOWN = 1")
                else:
                    v1_t = int(target_input[in1]) if in1 != -1 else 0
                    v2_t = int(target_input[in2]) if in2 != -1 else 0
                    res = self.ops[op](v1_t, v2_t)
                    target_output_arr.append(str(res))
                    
                    if op in ['C0', 'C1']: cot.append(f"{bit_idx} {op} = {res}")
                    elif op in ['I', 'NOT']: cot.append(f"{bit_idx} {op}{in1} = {op}({v1_t}) = {res}")
                    else: cot.append(f"{bit_idx} {op}{in1}{in2} = {op}({v1_t},{v2_t}) = {res}")

            target_output = "".join(target_output_arr)

        cot.append(f"\nThe final answer: {target_output}")
        
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in cot_text:
            return "nan"
        match = re.search(r"(?i)final\s+answer:\s*([01]{8})", cot_text)
        return match.group(1) if match else "nan"