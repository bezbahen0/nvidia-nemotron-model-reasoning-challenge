import re


class BitManipulationSolver:
    def __init__(self, max_macro_proof_examples=10):
        self.ops = {
            'I': lambda a, b: a,
            'NOT': lambda a, b: 1 - a,
            'C0': lambda a, b: 0,
            'C1': lambda a, b: 1,
            'AND': lambda a, b: a & b,
            'OR': lambda a, b: a | b,
            'XOR': lambda a, b: a ^ b,
            'AND-NOT': lambda a, b: a & (1 - b),
            'OR-NOT': lambda a, b: a | (1 - b),
            'XOR-NOT': lambda a, b: a ^ (1 - b),
        }

        self.bin_ops_keys = ['AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']
        self.SECTION_ORDER = ['I', 'NOT', 'C0', 'C1'] + self.bin_ops_keys
        self.max_macro_proof_examples = max_macro_proof_examples

    def _normalize_bits(self, value):
        bits = re.sub(r'[^01]', '', value)

        if not bits:
            return ''

        if len(bits) < 8:
            return bits.zfill(8)

        if len(bits) == 8:
            return bits

        return bits[-8:]

    def _parse_prompt(self, prompt):
        examples = []

        for line in prompt.splitlines():
            if '->' not in line:
                continue

            left, right = line.split('->', 1)

            left_tokens = re.findall(r'(?<![01])([01]{1,8})(?![01])', left)
            right_tokens = re.findall(r'(?<![01])([01]{1,8})(?![01])', right)

            if left_tokens and right_tokens:
                in_str = self._normalize_bits(left_tokens[-1])
                out_str = self._normalize_bits(right_tokens[0])
            else:
                in_str = self._normalize_bits(left)
                out_str = self._normalize_bits(right)

            if len(in_str) == 8 and len(out_str) == 8:
                examples.append((in_str, out_str))

        target_match = re.search(
            r'(?:output\s+for|target)\s*:?\s*([01]{1,8})(?![01])',
            prompt,
            re.IGNORECASE,
        )

        if not target_match:
            return examples, ''

        target_input = self._normalize_bits(target_match.group(1))
        return examples, target_input

    def _get_valid_rules(self, examples, out_idx):
        valid = []

        for op_name in self.SECTION_ORDER:
            op_func = self.ops[op_name]

            if op_name in ['C0', 'C1']:
                if all(int(ex_out[out_idx]) == op_func(0, 0) for _, ex_out in examples):
                    valid.append((op_name, -1, -1))

            elif op_name in ['I', 'NOT']:
                for in1 in range(8):
                    if all(
                        int(ex_out[out_idx]) == op_func(int(ex_in[in1]), 0)
                        for ex_in, ex_out in examples
                    ):
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
        if all(
            ex_out == ''.join('1' if bit == '0' else '0' for bit in ex_in)
            for ex_in, ex_out in examples
        ):
            return 'NOT_ALL', 'invert all bits'

        for shift in range(1, 8):
            if all(ex_out == ex_in[shift:] + ex_in[:shift] for ex_in, ex_out in examples):
                return f'ROL_{shift}', f'rotate left by {shift}'

            if all(ex_out == ex_in[-shift:] + ex_in[:-shift] for ex_in, ex_out in examples):
                return f'ROR_{shift}', f'rotate right by {shift}'

            if all(ex_out == ex_in[shift:] + '0' * shift for ex_in, ex_out in examples):
                return f'SHL_{shift}', f'shift left by {shift}, padding with 0'

            if all(ex_out == '0' * shift + ex_in[:-shift] for ex_in, ex_out in examples):
                return f'SHR_{shift}', f'shift right by {shift}, padding with 0'

            if all(ex_out == ex_in[shift:] + '1' * shift for ex_in, ex_out in examples):
                return f'SHL_1_{shift}', f'shift left by {shift}, padding with 1'

            if all(ex_out == '1' * shift + ex_in[:-shift] for ex_in, ex_out in examples):
                return f'SHR_1_{shift}', f'shift right by {shift}, padding with 1'

        return None, None

    def _apply_macro_pattern(self, pattern_id, target_input):
        if pattern_id == 'NOT_ALL':
            return ''.join('1' if bit == '0' else '0' for bit in target_input)

        if pattern_id.startswith('ROL_'):
            shift = int(pattern_id.split('_')[1])
            return target_input[shift:] + target_input[:shift]

        if pattern_id.startswith('ROR_'):
            shift = int(pattern_id.split('_')[1])
            return target_input[-shift:] + target_input[:-shift]

        if pattern_id.startswith('SHL_1_'):
            shift = int(pattern_id.split('_')[2])
            return target_input[shift:] + '1' * shift

        if pattern_id.startswith('SHR_1_'):
            shift = int(pattern_id.split('_')[2])
            return '1' * shift + target_input[:-shift]

        if pattern_id.startswith('SHL_'):
            shift = int(pattern_id.split('_')[1])
            return target_input[shift:] + '0' * shift

        if pattern_id.startswith('SHR_'):
            shift = int(pattern_id.split('_')[1])
            return '0' * shift + target_input[:-shift]

        return ''

    def _find_best_runs(self, flat_matches):
        best_left_run = []
        best_right_run = []

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

        return best_left_run, best_right_run

    def _select_final_rules(self, flat_matches):
        best_left_run, best_right_run = self._find_best_runs(flat_matches)

        len_l = len(best_left_run)
        len_r = len(best_right_run)

        if len_l + len_r > 8:
            if len_r > len_l:
                len_l = 8 - len_r
                best_left_run = best_left_run[:len_l]
            else:
                len_r = 8 - len_l
                best_right_run = best_right_run[-len_r:] if len_r > 0 else []

        final_rules = [None] * 8

        for i in range(len_l):
            final_rules[i] = best_left_run[i]

        for i in range(len_r):
            out_idx = 8 - len_r + i
            final_rules[out_idx] = best_right_run[i]

        for out_idx in range(8):
            if final_rules[out_idx] is None:
                if flat_matches[out_idx]:
                    final_rules[out_idx] = flat_matches[out_idx][0]
                else:
                    final_rules[out_idx] = ('UNKNOWN', -1, -1)

        return final_rules

    def _expected_column(self, examples, out_idx):
        return ''.join(ex_out[out_idx] for _, ex_out in examples)

    def _apply_rule(self, rule, input_bits):
        op, in1, in2 = rule

        if op == 'UNKNOWN':
            return 1

        if op in ['C0', 'C1']:
            return self.ops[op](0, 0)

        if op in ['I', 'NOT']:
            return self.ops[op](int(input_bits[in1]), 0)

        return self.ops[op](int(input_bits[in1]), int(input_bits[in2]))

    def _rule_column(self, rule, examples):
        op, _, _ = rule

        if op == 'UNKNOWN':
            return '?' * len(examples)

        return ''.join(str(self._apply_rule(rule, ex_in)) for ex_in, _ in examples)

    def _rule_to_text(self, rule):
        op, in1, in2 = rule

        if op == 'UNKNOWN':
            return 'UNKNOWN'

        if op == 'C0':
            return '0'

        if op == 'C1':
            return '1'

        if op == 'I':
            return f'in[{in1}]'

        if op == 'NOT':
            return f'NOT(in[{in1}])'

        return f'{op}(in[{in1}], in[{in2}])'

    def _rule_assignment_text(self, out_idx, rule):
        return f'out[{out_idx}] = {self._rule_to_text(rule)}'

    def _operation_eval_text(self, rule, input_bits):
        op, in1, in2 = rule

        if op == 'UNKNOWN':
            return 'UNKNOWN -> fallback 1', 1

        if op == 'C0':
            return '0', 0

        if op == 'C1':
            return '1', 1

        if op == 'I':
            v1 = int(input_bits[in1])
            return f'in[{in1}] = {v1}', v1

        if op == 'NOT':
            v1 = int(input_bits[in1])
            result = self.ops[op](v1, 0)
            return f'NOT(in[{in1}]) = NOT({v1}) = {result}', result

        v1 = int(input_bits[in1])
        v2 = int(input_bits[in2])
        result = self.ops[op](v1, v2)

        if op == 'AND':
            return f'AND(in[{in1}], in[{in2}]) = AND({v1}, {v2}) = {result}', result

        if op == 'OR':
            return f'OR(in[{in1}], in[{in2}]) = OR({v1}, {v2}) = {result}', result

        if op == 'XOR':
            return f'XOR(in[{in1}], in[{in2}]) = XOR({v1}, {v2}) = {result}', result

        if op == 'AND-NOT':
            return f'AND-NOT(in[{in1}], in[{in2}]) = {v1} AND NOT({v2}) = {result}', result

        if op == 'OR-NOT':
            return f'OR-NOT(in[{in1}], in[{in2}]) = {v1} OR NOT({v2}) = {result}', result

        if op == 'XOR-NOT':
            return f'XOR-NOT(in[{in1}], in[{in2}]) = {v1} XOR NOT({v2}) = {result}', result

        return f'{op}(in[{in1}], in[{in2}]) = {result}', result

    def _rule_input_signature(self, rule):
        op, in1, in2 = rule

        if op in ['C0', 'C1', 'UNKNOWN']:
            return None

        if op in ['I', 'NOT']:
            return (in1,)

        return (in1, in2)

    def _is_shifted_from_previous(self, previous_rule, current_rule):
        previous_op, _, _ = previous_rule
        current_op, _, _ = current_rule

        if previous_op != current_op:
            return False

        previous_signature = self._rule_input_signature(previous_rule)
        current_signature = self._rule_input_signature(current_rule)

        if previous_signature is None or current_signature is None:
            return False

        if len(previous_signature) != len(current_signature):
            return False

        return all(
            (previous_signature[idx] + 1) % 8 == current_signature[idx]
            for idx in range(len(previous_signature))
        )

    def _is_same_constant_or_unknown(self, previous_rule, current_rule):
        previous_op, _, _ = previous_rule
        current_op, _, _ = current_rule

        if previous_op not in ['C0', 'C1', 'UNKNOWN']:
            return False

        return previous_op == current_op

    def _pattern_group_name(self, rules):
        first_rule = rules[0]
        op, _, _ = first_rule

        if op == 'I':
            return 'shifted copy pattern'

        if op == 'NOT':
            return 'shifted NOT pattern'

        if op == 'C0':
            return 'constant 0 pattern'

        if op == 'C1':
            return 'constant 1 pattern'

        if op == 'UNKNOWN':
            return 'fallback pattern'

        return f'shifted {op} pattern'

    def _pattern_summary_lines(self, final_rules):
        lines = []
        start = 0

        while start < 8:
            end = start

            while end + 1 < 8 and self._is_shifted_from_previous(final_rules[end], final_rules[end + 1]):
                end += 1

            if end > start:
                group_rules = final_rules[start:end + 1]
                group_name = self._pattern_group_name(group_rules)
                lines.append(f'out[{start}..{end}] use a {group_name}:')
                for out_idx in range(start, end + 1):
                    lines.append(self._rule_assignment_text(out_idx, final_rules[out_idx]))
                start = end + 1
                continue

            end = start

            while end + 1 < 8 and self._is_same_constant_or_unknown(final_rules[end], final_rules[end + 1]):
                end += 1

            if end > start:
                group_rules = final_rules[start:end + 1]
                group_name = self._pattern_group_name(group_rules)
                lines.append(f'out[{start}..{end}] use a {group_name}:')
                for out_idx in range(start, end + 1):
                    lines.append(self._rule_assignment_text(out_idx, final_rules[out_idx]))
                start = end + 1
                continue

            lines.append(f'out[{start}] is a single rule:')
            lines.append(self._rule_assignment_text(start, final_rules[start]))
            start += 1

        return lines

    def _solve(self, prompt):
        examples, target_input = self._parse_prompt(prompt)

        if not examples or len(target_input) != 8:
            return {
                'status': 'parse_error',
                'task_type': None,
                'examples': examples,
                'target_input': target_input,
                'target_output': 'nan',
                'macro_id': None,
                'macro_desc': None,
                'final_rules': None,
            }

        macro_id, macro_desc = self._detect_macro_pattern(examples)

        if macro_id is not None:
            target_output = self._apply_macro_pattern(macro_id, target_input)

            return {
                'status': 'ok',
                'task_type': 'macro',
                'examples': examples,
                'target_input': target_input,
                'target_output': target_output,
                'macro_id': macro_id,
                'macro_desc': macro_desc,
                'final_rules': None,
            }

        flat_matches = [self._get_valid_rules(examples, out_idx) for out_idx in range(8)]
        final_rules = self._select_final_rules(flat_matches)

        target_output = ''.join(str(self._apply_rule(rule, target_input)) for rule in final_rules)

        return {
            'status': 'ok',
            'task_type': 'bit_rules',
            'examples': examples,
            'target_input': target_input,
            'target_output': target_output,
            'macro_id': None,
            'macro_desc': None,
            'final_rules': final_rules,
        }

    def _render_macro_cot(self, solution):
        examples = solution['examples']
        target_input = solution['target_input']
        target_output = solution['target_output']
        macro_id = solution['macro_id']
        macro_desc = solution['macro_desc']

        cot = []

        cot.append('Task type: macro')
        cot.append('')
        cot.append('Indexing:')
        cot.append('Bits are indexed left-to-right as 0..7.')
        cot.append('')
        cot.append('Macro rule matching:')
        cot.append(f'{macro_desc} maps each given input to its expected output.')
        cot.append(f'Chosen macro: {macro_desc}')
        cot.append(f'Macro id: {macro_id}')
        cot.append('')
        cot.append('Macro check:')

        proof_examples = examples[:self.max_macro_proof_examples]

        for ex_idx, (ex_in, ex_out) in enumerate(proof_examples, start=1):
            predicted = self._apply_macro_pattern(macro_id, ex_in)
            status = 'ok' if predicted == ex_out else 'mismatch'
            cot.append(f'ex{ex_idx}: {ex_in} -> {predicted}; expected {ex_out}; {status}')

        if len(examples) > len(proof_examples):
            remaining = len(examples) - len(proof_examples)
            cot.append(f'... {remaining} more example(s) also match this macro rule.')

        cot.append('')
        cot.append('Apply to target:')
        cot.append(f'target = {target_input}')
        cot.append(f'{macro_desc}: {target_input} -> {target_output}')
        cot.append('')
        cot.append(f'The final answer: {target_output}')

        return '\n'.join(cot)

    def _render_bit_rules_cot(self, solution):
        examples = solution['examples']
        target_input = solution['target_input']
        target_output = solution['target_output']
        final_rules = solution['final_rules']

        cot = []

        cot.append('Task type: bit_rules')
        cot.append('')
        cot.append('Indexing:')
        cot.append('Bits are indexed left-to-right as 0..7.')
        cot.append('')
        cot.append('Output columns:')
        cot.append('Each output bit is checked as a column across all examples.')

        for out_idx in range(8):
            expected = self._expected_column(examples, out_idx)
            cot.append(f'out[{out_idx}] expected = {expected}')

        cot.append('')
        cot.append('Rule matching:')

        for out_idx, selected_rule in enumerate(final_rules):
            expected = self._expected_column(examples, out_idx)
            selected_column = self._rule_column(selected_rule, examples)
            rule_text = self._rule_to_text(selected_rule)

            if selected_rule[0] == 'UNKNOWN':
                cot.append(
                    f'out[{out_idx}]: expected {expected}; no matching standard rule -> '
                    f'choose out[{out_idx}] = UNKNOWN fallback 1'
                )
            else:
                cot.append(
                    f'out[{out_idx}]: expected {expected}; {rule_text} gives {selected_column} -> '
                    f'choose out[{out_idx}] = {rule_text}'
                )

        cot.append('')
        cot.append('Pattern summary:')

        for line in self._pattern_summary_lines(final_rules):
            cot.append(line)

        cot.append('')
        cot.append('Apply to target:')
        cot.append(f'target = {target_input}')
        cot.append('target bits:')
        cot.append(' '.join(f'in[{idx}]={bit}' for idx, bit in enumerate(target_input)))
        cot.append('')

        output_bits = []

        for out_idx, rule in enumerate(final_rules):
            eval_text, result = self._operation_eval_text(rule, target_input)
            output_bits.append(str(result))
            cot.append(f'out[{out_idx}] = {eval_text}')

        recomputed_output = ''.join(output_bits)

        cot.append('')
        cot.append(f'result = {recomputed_output}')
        cot.append(f'The final answer: {target_output}')

        return '\n'.join(cot)

    def generate_cot(self, prompt: str) -> str:
        solution = self._solve(prompt)

        if solution['status'] == 'parse_error':
            return '\n'.join([
                'Task type: parse_error',
                'Reason: could not parse at least one example pair or the target input.',
                'The final answer: nan',
            ])

        if solution['task_type'] == 'macro':
            return self._render_macro_cot(solution)

        return self._render_bit_rules_cot(solution)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text:
            return 'nan'

        match = re.search(r'(?i)final\s+answer:\s*([01]{8})', cot_text)
        return match.group(1) if match else 'nan'