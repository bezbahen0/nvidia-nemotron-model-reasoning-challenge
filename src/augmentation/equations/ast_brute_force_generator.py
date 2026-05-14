import hashlib
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pandarallel import pandarallel

from src.solvers.equations.ast_brute_force import ASTBruteForceSolver



ast_brute_force_prompt_template = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
{examples}
Now, determine the result for: {target}"""

AST_OPERATOR_CHARS = [
    "!", '"', "#", "$", "%", "&", "'", "(", ")",
    "*", "+", "-", "/", ":", "<", ">", "?",
    "@", "[", "\\", "]", "^", "`", "{", "|", "}"
]
AST_CONFIGS = ["fwd", "swap_ops", "rev_digits", "swap_rev"]
AST_FALLBACK_POOL = ["add", "sub", "cat", "mul", "div"]


@dataclass(frozen=True)
class ASTRule:
    config: str
    operation: str
    output_format: str


@dataclass(frozen=True)
class ASTExpression:
    left: str
    operator: str
    right: str

    def as_target_text(self) -> str:
        return f"{self.left}{self.operator}{self.right}"

    def as_example_line(self, output: str) -> str:
        return f"{self.left}{self.operator}{self.right} = {output}"


class ASTBruteForceTaskGenerator:
    """
    Dataset generator for ASTBruteForceSolver.

    It samples from the same hypothesis space that the solver searches:
      - operand configs: fwd, swap_ops, rev_digits, swap_rev;
      - operations returned by ASTBruteForceSolver._get_operations;
      - output formats returned by ASTBruteForceSolver._get_formats;
      - operator symbols from the user's punctuation dictionary;
      - the solver's unseen-target-operator fallback path.

    Every emitted task is verified by ASTBruteForceSolver.solve(...).
    """

    def __init__(self, seed: Optional[int] = None, max_task_attempts: int = 800):
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_task_attempts = max_task_attempts
        self.solver = ASTBruteForceSolver()
        self.operation_names = list(self.solver._get_operations(12, 34, "12", "34").keys())
        self.format_names = list(self.solver._get_formats(1, "+").keys())

    def _sample_operator(self, exclude: Sequence[str] = ()) -> str:
        candidates = [op for op in AST_OPERATOR_CHARS if op not in set(exclude)]
        if not candidates:
            raise ValueError("No operator characters left to sample.")
        return self.rng.choice(candidates)

    def _sample_operand(self) -> str:
        return f"{self.rng.randint(0, 99):02d}"

    def _sample_expression(self, operator: str) -> ASTExpression:
        return ASTExpression(self._sample_operand(), operator, self._sample_operand())

    def _sample_rule(
        self,
        allowed_operations: Optional[Sequence[str]] = None,
        allowed_configs: Optional[Sequence[str]] = None,
        allowed_formats: Optional[Sequence[str]] = None,
    ) -> ASTRule:
        return ASTRule(
            config=self.rng.choice(list(allowed_configs or AST_CONFIGS)),
            operation=self.rng.choice(list(allowed_operations or self.operation_names)),
            output_format=self.rng.choice(list(allowed_formats or self.format_names)),
        )

    def _apply_rule(self, expr: ASTExpression, rule: ASTRule) -> Optional[str]:
        try:
            cfg = self.solver._get_operand_configs(expr.left, expr.right)[rule.config]
            ops = self.solver._get_operations(*cfg)
            if rule.operation not in ops:
                return None
            formats = self.solver._get_formats(ops[rule.operation], expr.operator)
            return formats.get(rule.output_format)
        except (ValueError, KeyError, ZeroDivisionError):
            return None

    def _make_examples_for_rule(
        self,
        operator: str,
        rule: ASTRule,
        num_examples: int,
        used_expressions: Optional[set] = None,
    ) -> Tuple[List[str], List[ASTExpression]]:
        used = used_expressions if used_expressions is not None else set()
        lines: List[str] = []
        expressions: List[ASTExpression] = []

        for _ in range(self.max_task_attempts):
            if len(lines) >= num_examples:
                return lines, expressions
            expr = self._sample_expression(operator)
            key = (expr.left, expr.operator, expr.right)
            if key in used:
                continue
            output = self._apply_rule(expr, rule)
            if output is None:
                continue
            used.add(key)
            expressions.append(expr)
            lines.append(expr.as_example_line(output))

        raise RuntimeError(f"Could not generate {num_examples} examples for rule {rule}.")

    def _make_target_for_rule(
        self,
        operator: str,
        rule: ASTRule,
        used_expressions: Optional[set] = None,
    ) -> Tuple[ASTExpression, str]:
        used = used_expressions if used_expressions is not None else set()
        for _ in range(self.max_task_attempts):
            expr = self._sample_expression(operator)
            key = (expr.left, expr.operator, expr.right)
            if key in used:
                continue
            output = self._apply_rule(expr, rule)
            if output is None:
                continue
            used.add(key)
            return expr, output
        raise RuntimeError(f"Could not generate a target for rule {rule}.")

    def _solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        result = self.solver.solve(examples_text, target_text)
        debug_lines = result.get("debug") or []
        return {
            "answer": result.get("answer"),
            "generated_cot": "\n".join(str(line) for line in debug_lines),
            "debug_lines": debug_lines,
        }

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]

    def _resolved_rule_for_operator(self, debug_lines: Sequence[str], operator: str) -> Optional[ASTRule]:
        pattern = re.compile(
            rf"^Rule identified for '{re.escape(operator)}': "
            r"(?P<config>[^ ]+) -> (?P<operation>[^ ]+) -> (?P<fmt>[^ ]+)$"
        )
        for line in debug_lines:
            match = pattern.match(str(line))
            if match:
                return ASTRule(match.group("config"), match.group("operation"), match.group("fmt"))
        return None

    def _fallback_operation_from_debug(self, debug_lines: Sequence[str]) -> Optional[str]:
        pattern = re.compile(r"^Selected fallback operation: '(?P<operation>[^']+)'")
        for line in debug_lines:
            match = pattern.match(str(line))
            if match:
                return match.group("operation")
        return None

    def _base_operations_used_by_rules(self, rules: Sequence[ASTRule]) -> set:
        used = set()
        for rule in rules:
            for base_op in ["add", "sub", "mul", "div", "mod", "cat"]:
                if base_op in rule.operation:
                    used.add(base_op)
        return used

    def _apply_fallback_target(
        self,
        expr: ASTExpression,
        used_base_ops: set,
        global_config: str,
        global_format: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        available = [op for op in AST_FALLBACK_POOL if op not in used_base_ops]
        selected_operation = available[0] if available else AST_FALLBACK_POOL[0]
        answer = self._apply_rule(expr, ASTRule(global_config, selected_operation, global_format))
        return answer, selected_operation

    def _examples_count_for_rule(self, rule: ASTRule, min_examples: int, max_examples: int) -> int:
        lower_bound = max(min_examples, 3 if rule.output_format in {"first_digit", "last_digit"} else 1)
        upper_bound = max(lower_bound, max_examples)
        return self.rng.randint(lower_bound, upper_bound)

    def generate_task(
        self,
        mode: str = "random",
        min_examples_per_operator: int = 2,
        max_examples_per_operator: int = 4,
        max_operators: int = 4,
        force_rule: Optional[ASTRule] = None,
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        valid_modes = ["known_operator", "multi_operator", "fallback"]
        if mode == "random":
            mode = self.rng.choices(valid_modes, weights=[0.50, 0.40, 0.10], k=1)[0]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes + ['random']}.")

        if mode == "known_operator":
            return self._generate_known_operator_task(min_examples_per_operator, max_examples_per_operator, force_rule)
        if mode == "multi_operator":
            return self._generate_multi_operator_task(min_examples_per_operator, max_examples_per_operator, max_operators, force_rule)
        return self._generate_fallback_task(min_examples_per_operator, max_examples_per_operator, max_operators)

    def _generate_known_operator_task(
        self,
        min_examples_per_operator: int,
        max_examples_per_operator: int,
        force_rule: Optional[ASTRule] = None,
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        for _ in range(self.max_task_attempts):
            operator = self._sample_operator()
            rule = force_rule or self._sample_rule()
            num_examples = self._examples_count_for_rule(rule, min_examples_per_operator, max_examples_per_operator)
            used = set()
            try:
                example_lines, _ = self._make_examples_for_rule(operator, rule, num_examples, used)
                target_expr, answer = self._make_target_for_rule(operator, rule, used)
            except RuntimeError:
                continue

            examples_text = "\n".join(example_lines)
            target_text = target_expr.as_target_text()
            solved = self._solve(examples_text, target_text)
            resolved_rule = self._resolved_rule_for_operator(solved["debug_lines"], operator)

            if solved["answer"] == answer and (force_rule is None or resolved_rule == force_rule):
                metadata = {
                    "target_operator": operator,
                    "target_rule": asdict(rule),
                    "solver_resolved_target_rule": asdict(resolved_rule) if resolved_rule else None,
                    "num_example_operators": 1,
                    "num_examples": len(example_lines),
                }
                return examples_text, answer, target_text, metadata, "known_operator"
        raise RuntimeError("Failed to generate a verified known_operator task.")

    def _generate_multi_operator_task(
        self,
        min_examples_per_operator: int,
        max_examples_per_operator: int,
        max_operators: int,
        force_rule: Optional[ASTRule] = None,
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        for _ in range(self.max_task_attempts):
            num_ops = self.rng.randint(2, max(2, max_operators))
            operators: List[str] = []
            while len(operators) < num_ops:
                operators.append(self._sample_operator(exclude=operators))

            target_operator = self.rng.choice(operators)
            rules = {
                op: (force_rule if force_rule is not None and op == target_operator else self._sample_rule())
                for op in operators
            }
            used = set()
            example_lines: List[str] = []
            try:
                for op in operators:
                    rule = rules[op]
                    num_examples = self._examples_count_for_rule(rule, min_examples_per_operator, max_examples_per_operator)
                    op_lines, _ = self._make_examples_for_rule(op, rule, num_examples, used)
                    example_lines.extend(op_lines)
                self.rng.shuffle(example_lines)
                target_expr, answer = self._make_target_for_rule(target_operator, rules[target_operator], used)
            except RuntimeError:
                continue

            examples_text = "\n".join(example_lines)
            target_text = target_expr.as_target_text()
            solved = self._solve(examples_text, target_text)
            resolved_rule = self._resolved_rule_for_operator(solved["debug_lines"], target_operator)

            if solved["answer"] == answer and (force_rule is None or resolved_rule == force_rule):
                metadata = {
                    "target_operator": target_operator,
                    "target_rule": asdict(rules[target_operator]),
                    "all_intended_rules": {op: asdict(rule) for op, rule in rules.items()},
                    "solver_resolved_target_rule": asdict(resolved_rule) if resolved_rule else None,
                    "num_example_operators": len(operators),
                    "num_examples": len(example_lines),
                }
                return examples_text, answer, target_text, metadata, "multi_operator"
        raise RuntimeError("Failed to generate a verified multi_operator task.")

    def _generate_fallback_task(
        self,
        min_examples_per_operator: int,
        max_examples_per_operator: int,
        max_operators: int,
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        for _ in range(self.max_task_attempts):
            target_fallback_operation = self.rng.choice(AST_FALLBACK_POOL)
            required_used_ops = AST_FALLBACK_POOL[:AST_FALLBACK_POOL.index(target_fallback_operation)]

            example_operators: List[str] = []
            while len(example_operators) < max(1, len(required_used_ops)):
                example_operators.append(self._sample_operator(exclude=example_operators))
            target_operator = self._sample_operator(exclude=example_operators)

            global_config = self.rng.choice(AST_CONFIGS)
            global_format = self.rng.choice(self.format_names)
            rules: Dict[str, ASTRule] = {}

            if required_used_ops:
                for op_char, op_name in zip(example_operators, required_used_ops):
                    rules[op_char] = ASTRule(global_config, op_name, global_format)
            else:
                non_base_ops = [
                    op for op in self.operation_names
                    if not any(base in op for base in ["add", "sub", "mul", "div", "mod", "cat"])
                ]
                rules[example_operators[0]] = ASTRule(global_config, self.rng.choice(non_base_ops), global_format)

            used = set()
            example_lines: List[str] = []
            try:
                for op_char, rule in rules.items():
                    num_examples = self._examples_count_for_rule(rule, min_examples_per_operator, max_examples_per_operator)
                    op_lines, _ = self._make_examples_for_rule(op_char, rule, num_examples, used)
                    example_lines.extend(op_lines)
                self.rng.shuffle(example_lines)

                fallback_answer = None
                selected_operation = None
                target_expr = self._sample_expression(target_operator)
                for _target_attempt in range(self.max_task_attempts):
                    target_expr = self._sample_expression(target_operator)
                    if (target_expr.left, target_expr.operator, target_expr.right) in used:
                        continue
                    fallback_answer, selected_operation = self._apply_fallback_target(
                        target_expr,
                        used_base_ops=self._base_operations_used_by_rules(list(rules.values())),
                        global_config=global_config,
                        global_format=global_format,
                    )
                    if fallback_answer is not None:
                        break
                if fallback_answer is None:
                    continue
            except RuntimeError:
                continue

            examples_text = "\n".join(example_lines)
            target_text = target_expr.as_target_text()
            solved = self._solve(examples_text, target_text)
            solver_fallback_operation = self._fallback_operation_from_debug(solved["debug_lines"])

            if solved["answer"] == fallback_answer and solver_fallback_operation == selected_operation:
                metadata = {
                    "target_operator": target_operator,
                    "target_rule": {
                        "config": global_config,
                        "operation": selected_operation,
                        "output_format": global_format,
                        "source": "fallback_unseen_operator",
                    },
                    "all_intended_rules": {op: asdict(rule) for op, rule in rules.items()},
                    "solver_selected_fallback_operation": solver_fallback_operation,
                    "num_example_operators": len(rules),
                    "num_examples": len(example_lines),
                }
                return examples_text, fallback_answer, target_text, metadata, "fallback"
        raise RuntimeError("Failed to generate a verified fallback task.")

    def _generate_verified_row(
        self,
        sample_idx: int,
        mode: str,
        min_examples_per_operator: int,
        max_examples_per_operator: int,
        max_operators: int,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for _ in range(self.max_task_attempts):
            try:
                examples_text, answer, target_text, metadata, task_mode = self.generate_task(
                    mode=mode,
                    min_examples_per_operator=min_examples_per_operator,
                    max_examples_per_operator=max_examples_per_operator,
                    max_operators=max_operators,
                )
                prompt = ast_brute_force_prompt_template.format(examples=examples_text, target=target_text)
                solved = self._solve(examples_text, target_text)
                computed_answer = solved["answer"]
                if computed_answer != answer:
                    continue
                return {
                    "id": self._hash_prompt(prompt),
                    "prompt": prompt,
                    "examples": examples_text,
                    "target": target_text,
                    "answer": answer,
                    "generated_cot": solved["generated_cot"],
                    "computed_answer": computed_answer,
                    "is_correct": computed_answer == answer,
                    "task_mode": task_mode,
                    "metadata": metadata,
                    "label": "ast brute force",
                    "sample_idx": sample_idx,
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not generate a verified row. Last error: {last_error}")

    def generate_dataset(
        self,
        num_samples: int,
        mode: str = "random",
        min_examples_per_operator: int = 2,
        max_examples_per_operator: int = 4,
        max_operators: int = 4,
        nb_workers: Optional[int] = None,
        progress_bar: bool = True,
        use_parallel: bool = True,
    ) -> pd.DataFrame:
        """
        Generates a verified dataframe.

        By default this method uses pandas Series.parallel_apply through pandarallel.
        Install pandarallel with: pip install pandarallel
        """
        if num_samples <= 0:
            return pd.DataFrame.from_records([])

        base_seed = self.seed if self.seed is not None else random.SystemRandom().randint(1, 2_000_000_000)
        work_items = pd.Series([
            {
                "sample_idx": i,
                "seed": base_seed + i * 1_000_003,
                "max_task_attempts": self.max_task_attempts,
                "mode": mode,
                "min_examples_per_operator": min_examples_per_operator,
                "max_examples_per_operator": max_examples_per_operator,
                "max_operators": max_operators,
            }
            for i in range(num_samples)
        ])

        
        pandarallel.initialize(nb_workers=nb_workers or os.cpu_count(), progress_bar=progress_bar, verbose=0)
        rows = work_items.parallel_apply(_generate_ast_row_from_work_item)
        
        df = pd.DataFrame.from_records(rows.tolist())
        return df.sort_values("sample_idx").drop(columns=["sample_idx"]).reset_index(drop=True)


def _generate_ast_row_from_work_item(work_item: Dict[str, Any]) -> Dict[str, Any]:
    generator = ASTBruteForceTaskGenerator(
        seed=int(work_item["seed"]),
        max_task_attempts=int(work_item["max_task_attempts"]),
    )
    return generator._generate_verified_row(
        sample_idx=int(work_item["sample_idx"]),
        mode=str(work_item["mode"]),
        min_examples_per_operator=int(work_item["min_examples_per_operator"]),
        max_examples_per_operator=int(work_item["max_examples_per_operator"]),
        max_operators=int(work_item["max_operators"]),
    )


if __name__ == "__main__":
    generator = ASTBruteForceTaskGenerator(seed=42)
    df = generator.generate_dataset(num_samples=10, use_parallel=False)
    print(f"Accuracy: {df['is_correct'].mean() * 100:.1f}%")
    print(df[["task_mode", "target", "answer", "computed_answer", "is_correct"]].head(10).to_string(index=False))
