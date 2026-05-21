import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import Counter

import pandas as pd

from src.solvers.equations.ast_brute_force import ASTBruteForceSolver


ast_brute_force_prompt_template = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
{examples}
Now, determine the result for: {target}"""


AST_OPERATOR_COUNTS = Counter({
    "-": 423,
    "*": 374,
    "+": 374,
    "$": 80,
    '"': 76,
    "%": 74,
    "`": 73,
    "{": 68,
    "!": 68,
    "^": 67,
    "}": 66,
    "'": 65,
    "[": 64,
    "/": 62,
    "<": 62,
    "#": 61,
    "?": 60,
    ">": 59,
    ":": 57,
    "\\": 56,
    "]": 56,
    "@": 55,
    "(": 52,
    "&": 51,
    ")": 43,
    "|": 39,
})

MODE_COUNTS = Counter({
    "multi_operator": 462,
    "fallback": 37,
    "known_operator": 12,
})

SHAPE_COUNTS = Counter({
    (2, 1): 102,
    (2, 1, 1): 81,
    (2, 2, 1): 60,
    (3, 1, 1): 57,
    (3, 1): 49,
    (3, 2): 46,
    (1, 1, 1): 42,
    (2, 2): 36,
    (4, 1): 21,
    (3,): 12,
    (4,): 3,
    (5,): 2,
})

MULTI_TARGET_COUNT_WEIGHTS = Counter({
    1: 223,
    2: 166,
    3: 62,
    4: 11,
})

OPERATOR_ORDER_COUNTS = Counter({
    "shuffled": 265,
    "grouped": 246,
})

TARGET_RULE_COUNTS = Counter({
    ("rev_digits", "add", "sign_pref_rev"): 73,
    ("rev_digits", "mul", "sign_pref_rev"): 63,
    ("rev_digits", "sub", "sign_pref_rev"): 56,
    ("rev_digits", "mulm1", "sign_pref_rev"): 27,
    ("rev_digits", "addm1", "sign_pref_rev"): 21,
    ("rev_digits", "mul1", "sign_pref_rev"): 20,
    ("fwd", "mul", "raw"): 17,
    ("rev_digits", "add1", "sign_pref_rev"): 17,
    ("fwd", "sub", "sign_pref_raw"): 17,
    ("fwd", "add", "raw"): 17,
    ("fwd", "sub", "raw"): 14,
    ("fwd", "add1", "raw"): 12,
    ("fwd", "abs_diff", "raw"): 12,
    ("fwd", "mul1", "raw"): 12,
    ("fwd", "addm1", "raw"): 10,
    ("rev_digits", "abs_diff", "sign_pref_rev"): 10,
    ("fwd", "mulm1", "raw"): 8,
    ("fwd", "max_mod_min", "raw"): 8,
    ("rev_digits", "rev_sub", "sign_suff_rev"): 7,
    ("rev_digits", "max_mod_min", "sign_pref_rev"): 7,
    ("rev_digits", "neg_abs_diff", "sign_pref_rev"): 6,
    ("fwd", "addm1", "sign_pref_raw"): 5,
    ("fwd", "mulm1", "sign_pref_raw"): 5,
    ("fwd", "mul1", "sign_pref_raw"): 4,
    ("rev_digits", "add", "sign_suff_rev"): 4,
    ("fwd", "add1", "sign_pref_raw"): 3,
    ("fwd", "sub", "raw_pref"): 3,
    ("fwd", "sub", "abs_pref"): 3,
    ("rev_digits", "mul", "sign_suff_rev"): 3,
    ("rev_digits", "mul1", "sign_suff_rev"): 2,
    ("rev_digits", "rev_sub", "sign_pref_rev"): 2,
    ("fwd", "sub", "zpad2"): 1,
    ("rev_digits", "addm1", "sign_suff_rev"): 1,
    ("fwd", "sub", "abs_suff"): 1,
    ("fwd", "neg_abs_diff", "raw"): 1,
    ("fwd", "cross_diff_abs", "raw"): 1,
    ("fwd", "mul", "sign_pref_raw"): 1,
})

EXAMPLE_RULE_COUNTS = Counter({
    ("rev_digits", "add", "sign_pref_rev"): 150,
    ("rev_digits", "sub", "sign_pref_rev"): 141,
    ("rev_digits", "mul", "sign_pref_rev"): 126,
    ("rev_digits", "mul1", "sign_pref_rev"): 60,
    ("rev_digits", "mulm1", "sign_pref_rev"): 58,
    ("rev_digits", "add1", "sign_pref_rev"): 49,
    ("fwd", "mul", "raw"): 47,
    ("rev_digits", "cat", "sign_pref_rev"): 47,
    ("rev_digits", "abs_diff", "sign_pref_rev"): 45,
    ("rev_digits", "addm1", "sign_pref_rev"): 40,
    ("fwd", "add", "raw"): 38,
    ("fwd", "sub", "sign_pref_raw"): 37,
    ("fwd", "sub", "raw"): 36,
    ("fwd", "add1", "raw"): 33,
    ("fwd", "abs_diff", "raw"): 28,
    ("fwd", "cat", "raw"): 27,
    ("fwd", "mul1", "raw"): 25,
    ("fwd", "addm1", "raw"): 22,
    ("rev_digits", "rev_sub", "sign_suff_rev"): 22,
    ("fwd", "mulm1", "raw"): 20,
    ("rev_digits", "max_mod_min", "sign_pref_rev"): 19,
    ("fwd", "max_mod_min", "raw"): 16,
    ("rev_digits", "rev_sub", "sign_pref_rev"): 15,
    ("fwd", "cat", "sign_pref_raw"): 14,
    ("rev_digits", "neg_abs_diff", "sign_pref_rev"): 12,
    ("fwd", "addm1", "sign_pref_raw"): 12,
    ("fwd", "add1", "sign_pref_raw"): 12,
    ("fwd", "sub", "raw_pref"): 12,
    ("fwd", "mulm1", "sign_pref_raw"): 11,
    ("rev_digits", "mul", "sign_suff_rev"): 9,
    ("rev_digits", "rev_cat", "sign_pref_rev"): 8,
    ("fwd", "mul1", "sign_pref_raw"): 7,
    ("rev_digits", "add", "sign_suff_rev"): 7,
    ("fwd", "sub", "abs_pref"): 6,
    ("fwd", "sub", "raw_suff"): 4,
    ("rev_digits", "rev_cat", "sign_suff_rev"): 3,
    ("rev_digits", "cross_diff_abs", "zpad2"): 3,
    ("rev_digits", "sub", "sign_suff_rev"): 3,
    ("rev_digits", "mul1", "sign_suff_rev"): 2,
    ("fwd", "rev_cat", "raw"): 2,
    ("rev_digits", "addm1", "sign_suff_rev"): 2,
    ("rev_digits", "sub", "raw_suff"): 2,
    ("fwd", "rev_sub", "sign_suff_raw"): 1,
    ("rev_digits", "sub", "sign_pref_raw"): 1,
    ("fwd", "sub", "zpad2"): 1,
    ("fwd", "sub", "sign_suff_raw"): 1,
    ("fwd", "rev_sub", "zpad2"): 1,
    ("fwd", "rev_cat", "sign_pref_raw"): 1,
    ("fwd", "sub", "abs_suff"): 1,
    ("fwd", "neg_abs_diff", "raw"): 1,
    ("fwd", "cross_diff_abs", "raw"): 1,
    ("rev_digits", "neg_abs_diff", "sign_suff_rev"): 1,
    ("rev_digits", "mod", "sign_pref_rev"): 1,
    ("fwd", "rev_sub", "raw"): 1,
    ("fwd", "mul", "sign_pref_raw"): 1,
})

FALLBACK_RULE_COUNTS = Counter({
    ("rev_digits", "add", "sign_pref_rev"): 15,
    ("rev_digits", "sub", "sign_pref_rev"): 10,
    ("fwd", "sub", "raw"): 5,
    ("fwd", "add", "raw"): 2,
    ("rev_digits", "add", "sign_suff_rev"): 2,
    ("rev_digits", "cat", "sign_pref_rev"): 2,
    ("fwd", "cat", "raw"): 1,
})


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


def weighted_choice(rng: random.Random, counter: Counter):
    total = sum(counter.values())
    x = rng.uniform(0, total)

    acc = 0
    for item, weight in counter.items():
        acc += weight
        if x <= acc:
            return item

    return next(reversed(counter))


class ASTBruteForceTaskGenerator:
    """Generate AST equation tasks from the solver rule space.

    The generator does not parse or depend on CoT text. It samples operators,
    rules, shapes, prompts, targets, and ground-truth answers directly from the
    rule space. The solver is used only to produce optional CoT and to verify
    that the generated answer is recoverable by the current solver.
    """
    def __init__(self, seed: Optional[int] = None, max_task_attempts: int = 2000):
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_task_attempts = max_task_attempts
        self.solver = ASTBruteForceSolver()

        self.operation_names = list(self.solver._get_operations(12, 34, "12", "34").keys())
        self._last_task_metadata: Dict[str, Any] = {}

    def _sample_operand(self) -> str:
        return f"{self.rng.randint(0, 99):02d}"

    def _sample_operator(self, exclude: Sequence[str] = ()) -> str:
        excluded = set(exclude)
        candidates = Counter({
            op: count
            for op, count in AST_OPERATOR_COUNTS.items()
            if op not in excluded
        })

        if not candidates:
            raise RuntimeError("No operators left to sample")

        return weighted_choice(self.rng, candidates)

    def _sample_expression(self, operator: str) -> ASTExpression:
        return ASTExpression(self._sample_operand(), operator, self._sample_operand())

    def _sample_rule(self, counter: Counter) -> ASTRule:
        config, operation, output_format = weighted_choice(self.rng, counter)
        return ASTRule(config, operation, output_format)

    def _sample_shape(self, mode: str) -> Tuple[int, ...]:
        if mode == "known_operator":
            compatible = Counter({s: c for s, c in SHAPE_COUNTS.items() if len(s) == 1})
        elif mode == "multi_operator":
            compatible = Counter({s: c for s, c in SHAPE_COUNTS.items() if len(s) >= 2})
        elif mode == "fallback":
            compatible = Counter({s: c for s, c in SHAPE_COUNTS.items() if len(s) >= 1})
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return weighted_choice(self.rng, compatible)

    def _apply_rule(self, expr: ASTExpression, rule: ASTRule) -> Optional[str]:
        try:
            cfg = self.solver._get_operand_configs(expr.left, expr.right)[rule.config]
            ops = self.solver._get_operations(*cfg)

            if rule.operation not in ops:
                return None

            return self.solver._get_formats(ops[rule.operation], expr.operator).get(rule.output_format)
        except Exception:
            return None

    def _make_examples(
        self,
        operator: str,
        rule: ASTRule,
        count: int,
        used: set,
    ) -> List[str]:
        lines = []

        for _ in range(self.max_task_attempts):
            if len(lines) == count:
                return lines

            expr = self._sample_expression(operator)
            key = (expr.left, expr.operator, expr.right)

            if key in used:
                continue

            output = self._apply_rule(expr, rule)

            if output is None:
                continue

            used.add(key)
            lines.append(expr.as_example_line(output))

        raise RuntimeError("Could not generate examples")

    def _make_target(self, operator: str, rule: ASTRule, used: set) -> Tuple[ASTExpression, str]:
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

        raise RuntimeError("Could not generate target")

    def _solve(self, examples_text: str, target_text: str) -> Dict[str, Any]:
        result = self.solver.solve(examples_text, target_text)
        debug_lines = result.get("debug") or []

        return {
            "answer": result.get("answer"),
            "generated_cot": "\n".join(str(line) for line in debug_lines),
            "debug_lines": debug_lines,
            "rule_source": result.get("rule_source"),
            "training_category": result.get("training_category"),
            "metadata": result.get("metadata") or {},
        }

    def _maybe_shuffle(self, lines: List[str]) -> List[str]:
        if weighted_choice(self.rng, OPERATOR_ORDER_COUNTS) == "grouped":
            return lines

        shuffled = list(lines)
        self.rng.shuffle(shuffled)
        return shuffled

    def _generate_known_operator_task(self) -> Tuple[str, str, str, str]:
        for _ in range(self.max_task_attempts):
            shape = self._sample_shape("known_operator")
            operator = self._sample_operator()
            rule = self._sample_rule(TARGET_RULE_COUNTS)
            used = set()

            try:
                example_lines = self._make_examples(operator, rule, shape[0], used)
                target_expr, answer = self._make_target(operator, rule, used)
            except RuntimeError:
                continue

            examples_text = "\n".join(example_lines)
            target_text = target_expr.as_target_text()
            self._last_task_metadata = {
                "generation_mode": "known_operator",
                "target_operator": operator,
                "target_operator_seen_in_examples": True,
                "uses_fallback_inference": False,
                "target_rule": {
                    "config": rule.config,
                    "operation": rule.operation,
                    "output_format": rule.output_format,
                },
                "example_rules": {
                    operator: {
                        "config": rule.config,
                        "operation": rule.operation,
                        "output_format": rule.output_format,
                    }
                },
                "example_shape": shape,
            }
            return examples_text, answer, target_text, "known_operator"

        raise RuntimeError("Failed to generate known_operator task")

    def _assign_target_count(self, shape: Tuple[int, ...]) -> Optional[int]:
        compatible = Counter({
            count: MULTI_TARGET_COUNT_WEIGHTS[count]
            for count in shape
            if count in MULTI_TARGET_COUNT_WEIGHTS
        })

        if not compatible:
            return None

        return weighted_choice(self.rng, compatible)

    def _generate_multi_operator_task(self) -> Tuple[str, str, str, str]:
        for _ in range(self.max_task_attempts):
            shape = self._sample_shape("multi_operator")
            target_count = self._assign_target_count(shape)

            if target_count is None:
                continue

            operators = []

            while len(operators) < len(shape):
                operators.append(self._sample_operator(exclude=operators))

            target_operator = self.rng.choice(operators)
            target_rule = self._sample_rule(TARGET_RULE_COUNTS)

            remaining_counts = list(shape)
            remaining_counts.remove(target_count)
            self.rng.shuffle(remaining_counts)

            op_counts = {target_operator: target_count}
            other_operators = [op for op in operators if op != target_operator]

            for op, count in zip(other_operators, remaining_counts):
                op_counts[op] = count

            rules = {target_operator: target_rule}

            for op in other_operators:
                rules[op] = self._sample_rule(EXAMPLE_RULE_COUNTS)

            used = set()
            example_lines = []

            try:
                for op in operators:
                    example_lines.extend(self._make_examples(op, rules[op], op_counts[op], used))

                example_lines = self._maybe_shuffle(example_lines)
                target_expr, answer = self._make_target(target_operator, target_rule, used)
            except RuntimeError:
                continue

            examples_text = "\n".join(example_lines)
            target_text = target_expr.as_target_text()
            self._last_task_metadata = {
                "generation_mode": "multi_operator",
                "target_operator": target_operator,
                "target_operator_seen_in_examples": True,
                "uses_fallback_inference": False,
                "target_rule": {
                    "config": target_rule.config,
                    "operation": target_rule.operation,
                    "output_format": target_rule.output_format,
                },
                "example_rules": {
                    op: {
                        "config": rule.config,
                        "operation": rule.operation,
                        "output_format": rule.output_format,
                    }
                    for op, rule in rules.items()
                },
                "example_shape": shape,
                "target_example_count": target_count,
            }
            return examples_text, answer, target_text, "multi_operator"

        raise RuntimeError("Failed to generate multi_operator task")

    def _operation_has_base(self, operation: str, base: str) -> bool:
        return base in operation

    def _sample_fallback_filler_operation(self, forbidden_base: str) -> str:
        candidates = [
            op
            for op in self.operation_names
            if not self._operation_has_base(op, forbidden_base)
        ]

        if not candidates:
            raise RuntimeError("No fallback filler operations")

        return self.rng.choice(candidates)

    def _generate_fallback_task(self) -> Tuple[str, str, str, str]:
        strict_pool = ["add", "sub", "cat", "mul", "div"]

        for _ in range(self.max_task_attempts):
            fallback_rule = self._sample_rule(FALLBACK_RULE_COUNTS)

            if fallback_rule.operation not in strict_pool:
                continue

            selected_operation = fallback_rule.operation
            required_operations = strict_pool[:strict_pool.index(selected_operation)]

            compatible_shapes = Counter({
                shape: count
                for shape, count in SHAPE_COUNTS.items()
                if len(shape) >= max(1, len(required_operations))
            })

            shape = weighted_choice(self.rng, compatible_shapes)

            example_operators = []

            while len(example_operators) < len(shape):
                example_operators.append(self._sample_operator(exclude=example_operators))

            target_operator = self._sample_operator(exclude=example_operators)

            rules = {}

            for op_char, required_operation in zip(example_operators, required_operations):
                rules[op_char] = ASTRule(
                    fallback_rule.config,
                    required_operation,
                    fallback_rule.output_format,
                )

            for op_char in example_operators:
                if op_char in rules:
                    continue

                filler_operation = self._sample_fallback_filler_operation(selected_operation)

                rules[op_char] = ASTRule(
                    fallback_rule.config,
                    filler_operation,
                    fallback_rule.output_format,
                )

            used = set()
            example_lines = []

            try:
                for op_char, count in zip(example_operators, shape):
                    example_lines.extend(self._make_examples(op_char, rules[op_char], count, used))

                example_lines = self._maybe_shuffle(example_lines)

                target_expr, answer = self._make_target(
                    target_operator,
                    fallback_rule,
                    used,
                )
            except RuntimeError:
                continue

            examples_text = "\n".join(example_lines)
            target_text = target_expr.as_target_text()
            self._last_task_metadata = {
                "generation_mode": "fallback",
                "target_operator": target_operator,
                "target_operator_seen_in_examples": False,
                "uses_fallback_inference": True,
                "fallback_operation": selected_operation,
                "target_rule": {
                    "config": fallback_rule.config,
                    "operation": fallback_rule.operation,
                    "output_format": fallback_rule.output_format,
                },
                "example_rules": {
                    op: {
                        "config": rule.config,
                        "operation": rule.operation,
                        "output_format": rule.output_format,
                    }
                    for op, rule in rules.items()
                },
                "example_shape": shape,
                "strict_pool": strict_pool,
                "required_operations": required_operations,
            }
            return examples_text, answer, target_text, "fallback"

        raise RuntimeError("Failed to generate fallback task")

    def generate_task(self, mode: str = "random") -> Tuple[str, str, str, str]:
        if mode == "random":
            mode = weighted_choice(self.rng, MODE_COUNTS)

        if mode == "known_operator":
            return self._generate_known_operator_task()

        if mode == "multi_operator":
            return self._generate_multi_operator_task()

        if mode == "fallback":
            return self._generate_fallback_task()

        raise ValueError(f"Unknown mode: {mode}")

    def _generate_row(self, sample_idx: int, mode: str, label: str) -> Dict[str, Any]:
        last_error = None

        for _ in range(self.max_task_attempts):
            try:
                examples_text, answer, target_text, task_mode = self.generate_task(mode=mode)

                prompt = ast_brute_force_prompt_template.format(
                    examples=examples_text,
                    target=target_text,
                )

                solved = self._solve(examples_text, target_text)
                computed_answer = solved["answer"]

                if computed_answer != answer:
                    continue

                solver_metadata = solved.get("metadata") or {}
                generation_metadata = dict(self._last_task_metadata)

                return {
                    "prompt": prompt,
                    "answer": answer,
                    "generated_cot": solved["generated_cot"],
                    "computed_answer": computed_answer,
                    "task_mode": task_mode,
                    "label": label,
                    "target_operator": generation_metadata.get("target_operator"),
                    "target_operator_seen_in_examples": generation_metadata.get("target_operator_seen_in_examples"),
                    "uses_fallback_inference": generation_metadata.get("uses_fallback_inference"),
                    "target_rule": generation_metadata.get("target_rule"),
                    "example_rules": generation_metadata.get("example_rules"),
                    "generation_metadata": generation_metadata,
                    "solver_rule_source": solved.get("rule_source"),
                    "solver_training_category": solved.get("training_category"),
                    "solver_metadata": solver_metadata,
                }

            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Could not generate verified row. Last error: {last_error}")

    def generate_dataset(
        self,
        num_samples: int,
        mode: str = "random",
        nb_workers: Optional[int] = None,
        progress_bar: bool = True,
        use_parallel: bool = True,
        label: str = "",
    ) -> pd.DataFrame:
        if num_samples <= 0:
            return pd.DataFrame.from_records([])

        base_seed = self.seed if self.seed is not None else random.SystemRandom().randint(1, 2_000_000_000)

        work_items = [
            {
                "sample_idx": i,
                "seed": base_seed + i * 1_000_003,
                "max_task_attempts": self.max_task_attempts,
                "mode": mode,
                "label": label,
            }
            for i in range(num_samples)
        ]

        if use_parallel:
            try:
                from pandarallel import pandarallel

                pandarallel.initialize(
                    nb_workers=nb_workers,
                    progress_bar=progress_bar,
                    verbose=0,
                )

                rows = pd.Series(work_items).parallel_apply(_generate_ast_row_from_work_item).tolist()
                return pd.DataFrame.from_records(rows)

            except ImportError:
                pass

        rows = [_generate_ast_row_from_work_item(item) for item in work_items]
        return pd.DataFrame.from_records(rows)


def _generate_ast_row_from_work_item(work_item: Dict[str, Any]) -> Dict[str, Any]:
    generator = ASTBruteForceTaskGenerator(
        seed=int(work_item["seed"]),
        max_task_attempts=int(work_item["max_task_attempts"]),
    )

    return generator._generate_row(
        sample_idx=int(work_item["sample_idx"]),
        mode=str(work_item["mode"]),
        label=str(work_item["label"]),
    )