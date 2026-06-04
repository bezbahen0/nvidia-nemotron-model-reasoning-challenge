from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from src.solvers.equations.cryptarithm import CryptarithmSolver


CHARSET: Tuple[str, ...] = (
    "!", '"', "#", "$", "%", "&", "'", "(", ")",
    "*", "+", "-", "/", ":", "<", ">", "?",
    "@", "[", "\\", "]", "^", "`", "{", "|", "}",
)

# Cat-only rules. Both are sampled with equal probability.
RULES: Tuple[str, ...] = (
    "std/cat",  # AB op CD -> ABCD
    "rev/cat",  # AB op CD -> CDAB
)


class CryptarithmAugmentGenerator:
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        charset: Tuple[str, ...] = CHARSET,
        rules: Tuple[str, ...] = RULES,
        operand_len: int = 2,
        solver: Optional[CryptarithmSolver] = None,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.charset = tuple(charset)
        self.rules = tuple(rules)
        self.operand_len = operand_len
        self.solver = solver or CryptarithmSolver()

        if operand_len < 1:
            raise ValueError("operand_len must be >= 1")
        if len(self.charset) < 2:
            raise ValueError("charset must contain at least 2 symbols")
        if not self.rules:
            raise ValueError("rules must not be empty")

    def generate_dataset(
        self,
        n: int = 100,
        *,
        min_examples: int = 3,
        max_examples: int = 5,
        min_same_op_examples: int = 1,
        include_metadata: bool = True,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate tasks and immediately solve them with CryptarithmSolver.

        Returns rows with:
          - prompt: task text
          - generated_cot: solver's reasoning text
          - answer: generator's intended answer
          - computed_answer: solver-computed answer

        With the same seed and the same parameters, repeated script runs produce
        the same dataset. Repeated calls on the same generator instance continue
        from the current RNG state, as usual for random.Random.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        if min_examples < 1 or max_examples < min_examples:
            raise ValueError("Use 1 <= min_examples <= max_examples")
        if min_same_op_examples < 1:
            raise ValueError("min_same_op_examples must be >= 1")
        if min_same_op_examples > max_examples:
            raise ValueError("min_same_op_examples cannot exceed max_examples")

        rows: List[Dict[str, Any]] = []
        for i in range(n):
            row = self._generate_one(
                index=i,
                min_examples=min_examples,
                max_examples=max_examples,
                min_same_op_examples=min_same_op_examples,
                include_metadata=include_metadata,
            )

            solved = self.solver.solve(row["prompt"])
            computed_answer = solved.get("answer") or "nan"
            row["generated_cot"] = solved.get("solution_text", "Final answer: nan")
            row["computed_answer"] = computed_answer
            row["solver_strategy"] = solved.get("strategy")

            if validate and row["answer"] != row["computed_answer"]:
                raise AssertionError(
                    "Generator/solver mismatch: "
                    f"answer={row['answer']!r}, computed_answer={row['computed_answer']!r}, "
                    f"prompt={row['prompt']!r}"
                )

            rows.append(row)
        return rows

    def _generate_one(
        self,
        *,
        index: int,
        min_examples: int,
        max_examples: int,
        min_same_op_examples: int,
        include_metadata: bool,
    ) -> Dict[str, Any]:
        rule = self.rng.choice(self.rules)
        target_op = self._sample_symbol()
        target_left, target_right = self._sample_distinct_operands()
        target_expr = f"{target_left}{target_op}{target_right}"
        answer = self._apply_rule(rule, target_left, target_right)

        total_examples = self.rng.randint(min_examples, max_examples)
        same_op_count = self.rng.randint(
            min_same_op_examples,
            max(min_same_op_examples, total_examples),
        )

        examples: List[str] = []
        used_left_sides = {target_expr}

        for _ in range(same_op_count):
            left, right = self._sample_distinct_operands()
            expr = f"{left}{target_op}{right}"
            while expr in used_left_sides:
                left, right = self._sample_distinct_operands()
                expr = f"{left}{target_op}{right}"
            used_left_sides.add(expr)
            examples.append(f"{expr} = {self._apply_rule(rule, left, right)}")

        for _ in range(total_examples - same_op_count):
            op = self._sample_symbol(exclude={target_op})
            left = self._sample_operand()
            right = self._sample_operand()
            expr = f"{left}{op}{right}"
            while expr in used_left_sides:
                op = self._sample_symbol(exclude={target_op})
                left = self._sample_operand()
                right = self._sample_operand()
                expr = f"{left}{op}{right}"
            used_left_sides.add(expr)

            # Distractors may follow either cat rule or be random noise. The solver
            # only uses examples with the target operator, so this does not affect
            # the intended solution.
            distractor_rule = self.rng.choice((*self.rules, "noise"))
            if distractor_rule == "noise":
                result = self._sample_operand(self.rng.randint(1, self.operand_len * 2))
            else:
                result = self._apply_rule(distractor_rule, left, right)
            examples.append(f"{expr} = {result}")

        self.rng.shuffle(examples)
        prompt = self._format_prompt(examples, target_expr)

        item_id = self._stable_id(prompt)
        row: Dict[str, Any] = {
            "prompt": prompt,
            "generated_cot": "",       # filled by generate_dataset()
            "answer": answer,           # generator-intended answer
            "computed_answer": "",     # filled by generate_dataset()
        }
        if include_metadata:
            row.update({
                "id": item_id,
                "rule": rule,
                "target": target_expr,
                "target_op": target_op,
                "examples": examples,
            })
        return row

    @staticmethod
    def _format_prompt(examples: List[str], target_expr: str) -> str:
        examples_text = "\n".join(examples)
        return (
            "In Alice's Wonderland, a secret set of transformation rules is applied "
            "to equations. Below are a few examples:\n"
            f"{examples_text}\n"
            f"Now, determine the result for: {target_expr}"
        )

    @staticmethod
    def _apply_rule(rule: str, left: str, right: str) -> str:
        if rule == "std/cat":
            return left + right
        if rule == "rev/cat":
            return right + left
        raise ValueError(f"Unsupported rule: {rule!r}")

    def _sample_operand(self, length: Optional[int] = None) -> str:
        length = self.operand_len if length is None else length
        return "".join(self.rng.choice(self.charset) for _ in range(length))

    def _sample_distinct_operands(self) -> Tuple[str, str]:
        left = self._sample_operand()
        right = self._sample_operand()
        # Avoid ambiguous proof examples where left+right == right+left.
        while left + right == right + left:
            right = self._sample_operand()
        return left, right

    def _sample_symbol(self, exclude: Optional[set[str]] = None) -> str:
        exclude = exclude or set()
        choices = [ch for ch in self.charset if ch not in exclude]
        if not choices:
            raise ValueError("No symbols left to sample after exclude")
        return self.rng.choice(choices)

    @staticmethod
    def _stable_id(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


if __name__ == "__main__":
    generator = CryptarithmAugmentGenerator(seed=42)
    dataset = generator.generate_dataset(n=3)
    print(json.dumps(dataset, ensure_ascii=False, indent=2))
