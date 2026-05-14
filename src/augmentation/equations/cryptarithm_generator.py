from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandarallel import pandarallel


from src.solvers.equations.cryptarithm import CryptarithmCSPSolver, TraceConfig


cryptarithm_prompt_template = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
{examples}
Now, determine the result for: {target_input}"""


@dataclass(frozen=True)
class CryptarithmRuleSpec:
    op_symbol: str
    rule_name: str
    orientation: str

    @property
    def reverse(self) -> bool:
        return self.orientation == "little_endian"

    @property
    def solver_desc(self) -> str:
        return f"{self.orientation}(A,B) \\mapsto {self.rule_name}"


@dataclass(frozen=True)
class GeneratedEquation:
    line: str
    left_value: int
    right_value: int
    numeric_value: int
    op_symbol: str
    rule_name: str
    orientation: str


class CryptarithmTaskGenerator:
    """
    Generator for the DeductiveCryptarithmSolver/CryptarithmCSPSolver task space.

    Covered solver rule space:
    - operations: add, add_p1, add_m1, mul, mul_p1, mul_m1, absdiff, sub_signed, concat_fwd;
    - orientations: standard and little_endian;
    - one or several operator symbols in the same task;
    - operator symbols and digit symbols are sampled from the 26-character alphabet used in the examples.
    """

    CHARSET: Tuple[str, ...] = (
        "!", '"', "#", "$", "%", "&", "'", "(", ")",
        "*", "+", "-", "/", ":", "<", ">", "?",
        "@", "[", "\\", "]", "^", "`", "{", "|", "}",
    )

    RULE_NAMES: Tuple[str, ...] = (
        "add", "add_p1", "add_m1",
        "mul", "mul_p1", "mul_m1",
        "absdiff", "sub_signed", "concat_fwd",
    )

    NON_CONCAT_RULE_NAMES: Tuple[str, ...] = (
        "add", "add_p1", "add_m1",
        "mul", "mul_p1", "mul_m1",
        "absdiff", "sub_signed",
    )

    MODES: Tuple[str, ...] = (
        "random",
        "arithmetic",
        "little_endian",
        "mixed_concat",
        "pure_concat",
        "full_rule_space",
    )

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    @staticmethod
    def _apply_rule(rule_name: str, a: int, b: int) -> Optional[int]:
        if rule_name == "add":
            return a + b
        if rule_name == "add_p1":
            return a + b + 1
        if rule_name == "add_m1":
            value = a + b - 1
            return value if value >= 0 else None
        if rule_name == "mul":
            return a * b
        if rule_name == "mul_p1":
            return a * b + 1
        if rule_name == "mul_m1":
            value = a * b - 1
            return value if value >= 0 else None
        if rule_name == "absdiff":
            return abs(a - b)
        if rule_name == "sub_signed":
            return a - b
        if rule_name == "concat_fwd":
            return int(f"{a}{b}")
        raise ValueError(f"Unsupported rule_name: {rule_name}")

    @staticmethod
    def _random_number(rng: random.Random, length: int) -> int:
        if length <= 0:
            raise ValueError("length must be positive")
        if length == 1:
            return rng.randint(0, 9)
        return rng.randint(10 ** (length - 1), 10**length - 1)

    @staticmethod
    def _encode_number(value: int, digit_to_symbol: Dict[int, str], reverse: bool) -> str:
        if value < 0:
            raise ValueError("_encode_number accepts only non-negative integers")
        encoded = "".join(digit_to_symbol[int(ch)] for ch in str(value))
        return encoded[::-1] if reverse else encoded

    @classmethod
    def _format_result(
        cls,
        value: int,
        rule: CryptarithmRuleSpec,
        digit_to_symbol: Dict[int, str],
    ) -> Optional[str]:
        if rule.rule_name == "sub_signed" and value < 0:
            return rule.op_symbol + cls._encode_number(abs(value), digit_to_symbol, rule.reverse)
        if value < 0:
            return None
        return cls._encode_number(value, digit_to_symbol, rule.reverse)

    @classmethod
    def _format_equation(
        cls,
        a: int,
        b: int,
        rule: CryptarithmRuleSpec,
        digit_to_symbol: Dict[int, str],
    ) -> Optional[GeneratedEquation]:
        value = cls._apply_rule(rule.rule_name, a, b)
        if value is None:
            return None

        left = cls._encode_number(a, digit_to_symbol, rule.reverse)
        right = cls._encode_number(b, digit_to_symbol, rule.reverse)
        result = cls._format_result(value, rule, digit_to_symbol)
        if result is None:
            return None

        return GeneratedEquation(
            line=f"{left}{rule.op_symbol}{right} = {result}",
            left_value=a,
            right_value=b,
            numeric_value=value,
            op_symbol=rule.op_symbol,
            rule_name=rule.rule_name,
            orientation=rule.orientation,
        )

    def _sample_symbol_mapping(self, num_digit_symbols: int) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
        if not 1 <= num_digit_symbols <= 10:
            raise ValueError("num_digit_symbols must be in [1, 10]")
        digit_symbols = self.rng.sample(list(self.CHARSET), num_digit_symbols)
        digits = list(range(10))
        self.rng.shuffle(digits)
        used_digits = digits[:num_digit_symbols]

        symbol_to_digit = {symbol: digit for symbol, digit in zip(digit_symbols, used_digits)}
        digit_to_symbol = {digit: symbol for symbol, digit in symbol_to_digit.items()}

        # The solver encodes every decimal digit of the numeric result back to a symbol.
        # Therefore the generator normally uses all ten digits. If a smaller alphabet is
        # requested, fill missing digits with extra non-operator symbols so generated
        # arithmetic never becomes unencodable.
        if len(digit_to_symbol) < 10:
            remaining_symbols = [ch for ch in self.CHARSET if ch not in symbol_to_digit]
            for digit in range(10):
                if digit in digit_to_symbol:
                    continue
                symbol = remaining_symbols.pop(0)
                symbol_to_digit[symbol] = digit
                digit_to_symbol[digit] = symbol

        operator_pool = [ch for ch in self.CHARSET if ch not in set(digit_to_symbol.values())]
        if not operator_pool:
            raise ValueError("No characters left for operator symbols; use fewer digit symbols")
        return symbol_to_digit, digit_to_symbol, operator_pool

    def _sample_rules(
        self,
        mode: str,
        operator_pool: Sequence[str],
        min_operators: int,
        max_operators: int,
        forced_rule_name: Optional[str],
        forced_orientation: Optional[str],
    ) -> Tuple[str, List[CryptarithmRuleSpec]]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if forced_rule_name is not None and forced_rule_name not in self.RULE_NAMES:
            raise ValueError(f"forced_rule_name must be one of {self.RULE_NAMES}")
        if forced_orientation is not None and forced_orientation not in ("standard", "little_endian"):
            raise ValueError("forced_orientation must be 'standard' or 'little_endian'")
        if min_operators < 1 or max_operators < min_operators:
            raise ValueError("Invalid operator-count range")

        resolved_mode = mode
        if resolved_mode == "random":
            resolved_mode = self.rng.choices(
                population=["arithmetic", "little_endian", "mixed_concat", "pure_concat", "full_rule_space"],
                weights=[4, 4, 3, 1, 2],
                k=1,
            )[0]

        min_ops = min_operators
        if resolved_mode == "mixed_concat":
            min_ops = max(min_ops, 2)
        op_count = self.rng.randint(min_ops, max_operators)
        op_count = min(op_count, len(operator_pool))
        op_symbols = self.rng.sample(list(operator_pool), op_count)

        specs: List[CryptarithmRuleSpec] = []

        def choose_orientation(default: Optional[str] = None) -> str:
            if forced_orientation is not None:
                return forced_orientation
            if default is not None:
                return default
            return self.rng.choice(["standard", "little_endian"])

        def choose_non_concat() -> str:
            if forced_rule_name is not None:
                return forced_rule_name
            return self.rng.choice(list(self.NON_CONCAT_RULE_NAMES))

        if forced_rule_name is not None:
            for op_symbol in op_symbols:
                specs.append(CryptarithmRuleSpec(op_symbol, forced_rule_name, choose_orientation()))
            return resolved_mode, specs

        if resolved_mode == "arithmetic":
            for op_symbol in op_symbols:
                specs.append(CryptarithmRuleSpec(op_symbol, choose_non_concat(), choose_orientation("standard")))
        elif resolved_mode == "little_endian":
            for op_symbol in op_symbols:
                specs.append(CryptarithmRuleSpec(op_symbol, choose_non_concat(), choose_orientation("little_endian")))
        elif resolved_mode == "mixed_concat":
            specs.append(CryptarithmRuleSpec(op_symbols[0], "concat_fwd", choose_orientation()))
            for op_symbol in op_symbols[1:]:
                specs.append(CryptarithmRuleSpec(op_symbol, choose_non_concat(), choose_orientation()))
        elif resolved_mode == "pure_concat":
            for op_symbol in op_symbols:
                specs.append(CryptarithmRuleSpec(op_symbol, "concat_fwd", choose_orientation()))
        elif resolved_mode == "full_rule_space":
            for op_symbol in op_symbols:
                specs.append(CryptarithmRuleSpec(op_symbol, self.rng.choice(list(self.RULE_NAMES)), choose_orientation()))
        else:
            raise AssertionError(f"Unhandled mode: {resolved_mode}")

        return resolved_mode, specs

    def _sample_operands(
        self,
        rule: CryptarithmRuleSpec,
        force_negative_subtraction: bool,
        max_result_digits: int,
    ) -> Tuple[int, int]:
        for _ in range(2000):
            left_len = 2
            right_len = self.rng.choice([1, 2])
            a = self._random_number(self.rng, left_len)
            b = self._random_number(self.rng, right_len)

            if rule.rule_name == "sub_signed" and force_negative_subtraction and a >= b:
                a, b = b, a
                if a < 10:
                    a = self._random_number(self.rng, 2)
                    b = self._random_number(self.rng, 2)
                    if a >= b:
                        a, b = min(a, b), max(a, b)

            value = self._apply_rule(rule.rule_name, a, b)
            if value is None:
                continue
            if len(str(abs(value))) > max_result_digits:
                continue
            if rule.rule_name == "sub_signed" and force_negative_subtraction and value >= 0:
                continue
            return a, b

        raise RuntimeError(f"Could not sample operands for rule={rule} within constraints")

    def _generate_equation_for_rule(
        self,
        rule: CryptarithmRuleSpec,
        digit_to_symbol: Dict[int, str],
        force_negative_subtraction: bool,
        max_result_digits: int,
        used_lines: set[str],
    ) -> GeneratedEquation:
        for _ in range(2000):
            a, b = self._sample_operands(rule, force_negative_subtraction, max_result_digits)
            equation = self._format_equation(a, b, rule, digit_to_symbol)
            if equation is not None and equation.line not in used_lines:
                used_lines.add(equation.line)
                return equation
        raise RuntimeError(f"Could not generate unique equation for rule={rule}")

    def generate_task(
        self,
        num_examples: Optional[int] = None,
        mode: str = "random",
        min_operators: int = 1,
        max_operators: int = 3,
        forced_rule_name: Optional[str] = None,
        forced_orientation: Optional[str] = None,
        max_result_digits: int = 4,
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        """
        Returns: examples_text, answer, target_input, metadata, resolved_mode.
        """
        if num_examples is None:
            num_examples = self.rng.randint(3, 5)
        if num_examples < 1:
            raise ValueError("num_examples must be positive")

        symbol_to_digit, digit_to_symbol, operator_pool = self._sample_symbol_mapping(num_digit_symbols=10)
        resolved_mode, rules = self._sample_rules(
            mode=mode,
            operator_pool=operator_pool,
            min_operators=min_operators,
            max_operators=max_operators,
            forced_rule_name=forced_rule_name,
            forced_orientation=forced_orientation,
        )
        if num_examples < len(rules):
            num_examples = len(rules)

        used_lines: set[str] = set()
        equations: List[GeneratedEquation] = []

        # First, add at least one example for every operator symbol so the uploaded
        # solver can build candidates for every target operator it may see.
        for rule in rules:
            equations.append(
                self._generate_equation_for_rule(
                    rule=rule,
                    digit_to_symbol=digit_to_symbol,
                    force_negative_subtraction=rule.rule_name == "sub_signed",
                    max_result_digits=max_result_digits,
                    used_lines=used_lines,
                )
            )

        while len(equations) < num_examples:
            rule = self.rng.choice(rules)
            equations.append(
                self._generate_equation_for_rule(
                    rule=rule,
                    digit_to_symbol=digit_to_symbol,
                    force_negative_subtraction=False,
                    max_result_digits=max_result_digits,
                    used_lines=used_lines,
                )
            )

        self.rng.shuffle(equations)
        target_rule = self.rng.choice(rules)
        target_equation = self._generate_equation_for_rule(
            rule=target_rule,
            digit_to_symbol=digit_to_symbol,
            force_negative_subtraction=False,
            max_result_digits=max_result_digits,
            used_lines=used_lines,
        )
        target_input = target_equation.line.split("=", 1)[0].replace(" ", "")
        answer = target_equation.line.split("=", 1)[1].strip()

        op_counts: Dict[str, int] = {}
        for equation in equations:
            op_counts[equation.op_symbol] = op_counts.get(equation.op_symbol, 0) + 1

        metadata: Dict[str, Any] = {
            "rules": {rule.op_symbol: rule.solver_desc for rule in rules},
            "rule_specs": [asdict(rule) for rule in rules],
            "target_rule": asdict(target_rule),
            "symbol_to_digit": dict(sorted(symbol_to_digit.items(), key=lambda item: item[0])),
            "digit_to_symbol": {str(k): v for k, v in sorted(digit_to_symbol.items())},
            "op_counts": op_counts,
            "equation_debug": [asdict(eq) for eq in equations],
            "target_debug": asdict(target_equation),
        }

        examples_text = "\n".join(eq.line for eq in equations)
        return examples_text, answer, target_input, metadata, resolved_mode

    @staticmethod
    def _make_solver() -> Any:
        if TraceConfig is None:
            return CryptarithmCSPSolver()
        return CryptarithmCSPSolver(
            trace_config=TraceConfig(
                max_steps=180,
                max_solution_chars=24000,
                max_examples_per_step=0,
                include_rejected_hypotheses=False,
                include_search_branches=True,
            )
        )

    @classmethod
    def solve_generated_prompt(
        cls,
        examples_text: str,
        prompt: str,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        solver = cls._make_solver()
        result = solver.solve(examples_text, prompt, timeout_seconds=timeout_seconds)
        solution = result.get("solution")
        if not solution:
            debug = result.get("debug", [])
            solution = "\n".join(debug) if isinstance(debug, list) else str(debug)
        return {
            "computed_answer": result.get("answer"),
            "generated_cot": solution,
            "solver_result": result,
        }

    def generate_dataset(
        self,
        num_samples: int,
        mode: str = "random",
        min_examples: int = 4,
        max_examples: int = 6,
        min_operators: int = 1,
        max_operators: int = 3,
        forced_rule_name: Optional[str] = None,
        forced_orientation: Optional[str] = None,
        max_result_digits: int = 4,
        solver_timeout_seconds: float = 30.0,
        max_attempts_per_sample: int = 30,
        require_solver_success: bool = True,
        nb_workers: Optional[int] = None,
        progress_bar: bool = True,
        use_parallel: bool = True,
    ) -> pd.DataFrame:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        if min_examples < 1 or max_examples < min_examples:
            raise ValueError("Invalid examples-count range")
        if max_attempts_per_sample < 1:
            raise ValueError("max_attempts_per_sample must be positive")

        jobs = pd.DataFrame(
            {
                "sample_index": list(range(num_samples)),
                "seed": [self.rng.randint(0, 2**31 - 1) for _ in range(num_samples)],
            }
        )
        if jobs.empty:
            return pd.DataFrame()

        config = {
            "mode": mode,
            "min_examples": min_examples,
            "max_examples": max_examples,
            "min_operators": min_operators,
            "max_operators": max_operators,
            "forced_rule_name": forced_rule_name,
            "forced_orientation": forced_orientation,
            "max_result_digits": max_result_digits,
            "solver_timeout_seconds": solver_timeout_seconds,
            "max_attempts_per_sample": max_attempts_per_sample,
            "require_solver_success": require_solver_success,
        }

        if use_parallel:
            pandarallel.initialize(nb_workers=nb_workers, progress_bar=progress_bar, verbose=0)
            rows = jobs.parallel_apply(
                lambda row: _generate_and_solve_cryptarithm_row(row.to_dict(), config),
                axis=1,
            )
        else:
            rows = jobs.apply(
                lambda row: _generate_and_solve_cryptarithm_row(row.to_dict(), config),
                axis=1,
            )

        return pd.DataFrame.from_records(rows.tolist())


def _generate_and_solve_cryptarithm_row(row: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(row["seed"])
    sample_index = int(row["sample_index"])
    generator = CryptarithmTaskGenerator(seed=seed)
    last_record: Optional[Dict[str, Any]] = None

    for attempt in range(1, int(config["max_attempts_per_sample"]) + 1):
        num_examples = generator.rng.randint(int(config["min_examples"]), int(config["max_examples"]))
        try:
            examples_text, answer, target_input, metadata, task_mode = generator.generate_task(
                num_examples=num_examples,
                mode=str(config["mode"]),
                min_operators=int(config["min_operators"]),
                max_operators=int(config["max_operators"]),
                forced_rule_name=config["forced_rule_name"],
                forced_orientation=config["forced_orientation"],
                max_result_digits=int(config["max_result_digits"]),
            )
            prompt = cryptarithm_prompt_template.format(examples=examples_text, target_input=target_input)
            solved = CryptarithmTaskGenerator.solve_generated_prompt(
                examples_text=examples_text,
                prompt=prompt,
                timeout_seconds=float(config["solver_timeout_seconds"]),
            )
            computed_answer = solved["computed_answer"]
            is_correct = computed_answer == answer
            solver_result = solved["solver_result"]

            last_record = {
                "sample_index": sample_index,
                "generator_seed": seed,
                "attempts": attempt,
                "prompt": prompt,
                "examples": examples_text,
                "target_input": target_input,
                "answer": answer,
                "generated_cot": solved["generated_cot"],
                "computed_answer": computed_answer,
                "is_correct": is_correct,
                "accepted": is_correct or not bool(config["require_solver_success"]),
                "task_mode": task_mode,
                "label": "cryptarithm",
                "rules": metadata["rules"],
                "rule_specs": metadata["rule_specs"],
                "solver_mapping": json.dumps(metadata["symbol_to_digit"], ensure_ascii=False, sort_keys=True),
                "digit_to_symbol": json.dumps(metadata["digit_to_symbol"], ensure_ascii=False, sort_keys=True),
                "op_counts": metadata["op_counts"],
                "target_rule": metadata["target_rule"],
                "equation_debug": metadata["equation_debug"],
                "target_debug": metadata["target_debug"],
                "solver_rules": solver_result.get("rules"),
                "solver_stats": solver_result.get("stats"),
                "solver_elapsed_seconds": solver_result.get("elapsed_seconds"),
                "solver_error": None if computed_answer is not None else solver_result.get("debug"),
            }

            if is_correct or not bool(config["require_solver_success"]):
                return last_record
        except Exception as exc:
            last_record = {
                "sample_index": sample_index,
                "generator_seed": seed,
                "attempts": attempt,
                "prompt": None,
                "examples": None,
                "target_input": None,
                "answer": None,
                "generated_cot": None,
                "computed_answer": None,
                "is_correct": False,
                "accepted": False,
                "task_mode": str(config["mode"]),
                "label": "cryptarithm",
                "rules": None,
                "rule_specs": None,
                "solver_mapping": None,
                "digit_to_symbol": None,
                "op_counts": None,
                "target_rule": None,
                "equation_debug": None,
                "target_debug": None,
                "solver_rules": None,
                "solver_stats": None,
                "solver_elapsed_seconds": None,
                "solver_error": f"{type(exc).__name__}: {exc}",
            }

    if last_record is None:
        raise RuntimeError("No generation attempt was executed")
    last_record["accepted"] = False
    return last_record


if __name__ == "__main__":
    generator = CryptarithmTaskGenerator(seed=42)
    df = generator.generate_dataset(num_samples=5, use_parallel=False, solver_timeout_seconds=10.0)
    print(f"Accuracy: {df['is_correct'].mean() * 100:.1f}%")
    print(df[["task_mode", "answer", "computed_answer", "is_correct", "attempts"]].to_string(index=False))
