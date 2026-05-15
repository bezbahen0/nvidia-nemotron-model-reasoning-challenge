from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pandarallel import pandarallel
except ImportError:
    pandarallel = None

from src.solvers.equations.cryptarithm import CryptarithmCSPSolver, TraceConfig


cryptarithm_prompt_template = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
{examples}
Now, determine the result for: {target_input}"""


CHARSET: Tuple[str, ...] = (
    "!", '"', "#", "$", "%", "&", "'", "(", ")",
    "*", "+", "-", "/", ":", "<", ">", "?",
    "@", "[", "\\", "]", "^", "`", "{", "|", "}",
)


EXAMPLE_COUNT_DIST = {
    3: 0.2064,
    4: 0.3750,
    5: 0.4186,
}


OPERATOR_COUNT_DIST = {
    1: 0.0174,
    2: 0.4215,
    3: 0.5610,
}


TARGET_OP_EXAMPLE_COUNT_DIST = {
    1: 0.4419,
    2: 0.3837,
    3: 0.1424,
    4: 0.0262,
    5: 0.0058,
}


VISIBLE_DIGIT_SYMBOL_COUNT_DIST = {
    10: 0.6686,
    9: 0.2733,
    8: 0.0552,
    6: 0.0029,
}


TARGET_OPERATOR_SYMBOL_DIST = {
    "!": 0.0174,
    '"': 0.0087,
    "#": 0.0203,
    "$": 0.0116,
    "%": 0.0174,
    "&": 0.0087,
    "'": 0.0116,
    "(": 0.0145,
    ")": 0.0233,
    "*": 0.2529,
    "+": 0.2122,
    "-": 0.1744,
    "/": 0.0058,
    ":": 0.0320,
    "<": 0.0087,
    ">": 0.0116,
    "?": 0.0145,
    "@": 0.0174,
    "[": 0.0087,
    "\\": 0.0291,
    "]": 0.0058,
    "^": 0.0029,
    "`": 0.0262,
    "{": 0.0174,
    "|": 0.0203,
    "}": 0.0262,
}


NON_TARGET_OPERATOR_SYMBOL_DIST = {
    "!": 0.0151,
    '"': 0.0056,
    "#": 0.0132,
    "$": 0.0151,
    "%": 0.0132,
    "&": 0.0113,
    "'": 0.0169,
    "(": 0.0132,
    ")": 0.0226,
    "*": 0.1940,
    "+": 0.2185,
    "-": 0.2392,
    "/": 0.0169,
    ":": 0.0132,
    "<": 0.0188,
    ">": 0.0132,
    "?": 0.0169,
    "@": 0.0226,
    "[": 0.0188,
    "\\": 0.0113,
    "]": 0.0075,
    "^": 0.0169,
    "`": 0.0151,
    "{": 0.0132,
    "|": 0.0188,
    "}": 0.0188,
}


TARGET_RULE_PAIR_DIST = {
    "little_endian:absdiff": 0.0320,
    "little_endian:add": 0.0814,
    "little_endian:add_m1": 0.0203,
    "little_endian:add_p1": 0.0233,
    "little_endian:concat_fwd": 0.0320,
    "little_endian:mul": 0.1512,
    "little_endian:mul_m1": 0.0058,
    "little_endian:mul_p1": 0.0145,
    "little_endian:sub_signed": 0.0610,
    "standard:absdiff": 0.1076,
    "standard:add": 0.0698,
    "standard:add_m1": 0.0349,
    "standard:add_p1": 0.0320,
    "standard:concat_fwd": 0.1134,
    "standard:mul": 0.0988,
    "standard:mul_m1": 0.0087,
    "standard:mul_p1": 0.0465,
    "standard:sub_signed": 0.0669,
}


NON_TARGET_RULE_PAIR_DIST = {
    "little_endian:absdiff": 0.0471,
    "little_endian:add": 0.0904,
    "little_endian:add_m1": 0.0132,
    "little_endian:add_p1": 0.0207,
    "little_endian:concat_fwd": 0.0113,
    "little_endian:mul": 0.0960,
    "little_endian:mul_m1": 0.0038,
    "little_endian:mul_p1": 0.0113,
    "little_endian:sub_signed": 0.0753,
    "standard:absdiff": 0.1205,
    "standard:add": 0.1073,
    "standard:add_m1": 0.0358,
    "standard:add_p1": 0.0245,
    "standard:concat_fwd": 0.0414,
    "standard:mul": 0.1243,
    "standard:mul_m1": 0.0264,
    "standard:mul_p1": 0.0358,
    "standard:sub_signed": 0.1149,
}


RESULT_LEN_BY_RULE_PAIR = {
    "standard:mul": {4: 0.831, 3: 0.169},
    "little_endian:mul": {4: 0.832, 3: 0.168},

    "standard:mul_p1": {4: 0.755, 3: 0.245},
    "little_endian:mul_p1": {4: 0.864, 3: 0.136},

    "standard:mul_m1": {4: 0.741, 3: 0.259},
    "little_endian:mul_m1": {4: 0.857, 3: 0.143},

    "standard:add": {3: 0.556, 2: 0.444},
    "little_endian:add": {3: 0.610, 2: 0.390},

    "standard:add_p1": {3: 0.634, 2: 0.366},
    "little_endian:add_p1": {3: 0.611, 2: 0.389},

    "standard:add_m1": {3: 0.660, 2: 0.340},
    "little_endian:add_m1": {3: 0.760, 2: 0.240},

    "standard:absdiff": {2: 0.857, 1: 0.143},
    "little_endian:absdiff": {2: 0.686, 1: 0.314},

    "standard:sub_signed": {2: 0.826, 1: 0.174},
    "little_endian:sub_signed": {2: 0.763, 1: 0.237},

    "standard:concat_fwd": {4: 1.0},
    "little_endian:concat_fwd": {4: 1.0},
}


EXAMPLE_SUB_SIGN_RATE_BY_RULE_PAIR = {
    "standard:sub_signed": 0.742,
    "little_endian:sub_signed": 0.695,
}


TARGET_SUB_SIGN_RATE = 0.455


def _weighted_choice(rng: random.Random, dist: Dict[Any, float]) -> Any:
    items = list(dist.items())
    return rng.choices(
        [item[0] for item in items],
        weights=[item[1] for item in items],
        k=1,
    )[0]


def _weighted_sample_without_replacement(
    rng: random.Random,
    dist: Dict[str, float],
    allowed: Sequence[str],
    k: int,
) -> List[str]:
    selected: List[str] = []

    for _ in range(k):
        local_dist = {
            value: dist.get(value, 0.000001)
            for value in allowed
            if value not in selected
        }

        selected.append(_weighted_choice(rng, local_dist))

    return selected


def _split_rule_pair(rule_pair: str) -> Tuple[str, str]:
    orientation, rule_name = rule_pair.split(":", 1)
    return orientation, rule_name


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

    raise ValueError(f"unknown rule_name: {rule_name}")


def _encode_number(value: int, digit_to_symbol: Dict[int, str], reverse: bool) -> str:
    if value < 0:
        raise ValueError("cannot encode negative number")

    encoded = "".join(digit_to_symbol[int(ch)] for ch in str(value))
    return encoded[::-1] if reverse else encoded


def _format_result(
    value: int,
    op_symbol: str,
    rule_name: str,
    reverse: bool,
    digit_to_symbol: Dict[int, str],
) -> Optional[str]:
    if rule_name == "sub_signed" and value < 0:
        return op_symbol + _encode_number(abs(value), digit_to_symbol, reverse)

    if value < 0:
        return None

    return _encode_number(value, digit_to_symbol, reverse)


def _result_body(result: str, op_symbol: str) -> str:
    if len(result) > 1 and result[0] == op_symbol:
        return result[1:]

    return result


def _format_equation(
    a: int,
    b: int,
    op_symbol: str,
    rule_pair: str,
    digit_to_symbol: Dict[int, str],
) -> Optional[Dict[str, Any]]:
    orientation, rule_name = _split_rule_pair(rule_pair)
    reverse = orientation == "little_endian"

    value = _apply_rule(rule_name, a, b)
    if value is None:
        return None

    left = _encode_number(a, digit_to_symbol, reverse)
    right = _encode_number(b, digit_to_symbol, reverse)
    result = _format_result(value, op_symbol, rule_name, reverse, digit_to_symbol)

    if result is None:
        return None

    body = _result_body(result, op_symbol)

    return {
        "line": f"{left}{op_symbol}{right} = {result}",
        "input": f"{left}{op_symbol}{right}",
        "answer": result,
        "answer_body": body,
        "has_sign": result != body,
        "result_body_len": len(body),
        "a": a,
        "b": b,
        "value": value,
        "op_symbol": op_symbol,
        "rule_pair": rule_pair,
    }


def _sample_operands(
    rng: random.Random,
    rule_pair: str,
    want_signed: Optional[bool],
) -> Tuple[int, int]:
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)

    if rule_pair.endswith(":sub_signed"):
        if want_signed is True and a >= b:
            a, b = b, a
        elif want_signed is False and a < b:
            a, b = b, a

    return a, b


def _generate_equation(
    rng: random.Random,
    op_symbol: str,
    rule_pair: str,
    digit_to_symbol: Dict[int, str],
    used_lines: set[str],
    for_target: bool,
    max_attempts: int = 5000,
) -> Optional[Dict[str, Any]]:
    desired_len = _weighted_choice(rng, RESULT_LEN_BY_RULE_PAIR[rule_pair])

    want_signed = None
    if rule_pair.endswith(":sub_signed"):
        if for_target:
            want_signed = rng.random() < TARGET_SUB_SIGN_RATE
        else:
            want_signed = rng.random() < EXAMPLE_SUB_SIGN_RATE_BY_RULE_PAIR[rule_pair]

    for _ in range(max_attempts):
        a, b = _sample_operands(rng, rule_pair, want_signed)
        equation = _format_equation(a, b, op_symbol, rule_pair, digit_to_symbol)

        if equation is None:
            continue

        if equation["line"] in used_lines:
            continue

        if equation["result_body_len"] != desired_len:
            continue

        if want_signed is not None and equation["has_sign"] != want_signed:
            continue

        used_lines.add(equation["line"])
        return equation

    return None


def _visible_digit_symbols(equations: Sequence[Dict[str, Any]], target_input: str) -> set[str]:
    symbols: set[str] = set()

    for equation in equations:
        expression = equation["input"]
        symbols.update(expression[:2])
        symbols.update(expression[3:])
        symbols.update(equation["answer_body"])

    symbols.update(target_input[:2])
    symbols.update(target_input[3:])

    return symbols


def _make_symbol_mapping(
    rng: random.Random,
    op_count: int,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    target_op = _weighted_choice(rng, TARGET_OPERATOR_SYMBOL_DIST)

    non_target_ops = _weighted_sample_without_replacement(
        rng=rng,
        dist=NON_TARGET_OPERATOR_SYMBOL_DIST,
        allowed=[ch for ch in CHARSET if ch != target_op],
        k=op_count - 1,
    )

    op_symbols = [target_op] + non_target_ops

    digit_symbol_pool = [ch for ch in CHARSET if ch not in set(op_symbols)]
    digit_symbols = rng.sample(digit_symbol_pool, 10)

    digits = list(range(10))
    rng.shuffle(digits)

    symbol_to_digit = {
        symbol: digit
        for symbol, digit in zip(digit_symbols, digits)
    }

    digit_to_symbol = {
        digit: symbol
        for symbol, digit in symbol_to_digit.items()
    }

    return op_symbols, symbol_to_digit, digit_to_symbol


def _allocate_example_ops(
    rng: random.Random,
    num_examples: int,
    op_count: int,
    target_index: int,
) -> List[int]:
    if op_count == 1:
        return [target_index] * num_examples

    target_count = _weighted_choice(rng, TARGET_OP_EXAMPLE_COUNT_DIST)
    target_count = min(target_count, num_examples - (op_count - 1))
    target_count = max(1, target_count)

    op_indices: List[int] = [target_index] * target_count

    for op_index in range(op_count):
        if op_index != target_index:
            op_indices.append(op_index)

    while len(op_indices) < num_examples:
        op_indices.append(rng.randrange(op_count))

    rng.shuffle(op_indices)
    return op_indices


class CryptarithmTaskGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    @staticmethod
    def _make_solver() -> CryptarithmCSPSolver:
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
        result = cls._make_solver().solve(
            examples_text,
            prompt,
            timeout_seconds=timeout_seconds,
        )

        solution = result.get("solution")
        if not solution:
            debug = result.get("debug", [])
            solution = "\n".join(debug) if isinstance(debug, list) else str(debug)

        return {
            "computed_answer": result.get("answer"),
            "generated_cot": solution,
            "solver_result": result,
        }

    def generate_task(
        self,
        max_attempts: int = 500,
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        for _ in range(max_attempts):
            num_examples = _weighted_choice(self.rng, EXAMPLE_COUNT_DIST)
            op_count = _weighted_choice(self.rng, OPERATOR_COUNT_DIST)
            op_count = min(op_count, num_examples)

            op_symbols, symbol_to_digit, digit_to_symbol = _make_symbol_mapping(
                rng=self.rng,
                op_count=op_count,
            )

            target_index = 0
            rules: List[Dict[str, str]] = []

            for op_index, op_symbol in enumerate(op_symbols):
                rule_pair = _weighted_choice(
                    self.rng,
                    TARGET_RULE_PAIR_DIST if op_index == target_index else NON_TARGET_RULE_PAIR_DIST,
                )

                rules.append({
                    "op_symbol": op_symbol,
                    "rule_pair": rule_pair,
                })

            used_lines: set[str] = set()
            example_op_indices = _allocate_example_ops(
                rng=self.rng,
                num_examples=num_examples,
                op_count=op_count,
                target_index=target_index,
            )

            equations: List[Dict[str, Any]] = []
            failed = False

            for op_index in example_op_indices:
                rule = rules[op_index]
                equation = _generate_equation(
                    rng=self.rng,
                    op_symbol=rule["op_symbol"],
                    rule_pair=rule["rule_pair"],
                    digit_to_symbol=digit_to_symbol,
                    used_lines=used_lines,
                    for_target=False,
                )

                if equation is None:
                    failed = True
                    break

                equations.append(equation)

            if failed:
                continue

            target_rule = rules[target_index]
            target_equation = _generate_equation(
                rng=self.rng,
                op_symbol=target_rule["op_symbol"],
                rule_pair=target_rule["rule_pair"],
                digit_to_symbol=digit_to_symbol,
                used_lines=used_lines,
                for_target=True,
            )

            if target_equation is None:
                continue

            visible_symbols = _visible_digit_symbols(equations, target_equation["input"])

            if not set(target_equation["answer_body"]).issubset(visible_symbols):
                continue

            desired_visible_count = _weighted_choice(self.rng, VISIBLE_DIGIT_SYMBOL_COUNT_DIST)
            if len(visible_symbols) != desired_visible_count:
                continue

            examples_text = "\n".join(equation["line"] for equation in equations)
            prompt = cryptarithm_prompt_template.format(
                examples=examples_text,
                target_input=target_equation["input"],
            )

            metadata = {
                "rules": rules,
                "target_rule": target_rule,
                "symbol_to_digit": symbol_to_digit,
                "digit_to_symbol": digit_to_symbol,
                "visible_digit_symbol_count": len(visible_symbols),
                "equations": equations,
                "target": target_equation,
            }

            return (
                examples_text,
                target_equation["answer"],
                target_equation["input"],
                metadata,
                "profiled_2x2",
            )

        raise RuntimeError("could not generate profiled cryptarithm task")

    def generate_dataset(
        self,
        num_samples: int,
        mode: str = "random",
        solver_timeout_seconds: float = 30.0,
        max_attempts_per_sample: int = 50,
        require_solver_success: bool = True,
        nb_workers: Optional[int] = None,
        progress_bar: bool = True,
        use_parallel: bool = True,
    ) -> pd.DataFrame:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")

        jobs = pd.DataFrame({
            "sample_index": list(range(num_samples)),
            "seed": [self.rng.randint(0, 2**31 - 1) for _ in range(num_samples)],
        })

        if jobs.empty:
            return pd.DataFrame(
                columns=[
                    "prompt",
                    "answer",
                    "generated_cot",
                    "computed_answer",
                    "task_mode",
                    "label",
                ]
            )

        config = {
            "solver_timeout_seconds": solver_timeout_seconds,
            "max_attempts_per_sample": max_attempts_per_sample,
            "require_solver_success": require_solver_success,
            "label": "equations transformation",
        }

        if use_parallel and nb_workers != 1 and pandarallel is not None:
            pandarallel.initialize(
                nb_workers=nb_workers,
                progress_bar=progress_bar,
                verbose=0,
            )

            rows = jobs.parallel_apply(
                lambda row: _generate_cryptarithm_row(row.to_dict(), config),
                axis=1,
            )
        else:
            rows = jobs.apply(
                lambda row: _generate_cryptarithm_row(row.to_dict(), config),
                axis=1,
            )

        return pd.DataFrame.from_records(rows.tolist())


def _generate_cryptarithm_row(row: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    generator = CryptarithmTaskGenerator(seed=int(row["seed"]))
    last_error = None

    for _ in range(int(config["max_attempts_per_sample"])):
        try:
            examples_text, answer, target_input, metadata, task_mode = generator.generate_task()

            prompt = cryptarithm_prompt_template.format(
                examples=examples_text,
                target_input=target_input,
            )

            solved = CryptarithmTaskGenerator.solve_generated_prompt(
                examples_text=examples_text,
                prompt=prompt,
                timeout_seconds=float(config["solver_timeout_seconds"]),
            )

            computed_answer = solved["computed_answer"]

            if computed_answer == answer or not bool(config["require_solver_success"]):
                return {
                    "prompt": prompt,
                    "answer": answer,
                    "generated_cot": solved["generated_cot"],
                    "computed_answer": computed_answer,
                    "task_mode": task_mode,
                    "label": config["label"],
                }

            last_error = {
                "expected": answer,
                "computed": computed_answer,
                "metadata": metadata,
                "solver_debug": solved["solver_result"].get("debug"),
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "prompt": None,
        "answer": None,
        "generated_cot": json.dumps(last_error, ensure_ascii=False),
        "computed_answer": None,
        "task_mode": "profiled_2x2",
        "label": config["label"],
    }


if __name__ == "__main__":
    generator = CryptarithmTaskGenerator(seed=42)

    df = generator.generate_dataset(
        num_samples=10,
        nb_workers=1,
        progress_bar=False,
    )

    print(df.columns.tolist())
    print(df["task_mode"].value_counts(dropna=False))
    print((df["answer"] == df["computed_answer"]).mean())
    print(df[["prompt", "answer", "computed_answer"]].head(3).to_string(index=False))