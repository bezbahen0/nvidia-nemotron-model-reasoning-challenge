from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pandarallel import pandarallel
except ImportError:  # optional dependency
    pandarallel = None


CRYPTARITHM_PROMPT_TEMPLATE = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
{examples}
Now, determine the result for: {target_input}"""

CHARSET: Tuple[str, ...] = (
    "!", '"', "#", "$", "%", "&", "'", "(", ")",
    "*", "+", "-", "/", ":", "<", ">", "?",
    "@", "[", "\\", "]", "^", "`", "{", "|", "}",
)

# Rule names are intentionally identical to the replay solver names.
RULES: Tuple[str, ...] = (
    "std/cat", "rev/cat",
    "std/add", "std/add1", "std/addm1", "rev/add", "rev/add1", "rev/addm1",
    "std/abs", "std/sub", "rev/abs", "rev/sub",
    "std/mul", "std/mul1", "std/mulm1", "rev/mul", "rev/mul1", "rev/mulm1",
)

# Conservative defaults close to the real 2x2 cryptarithm distribution.  These
# can be replaced by a profile fitted from your 317 real solved rows.
DEFAULT_EXAMPLE_COUNT_DIST = {3: 0.22, 4: 0.38, 5: 0.40}
DEFAULT_OPERATOR_COUNT_DIST = {1: 0.02, 2: 0.42, 3: 0.56}
DEFAULT_TARGET_OP_EXAMPLE_COUNT_DIST = {1: 0.44, 2: 0.38, 3: 0.14, 4: 0.03, 5: 0.01}
DEFAULT_VISIBLE_DIGIT_SYMBOL_COUNT_DIST = {8: 0.06, 9: 0.27, 10: 0.67}

DEFAULT_RULE_DIST = {
    "std/cat": 0.08,
    "rev/cat": 0.025,
    "std/add": 0.08,
    "std/add1": 0.03,
    "std/addm1": 0.03,
    "rev/add": 0.085,
    "rev/add1": 0.025,
    "rev/addm1": 0.02,
    "std/abs": 0.11,
    "rev/abs": 0.05,
    "std/sub": 0.08,
    "rev/sub": 0.07,
    "std/mul": 0.11,
    "std/mul1": 0.04,
    "std/mulm1": 0.02,
    "rev/mul": 0.12,
    "rev/mul1": 0.02,
    "rev/mulm1": 0.01,
}

# Prefer the visually meaningful operators, but still allow every character.
DEFAULT_TARGET_OPERATOR_SYMBOL_DIST = {
    "*": 0.25,
    "+": 0.22,
    "-": 0.18,
    ":": 0.04,
    "\\": 0.03,
    "`": 0.03,
    "}": 0.03,
    ")": 0.025,
    "|": 0.025,
    "#": 0.025,
    "!": 0.02,
    "%": 0.02,
    "@": 0.02,
}
DEFAULT_NON_TARGET_OPERATOR_SYMBOL_DIST = {
    "-": 0.24,
    "+": 0.22,
    "*": 0.19,
    ")": 0.03,
    "@": 0.025,
    "|": 0.025,
    "<": 0.02,
    "[": 0.02,
    "}": 0.02,
    "/": 0.02,
}


@dataclass(frozen=True)
class GeneratedEquation:
    line: str
    input: str
    answer: str
    answer_body: str
    has_sign: bool
    result_body_len: int
    a: int
    b: int
    value: int
    op_symbol: str
    rule: str


@dataclass
class CryptarithmGenerationProfile:
    example_count_dist: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_EXAMPLE_COUNT_DIST))
    operator_count_dist: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_OPERATOR_COUNT_DIST))
    target_op_example_count_dist: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_TARGET_OP_EXAMPLE_COUNT_DIST))
    visible_digit_symbol_count_dist: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_VISIBLE_DIGIT_SYMBOL_COUNT_DIST))
    target_rule_dist: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RULE_DIST))
    non_target_rule_dist: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RULE_DIST))
    target_operator_symbol_dist: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TARGET_OPERATOR_SYMBOL_DIST))
    non_target_operator_symbol_dist: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NON_TARGET_OPERATOR_SYMBOL_DIST))

    @staticmethod
    def _normalize_counter(counter: Dict[Any, float], fallback: Dict[Any, float], smoothing: float = 0.05) -> Dict[Any, float]:
        merged: Dict[Any, float] = {}
        for key, value in fallback.items():
            merged[key] = float(value) * smoothing
        for key, value in counter.items():
            merged[key] = merged.get(key, 0.0) + float(value)
        total = sum(v for v in merged.values() if v > 0)
        if total <= 0:
            return dict(fallback)
        return {k: v / total for k, v in merged.items() if v > 0}

    @classmethod
    def from_source_data(cls, source_data: pd.DataFrame, smoothing: float = 0.05) -> "CryptarithmGenerationProfile":
        """Fit a lightweight sampling profile from solved cryptarithm rows.

        Expected columns: prompt, generated_cot/solution/cot, label optional.
        The profile intentionally uses only stable observables: example counts,
        operator counts, target operator coverage, visible-symbol counts, selected
        rules in CoT verification, and operator-symbol frequencies.
        """
        ex_counts: Dict[int, int] = {}
        op_counts: Dict[int, int] = {}
        target_op_ex_counts: Dict[int, int] = {}
        visible_counts: Dict[int, int] = {}
        target_rule_counts: Dict[str, int] = {}
        non_target_rule_counts: Dict[str, int] = {}
        target_op_symbol_counts: Dict[str, int] = {}
        non_target_op_symbol_counts: Dict[str, int] = {}

        for _, row in source_data.iterrows():
            prompt = str(row.get("prompt", "") or "")
            cot = str(row.get("generated_cot", row.get("solution", row.get("cot", ""))) or "")
            parsed = _parse_prompt(prompt)
            if not parsed:
                continue
            equations, target_input = parsed
            if not equations or not target_input:
                continue
            target_op = _parse_target_op(target_input, {eq["op"] for eq in equations})
            if target_op is None:
                continue
            ops = sorted({eq["op"] for eq in equations})
            ex_counts[len(equations)] = ex_counts.get(len(equations), 0) + 1
            op_counts[len(ops)] = op_counts.get(len(ops), 0) + 1
            target_op_ex_counts[sum(1 for eq in equations if eq["op"] == target_op)] = target_op_ex_counts.get(sum(1 for eq in equations if eq["op"] == target_op), 0) + 1
            visible = _visible_symbols_from_prompt(equations, target_input)
            visible_counts[len(visible)] = visible_counts.get(len(visible), 0) + 1
            target_op_symbol_counts[target_op] = target_op_symbol_counts.get(target_op, 0) + 1
            for op in ops:
                if op != target_op:
                    non_target_op_symbol_counts[op] = non_target_op_symbol_counts.get(op, 0) + 1

            rules = _parse_selected_rules_from_cot(cot)
            for op, rule in rules.items():
                if rule not in RULES:
                    continue
                if op == target_op:
                    target_rule_counts[rule] = target_rule_counts.get(rule, 0) + 1
                else:
                    non_target_rule_counts[rule] = non_target_rule_counts.get(rule, 0) + 1

        return cls(
            example_count_dist=cls._normalize_counter(ex_counts, DEFAULT_EXAMPLE_COUNT_DIST, smoothing=smoothing),
            operator_count_dist=cls._normalize_counter(op_counts, DEFAULT_OPERATOR_COUNT_DIST, smoothing=smoothing),
            target_op_example_count_dist=cls._normalize_counter(target_op_ex_counts, DEFAULT_TARGET_OP_EXAMPLE_COUNT_DIST, smoothing=smoothing),
            visible_digit_symbol_count_dist=cls._normalize_counter(visible_counts, DEFAULT_VISIBLE_DIGIT_SYMBOL_COUNT_DIST, smoothing=smoothing),
            target_rule_dist=cls._normalize_counter(target_rule_counts, DEFAULT_RULE_DIST, smoothing=smoothing),
            non_target_rule_dist=cls._normalize_counter(non_target_rule_counts, DEFAULT_RULE_DIST, smoothing=smoothing),
            target_operator_symbol_dist=cls._normalize_counter(target_op_symbol_counts, DEFAULT_TARGET_OPERATOR_SYMBOL_DIST, smoothing=smoothing),
            non_target_operator_symbol_dist=cls._normalize_counter(non_target_op_symbol_counts, DEFAULT_NON_TARGET_OPERATOR_SYMBOL_DIST, smoothing=smoothing),
        )


@dataclass
class CryptarithmTaskGeneratorConfig:
    sample_frac: float = 1.0
    solver_timeout_seconds: float = 10.0
    max_attempts_per_row: int = 100
    require_solver_success: bool = True
    require_answer_symbols_visible: bool = True
    min_visible_digit_symbols: int = 8
    max_visible_digit_symbols: int = 10
    # If not None, reject solved CoTs longer than this many characters. Useful
    # for avoiding synthetic tail cases before subtask augmentation.
    max_cot_chars: Optional[int] = None


def _weighted_choice(rng: random.Random, dist: Dict[Any, float]) -> Any:
    items = [(k, float(v)) for k, v in dist.items() if float(v) > 0]
    if not items:
        raise ValueError("empty distribution")
    return rng.choices([k for k, _ in items], weights=[v for _, v in items], k=1)[0]


def _weighted_sample_without_replacement(
    rng: random.Random,
    dist: Dict[str, float],
    allowed: Sequence[str],
    k: int,
) -> List[str]:
    selected: List[str] = []
    for _ in range(k):
        local = {x: max(float(dist.get(x, 0.0)), 1e-6) for x in allowed if x not in selected}
        selected.append(_weighted_choice(rng, local))
    return selected


def _split_rule(rule: str) -> Tuple[str, str]:
    orientation, name = rule.split("/", 1)
    if orientation not in {"std", "rev"}:
        raise ValueError(f"bad orientation in rule {rule!r}")
    return orientation, name


def _apply_rule(rule_name: str, a: int, b: int) -> Optional[int]:
    if rule_name == "add":
        return a + b
    if rule_name == "add1":
        return a + b + 1
    if rule_name == "addm1":
        v = a + b - 1
        return v if v >= 0 else None
    if rule_name == "mul":
        return a * b
    if rule_name == "mul1":
        return a * b + 1
    if rule_name == "mulm1":
        v = a * b - 1
        return v if v >= 0 else None
    if rule_name == "abs":
        return abs(a - b)
    if rule_name == "sub":
        return a - b
    if rule_name == "cat":
        return int(f"{a}{b}")
    raise ValueError(f"unknown rule: {rule_name}")


def _encode_number(value: int, digit_to_symbol: Dict[int, str], reverse: bool) -> str:
    if value < 0:
        raise ValueError("cannot encode negative number")
    chars = "".join(digit_to_symbol[int(ch)] for ch in str(value))
    return chars[::-1] if reverse else chars


def _format_result(value: int, op_symbol: str, rule_name: str, digit_to_symbol: Dict[int, str], reverse: bool) -> Optional[str]:
    if rule_name == "sub" and value < 0:
        return op_symbol + _encode_number(abs(value), digit_to_symbol, reverse)
    if value < 0:
        return None
    return _encode_number(value, digit_to_symbol, reverse)


def _result_body(result: str, op_symbol: str) -> str:
    if len(result) > 1 and result[0] == op_symbol:
        return result[1:]
    return result


def _format_equation(a: int, b: int, op_symbol: str, rule: str, digit_to_symbol: Dict[int, str]) -> Optional[GeneratedEquation]:
    orientation, rule_name = _split_rule(rule)
    reverse = orientation == "rev"
    value = _apply_rule(rule_name, a, b)
    if value is None:
        return None
    left = _encode_number(a, digit_to_symbol, reverse)
    right = _encode_number(b, digit_to_symbol, reverse)
    answer = _format_result(value, op_symbol, rule_name, digit_to_symbol, reverse)
    if answer is None:
        return None
    body = _result_body(answer, op_symbol)
    return GeneratedEquation(
        line=f"{left}{op_symbol}{right} = {answer}",
        input=f"{left}{op_symbol}{right}",
        answer=answer,
        answer_body=body,
        has_sign=answer != body,
        result_body_len=len(body),
        a=a,
        b=b,
        value=value,
        op_symbol=op_symbol,
        rule=rule,
    )


def _sample_operands(rng: random.Random, rule: str, want_negative: Optional[bool]) -> Tuple[int, int]:
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    if rule.endswith("/sub"):
        if want_negative is True and a >= b:
            a, b = b, a
        elif want_negative is False and a < b:
            a, b = b, a
    return a, b


def _result_length_allowed(rule: str, length: int) -> bool:
    _, name = _split_rule(rule)
    if name == "cat":
        return length == 4
    if name in {"mul", "mul1"}:
        return length in {3, 4}
    if name == "mulm1":
        return length in {2, 3, 4}
    if name in {"add", "add1", "addm1"}:
        return length in {2, 3}
    if name in {"abs", "sub"}:
        return length in {1, 2}
    return True


def _generate_equation(
    rng: random.Random,
    op_symbol: str,
    rule: str,
    digit_to_symbol: Dict[int, str],
    used_lines: set[str],
    for_target: bool,
    max_attempts: int = 3000,
) -> Optional[GeneratedEquation]:
    # Keep signed-sub examples mixed; target sub signs are slightly less frequent.
    want_negative: Optional[bool] = None
    if rule.endswith("/sub"):
        want_negative = rng.random() < (0.45 if for_target else 0.70)

    for _ in range(max_attempts):
        a, b = _sample_operands(rng, rule, want_negative)
        eq = _format_equation(a, b, op_symbol, rule, digit_to_symbol)
        if eq is None:
            continue
        if eq.line in used_lines:
            continue
        if not _result_length_allowed(rule, eq.result_body_len):
            continue
        if want_negative is not None and eq.has_sign != want_negative:
            continue
        used_lines.add(eq.line)
        return eq
    return None


def _visible_digit_symbols(equations: Sequence[GeneratedEquation], target_input: str) -> set[str]:
    symbols: set[str] = set()
    for eq in equations:
        compact = eq.input
        # inputs are always 2-symbol operand, op, 2-symbol operand
        symbols.update(compact[:2])
        symbols.update(compact[3:])
        symbols.update(eq.answer_body)
    symbols.update(target_input[:2])
    symbols.update(target_input[3:])
    return symbols


def _make_symbol_mapping(rng: random.Random, op_count: int, profile: CryptarithmGenerationProfile) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    target_op = _weighted_choice(rng, profile.target_operator_symbol_dist)
    non_target_ops = _weighted_sample_without_replacement(
        rng,
        profile.non_target_operator_symbol_dist,
        allowed=[ch for ch in CHARSET if ch != target_op],
        k=max(0, op_count - 1),
    )
    op_symbols = [target_op] + non_target_ops
    digit_pool = [ch for ch in CHARSET if ch not in set(op_symbols)]
    digit_symbols = rng.sample(digit_pool, 10)
    digits = list(range(10))
    rng.shuffle(digits)
    symbol_to_digit = {sym: digit for sym, digit in zip(digit_symbols, digits)}
    digit_to_symbol = {digit: sym for sym, digit in symbol_to_digit.items()}
    return op_symbols, symbol_to_digit, digit_to_symbol


def _allocate_example_ops(rng: random.Random, num_examples: int, op_count: int, profile: CryptarithmGenerationProfile) -> List[int]:
    if op_count <= 1:
        return [0] * num_examples
    target_count = int(_weighted_choice(rng, profile.target_op_example_count_dist))
    target_count = min(target_count, num_examples - (op_count - 1))
    target_count = max(1, target_count)
    ops = [0] * target_count
    for op_index in range(1, op_count):
        ops.append(op_index)
    while len(ops) < num_examples:
        ops.append(rng.randrange(op_count))
    rng.shuffle(ops)
    return ops


def _stable_id(*parts: Any) -> str:
    raw = "\n".join(str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


def _parse_prompt(prompt: str) -> Optional[Tuple[List[Dict[str, str]], str]]:
    text = str(prompt or "")
    m = re.search(r"Below are a few examples:\s*(.*?)\s*Now, determine the result for:\s*(\S+)", text, flags=re.S)
    if not m:
        return None
    example_text, target = m.group(1), m.group(2).strip().rstrip(".")
    lines = [ln.strip() for ln in example_text.splitlines() if ln.strip() and "=" in ln]
    known_ops: set[str] = set()
    first_pass: List[Dict[str, str]] = []
    for ln in lines:
        eq = _parse_equation_line(ln, known_ops=None)
        if eq:
            first_pass.append(eq)
            known_ops.add(eq["op"])
    equations = [_parse_equation_line(ln, known_ops=known_ops) for ln in lines]
    equations = [eq for eq in equations if eq]
    return equations, target


def _preferred_operator_index(left_side: str) -> int:
    return 2 if len(left_side) >= 5 else len(left_side) // 2


def _parse_equation_line(line: str, known_ops: Optional[set[str]] = None) -> Optional[Dict[str, str]]:
    compact = str(line).strip().replace(" ", "")
    if "=" not in compact:
        return None
    lhs, rhs = compact.split("=", 1)
    idx = None
    pref = _preferred_operator_index(lhs)
    if known_ops:
        if 0 < pref < len(lhs) - 1 and lhs[pref] in known_ops:
            idx = pref
        else:
            idx = next((i for i, ch in enumerate(lhs) if 0 < i < len(lhs) - 1 and ch in known_ops), None)
    if idx is None and 0 < pref < len(lhs) - 1:
        idx = pref
    if idx is None:
        return None
    op = lhs[idx]
    has_sign = len(rhs) > 1 and rhs[0] == op
    result = rhs[1:] if has_sign else rhs
    return {"raw": compact, "left": lhs[:idx], "op": op, "right": lhs[idx + 1 :], "output": result, "signed": str(has_sign)}


def _parse_target_op(target_input: str, known_ops: set[str]) -> Optional[str]:
    expr = target_input.strip().replace(" ", "")
    pref = _preferred_operator_index(expr)
    if 0 < pref < len(expr) - 1 and expr[pref] in known_ops:
        return expr[pref]
    for i, ch in enumerate(expr):
        if 0 < i < len(expr) - 1 and ch in known_ops:
            return ch
    return expr[pref] if 0 < pref < len(expr) - 1 else None


def _visible_symbols_from_prompt(equations: Sequence[Dict[str, str]], target_input: str) -> set[str]:
    ops = {eq["op"] for eq in equations}
    visible: set[str] = set()
    for eq in equations:
        visible.update(ch for ch in eq["left"] + eq["right"] + eq["output"] if ch not in ops)
    target_op = _parse_target_op(target_input, ops)
    if target_op:
        visible.update(ch for ch in target_input if ch != target_op)
    return visible


def _parse_selected_rules_from_cot(cot: str) -> Dict[str, str]:
    rules: Dict[str, str] = {}
    # v2.3 verification section: Operator '*' uses rev/mul
    for m in re.finditer(r"(?m)^Operator\s+(.+?)\s+uses\s+([a-z]+/[a-z0-9]+)\s*$", str(cot or "")):
        op_token = m.group(1).strip()
        op = op_token[1:-1] if len(op_token) >= 2 and op_token[0] in {"'", '"'} and op_token[-1] == op_token[0] else op_token
        rules[op] = m.group(2)
    return rules


def _default_solver_factory(timeout_seconds: float) -> Any:
    """Import your replay solver from common repo locations.

    Put cryptarithm_replay_v2_3.py into one of these locations:
      - src/solvers/equations/cryptarithm_replay_v2_3.py
      - src/solvers/equations/cryptarithm.py with CryptarithmReplaySolver
      - local module cryptarithm_replay_v2_3.py
    """
    module_names = [
        "src.solvers.equations.cryptarithm_replay_v2_3",
        "src.solvers.equations.cryptarithm_replay",
        "src.solvers.equations.cryptarithm",
        "cryptarithm_replay_v2_3",
    ]
    last_error: Optional[Exception] = None
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            solver_cls = getattr(mod, "CryptarithmReplaySolver", None) or getattr(mod, "CryptarithmSolver", None)
            config_cls = getattr(mod, "ReplayConfig", None) or getattr(mod, "SolverConfig", None)
            if solver_cls is None:
                continue
            if config_cls is not None:
                try:
                    config = config_cls(timeout_seconds=timeout_seconds)
                    return solver_cls(config)
                except TypeError:
                    return solver_cls()
            return solver_cls()
        except Exception as exc:
            last_error = exc
    raise ImportError(f"Could not import cryptarithm replay solver. Last error: {last_error}")


class CryptarithmReplayTaskGenerator:
    """Profiled generator for new full cryptarithm tasks.

    The generator creates synthetic Alice-style full cryptarithm prompts and then
    validates them with the replay solver.  Accepted rows can be used directly as
    full cryptarithm training examples and/or passed through CryptarithmAugmentGenerator
    to create local subtasks.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        profile: Optional[CryptarithmGenerationProfile] = None,
        config: Optional[CryptarithmTaskGeneratorConfig] = None,
        solver_factory: Optional[Callable[[float], Any]] = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.profile = profile or CryptarithmGenerationProfile()
        self.config = config or CryptarithmTaskGeneratorConfig()
        self.solver_factory = solver_factory or _default_solver_factory

    @classmethod
    def from_source_data(
        cls,
        source_data: pd.DataFrame,
        seed: Optional[int] = None,
        config: Optional[CryptarithmTaskGeneratorConfig] = None,
        solver_factory: Optional[Callable[[float], Any]] = None,
        smoothing: float = 0.05,
    ) -> "CryptarithmReplayTaskGenerator":
        profile = CryptarithmGenerationProfile.from_source_data(source_data, smoothing=smoothing)
        return cls(seed=seed, profile=profile, config=config, solver_factory=solver_factory)

    def _make_candidate(self) -> Optional[Dict[str, Any]]:
        num_examples = int(_weighted_choice(self.rng, self.profile.example_count_dist))
        op_count = int(_weighted_choice(self.rng, self.profile.operator_count_dist))
        op_count = max(1, min(op_count, num_examples, 3))
        op_symbols, symbol_to_digit, digit_to_symbol = _make_symbol_mapping(self.rng, op_count, self.profile)
        target_index = 0

        rules: List[Dict[str, str]] = []
        for i, op_symbol in enumerate(op_symbols):
            dist = self.profile.target_rule_dist if i == target_index else self.profile.non_target_rule_dist
            rules.append({"op_symbol": op_symbol, "rule": _weighted_choice(self.rng, dist)})

        used_lines: set[str] = set()
        op_indices = _allocate_example_ops(self.rng, num_examples, op_count, self.profile)
        equations: List[GeneratedEquation] = []
        for op_index in op_indices:
            rule = rules[op_index]
            eq = _generate_equation(
                self.rng,
                op_symbol=rule["op_symbol"],
                rule=rule["rule"],
                digit_to_symbol=digit_to_symbol,
                used_lines=used_lines,
                for_target=False,
            )
            if eq is None:
                return None
            equations.append(eq)

        target_rule = rules[target_index]
        target_eq = _generate_equation(
            self.rng,
            op_symbol=target_rule["op_symbol"],
            rule=target_rule["rule"],
            digit_to_symbol=digit_to_symbol,
            used_lines=used_lines,
            for_target=True,
        )
        if target_eq is None:
            return None

        visible = _visible_digit_symbols(equations, target_eq.input)
        if not (self.config.min_visible_digit_symbols <= len(visible) <= self.config.max_visible_digit_symbols):
            return None
        desired_visible = int(_weighted_choice(self.rng, self.profile.visible_digit_symbol_count_dist))
        # Soft profile matching: do not force rare counts too hard, but keep the
        # generated distribution close to source data.
        if len(visible) != desired_visible and self.rng.random() < 0.75:
            return None
        if self.config.require_answer_symbols_visible and not set(target_eq.answer_body).issubset(visible):
            return None

        examples_text = "\n".join(eq.line for eq in equations)
        prompt = CRYPTARITHM_PROMPT_TEMPLATE.format(examples=examples_text, target_input=target_eq.input)
        metadata = {
            "rules": rules,
            "target_rule": target_rule,
            "symbol_to_digit": symbol_to_digit,
            "digit_to_symbol": digit_to_symbol,
            "visible_digit_symbol_count": len(visible),
            "equations": [asdict(eq) for eq in equations],
            "target": asdict(target_eq),
        }
        return {
            "prompt": prompt,
            "examples_text": examples_text,
            "target_input": target_eq.input,
            "answer": target_eq.answer,
            "metadata": metadata,
            "task_mode": "profiled_replay_2x2",
            "label": "cryptarithm",
        }

    def solve_prompt(self, prompt: str, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        solver = self.solver_factory(float(timeout_seconds or self.config.solver_timeout_seconds))
        result = solver.solve(prompt, None, timeout_seconds=float(timeout_seconds or self.config.solver_timeout_seconds))
        solution = result.get("solution") or ""
        computed = result.get("answer")
        if computed is None:
            # Some solvers only expose extract_answer.
            extract = getattr(solver, "extract_answer", None)
            if callable(extract):
                ans = extract(solution)
                computed = None if ans == "nan" else ans
        return {"computed_answer": computed, "generated_cot": solution, "solver_result": result}

    def generate_task(self, max_attempts: Optional[int] = None) -> Dict[str, Any]:
        max_attempts = int(max_attempts or self.config.max_attempts_per_row)
        last_error: Any = None
        for _ in range(max_attempts):
            candidate = self._make_candidate()
            if candidate is None:
                continue
            try:
                solved = self.solve_prompt(candidate["prompt"])
                computed = solved["computed_answer"]
                cot = solved["generated_cot"]
                if self.config.max_cot_chars is not None and len(cot) > self.config.max_cot_chars:
                    last_error = {"reason": "cot_too_long", "chars": len(cot)}
                    continue
                if self.config.require_solver_success and computed != candidate["answer"]:
                    last_error = {"reason": "solver_mismatch", "expected": candidate["answer"], "computed": computed}
                    continue
                row = dict(candidate)
                row.update({
                    "computed_answer": computed,
                    "generated_cot": cot,
                    "metadata_json": json.dumps(candidate["metadata"], ensure_ascii=False, sort_keys=True),
                })
                row["id"] = "crypt_gen_" + _stable_id(row["prompt"], row["answer"], row["generated_cot"])
                return row
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"could not generate accepted cryptarithm task after {max_attempts} attempts; last_error={last_error}")

    def generate_dataset(
        self,
        num_samples: int,
        progress_bar: bool = False,
        nb_workers: Optional[int] = None,
        use_parallel: bool = True,
    ) -> pd.DataFrame:
        """Generate validated full cryptarithm tasks.

        This method intentionally generates only full cryptarithm rows.  Subtask
        extraction is handled later by CryptarithmAugmentGenerator in the dataset
        preparation pipeline, so source can be assigned correctly.
        """
        num_samples = int(num_samples)
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        if num_samples == 0:
            return pd.DataFrame(columns=[
                "id", "prompt", "answer", "computed_answer", "generated_cot",
                "task_mode", "label", "metadata_json",
            ])

        jobs = pd.DataFrame({
            "sample_index": list(range(num_samples)),
            "seed": [self.rng.randint(0, 2**31 - 1) for _ in range(num_samples)],
        })
        config_dict = asdict(self.config)
        profile_dict = asdict(self.profile)

        if use_parallel and nb_workers != 1 and pandarallel is not None:
            pandarallel.initialize(
                nb_workers=nb_workers,
                progress_bar=progress_bar,
                verbose=0,
            )
            rows = jobs.parallel_apply(
                lambda row: _generate_cryptarithm_replay_row(
                    row.to_dict(), profile_dict=profile_dict, config_dict=config_dict
                ),
                axis=1,
            )
        else:
            rows = jobs.apply(
                lambda row: _generate_cryptarithm_replay_row(
                    row.to_dict(), profile_dict=profile_dict, config_dict=config_dict
                ),
                axis=1,
            )

        df = pd.DataFrame.from_records(rows.tolist())
        if "computed_answer" in df.columns:
            df = df[~df["computed_answer"].isna()].copy()
        return df



def _generate_cryptarithm_replay_row(
    row: Dict[str, Any],
    profile_dict: Dict[str, Any],
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Worker-safe row generator for pandas/pandarallel.

    Each worker constructs its own generator and solver instances.  This avoids
    sharing solver state across processes and keeps the generation deterministic
    from the per-row seed.
    """
    sample_index = int(row.get("sample_index", 0))
    seed = int(row["seed"])
    try:
        profile = CryptarithmGenerationProfile(**profile_dict)
        config = CryptarithmTaskGeneratorConfig(**config_dict)
        gen = CryptarithmReplayTaskGenerator(seed=seed, profile=profile, config=config)
        return gen.generate_task()
    except Exception as exc:
        return {
            "id": "crypt_gen_failed_" + _stable_id(seed, sample_index, exc),
            "prompt": None,
            "answer": None,
            "computed_answer": None,
            "generated_cot": json.dumps({"error": str(exc)}, ensure_ascii=False),
            "task_mode": "profiled_replay_2x2",
            "label": "cryptarithm",
            "metadata_json": "{}",
        }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate new validated Alice cryptarithm tasks using replay solver.")
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source_profile_csv", type=str, default=None, help="Optional real solved cryptarithm CSV to fit sampling profile.")
    p.add_argument("--solver_timeout_seconds", type=float, default=10.0)
    p.add_argument("--max_attempts_per_row", type=int, default=100)
    p.add_argument("--max_cot_chars", type=int, default=0)
    p.add_argument("--progress_bar", action="store_true")
    p.add_argument("--nb_workers", type=int, default=None)
    p.add_argument("--no_parallel", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = CryptarithmTaskGeneratorConfig(
        solver_timeout_seconds=args.solver_timeout_seconds,
        max_attempts_per_row=args.max_attempts_per_row,
        max_cot_chars=(args.max_cot_chars or None),
    )
    if args.source_profile_csv:
        src = pd.read_csv(args.source_profile_csv)
        if "label" in src.columns:
            src = src[src["label"].astype(str) == "cryptarithm"].copy()
        gen = CryptarithmReplayTaskGenerator.from_source_data(src, seed=args.seed, config=config)
    else:
        gen = CryptarithmReplayTaskGenerator(seed=args.seed, config=config)
    df = gen.generate_dataset(
        args.num_samples,
        progress_bar=args.progress_bar,
        nb_workers=args.nb_workers,
        use_parallel=not args.no_parallel,
    )
    df.to_csv(args.output_path, index=False)
    print(df["task_mode"].value_counts(dropna=False).to_string())
    print(f"saved {len(df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
