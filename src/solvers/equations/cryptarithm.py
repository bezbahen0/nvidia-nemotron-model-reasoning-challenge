from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Equation:
    raw: str
    left: str
    op: str
    right: str
    result: str
    has_sign: bool = False

    @property
    def expr(self) -> str:
        return f"{self.left}{self.op}{self.right}"

    def display(self) -> str:
        rhs = self.op + self.result if self.has_sign else self.result
        return f"{self.left}{self.op}{self.right}={rhs}"


@dataclass(frozen=True)
class Target:
    expr: str
    left: str
    op: str
    right: str


@dataclass
class ParsedState:
    prompt: str
    equations: List[Equation] = field(default_factory=list)
    known_ops: Set[str] = field(default_factory=set)
    target: Optional[Target] = None
    same_op_examples: List[Equation] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    has_final_answer: bool = False
    answer: str = "nan"
    strategy: str = "skip"
    confidence: float = 0.0


@dataclass
class PhaseEvent:
    status: str  # progress, continue, answer, skip
    reason: str = ""


class Phase:
    name = "phase"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        raise NotImplementedError


class IdentifyTargetOperator(Phase):
    name = "identify_target_operator"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        equations, known_ops = tutor.parse_examples(state.prompt)
        target = tutor.parse_target(state.prompt, known_ops)
        if not equations:
            return PhaseEvent("skip", "I could not read any example equations.")
        if target is None:
            return PhaseEvent("skip", "I could not identify the target expression.")
        state.equations = equations
        state.known_ops = known_ops
        state.target = target
        # Keep this as a mathematical setup, not a parser log.
        state.trace.append(
            f"The target uses the visible operator {target.op!r}: {target.left}{target.op}{target.right}."
        )
        return PhaseEvent("progress")


class CollectSameOperatorExamples(Phase):
    name = "collect_same_operator_examples"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        assert state.target is not None
        same = [e for e in state.equations if e.op == state.target.op]
        if not same:
            return PhaseEvent("skip", "The target operator never appears in the examples.")
        state.same_op_examples = same
        lines = [
            f"Only examples with the same operator {state.target.op!r} can teach that operator's rule."
        ]
        for e in same:
            lines.append(f"- {e.left} {e.op} {e.right} -> {e.result if not e.has_sign else e.op + e.result}")
        state.trace.extend(lines)
        return PhaseEvent("progress")


class KnownResultAnchorStrategy(Phase):
    name = "known_result_anchor"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        assert state.target is not None
        for e in state.same_op_examples:
            if e.expr == state.target.expr and not e.has_sign:
                state.trace.append(
                    f"The target expression already appears as an example: {e.expr} -> {e.result}."
                )
                state.answer = e.result
                state.strategy = self.name
                state.confidence = 1.0
                state.has_final_answer = True
                return PhaseEvent("answer")
        return PhaseEvent("continue")


class TryConcatDirectly(Phase):
    name = "try_concat_directly"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        assert state.target is not None
        same = state.same_op_examples
        if not same:
            return PhaseEvent("continue")

        direct_rows = [e for e in same if (not e.has_sign) and e.result == e.left + e.right]
        reverse_rows = [e for e in same if (not e.has_sign) and e.result == e.right + e.left]
        direct = len(direct_rows) == len(same)
        reverse = len(reverse_rows) == len(same)

        # If both are true, the examples are ambiguous, usually because left==right.
        # A real learner cannot decide which concat direction to apply to a new target.
        if direct and reverse:
            state.trace.append(
                "The visible rows are compatible with both left+right and right+left concat, so the concat direction is ambiguous."
            )
            return PhaseEvent("continue")

        if direct:
            ans = state.target.left + state.target.right
            lines = [
                f"The examples show a direct copy pattern for {state.target.op!r}.",
            ]
            for e in same:
                lines.append(
                    f"- {e.left} {e.op} {e.right} gives {e.result}, exactly the left part {e.left!r} followed by the right part {e.right!r}."
                )
            lines.append(
                f"Apply the same visible rule to the target: {state.target.left!r} followed by {state.target.right!r} gives {ans!r}."
            )
            state.trace.extend(lines)
            state.answer = ans
            state.strategy = "concat_fwd_symbolic"
            state.confidence = 1.0
            state.has_final_answer = True
            return PhaseEvent("answer")

        if reverse:
            ans = state.target.right + state.target.left
            lines = [
                f"The examples show a reverse copy pattern for {state.target.op!r}.",
            ]
            for e in same:
                lines.append(
                    f"- {e.left} {e.op} {e.right} gives {e.result}, exactly the right part {e.right!r} followed by the left part {e.left!r}."
                )
            lines.append(
                f"Apply the same visible rule to the target: {state.target.right!r} followed by {state.target.left!r} gives {ans!r}."
            )
            state.trace.extend(lines)
            state.answer = ans
            state.strategy = "concat_rev_symbolic"
            state.confidence = 1.0
            state.has_final_answer = True
            return PhaseEvent("answer")

        return PhaseEvent("continue")


class TryDifferenceIdentities(Phase):
    name = "try_difference_identities"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        # Deliberately conservative. We do not infer a difference rule from length alone.
        # That would produce fake textbook reasoning on this benchmark.
        return PhaseEvent("continue")


class TryAdditionCarryTrace(Phase):
    name = "try_addition_carry_trace"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        # Placeholder for a future real chapter. It should emit only column facts that
        # are actually derived, never a candidate dump or hidden search replay.
        return PhaseEvent("continue")


class TryMultiplicationUnitsTrace(Phase):
    name = "try_multiplication_units_trace"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        # Placeholder for a future real chapter. Multiplication traces need a cleaner
        # pedagogy than listing dozens of local digit rows.
        return PhaseEvent("continue")


class TrySmallLocalSymbolTable(Phase):
    name = "try_small_local_symbol_table"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        # Intentionally disabled in the clean tutor. Tables with 20-80 rows are solver
        # traces, not useful textbook CoT. Add narrow hand-written table chapters later.
        return PhaseEvent("continue")


class ApplyTarget(Phase):
    name = "apply_target"

    def step(self, state: ParsedState, tutor: "CryptarithmTutor") -> PhaseEvent:
        # Strategies that can answer already set the answer. Nothing generic to apply.
        return PhaseEvent("skip", "No clean linear reasoning strategy applied.")


class CryptarithmTutor:
    """Clean phase-based tutor for a high-precision subset of symbol cryptarithms.

    The trace is intended as training text, not as a solver log:
    - no DFS;
    - no hidden enumeration;
    - no candidate-family dumps;
    - if the visible examples do not yield a clean rule, return nan.
    """

    def __init__(self):
        self.phases: List[Phase] = [
            IdentifyTargetOperator(),
            CollectSameOperatorExamples(),
            KnownResultAnchorStrategy(),
            TryConcatDirectly(),
            TryDifferenceIdentities(),
            TryAdditionCarryTrace(),
            TryMultiplicationUnitsTrace(),
            TrySmallLocalSymbolTable(),
            ApplyTarget(),
        ]

    def solve(self, examples_text: Any, target_text: Optional[Any] = None) -> Dict[str, Any]:
        text = str(examples_text) if target_text is None else f"{examples_text}\nNow, determine the result for: {target_text}"
        state = ParsedState(prompt=text)
        phases_run: List[str] = []
        skip_reason = "No clean linear reasoning strategy applied."

        for phase in self.phases:
            phases_run.append(phase.name)
            event = phase.step(state, self)
            if state.has_final_answer:
                break
            if event.status == "skip":
                skip_reason = event.reason
                break

        if state.has_final_answer:
            solution_text = "\n".join(state.trace) + f"\nFinal answer: {state.answer}"
            return {
                "answer": state.answer,
                "solution_text": solution_text,
                "solution": solution_text,
                "debug": solution_text,
                "trace": list(state.trace),
                "strategy": state.strategy,
                "confidence": state.confidence,
                "phases": phases_run,
            }

        state.trace.append(f"I do not see a clean short deduction here, so I stop instead of inventing a search trace.")
        solution_text = "\n".join(state.trace) + "\nFinal answer: nan"
        return {
            "answer": "nan",
            "solution_text": solution_text,
            "solution": solution_text,
            "debug": solution_text,
            "trace": list(state.trace),
            "strategy": "skip",
            "confidence": 0.0,
            "phases": phases_run,
            "skip_reason": skip_reason,
        }

    # ---------- parsing ----------
    @staticmethod
    def _clean(x: Any) -> str:
        return "" if x is None else str(x).strip().replace(" ", "")

    @staticmethod
    def preferred_operator_index(expr: str) -> int:
        return 2 if len(expr) >= 5 else len(expr) // 2

    def parse_examples(self, text: str) -> Tuple[List[Equation], Set[str]]:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip() and "=" in ln]
        guessed: Set[str] = set()
        first: List[Equation] = []
        for line in lines:
            eq = self.parse_equation(line, None)
            if eq:
                first.append(eq)
                guessed.add(eq.op)
        equations: List[Equation] = []
        for line in lines:
            eq = self.parse_equation(line, guessed)
            if eq:
                equations.append(eq)
        return equations, guessed

    def parse_equation(self, line: str, known_ops: Optional[Set[str]]) -> Optional[Equation]:
        compact = self._clean(line)
        if "=" not in compact:
            return None
        lhs, rhs = compact.split("=", 1)
        if not lhs or not rhs:
            return None
        idx: Optional[int] = None
        preferred = self.preferred_operator_index(lhs)
        if known_ops:
            if 0 < preferred < len(lhs) - 1 and lhs[preferred] in known_ops:
                idx = preferred
            else:
                idx = next((i for i, ch in enumerate(lhs) if 0 < i < len(lhs) - 1 and ch in known_ops), None)
        if idx is None and 0 < preferred < len(lhs) - 1:
            idx = preferred
        if idx is None:
            return None
        op = lhs[idx]
        left, right = lhs[:idx], lhs[idx + 1:]
        has_sign = len(rhs) > 1 and rhs[0] == op
        result = rhs[1:] if has_sign else rhs
        if not left or not right or not result:
            return None
        return Equation(raw=compact, left=left, op=op, right=right, result=result, has_sign=has_sign)

    def parse_target(self, text: str, known_ops: Set[str]) -> Optional[Target]:
        raw = str(text)
        m = re.search(r"(?is)Now,\s*determine\s+the\s+result\s+for:\s*(\S+)", raw)
        expr = m.group(1).strip() if m else raw.strip().splitlines()[-1].strip()
        expr = expr.replace(" ", "").rstrip(".")
        if "=" in expr:
            expr = expr.split("=", 1)[0]
        if not expr:
            return None
        idx: Optional[int] = None
        preferred = self.preferred_operator_index(expr)
        if 0 < preferred < len(expr) - 1 and expr[preferred] in known_ops:
            idx = preferred
        if idx is None:
            idx = next((i for i, ch in enumerate(expr) if 0 < i < len(expr) - 1 and ch in known_ops), None)
        if idx is None and 0 < preferred < len(expr) - 1:
            idx = preferred
        if idx is None:
            return None
        return Target(expr=expr, left=expr[:idx], op=expr[idx], right=expr[idx + 1:])

    @staticmethod
    def extract_answer(cot_text: Any) -> str:
        text = "" if cot_text is None else str(cot_text)
        for pattern in [
            r"(?im)^\s*Final answer\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Computed output\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Answer\s*:\s*(\S+)\s*$",
        ]:
            matches = re.findall(pattern, text)
            if matches:
                ans = matches[-1].strip()
                return "nan" if ans.lower() in {"nan", "none", "stop_no_narrow_step"} else ans
        m = re.search(r"\\boxed\{([^{}\s]+)\}", text)
        if m:
            ans = m.group(1).strip()
            return "nan" if ans.lower() in {"nan", "none"} else ans
        return "nan"

    def generate_cot(self, prompt: Any) -> str:
        return self.solve(prompt)["solution_text"]


# Compatibility aliases.
CryptarithmReasoner = CryptarithmTutor
CryptarithmSolver = CryptarithmTutor
