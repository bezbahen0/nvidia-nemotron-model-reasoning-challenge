from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Equation:
    raw: str
    left: str
    op: str
    right: str
    result: str

    def display(self) -> str:
        return f"{self.left}{self.op}{self.right}={self.result}"


class CryptarithmSolver:
    """Minimal concat-only cryptarithm solver.

    It intentionally solves only the clean symbolic concatenation tasks:
      - direct concat:  AB op CD = ABCD
      - reverse concat: AB op CD = CDAB

    It does not attempt arithmetic, CSP, DFS, or hidden search. If the same-op
    examples do not uniquely prove one concat direction, it returns nan.
    """

    def solve(self, examples_text: Any, target_text: Optional[Any] = None) -> Dict[str, Any]:
        full_text = "" if examples_text is None else str(examples_text)
        if target_text is not None:
            full_text = f"{full_text}\n{target_text}"

        equations, known_ops = self._parse_examples(full_text)
        target = self._parse_target(full_text, known_ops)

        if target is None:
            return self._result("nan", [
                "Concat-only solver.",
                "I could not parse a target expression.",
                "Final answer: nan",
            ], strategy="skip_parse_target")

        target_expr, target_left, target_op, target_right = target
        same_op = [eq for eq in equations if eq.op == target_op]

        lines: List[str] = []
        add = lines.append
        add(f"Target: {target_expr}")
        add(f"The target operator is {target_op!r}.")
        add("Lets check if this a concatination")
        add("")

        if not same_op:
            add("No examples use the target operator, so the concat rule cannot be learned.")
            add("Final answer: nan")
            return self._result("nan", lines, strategy="skip_no_same_operator_examples")

        add("Same-operator examples:")
        for eq in same_op:
            add(f"- {eq.left} {eq.op} {eq.right} = {eq.result}")
        add("")

        direct_ok = all(eq.result == eq.left + eq.right for eq in same_op)
        reverse_ok = all(eq.result == eq.right + eq.left for eq in same_op)

        if direct_ok:
            add("Direct concat check:")
            for eq in same_op:
                add(f"- {eq.left} followed by {eq.right} gives {eq.left + eq.right}, matching {eq.result}.")
        else:
            add("Direct concat check fails: at least one output is not left followed by right.")

        add("")
        if reverse_ok:
            add("Reverse concat check:")
            for eq in same_op:
                add(f"- {eq.right} followed by {eq.left} gives {eq.right + eq.left}, matching {eq.result}.")
        else:
            add("Reverse concat check fails: at least one output is not right followed by left.")
        add("")

        if direct_ok and not reverse_ok:
            answer = target_left + target_right
            add("Only direct concat survives.")
            add(f"Apply it to the target: {target_left!r} followed by {target_right!r} gives {answer!r}.")
            add(f"Final answer: {answer}")
            return self._result(answer, lines, strategy="concat_fwd_symbolic")

        if reverse_ok and not direct_ok:
            answer = target_right + target_left
            add("Only reverse concat survives.")
            add(f"Apply it to the target: {target_right!r} followed by {target_left!r} gives {answer!r}.")
            add(f"Final answer: {answer}")
            return self._result(answer, lines, strategy="concat_rev_symbolic")

        if direct_ok and reverse_ok:
            direct_answer = target_left + target_right
            reverse_answer = target_right + target_left
            add("Both direct and reverse concat match the examples.")
            if direct_answer == reverse_answer:
                add("For this target both directions give the same visible output, so the answer is still fixed.")
                add(f"Final answer: {direct_answer}")
                return self._result(direct_answer, lines, strategy="concat_ambiguous_but_same_target")
            add(f"Direct would give {direct_answer!r}, but reverse would give {reverse_answer!r}.")
            add("Because the examples do not choose one direction uniquely, I skip instead of guessing.")
            add("Final answer: nan")
            return self._result("nan", lines, strategy="skip_ambiguous_concat_direction")

        add("Neither direct nor reverse concat explains all same-operator examples.")
        add("This concat-only solver does not attempt arithmetic or CSP search.")
        add("Final answer: nan")
        return self._result("nan", lines, strategy="skip_not_concat")

    def generate_cot(self, prompt: Any) -> str:
        return self.solve(prompt).get("solution_text", "Final answer: nan")

    @staticmethod
    def extract_answer(cot_text: Any) -> str:
        text = "" if cot_text is None else str(cot_text)
        for pattern in [
            r"(?im)^\s*Final answer\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Answer\s*:\s*(\S+)\s*$",
        ]:
            matches = re.findall(pattern, text)
            if matches:
                ans = matches[-1].strip()
                return "nan" if ans.lower() in {"nan", "none"} else ans
        m = re.search(r"\\boxed\{([^{}\s]+)\}", text)
        if m:
            ans = m.group(1).strip()
            return "nan" if ans.lower() in {"nan", "none"} else ans
        return "nan"

    def _result(self, answer: str, lines: Sequence[str], strategy: str) -> Dict[str, Any]:
        solution_text = "\n".join(lines)
        public_answer: Optional[str] = None if answer == "nan" else answer
        return {
            "answer": public_answer,
            "solution_text": solution_text,
            "solution": solution_text,
            "debug": list(lines),
            "trace": list(lines),
            "strategy": strategy,
            "confidence": 1.0 if public_answer is not None else 0.0,
        }

    def _parse_examples(self, text: str) -> Tuple[List[Equation], Set[str]]:
        lines = [line.strip() for line in str(text).splitlines() if line.strip() and "=" in line]
        guessed: Set[str] = set()
        first_pass: List[Equation] = []
        for line in lines:
            eq = self._parse_equation(line, None)
            if eq is not None:
                first_pass.append(eq)
                guessed.add(eq.op)

        equations: List[Equation] = []
        for line in lines:
            eq = self._parse_equation(line, guessed)
            if eq is not None:
                equations.append(eq)
        return equations, guessed

    def _parse_equation(self, line: str, known_ops: Optional[Set[str]] = None) -> Optional[Equation]:
        compact = str(line).strip().replace(" ", "")
        if "=" not in compact:
            return None
        left_side, result = compact.split("=", 1)
        if not left_side or not result:
            return None

        idx: Optional[int] = None
        preferred = self._preferred_operator_index(left_side)
        if known_ops:
            if 0 < preferred < len(left_side) - 1 and left_side[preferred] in known_ops:
                idx = preferred
            else:
                idx = next((i for i, ch in enumerate(left_side) if 0 < i < len(left_side) - 1 and ch in known_ops), None)
        if idx is None and 0 < preferred < len(left_side) - 1:
            idx = preferred
        if idx is None:
            return None

        left = left_side[:idx]
        op = left_side[idx]
        right = left_side[idx + 1:]
        if not left or not right:
            return None
        return Equation(raw=compact, left=left, op=op, right=right, result=result)

    def _parse_target(self, text: str, known_ops: Set[str]) -> Optional[Tuple[str, str, str, str]]:
        raw = str(text)
        patterns = [
            r"(?is).*?now,\s*d[e]?termine\s*the\s*result\s*for:\s*",
            r"(?is).*?determine\s*the\s*result\s*for:\s*",
            r"(?is).*?determine\s*the\s*output\s*for:\s*",
            r"(?is).*?for:\s*",
        ]
        expr = raw
        for pattern in patterns:
            new_expr = re.sub(pattern, "", raw).strip()
            if new_expr != raw.strip():
                expr = new_expr
                break
        expr = expr.splitlines()[-1].strip() if "\n" in expr else expr.strip()
        expr = expr.replace(" ", "").rstrip(".")
        if "=" in expr:
            expr = expr.split("=", 1)[0]
        if not expr:
            return None

        preferred = self._preferred_operator_index(expr)
        idx: Optional[int] = None
        if 0 < preferred < len(expr) - 1 and expr[preferred] in known_ops:
            idx = preferred
        if idx is None:
            idx = next((i for i, ch in enumerate(expr) if 0 < i < len(expr) - 1 and ch in known_ops), None)
        if idx is None and 0 < preferred < len(expr) - 1:
            idx = preferred
        if idx is None:
            return None
        return expr, expr[:idx], expr[idx], expr[idx + 1:]

    @staticmethod
    def _preferred_operator_index(expr: str) -> int:
        # The benchmark mainly uses two-symbol operands: AB?CD.
        return 2 if len(expr) >= 5 else len(expr) // 2


CryptarithmReasoner = CryptarithmSolver
CryptarithmTutor = CryptarithmSolver