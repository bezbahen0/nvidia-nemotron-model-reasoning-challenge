from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Digit = int
Assignment = Dict[str, Digit]
Domains = Dict[str, Set[Digit]]


@dataclass(frozen=True)
class Equation:
    raw: str
    left: str
    right: str
    op: str
    result: str
    has_sign: bool = False

    def display(self) -> str:
        rhs = f"{self.op}{self.result}" if self.has_sign else self.result
        return f"{self.left}{self.op}{self.right}={rhs}"


@dataclass(frozen=True)
class Rule:
    name: str
    orientation: str
    signed: bool
    tier: int
    func: Callable[[int, int], Optional[int]]

    @property
    def reverse(self) -> bool:
        return self.orientation == "rev"


@dataclass
class SolverConfig:
    timeout_seconds: float = 30.0
    max_trace_lines: int = 90
    max_trace_chars: int = 18000
    include_digit_matching_lines: bool = True
    include_rule_reject_summary: bool = True
    include_search_branches: bool = True
    max_kept_rules_shown: int = 16
    max_digit_lines_shown: int = 36


@dataclass
class Stats:
    rule_hypotheses: int = 0
    digit_branches: int = 0
    domain_reductions: int = 0
    support_checks: int = 0


@dataclass
class Trace:
    config: SolverConfig
    lines: List[str] = field(default_factory=list)
    omitted: int = 0
    chars: int = 0

    def add(self, line: str) -> None:
        line = str(line).strip()
        if not line:
            return
        if len(self.lines) < self.config.max_trace_lines and self.chars + len(line) + 1 <= self.config.max_trace_chars:
            self.lines.append(line)
            self.chars += len(line) + 1
        else:
            self.omitted += 1

    def extend(self, other: "Trace") -> None:
        for line in other.lines:
            self.add(line)
        self.omitted += other.omitted

    def finish(self) -> List[str]:
        out = list(self.lines)
        if self.omitted:
            out.append(f"... {self.omitted} repetitive matching/search lines compressed.")
        return out


class CryptarithmSolver:
    """
    Interface:
        solver = CryptarithmSolver()
        cot = solver.generate_cot(prompt)
        answer = solver.extract_answer(cot)
        result = solver.solve(examples_text, target_text)

    The solver path is CSP/backtracking for coverage. The renderer is a compact
    matching-style trace: operation matching, digit matching, selected map,
    verification, target application.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.rules = self._build_rules()
        self.stats = Stats()
        self._deadline = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cot(self, prompt: Any) -> str:
        result = self.solve(str(prompt), None, timeout_seconds=self.config.timeout_seconds)
        return result.get("solution", "")

    @staticmethod
    def extract_answer(cot_text: Any) -> str:
        text = "" if cot_text is None else str(cot_text)
        patterns = [
            r"\\boxed\{([^{}\s]+)\}",
            r"(?im)^\s*Final answer\s*:\s*([^\s]+)\s*$",
            r"(?im)^\s*Computed output\s*:\s*([^\s]+)\s*$",
            r"(?im)^\s*Answer\s*:\s*([^\s]+)\s*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                ans = m.group(1).strip()
                return "nan" if ans.lower() in {"nan", "none", "<no", "<invalid>"} else ans
        return "nan"

    def solve(self, examples_text: Any, target_text: Optional[Any] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        self.stats = Stats()
        self._deadline = time.time() + (timeout_seconds or self.config.timeout_seconds)
        started = time.time()

        try:
            full_text = str(examples_text) if target_text is None else f"{examples_text}\n{target_text}"
            examples_source = str(examples_text) if target_text is not None else full_text
            equations, known_ops = self._parse_examples(examples_source)
            target = self._parse_target(full_text if target_text is None else str(target_text), known_ops)

            if not equations:
                return self._failure("no examples were parsed", started)
            if target is None:
                return self._failure("target expression could not be parsed", started)

            target_expr, target_left, target_op, target_right = target
            symbols = self._symbols(equations, target_left, target_right)
            if len(symbols) > 10:
                return self._failure(f"too many distinct symbols: {len(symbols)} > 10", started)

            by_op: Dict[str, List[Equation]] = {}
            for eq in equations:
                by_op.setdefault(eq.op, []).append(eq)

            candidates, reject_summary = self._operation_candidates(by_op)
            if any(not cands for cands in candidates.values()):
                bad = [op for op, cands in candidates.items() if not cands][0]
                return self._failure(f"no structural rule remains for operator {repr(bad)}", started)
            if target_op not in candidates:
                return self._failure(f"target operator {repr(target_op)} was not seen in examples", started)

            domains: Domains = {s: set(range(10)) for s in symbols}
            ordered_ops = sorted(candidates, key=lambda op: (len(candidates[op]), -len(by_op[op]), op))
            trace = Trace(self.config)
            assignment, combo = self._search_rules(ordered_ops, candidates, equations, domains, {}, trace, 0)

            if assignment is None or combo is None:
                return self._failure("no rule/map combination satisfies all examples", started)

            answer, target_error, target_a, target_b, target_value = self._encode_target(
                target_left, target_op, target_right, combo[target_op], assignment
            )
            solution_lines = self._render_solution(
                equations=equations,
                target_expr=target_expr,
                target_left=target_left,
                target_op=target_op,
                target_right=target_right,
                candidates=candidates,
                reject_summary=reject_summary,
                combo=combo,
                assignment=assignment,
                trace_lines=trace.finish(),
                answer=answer,
                target_error=target_error,
                target_a=target_a,
                target_b=target_b,
                target_value=target_value,
            )
            return {
                "answer": answer,
                "solution": "\n".join(solution_lines),
                "debug": solution_lines,
                "trace": solution_lines,
                "mapping": assignment,
                "rules": {op: self._rule_name(rule) for op, rule in combo.items()},
                "stats": self.stats.__dict__,
                "elapsed_seconds": round(time.time() - started, 4),
                "training_category": "cryptarithm.hybrid",
            }
        except TimeoutError:
            return self._failure("solver timeout", started)
        except Exception as exc:
            return self._failure(f"execution error: {type(exc).__name__}: {exc}", started)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _preferred_operator_index(left_side: str) -> int:
        return 2 if len(left_side) >= 5 else len(left_side) // 2

    def _parse_equation(self, line: str, known_ops: Optional[Set[str]] = None) -> Optional[Equation]:
        compact = str(line).strip().replace(" ", "")
        if "=" not in compact:
            return None
        left_side, right_side = compact.split("=", 1)
        if not left_side or not right_side:
            return None

        idx: Optional[int] = None
        preferred = self._preferred_operator_index(left_side)
        if known_ops:
            if 0 < preferred < len(left_side) - 1 and left_side[preferred] in known_ops:
                idx = preferred
            else:
                idx = next((i for i, ch in enumerate(left_side) if 0 < i < len(left_side) - 1 and ch in known_ops), None)
        if idx is None:
            idx = preferred if 0 < preferred < len(left_side) - 1 else None
        if idx is None:
            return None

        op = left_side[idx]
        has_sign = len(right_side) > 1 and right_side[0] == op
        result = right_side[1:] if has_sign else right_side
        left, right = left_side[:idx], left_side[idx + 1 :]
        if not left or not right or not result:
            return None
        return Equation(raw=compact, left=left, right=right, op=op, result=result, has_sign=has_sign)

    def _parse_examples(self, text: str) -> Tuple[List[Equation], Set[str]]:
        lines = [line.strip() for line in str(text).splitlines() if line.strip() and "=" in line]
        guessed = {eq.op for line in lines if (eq := self._parse_equation(line, None))}
        equations = [eq for line in lines if (eq := self._parse_equation(line, guessed))]
        return equations, guessed

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
        return expr, expr[:idx], expr[idx], expr[idx + 1 :]

    # ------------------------------------------------------------------
    # Rule space and operation matching
    # ------------------------------------------------------------------

    def _build_rules(self) -> List[Rule]:
        def non_negative(v: int) -> Optional[int]:
            return v if v >= 0 else None

        base: List[Tuple[str, bool, Callable[[int, int], Optional[int]]]] = [
            ("add", False, lambda a, b: a + b),
            ("add1", False, lambda a, b: a + b + 1),
            ("addm1", False, lambda a, b: non_negative(a + b - 1)),
            ("mul", False, lambda a, b: a * b),
            ("mul1", False, lambda a, b: a * b + 1),
            ("mulm1", False, lambda a, b: non_negative(a * b - 1)),
            ("abs", False, lambda a, b: abs(a - b)),
            ("sub", True, lambda a, b: a - b),
            ("cat", False, lambda a, b: int(f"{a}{b}")),
        ]
        rules: List[Rule] = []
        for orientation in ("std", "rev"):
            for name, signed, func in base:
                tier = 0 if orientation == "std" else 1
                if name in {"cat", "sub"}:
                    tier += 1
                rules.append(Rule(name=name, orientation=orientation, signed=signed, tier=tier, func=func))
        return rules

    @staticmethod
    def _bounds(length: int) -> Tuple[int, int]:
        return (0, 9) if length <= 1 else (10 ** (length - 1), 10**length - 1)

    @staticmethod
    def _lengths_between(lo: int, hi: int) -> Set[int]:
        if hi < 0:
            return set()
        lo = max(0, lo)
        out = {len(str(hi))}
        n = lo
        while n <= hi:
            out.add(len(str(n)))
            n = 10 ** len(str(n))
        return out

    def _possible_result_lengths(self, rule: Rule, eq: Equation) -> Set[int]:
        ll, rr = len(eq.left), len(eq.right)
        lo_l, hi_l = self._bounds(ll)
        lo_r, hi_r = self._bounds(rr)

        if rule.name == "cat":
            return {ll + rr}
        if rule.name.startswith("add"):
            delta = 1 if rule.name == "add1" else -1 if rule.name == "addm1" else 0
            return self._lengths_between(lo_l + lo_r + delta, hi_l + hi_r + delta)
        if rule.name.startswith("mul"):
            delta = 1 if rule.name == "mul1" else -1 if rule.name == "mulm1" else 0
            return self._lengths_between(lo_l * lo_r + delta, hi_l * hi_r + delta)
        hi = max(abs(a - b) for a in (lo_l, hi_l) for b in (lo_r, hi_r))
        return self._lengths_between(0, hi)

    def _structural_status(self, rule: Rule, equations: Sequence[Equation]) -> Tuple[bool, str]:
        for eq in equations:
            if eq.has_sign and not rule.signed:
                return False, "sign"
            possible = self._possible_result_lengths(rule, eq)
            if len(eq.result) not in possible:
                if rule.name == "cat":
                    return False, "concat length"
                if rule.name.startswith("mul"):
                    return False, "product length"
                return False, "result length"
        return True, "keep"

    def _operation_candidates(self, by_op: Dict[str, List[Equation]]) -> Tuple[Dict[str, List[Rule]], Dict[str, Dict[str, int]]]:
        candidates: Dict[str, List[Rule]] = {}
        rejects: Dict[str, Dict[str, int]] = {}
        for op, eqs in by_op.items():
            kept: List[Rule] = []
            reasons: Dict[str, int] = {}
            for rule in self.rules:
                ok, reason = self._structural_status(rule, eqs)
                if ok:
                    kept.append(rule)
                else:
                    reasons[reason] = reasons.get(reason, 0) + 1
            kept.sort(key=lambda r: (r.tier, r.name, r.orientation))
            candidates[op] = kept
            rejects[op] = reasons
        return candidates, rejects

    # ------------------------------------------------------------------
    # CSP engine
    # ------------------------------------------------------------------

    @staticmethod
    def _symbols(equations: Sequence[Equation], target_left: str, target_right: str) -> List[str]:
        counts: Dict[str, int] = {}
        for eq in equations:
            for ch in eq.left + eq.right + eq.result:
                counts[ch] = counts.get(ch, 0) + 1
        for ch in target_left + target_right:
            counts[ch] = counts.get(ch, 0) + 1
        return sorted(counts, key=lambda ch: (-counts[ch], ch))

    @staticmethod
    def _copy_domains(domains: Domains) -> Domains:
        return {ch: set(vals) for ch, vals in domains.items()}

    @staticmethod
    def _number(text: str, assignment: Assignment, reverse: bool) -> int:
        digits = text[::-1] if reverse else text
        value = 0
        for ch in digits:
            value = value * 10 + assignment[ch]
        return value

    def _equation_matches(self, eq: Equation, rule: Rule, assignment: Assignment) -> bool:
        a = self._number(eq.left, assignment, rule.reverse)
        b = self._number(eq.right, assignment, rule.reverse)
        value = rule.func(a, b)
        if value is None:
            return False
        if rule.signed:
            if (value < 0) != eq.has_sign:
                return False
            value = abs(value)
        elif eq.has_sign or value < 0:
            return False
        digits = [int(c) for c in str(value)]
        if rule.reverse:
            digits.reverse()
        return digits == [assignment[ch] for ch in eq.result]

    def _leading_zero(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace) -> bool:
        changes: List[str] = []
        for eq in equations:
            rule = combo.get(eq.op)
            if rule is None:
                continue
            for term in (eq.left, eq.right, eq.result):
                if len(term) <= 1:
                    continue
                lead = term[-1] if rule.reverse else term[0]
                if 0 in domains[lead]:
                    before = set(domains[lead])
                    domains[lead].remove(0)
                    if not domains[lead]:
                        trace.add(f"leading-zero contradiction: {repr(lead)} cannot be zero in {repr(term)}")
                        return False
                    self.stats.domain_reductions += 1
                    changes.append(f"{repr(lead)} {self._dom(before)} -> {self._dom(domains[lead])}")
        if changes and self.config.include_digit_matching_lines:
            trace.add("No-leading-zero: " + "; ".join(changes))
        return True

    def _alldifferent(self, domains: Domains, trace: Trace) -> bool:
        singles = [next(iter(v)) for v in domains.values() if len(v) == 1]
        if len(singles) != len(set(singles)):
            trace.add("all-different contradiction: two symbols fixed to the same digit")
            return False
        fixed = set(singles)
        changes: List[str] = []
        for ch, vals in domains.items():
            if len(vals) == 1:
                continue
            before = set(vals)
            vals.difference_update(fixed)
            if not vals:
                trace.add(f"all-different contradiction: domain({repr(ch)}) became empty")
                return False
            if vals != before:
                self.stats.domain_reductions += 1
                changes.append(f"{repr(ch)} {self._dom(before)} -> {self._dom(vals)}")
        if changes and self.config.include_digit_matching_lines:
            trace.add("AllDifferent: " + "; ".join(changes))
        return True

    def _supports(
        self,
        scope: Sequence[str],
        domains: Domains,
        predicate: Callable[[Assignment], bool],
    ) -> Tuple[bool, Dict[str, Set[int]], int]:
        self.stats.support_checks += 1
        ordered = sorted(dict.fromkeys(scope), key=lambda ch: (len(domains[ch]), ch))
        supported: Dict[str, Set[int]] = {ch: set() for ch in ordered}
        local: Assignment = {}
        used: Set[int] = set()
        count = 0

        def bt(i: int) -> None:
            nonlocal count
            if time.time() > self._deadline:
                raise TimeoutError("timeout")
            if i == len(ordered):
                if predicate(local):
                    count += 1
                    for ch, d in local.items():
                        supported[ch].add(d)
                return
            ch = ordered[i]
            for digit in sorted(domains[ch]):
                if digit in used:
                    continue
                local[ch] = digit
                used.add(digit)
                bt(i + 1)
                used.remove(digit)
                del local[ch]

        bt(0)
        return count > 0, supported, count

    def _project(
        self,
        domains: Domains,
        scope: Sequence[str],
        predicate: Callable[[Assignment], bool],
        label: str,
        trace: Trace,
    ) -> Optional[bool]:
        ok, supported, count = self._supports(scope, domains, predicate)
        if not ok:
            trace.add(f"reject: {label} has no supported digit assignment")
            return None

        changes: List[str] = []
        changed = False
        for ch, allowed in supported.items():
            before = set(domains[ch])
            domains[ch].intersection_update(allowed)
            if not domains[ch]:
                trace.add(f"reject: {label} empties domain({repr(ch)})")
                return None
            if domains[ch] != before:
                self.stats.domain_reductions += 1
                changed = True
                changes.append(f"{repr(ch)} {self._dom(before)} -> {self._dom(domains[ch])}")
        if changed and self.config.include_digit_matching_lines:
            trace.add(f"{label}: {count} supports; " + "; ".join(changes))
        return changed

    def _suffix(self, term: str, k: int, reverse: bool) -> str:
        return term[:k] if reverse else term[-k:]

    def _modular_project(self, eq: Equation, rule: Rule, domains: Domains, trace: Trace) -> Optional[bool]:
        if rule.name not in {"add", "add1", "addm1", "mul", "mul1", "mulm1"}:
            return False
        any_changed = False
        max_k = max(len(eq.left), len(eq.right), len(eq.result))
        for k in range(1, max_k + 1):
            left = self._suffix(eq.left, k, rule.reverse)
            right = self._suffix(eq.right, k, rule.reverse)
            res = self._suffix(eq.result, k, rule.reverse)
            scope = list(dict.fromkeys(left + right + res))
            mod = 10**k

            def pred(local: Assignment, left=left, right=right, res=res, mod=mod) -> bool:
                value = rule.func(self._number(left, local, rule.reverse), self._number(right, local, rule.reverse))
                return value is not None and value % mod == self._number(res, local, rule.reverse) % mod

            changed = self._project(domains, scope, pred, f"column match {eq.display()} low{k} via {self._rule_name(rule)}", trace)
            if changed is None:
                return None
            any_changed = any_changed or changed
        return any_changed

    def _exact_project(self, eq: Equation, rule: Rule, domains: Domains, trace: Trace) -> Optional[bool]:
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        return self._project(domains, scope, lambda local: self._equation_matches(eq, rule, local), f"full match {eq.display()} via {self._rule_name(rule)}", trace)

    def _propagate(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace) -> bool:
        while True:
            before = {ch: tuple(sorted(vals)) for ch, vals in domains.items()}
            if not self._leading_zero(equations, combo, domains, trace):
                return False
            if not self._alldifferent(domains, trace):
                return False
            for eq in equations:
                rule = combo.get(eq.op)
                if rule is None:
                    continue
                if self._modular_project(eq, rule, domains, trace) is None:
                    return False
                if self._exact_project(eq, rule, domains, trace) is None:
                    return False
            after = {ch: tuple(sorted(vals)) for ch, vals in domains.items()}
            if before == after:
                return True

    def _verify_assignment(self, equations: Sequence[Equation], combo: Dict[str, Rule], assignment: Assignment) -> bool:
        if len(set(assignment.values())) != len(assignment):
            return False
        return all(self._equation_matches(eq, combo[eq.op], assignment) for eq in equations)

    def _solve_digits(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace) -> Optional[Assignment]:
        if time.time() > self._deadline:
            raise TimeoutError("timeout")
        unresolved = [ch for ch, vals in domains.items() if len(vals) > 1]
        if not unresolved:
            assignment = {ch: next(iter(vals)) for ch, vals in domains.items()}
            return assignment if self._verify_assignment(equations, combo, assignment) else None

        symbol = min(unresolved, key=lambda ch: (len(domains[ch]), ch))
        for digit in sorted(domains[symbol]):
            self.stats.digit_branches += 1
            branch_domains = self._copy_domains(domains)
            branch_domains[symbol] = {digit}
            branch_trace = Trace(self.config)
            if self.config.include_search_branches:
                branch_trace.add(f"Try digit match: {repr(symbol)}={digit}")
            if self._propagate(equations, combo, branch_domains, branch_trace):
                found = self._solve_digits(equations, combo, branch_domains, branch_trace)
                if found is not None:
                    trace.extend(branch_trace)
                    return found
        return None

    def _search_rules(
        self,
        ordered_ops: Sequence[str],
        candidates: Dict[str, List[Rule]],
        equations: Sequence[Equation],
        domains: Domains,
        combo: Dict[str, Rule],
        trace: Trace,
        index: int,
    ) -> Tuple[Optional[Assignment], Optional[Dict[str, Rule]]]:
        if time.time() > self._deadline:
            raise TimeoutError("timeout")
        if index == len(ordered_ops):
            domains2 = self._copy_domains(domains)
            trace2 = Trace(self.config)
            if not self._propagate(equations, combo, domains2, trace2):
                return None, None
            assignment = self._solve_digits(equations, combo, domains2, trace2)
            if assignment is not None:
                trace.extend(trace2)
                return assignment, dict(combo)
            return None, None

        op = ordered_ops[index]
        for rule in candidates[op]:
            self.stats.rule_hypotheses += 1
            combo2 = dict(combo)
            combo2[op] = rule
            domains2 = self._copy_domains(domains)
            trace2 = Trace(self.config)
            if self.config.include_search_branches:
                trace2.add(f"Try operation match: {repr(op)} -> {self._rule_name(rule)}")
            if self._propagate(equations, combo2, domains2, trace2):
                assignment, solved_combo = self._search_rules(ordered_ops, candidates, equations, domains2, combo2, trace2, index + 1)
                if assignment is not None and solved_combo is not None:
                    trace.extend(trace2)
                    return assignment, solved_combo
        return None, None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _literal(s: str) -> str:
        return repr(s)

    @staticmethod
    def _dom(values: Iterable[int]) -> str:
        vals = sorted(set(values))
        if vals == list(range(10)):
            return "{0..9}"
        return "{" + ",".join(str(v) for v in vals) + "}"

    @staticmethod
    def _rule_name(rule: Rule) -> str:
        return f"{rule.orientation}/{rule.name}"

    def _rule_meaning(self, rule: Rule) -> str:
        read = "read right-to-left" if rule.reverse else "read left-to-right"
        enc = "encode right-to-left" if rule.reverse else "encode left-to-right"
        action = {
            "add": "A+B",
            "add1": "A+B+1",
            "addm1": "A+B-1",
            "mul": "A*B",
            "mul1": "A*B+1",
            "mulm1": "A*B-1",
            "abs": "abs(A-B)",
            "sub": "A-B, with operator sign if negative",
            "cat": "concat(A,B)",
        }[rule.name]
        return f"{read}; {action}; {enc}"

    def _term_formula(self, term: str, rule: Rule) -> str:
        ordered = term[::-1] if rule.reverse else term
        parts: List[str] = []
        for i, ch in enumerate(ordered):
            power = len(ordered) - i - 1
            coeff = 10**power
            parts.append(repr(ch) if coeff == 1 else f"{coeff}*{repr(ch)}")
        return "+".join(parts) if parts else "0"

    def _constraint_formula(self, eq: Equation, rule: Rule) -> str:
        a = self._term_formula(eq.left, rule)
        b = self._term_formula(eq.right, rule)
        r = self._term_formula(eq.result, rule)
        if rule.name == "add":
            return f"{a}+{b}={r}"
        if rule.name == "add1":
            return f"{a}+{b}+1={r}"
        if rule.name == "addm1":
            return f"{a}+{b}-1={r}"
        if rule.name == "mul":
            return f"({a})*({b})={r}"
        if rule.name == "mul1":
            return f"({a})*({b})+1={r}"
        if rule.name == "mulm1":
            return f"({a})*({b})-1={r}"
        if rule.name == "abs":
            return f"abs(({a})-({b}))={r}"
        if rule.name == "sub":
            rhs = f"-({r})" if eq.has_sign else r
            return f"({a})-({b})={rhs}"
        if rule.name == "cat":
            return f"concat({a},{b})={r}"
        return f"{rule.name}({a},{b})={r}"

    def _operation_line(self, rule: Rule, a: int, b: int) -> Tuple[str, Optional[int]]:
        value = rule.func(a, b)
        if value is None:
            return "invalid", None
        if rule.name == "add":
            return f"{a}+{b}={value}", value
        if rule.name == "add1":
            return f"{a}+{b}+1={value}", value
        if rule.name == "addm1":
            return f"{a}+{b}-1={value}", value
        if rule.name == "mul":
            return f"{a}*{b}={value}", value
        if rule.name == "mul1":
            return f"{a}*{b}+1={value}", value
        if rule.name == "mulm1":
            return f"{a}*{b}-1={value}", value
        if rule.name == "abs":
            return f"abs({a}-{b})={value}", value
        if rule.name == "sub":
            return f"{a}-{b}={value}", value
        if rule.name == "cat":
            return f"concat({a},{b})={value}", value
        return f"{rule.name}({a},{b})={value}", value

    def _encode_value(self, value: int, op: str, rule: Rule, assignment: Assignment) -> Optional[str]:
        sign = ""
        if rule.signed and value < 0:
            sign = op
            value = abs(value)
        elif value < 0:
            return None
        inv = {d: ch for ch, d in assignment.items()}
        chars: List[str] = []
        for digit_char in str(value):
            digit = int(digit_char)
            if digit not in inv:
                return None
            chars.append(inv[digit])
        if rule.reverse:
            chars.reverse()
        return sign + "".join(chars)

    def _encode_target(self, left: str, op: str, right: str, rule: Rule, assignment: Assignment) -> Tuple[Optional[str], Optional[str], int, int, Optional[int]]:
        a = self._number(left, assignment, rule.reverse)
        b = self._number(right, assignment, rule.reverse)
        value = rule.func(a, b)
        if value is None:
            return None, "operation invalid", a, b, None
        ans = self._encode_value(value, op, rule, assignment)
        if ans is None:
            return None, "target value uses a digit not present in the solved symbol map", a, b, value
        return ans, None, a, b, value

    def _render_solution(
        self,
        equations: Sequence[Equation],
        target_expr: str,
        target_left: str,
        target_op: str,
        target_right: str,
        candidates: Dict[str, List[Rule]],
        reject_summary: Dict[str, Dict[str, int]],
        combo: Dict[str, Rule],
        assignment: Assignment,
        trace_lines: Sequence[str],
        answer: Optional[str],
        target_error: Optional[str],
        target_a: int,
        target_b: int,
        target_value: Optional[int],
    ) -> List[str]:
        lines: List[str] = []
        add = lines.append
        by_op: Dict[str, List[Equation]] = {}
        for eq in equations:
            by_op.setdefault(eq.op, []).append(eq)

        add("We need to solve the cryptarithm by matching operation rules and digit columns.")
        add("Each visible non-operator symbol is one decimal digit; different symbols use different digits.")
        add("")
        add("Examples")
        for i, eq in enumerate(equations, 1):
            add(f"{i}. {eq.display()}")
        add(f"Target: {target_expr}")
        add("")

        add("Operation matching")
        for op in sorted(candidates):
            selected = combo.get(op)
            kept = candidates[op]
            add(f"Operator {repr(op)}")
            add("examples: " + "; ".join(eq.display() for eq in by_op[op]))
            kept_text = ", ".join(self._rule_name(r) for r in kept[: self.config.max_kept_rules_shown])
            if len(kept) > self.config.max_kept_rules_shown:
                kept_text += f", ... ({len(kept)} total)"
            add(f"structural keep: {kept_text}")
            if self.config.include_rule_reject_summary:
                rej = reject_summary.get(op, {})
                if rej:
                    add("structural reject: " + ", ".join(f"{reason}={count}" for reason, count in sorted(rej.items())))
            if selected:
                add(f"Best: {self._rule_name(selected)}")
                add(f"meaning: {self._rule_meaning(selected)}")
            add("")

        add("Column and equation matching")
        for eq in equations:
            rule = combo[eq.op]
            add(f"{eq.display()} -> {self._rule_name(rule)}: {self._constraint_formula(eq, rule)}")
        add("")

        add("Digit matching")
        fixed = {ch: d for ch, d in sorted(assignment.items())}
        add("Selected map: " + "; ".join(f"{repr(ch)}={d}" for ch, d in fixed.items()))
        if self.config.include_digit_matching_lines and trace_lines:
            add("Useful narrowing steps")
            for line in list(trace_lines)[: self.config.max_digit_lines_shown]:
                add(line)
            if len(trace_lines) > self.config.max_digit_lines_shown:
                add(f"... {len(trace_lines) - self.config.max_digit_lines_shown} more narrowing lines compressed.")
        add("")

        add("Verify examples")
        for eq in equations:
            rule = combo[eq.op]
            a = self._number(eq.left, assignment, rule.reverse)
            b = self._number(eq.right, assignment, rule.reverse)
            op_text, value = self._operation_line(rule, a, b)
            encoded = None if value is None else self._encode_value(value, eq.op, rule, assignment)
            expected = f"{eq.op}{eq.result}" if eq.has_sign else eq.result
            status = "ok" if encoded == expected else "fail"
            add(f"{eq.display()}: decode {eq.left}->{a}, {eq.right}->{b}; {op_text}; encode->{encoded}; expected={expected}; {status}")
        add("")

        add("Target")
        target_rule = combo[target_op]
        add(f"Use operator {repr(target_op)}: {self._rule_name(target_rule)}")
        add(f"Decode: {target_left}->{target_a}, {target_right}->{target_b}")
        if target_error or answer is None or target_value is None:
            add(f"Apply: {target_error or 'invalid target'}")
            add("Computed output: nan")
            add("Final answer: nan")
        else:
            operation, _ = self._operation_line(target_rule, target_a, target_b)
            add(f"Apply: {operation}")
            add(f"Encode {target_value}: {answer}")
            add(f"Computed output: {answer}")
            add(f"Final answer: {answer}")
            add(f"\\boxed{{{answer}}}")
        return lines

    def _failure(self, reason: str, started: float) -> Dict[str, Any]:
        lines = [
            "We need to solve the cryptarithm by matching operation rules and digit columns.",
            f"Solver failed: {reason}",
            "Final answer: nan",
        ]
        return {
            "answer": None,
            "solution": "\n".join(lines),
            "debug": lines,
            "trace": lines,
            "stats": self.stats.__dict__,
            "elapsed_seconds": round(time.time() - started, 4),
        }
