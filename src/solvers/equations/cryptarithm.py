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


# ============================================================================
# Replay-CoT version
# ============================================================================
# The original class above is kept as a compatibility baseline.  The class below
# preserves its CSP/rule-search coverage, but replaces the compact post-hoc trace
# with deterministic replay events and a numeral-equations-style renderer.

from dataclasses import asdict

BaseCryptarithmSolver = CryptarithmSolver


@dataclass
class ReplayConfig(SolverConfig):
    """Configuration for replay-complete cryptarithm CoT generation.

    The defaults intentionally keep all domain-changing operations and all branch
    decisions, while skipping no-change projections.  No-change projections can be
    enabled for maximal audit/debug mode, but they are usually not helpful for
    training and make the CoT much longer.
    """

    # Disable legacy truncation semantics.  They are kept only for compatibility
    # with code that still reads these fields.
    max_trace_lines: int = 10_000_000
    max_trace_chars: int = 1_000_000_000
    max_digit_lines_shown: int = 10_000_000

    # Replay rendering controls.
    include_no_change_events: bool = False
    include_failed_attempt_events: bool = False
    max_projection_blocks: int = 10_000_000
    max_structural_tests_per_operator: int = 10_000_000

    # Optimization controls.
    use_support_cache: bool = True
    # v2: reduce real projection work by using exact/full projections directly
    # when their local scope is already small; otherwise run only a small
    # dominance-aware modular cascade instead of low1, low2, ..., lowK.
    projection_strategy: str = "adaptive"  # "legacy" or "adaptive"
    exact_dominates_modular_space: int = 1_000
    modular_anchor_ks: Tuple[int, ...] = (1, 2, -1)  # -1 means max_k

    # v2 renderer controls.  Supported projection + update is enough to replay
    # the state transition from the previous blocks, so current domains are off
    # by default to save tokens.
    compact_structural_matching: bool = True
    compact_projection_blocks: bool = True
    show_current_domains: bool = False
    show_projection_reason: bool = False


@dataclass
class ReplayEvent:
    kind: str
    title: str
    label: str = ""
    constraint: str = ""
    before: Optional[Dict[str, Tuple[int, ...]]] = None
    supported: Optional[Dict[str, Tuple[int, ...]]] = None
    after: Optional[Dict[str, Tuple[int, ...]]] = None
    changes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None
    decision: str = ""
    reason: str = ""
    count: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ReplayTrace:
    config: ReplayConfig
    events: List[ReplayEvent] = field(default_factory=list)
    legacy_lines: List[str] = field(default_factory=list)
    last_reject: str = ""

    def add(self, line: str) -> None:
        # Compatibility for inherited code paths.  These lines are not truncated.
        line = str(line).strip()
        if line:
            self.legacy_lines.append(line)

    def add_event(self, event: ReplayEvent) -> None:
        self.events.append(event)
        if event.decision == "reject" or "contradiction" in event.kind.lower() or "reject" in event.kind.lower():
            self.last_reject = event.reason or event.title or event.label

    def extend(self, other: "ReplayTrace") -> None:
        self.events.extend(other.events)
        self.legacy_lines.extend(other.legacy_lines)
        if other.last_reject:
            self.last_reject = other.last_reject

    def finish(self) -> List[ReplayEvent]:
        return list(self.events)

    def reject_reason(self) -> str:
        return self.last_reject or "no supported continuation"


class CryptarithmReplaySolver(BaseCryptarithmSolver):
    """Coverage-preserving cryptarithm solver with replay-complete CoT (v2.1 causal replay renderer).

    Design goals:
      * keep the same rule/CSP search space as cryptarithm(4).py;
      * record every state-changing local projection on the successful path;
      * record branch/rule accept-reject decisions in deterministic order;
      * avoid raw combinatorial tuple dumps by logging supported projections;
      * keep answer extraction API compatible with the old solver.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        if config is None:
            config = ReplayConfig()
        super().__init__(config)
        self._support_cache: Dict[Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]], Tuple[bool, Dict[str, Tuple[int, ...]], int]] = {}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    @staticmethod
    def _snapshot_scope(domains: Domains, scope: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
        return {ch: tuple(sorted(domains[ch])) for ch in dict.fromkeys(scope)}

    @staticmethod
    def _changes(before: Dict[str, Tuple[int, ...]], after: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        return {ch: (before[ch], after[ch]) for ch in before if before[ch] != after.get(ch, ())}

    def _support_key(self, label: str, domains: Domains, scope: Sequence[str]) -> Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]]:
        ordered = tuple((ch, tuple(sorted(domains[ch]))) for ch in dict.fromkeys(scope))
        return (label, ordered)

    @staticmethod
    def _domain_space(domains: Domains, scope: Sequence[str], cap: int = 10_000_000) -> int:
        """Approximate local search size for a support projection.

        This intentionally ignores AllDifferent permutations and uses the product
        of domain sizes.  It is a safe cheap upper-ish signal for deciding whether
        a full projection can dominate the modular low-k cascade.
        """
        total = 1
        for ch in dict.fromkeys(scope):
            total *= max(1, len(domains[ch]))
            if total > cap:
                return total
        return total

    def _rule_family_order(self) -> List[Tuple[str, List[str]]]:
        return [
            ("concat", ["cat"]),
            ("add_offset", ["add", "add1", "addm1"]),
            ("abs_sub", ["abs", "sub"]),
            ("mul_offset", ["mul", "mul1", "mulm1"]),
        ]

    def _rule_sort_key(self, rule: Rule) -> Tuple[int, int, str]:
        family_rank = {name: i for i, (_, names) in enumerate(self._rule_family_order()) for name in names}
        return (family_rank.get(rule.name, 99), 0 if rule.orientation == "std" else 1, rule.name)


    @staticmethod
    def extract_answer(cot_text: Any) -> str:
        text = "" if cot_text is None else str(cot_text)
        # Prefer explicit line-based answers.  Cryptarithm answers may themselves
        # contain braces such as "}", so a naive \boxed{...} regex can truncate.
        for pattern in [
            r"(?im)^\s*Final answer\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Computed output\s*:\s*(\S+)\s*$",
            r"(?im)^\s*Answer\s*:\s*(\S+)\s*$",
        ]:
            m = re.search(pattern, text)
            if m:
                ans = m.group(1).strip()
                return "nan" if ans.lower() in {"nan", "none", "<no", "<invalid>"} else ans
        # Last-resort boxed parsing for non-brace answers only.
        m = re.search(r"\\boxed\{([^{}\s]+)\}", text)
        if m:
            ans = m.group(1).strip()
            return "nan" if ans.lower() in {"nan", "none"} else ans
        return "nan"

    # ------------------------------------------------------------------
    # Public API override
    # ------------------------------------------------------------------

    def solve(self, examples_text: Any, target_text: Optional[Any] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        self.stats = Stats()
        # dynamic stats; Stats is intentionally not slotted.
        self.stats.support_cache_hits = 0
        self.stats.replay_events = 0
        self._support_cache = {}
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
            # Use the same rule ordering in solving and in CoT rendering.
            for op in list(candidates):
                candidates[op] = sorted(candidates[op], key=self._rule_sort_key)

            if any(not cands for cands in candidates.values()):
                bad = [op for op, cands in candidates.items() if not cands][0]
                return self._failure(f"no structural rule remains for operator {repr(bad)}", started)
            if target_op not in candidates:
                return self._failure(f"target operator {repr(target_op)} was not seen in examples", started)

            domains: Domains = {s: set(range(10)) for s in symbols}
            ordered_ops = sorted(candidates, key=lambda op: (0 if op == target_op else 1, len(candidates[op]), -len(by_op[op]), op))
            trace = ReplayTrace(self.config if isinstance(self.config, ReplayConfig) else ReplayConfig())
            trace.add_event(
                ReplayEvent(
                    kind="RULE_ORDER",
                    title="Rule-combo search order",
                    decision="order",
                    meta={"ordered_ops": tuple(ordered_ops), "target_op": target_op},
                )
            )
            assignment, combo = self._search_rules(ordered_ops, candidates, equations, domains, {}, trace, 0)

            if assignment is None or combo is None:
                return self._failure("no rule/map combination satisfies all examples", started)

            answer, target_error, target_a, target_b, target_value = self._encode_target(
                target_left, target_op, target_right, combo[target_op], assignment
            )
            events = trace.finish()
            self.stats.replay_events = len(events)
            solution_lines = self._render_replay_solution(
                equations=equations,
                target_expr=target_expr,
                target_left=target_left,
                target_op=target_op,
                target_right=target_right,
                candidates=candidates,
                reject_summary=reject_summary,
                combo=combo,
                assignment=assignment,
                events=events,
                answer=answer,
                target_error=target_error,
                target_a=target_a,
                target_b=target_b,
                target_value=target_value,
                ordered_ops=ordered_ops,
                symbols=symbols,
            )
            return {
                "answer": answer,
                "solution": "\n".join(solution_lines),
                "debug": solution_lines,
                "trace": solution_lines,
                "events": [asdict(e) for e in events],
                "mapping": assignment,
                "rules": {op: self._rule_name(rule) for op, rule in combo.items()},
                "stats": self.stats.__dict__,
                "elapsed_seconds": round(time.time() - started, 4),
                "training_category": "cryptarithm.replay_hybrid",
            }
        except TimeoutError:
            return self._failure("solver timeout", started)
        except Exception as exc:
            return self._failure(f"execution error: {type(exc).__name__}: {exc}", started)

    # ------------------------------------------------------------------
    # CSP engine with replay events
    # ------------------------------------------------------------------

    def _leading_zero(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: ReplayTrace) -> bool:
        scope: List[str] = []
        terms: List[str] = []
        for eq in equations:
            rule = combo.get(eq.op)
            if rule is None:
                continue
            for term in (eq.left, eq.right, eq.result):
                if len(term) <= 1:
                    continue
                lead = term[-1] if rule.reverse else term[0]
                scope.append(lead)
                terms.append(f"{repr(lead)} is leading digit of {repr(term)}")
        scope = list(dict.fromkeys(scope))
        if not scope:
            return True
        before = self._snapshot_scope(domains, scope)
        reason = "; ".join(terms)
        for lead in scope:
            if 0 in domains[lead]:
                domains[lead].remove(0)
                if not domains[lead]:
                    after = self._snapshot_scope(domains, scope)
                    trace.add_event(ReplayEvent(
                        kind="NO_LEADING_ZERO",
                        title="No-leading-zero block",
                        constraint="multi-digit numbers cannot start with 0",
                        before=before,
                        after=after,
                        changes=self._changes(before, after),
                        decision="reject",
                        reason=f"domain({repr(lead)}) became empty after removing 0",
                    ))
                    return False
        after = self._snapshot_scope(domains, scope)
        changes = self._changes(before, after)
        if changes:
            self.stats.domain_reductions += len(changes)
            trace.add_event(ReplayEvent(
                kind="NO_LEADING_ZERO",
                title="No-leading-zero block",
                constraint="multi-digit numbers cannot start with 0",
                before=before,
                after=after,
                changes=changes,
                decision="keep",
                reason=reason,
            ))
        elif self._cfg("include_no_change_events", False):
            trace.add_event(ReplayEvent(
                kind="NO_LEADING_ZERO",
                title="No-leading-zero block",
                constraint="multi-digit numbers cannot start with 0",
                before=before,
                after=after,
                changes={},
                decision="no-change",
                reason=reason,
            ))
        return True

    def _alldifferent(self, domains: Domains, trace: ReplayTrace) -> bool:
        scope = sorted(domains)
        before = self._snapshot_scope(domains, scope)
        singles = [next(iter(v)) for v in domains.values() if len(v) == 1]
        if len(singles) != len(set(singles)):
            trace.add_event(ReplayEvent(
                kind="ALLDIFFERENT",
                title="AllDifferent block",
                constraint="fixed digits must be unique",
                before=before,
                after=before,
                changes={},
                decision="reject",
                reason="two symbols are fixed to the same digit",
            ))
            return False
        fixed = set(singles)
        for ch, vals in domains.items():
            if len(vals) == 1:
                continue
            vals.difference_update(fixed)
            if not vals:
                after = self._snapshot_scope(domains, scope)
                trace.add_event(ReplayEvent(
                    kind="ALLDIFFERENT",
                    title="AllDifferent block",
                    constraint="remove fixed digits from every other symbol domain",
                    before=before,
                    after=after,
                    changes=self._changes(before, after),
                    decision="reject",
                    reason=f"domain({repr(ch)}) became empty",
                ))
                return False
        after = self._snapshot_scope(domains, scope)
        changes = self._changes(before, after)
        if changes:
            self.stats.domain_reductions += len(changes)
            trace.add_event(ReplayEvent(
                kind="ALLDIFFERENT",
                title="AllDifferent block",
                constraint="remove fixed digits from every other symbol domain",
                before=before,
                after=after,
                changes=changes,
                decision="keep",
                reason="fixed digits cannot be reused",
            ))
        elif self._cfg("include_no_change_events", False):
            trace.add_event(ReplayEvent(
                kind="ALLDIFFERENT",
                title="AllDifferent block",
                constraint="remove fixed digits from every other symbol domain",
                before=before,
                after=after,
                changes={},
                decision="no-change",
                reason="no fixed digit can prune another domain",
            ))
        return True

    def _project_replay(
        self,
        domains: Domains,
        scope: Sequence[str],
        predicate: Callable[[Assignment], bool],
        label: str,
        trace: ReplayTrace,
        constraint: str,
        kind: str,
    ) -> Optional[bool]:
        scope = list(dict.fromkeys(scope))
        before = self._snapshot_scope(domains, scope)
        key = self._support_key(label, domains, scope)
        if self._cfg("use_support_cache", True) and key in self._support_cache:
            ok, supported_tuple, count = self._support_cache[key]
            self.stats.support_cache_hits += 1
            supported = {ch: set(vals) for ch, vals in supported_tuple.items()}
        else:
            ok, supported, count = self._supports(scope, domains, predicate)
            if self._cfg("use_support_cache", True):
                self._support_cache[key] = (ok, {ch: tuple(sorted(vals)) for ch, vals in supported.items()}, count)

        if not ok:
            trace.add_event(ReplayEvent(
                kind=kind,
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={ch: tuple() for ch in scope},
                after=before,
                changes={},
                decision="reject",
                reason="no supported local digit assignment",
                count=0,
            ))
            return None

        for ch, allowed in supported.items():
            domains[ch].intersection_update(allowed)
            if not domains[ch]:
                after = self._snapshot_scope(domains, scope)
                trace.add_event(ReplayEvent(
                    kind=kind,
                    title="Projection block",
                    label=label,
                    constraint=constraint,
                    before=before,
                    supported={c: tuple(sorted(v)) for c, v in supported.items()},
                    after=after,
                    changes=self._changes(before, after),
                    decision="reject",
                    reason=f"domain({repr(ch)}) became empty",
                    count=count,
                ))
                return None

        after = self._snapshot_scope(domains, scope)
        changes = self._changes(before, after)
        if changes:
            self.stats.domain_reductions += len(changes)
            trace.add_event(ReplayEvent(
                kind=kind,
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={c: tuple(sorted(v)) for c, v in supported.items()},
                after=after,
                changes=changes,
                decision="keep",
                reason="supported projection narrows at least one domain",
                count=count,
            ))
            return True
        if self._cfg("include_no_change_events", False):
            trace.add_event(ReplayEvent(
                kind=kind,
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={c: tuple(sorted(v)) for c, v in supported.items()},
                after=after,
                changes={},
                decision="no-change",
                reason="supported projection equals current domains",
                count=count,
            ))
        return False

    def _project(
        self,
        domains: Domains,
        scope: Sequence[str],
        predicate: Callable[[Assignment], bool],
        label: str,
        trace: ReplayTrace,
    ) -> Optional[bool]:
        return self._project_replay(domains, scope, predicate, label, trace, constraint=label, kind="PROJECTION")

    def _modular_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        if rule.name not in {"add", "add1", "addm1", "mul", "mul1", "mulm1"}:
            return False

        max_k = max(len(eq.left), len(eq.right), len(eq.result))
        full_scope = list(dict.fromkeys(eq.left + eq.right + eq.result))

        # v2 real optimization: if the exact/full projection scope is already
        # small, do not run the dominated low1..lowK modular cascade.  The caller
        # will run _exact_project immediately after this method, which is a
        # stronger constraint than every modular suffix check.
        if self._cfg("projection_strategy", "adaptive") == "adaptive":
            if self._domain_space(domains, full_scope) <= int(self._cfg("exact_dominates_modular_space", 120_000)):
                trace.add_event(ReplayEvent(
                    kind="MODULAR_SKIP",
                    title="Modular cascade skipped",
                    decision="skip",
                    reason="exact full projection has small enough scope and dominates all low-k modular projections",
                    meta={"equation": eq.display(), "rule": self._rule_name(rule), "space": self._domain_space(domains, full_scope)},
                )) if self._cfg("include_no_change_events", False) else None
                return False

            ks: List[int] = []
            for raw_k in self._cfg("modular_anchor_ks", (1, -1)):
                k = max_k if int(raw_k) == -1 else int(raw_k)
                if 1 <= k <= max_k and k not in ks:
                    ks.append(k)
        else:
            ks = list(range(1, max_k + 1))

        any_changed = False
        for k in ks:
            left = self._suffix(eq.left, k, rule.reverse)
            right = self._suffix(eq.right, k, rule.reverse)
            res = self._suffix(eq.result, k, rule.reverse)
            scope = list(dict.fromkeys(left + right + res))
            mod = 10**k

            def pred(local: Assignment, left=left, right=right, res=res, mod=mod) -> bool:
                value = rule.func(self._number(left, local, rule.reverse), self._number(right, local, rule.reverse))
                return value is not None and value % mod == self._number(res, local, rule.reverse) % mod

            label = f"column match {eq.display()} low{k} via {self._rule_name(rule)}"
            if self._cfg("compact_projection_blocks", True):
                constraint = f"low{k}: rule({repr(left)}, {repr(right)}) ≡ {repr(res)} mod {mod}; dominates lower low-k checks when k is maximal"
            else:
                constraint = (
                    f"low{k} suffix: rule({repr(left)}, {repr(right)}) ≡ {repr(res)} (mod {mod}); "
                    f"full formula: {self._constraint_formula(eq, rule)}"
                )
            changed = self._project_replay(domains, scope, pred, label, trace, constraint=constraint, kind="COLUMN_PROJECT")
            if changed is None:
                return None
            any_changed = any_changed or changed
        return any_changed

    def _exact_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        label = f"full match {eq.display()} via {self._rule_name(rule)}"
        return self._project_replay(
            domains,
            scope,
            lambda local: self._equation_matches(eq, rule, local),
            label,
            trace,
            constraint=self._constraint_formula(eq, rule),
            kind="FULL_PROJECT",
        )

    def _solve_digits(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: ReplayTrace) -> Optional[Assignment]:
        if time.time() > self._deadline:
            raise TimeoutError("timeout")
        unresolved = [ch for ch, vals in domains.items() if len(vals) > 1]
        if not unresolved:
            assignment = {ch: next(iter(vals)) for ch, vals in domains.items()}
            if self._verify_assignment(equations, combo, assignment):
                trace.add_event(ReplayEvent(
                    kind="MAP_ACCEPT",
                    title="Digit map accepted",
                    decision="keep",
                    reason="all domains are singleton and every example verifies",
                    meta={"assignment": tuple(sorted(assignment.items()))},
                ))
                return assignment
            trace.add_event(ReplayEvent(
                kind="MAP_REJECT",
                title="Digit map rejected",
                decision="reject",
                reason="singleton domains do not verify every example",
                meta={"assignment": tuple(sorted(assignment.items()))},
            ))
            return None

        symbol = min(unresolved, key=lambda ch: (len(domains[ch]), ch))
        trace.add_event(ReplayEvent(
            kind="BRANCH_SELECT",
            title="Branch symbol selection",
            decision="select",
            reason=f"choose smallest domain, then symbol order: {repr(symbol)} in {self._dom(domains[symbol])}",
            meta={"symbol": symbol, "domain": tuple(sorted(domains[symbol]))},
        ))
        for digit in sorted(domains[symbol]):
            self.stats.digit_branches += 1
            old_domain = tuple(sorted(domains[symbol]))
            new_domain = (digit,)
            branch_domains = self._copy_domains(domains)
            branch_domains[symbol] = {digit}
            branch_trace = ReplayTrace(self.config if isinstance(self.config, ReplayConfig) else ReplayConfig())
            branch_trace.add_event(ReplayEvent(
                kind="DIGIT_TRY",
                title="Digit branch",
                before={symbol: old_domain},
                after={symbol: new_domain},
                changes={symbol: (old_domain, new_domain)},
                decision="try",
                reason=f"set {repr(symbol)}={digit}",
                meta={"symbol": symbol, "digit": digit},
            ))
            if self._propagate(equations, combo, branch_domains, branch_trace):
                found = self._solve_digits(equations, combo, branch_domains, branch_trace)
                if found is not None:
                    trace.extend(branch_trace)
                    return found
                reject_reason = branch_trace.reject_reason()
            else:
                reject_reason = branch_trace.reject_reason()
            failure_event = branch_trace.events[-1] if branch_trace.events else None
            trace.add_event(ReplayEvent(
                kind="DIGIT_REJECT",
                title="Digit branch rejected",
                label=(failure_event.label if failure_event else ""),
                constraint=(failure_event.constraint if failure_event else ""),
                before=(failure_event.before if failure_event else None),
                supported=(failure_event.supported if failure_event else None),
                after=(failure_event.after if failure_event else None),
                changes=(failure_event.changes if failure_event else None),
                count=(failure_event.count if failure_event else None),
                decision="reject",
                reason=f"{repr(symbol)}={digit}: {reject_reason}",
                meta={"symbol": symbol, "digit": digit, "failure_kind": failure_event.kind if failure_event else None},
            ))
            if self._cfg("include_failed_attempt_events", False):
                trace.extend(branch_trace)
        return None

    def _search_rules(
        self,
        ordered_ops: Sequence[str],
        candidates: Dict[str, List[Rule]],
        equations: Sequence[Equation],
        domains: Domains,
        combo: Dict[str, Rule],
        trace: ReplayTrace,
        index: int,
    ) -> Tuple[Optional[Assignment], Optional[Dict[str, Rule]]]:
        if time.time() > self._deadline:
            raise TimeoutError("timeout")
        if index == len(ordered_ops):
            domains2 = self._copy_domains(domains)
            trace2 = ReplayTrace(self.config if isinstance(self.config, ReplayConfig) else ReplayConfig())
            trace2.add_event(ReplayEvent(
                kind="COMBO_PROPAGATE",
                title="Rule combo propagation",
                decision="try",
                reason="all operators have candidate rules; start digit-domain solving",
                meta={"combo": tuple(sorted((op, self._rule_name(rule)) for op, rule in combo.items()))},
            ))
            if not self._propagate(equations, combo, domains2, trace2):
                failure_event = trace2.events[-1] if trace2.events else None
                trace.add_event(ReplayEvent(
                    kind="COMBO_REJECT",
                    title="Rule combo rejected",
                    label=(failure_event.label if failure_event else ""),
                    constraint=(failure_event.constraint if failure_event else ""),
                    before=(failure_event.before if failure_event else None),
                    supported=(failure_event.supported if failure_event else None),
                    after=(failure_event.after if failure_event else None),
                    changes=(failure_event.changes if failure_event else None),
                    count=(failure_event.count if failure_event else None),
                    decision="reject",
                    reason=trace2.reject_reason(),
                    meta={"combo": tuple(sorted((op, self._rule_name(rule)) for op, rule in combo.items())), "failure_kind": failure_event.kind if failure_event else None},
                ))
                if self._cfg("include_failed_attempt_events", False):
                    trace.extend(trace2)
                return None, None
            assignment = self._solve_digits(equations, combo, domains2, trace2)
            if assignment is not None:
                trace.extend(trace2)
                trace.add_event(ReplayEvent(
                    kind="COMBO_ACCEPT",
                    title="Rule combo selected",
                    decision="keep",
                    reason="this is the first rule combo whose digit map verifies all examples",
                    meta={"combo": tuple(sorted((op, self._rule_name(rule)) for op, rule in combo.items()))},
                ))
                return assignment, dict(combo)
            trace.add_event(ReplayEvent(
                kind="COMBO_REJECT",
                title="Rule combo rejected",
                decision="reject",
                reason=trace2.reject_reason(),
                meta={"combo": tuple(sorted((op, self._rule_name(rule)) for op, rule in combo.items()))},
            ))
            if self._cfg("include_failed_attempt_events", False):
                trace.extend(trace2)
            return None, None

        op = ordered_ops[index]
        for rule in candidates[op]:
            self.stats.rule_hypotheses += 1
            combo2 = dict(combo)
            combo2[op] = rule
            domains2 = self._copy_domains(domains)
            trace2 = ReplayTrace(self.config if isinstance(self.config, ReplayConfig) else ReplayConfig())
            trace2.add_event(ReplayEvent(
                kind="RULE_TRY",
                title="Rule candidate",
                decision="try",
                reason=f"operator {repr(op)} -> {self._rule_name(rule)}",
                meta={"op": op, "rule": self._rule_name(rule), "depth": index},
            ))
            if self._propagate(equations, combo2, domains2, trace2):
                assignment, solved_combo = self._search_rules(ordered_ops, candidates, equations, domains2, combo2, trace2, index + 1)
                if assignment is not None and solved_combo is not None:
                    trace.extend(trace2)
                    return assignment, solved_combo
                reject_reason = trace2.reject_reason()
            else:
                reject_reason = trace2.reject_reason()
            failure_event = trace2.events[-1] if trace2.events else None
            trace.add_event(ReplayEvent(
                kind="RULE_REJECT",
                title="Rule candidate rejected",
                label=(failure_event.label if failure_event else ""),
                constraint=(failure_event.constraint if failure_event else ""),
                before=(failure_event.before if failure_event else None),
                supported=(failure_event.supported if failure_event else None),
                after=(failure_event.after if failure_event else None),
                changes=(failure_event.changes if failure_event else None),
                count=(failure_event.count if failure_event else None),
                decision="reject",
                reason=f"operator {repr(op)} -> {self._rule_name(rule)}: {reject_reason}",
                meta={"op": op, "rule": self._rule_name(rule), "depth": index, "failure_kind": failure_event.kind if failure_event else None},
            ))
            if self._cfg("include_failed_attempt_events", False):
                trace.extend(trace2)
        return None, None

    # ------------------------------------------------------------------
    # Replay renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_text_map(domains: Optional[Dict[str, Tuple[int, ...]]]) -> str:
        if not domains:
            return "none"
        pieces = []
        for ch, vals in domains.items():
            if tuple(vals) == tuple(range(10)):
                txt = "{0..9}"
            else:
                txt = "{" + ",".join(str(v) for v in vals) + "}"
            pieces.append(f"D[{repr(ch)}]={txt}")
        return "; ".join(pieces)

    @staticmethod
    def _changes_text(changes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]) -> str:
        if not changes:
            return "none"
        pieces = []
        for ch, (before, after) in changes.items():
            b = "{" + ",".join(str(v) for v in before) + "}"
            a = "{" + ",".join(str(v) for v in after) + "}"
            if before == tuple(range(10)):
                b = "{0..9}"
            if after == tuple(range(10)):
                a = "{0..9}"
            pieces.append(f"D[{repr(ch)}]: {b} -> {a}")
        return "; ".join(pieces)

    def _structural_reason(self, rule: Rule, eq: Equation) -> str:
        if eq.has_sign and not rule.signed:
            return f"signed output is present, but {self._rule_name(rule)} is unsigned -> no"
        possible = self._possible_result_lengths(rule, eq)
        actual = len(eq.result)
        lengths = "{" + ",".join(str(x) for x in sorted(possible)) + "}"
        if actual in possible:
            return f"lengths {len(eq.left)} and {len(eq.right)} can produce result length in {lengths}; actual length {actual} -> possible"
        return f"lengths {len(eq.left)} and {len(eq.right)} can produce result length in {lengths}; actual length {actual} -> no"

    def _family_structural_summary(self, rules: Sequence[Rule], eqs: Sequence[Equation]) -> Dict[str, Any]:
        kept: List[Rule] = []
        rejected: Dict[str, List[Rule]] = {}
        reason_examples: Dict[str, List[str]] = {}
        for rule in sorted(rules, key=self._rule_sort_key):
            ok, reason = self._structural_status(rule, eqs)
            if ok:
                kept.append(rule)
            else:
                rejected.setdefault(reason, []).append(rule)
                if reason not in reason_examples:
                    # Keep only one representative local calculation per reason.
                    details = []
                    for eq in eqs:
                        detail = self._structural_reason(rule, eq)
                        if "-> no" in detail or "unsigned -> no" in detail:
                            details.append(detail)
                    reason_examples[reason] = details[:2] or [self._structural_reason(rule, eqs[0])]
        return {"kept": kept, "rejected": rejected, "reason_examples": reason_examples}

    def _render_structural_matching(self, by_op: Dict[str, List[Equation]], candidates: Dict[str, List[Rule]], combo: Dict[str, Rule]) -> List[str]:
        lines: List[str] = ["Rule matching"]
        compact = self._cfg("compact_structural_matching", True)
        for op in sorted(by_op):
            eqs = by_op[op]
            lines.append(f"Operator {repr(op)}")
            lines.append("examples: " + "; ".join(eq.display() for eq in eqs))
            lens = sorted({(len(eq.left), len(eq.right), len(eq.result), eq.has_sign) for eq in eqs})
            sig = "; ".join(f"{a}+{b}->len {r}, signed={sgn}" for a, b, r, sgn in lens)
            lines.append(f"structural signature: {sig}")
            lines.append("selection policy: structural filtering keeps rule groups; rule-combo search later chooses the first globally consistent combo")
            block_idx = 1
            for family, names in self._rule_family_order():
                family_rules = [r for r in self.rules if r.name in names]
                if not family_rules:
                    continue
                lines.append(f"Family block {block_idx}: {family}")
                if compact:
                    summary = self._family_structural_summary(family_rules, eqs)
                    kept = summary["kept"]
                    rejected = summary["rejected"]
                    if kept:
                        lines.append("  kept candidates: " + ", ".join(self._rule_name(r) for r in kept))
                    else:
                        lines.append("  kept candidates: none")
                    if rejected:
                        rej_parts = []
                        for reason, rules in sorted(rejected.items()):
                            rej_parts.append(f"{reason}: " + ", ".join(self._rule_name(r) for r in rules))
                        lines.append("  rejected candidates: " + "; ".join(rej_parts))
                        # Representative local calculations preserve replayability without repeating
                        # the same length/sign proof for every isomorphic rule.
                        for reason, details in sorted(summary["reason_examples"].items()):
                            lines.append(f"  representative {reason} check:")
                            for detail in details:
                                lines.append(f"    {detail}")
                    else:
                        lines.append("  rejected candidates: none")
                    lines.append(f"  decision: {'at least one candidate remains' if kept else 'no candidate in this family remains'}")
                else:
                    kept_any = False
                    for rule in sorted(family_rules, key=self._rule_sort_key):
                        ok, reason = self._structural_status(rule, eqs)
                        row_details = [self._structural_reason(rule, eq) for eq in eqs]
                        verdict = "keep" if ok else f"reject ({reason})"
                        if ok:
                            kept_any = True
                        lines.append(f"  test {self._rule_name(rule)}: {verdict}")
                        for detail in row_details:
                            lines.append(f"    {detail}")
                    lines.append(f"  decision: {'at least one candidate remains' if kept_any else 'no candidate in this family remains'}")
                block_idx += 1
            surviving = candidates.get(op, [])
            if surviving:
                lines.append("surviving after structural filtering: " + ", ".join(self._rule_name(r) for r in surviving))
                lines.append("final rule is selected later by rule-combo search")
            else:
                lines.append("surviving after structural filtering: none")
            lines.append("")
        return lines

    def _format_replay_event(self, event: ReplayEvent, block_num: int) -> List[str]:
        lines: List[str] = []
        kind = event.kind
        if kind == "RULE_ORDER":
            ordered = event.meta.get("ordered_ops", ()) if event.meta else ()
            lines.append(f"Rule-combo order: " + ", ".join(repr(op) for op in ordered))
            return lines
        if kind == "RULE_TRY":
            lines.append(f"Combo block {block_num}: try {event.reason}")
            return lines
        if kind == "RULE_REJECT":
            lines.append(f"Combo block {block_num}: reject {event.reason}")
            if event.label or event.constraint:
                lines.append("  rejecting local operation:")
                if event.label:
                    lines.append(f"    label: {event.label}")
                if event.constraint:
                    lines.append(f"    formula: {event.constraint}")
                if self._cfg("show_current_domains", False):
                    lines.append(f"    current local domains: {self._domain_text_map(event.before)}")
                if event.supported is not None:
                    count = "unknown" if event.count is None else str(event.count)
                    lines.append(f"    supported projection ({count} supports): {self._domain_text_map(event.supported)}")
                lines.append(f"    apply update: {self._changes_text(event.changes)}")
            return lines
        if kind == "COMBO_PROPAGATE":
            combo = event.meta.get("combo", ()) if event.meta else ()
            combo_text = "; ".join(f"{repr(op)}->{rule}" for op, rule in combo)
            lines.append(f"Combo block {block_num}: all operators assigned; start digit-domain solving")
            lines.append(f"  combo: {combo_text}")
            return lines
        if kind == "COMBO_REJECT":
            combo = event.meta.get("combo", ()) if event.meta else ()
            combo_text = "; ".join(f"{repr(op)}->{rule}" for op, rule in combo)
            lines.append(f"Combo block {block_num}: reject combo {combo_text}")
            lines.append(f"  reason: {event.reason}")
            if event.label or event.constraint:
                lines.append("  rejecting local operation:")
                if event.label:
                    lines.append(f"    label: {event.label}")
                if event.constraint:
                    lines.append(f"    formula: {event.constraint}")
                if self._cfg("show_current_domains", False):
                    lines.append(f"    current local domains: {self._domain_text_map(event.before)}")
                if event.supported is not None:
                    count = "unknown" if event.count is None else str(event.count)
                    lines.append(f"    supported projection ({count} supports): {self._domain_text_map(event.supported)}")
                lines.append(f"    apply update: {self._changes_text(event.changes)}")
            return lines
        if kind == "COMBO_ACCEPT":
            combo = event.meta.get("combo", ()) if event.meta else ()
            combo_text = "; ".join(f"{repr(op)}->{rule}" for op, rule in combo)
            lines.append(f"Combo block {block_num}: selected combo {combo_text}")
            lines.append(f"  decision: {event.reason}")
            return lines
        if kind in {"NO_LEADING_ZERO", "ALLDIFFERENT"}:
            lines.append(f"{event.title} {block_num}")
            lines.append(f"  constraint: {event.constraint}")
            if event.reason:
                lines.append(f"  reason: {event.reason}")
            if self._cfg("show_current_domains", False):
                lines.append(f"  current domains: {self._domain_text_map(event.before)}")
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            lines.append(f"  decision: {event.decision}")
            return lines
        if kind in {"COLUMN_PROJECT", "FULL_PROJECT", "PROJECTION"}:
            lines.append(f"Projection block {block_num}")
            if event.label:
                lines.append(f"  label: {event.label}")
            if event.constraint:
                lines.append(f"  formula: {event.constraint}")
            if self._cfg("show_current_domains", False):
                lines.append(f"  current local domains: {self._domain_text_map(event.before)}")
            if event.supported is not None:
                count = "unknown" if event.count is None else str(event.count)
                lines.append(f"  supported projection ({count} supports): {self._domain_text_map(event.supported)}")
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            if event.reason and self._cfg("show_projection_reason", False):
                lines.append(f"  reason: {event.reason}")
            lines.append(f"  decision: {event.decision}")
            return lines
        if kind == "BRANCH_SELECT":
            lines.append(f"Branch block {block_num}: choose symbol")
            lines.append(f"  reason: {event.reason}")
            return lines
        if kind == "DIGIT_TRY":
            lines.append(f"Branch block {block_num}: try {event.reason}")
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            lines.append("  decision: propagate this assignment")
            return lines
        if kind == "DIGIT_REJECT":
            lines.append(f"Branch block {block_num}: reject {event.reason}")
            if event.label or event.constraint:
                lines.append("  rejecting local operation:")
                if event.label:
                    lines.append(f"    label: {event.label}")
                if event.constraint:
                    lines.append(f"    formula: {event.constraint}")
                if self._cfg("show_current_domains", False):
                    lines.append(f"    current local domains: {self._domain_text_map(event.before)}")
                if event.supported is not None:
                    count = "unknown" if event.count is None else str(event.count)
                    lines.append(f"    supported projection ({count} supports): {self._domain_text_map(event.supported)}")
                lines.append(f"    apply update: {self._changes_text(event.changes)}")
            return lines
        if kind == "MAP_ACCEPT":
            assignment = event.meta.get("assignment", ()) if event.meta else ()
            lines.append(f"Map block {block_num}: digit map accepted")
            lines.append("  map: " + "; ".join(f"{repr(ch)}={d}" for ch, d in assignment))
            lines.append(f"  decision: {event.reason}")
            return lines
        if kind == "MAP_REJECT":
            lines.append(f"Map block {block_num}: reject candidate map")
            lines.append(f"  reason: {event.reason}")
            return lines
        lines.append(f"Replay block {block_num}: {event.kind}: {event.title}")
        if event.reason:
            lines.append(f"  reason: {event.reason}")
        if event.decision:
            lines.append(f"  decision: {event.decision}")
        return lines

    def _render_replay_solution(
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
        events: Sequence[ReplayEvent],
        answer: Optional[str],
        target_error: Optional[str],
        target_a: int,
        target_b: int,
        target_value: Optional[int],
        ordered_ops: Sequence[str],
        symbols: Sequence[str],
    ) -> List[str]:
        lines: List[str] = []
        add = lines.append
        by_op: Dict[str, List[Equation]] = {}
        for eq in equations:
            by_op.setdefault(eq.op, []).append(eq)

        add("Search setup")
        add("Each visible non-operator symbol is a digit variable.")
        add("The digit map is injective: different symbols denote different decimal digits.")
        add("Each operator has one selected arithmetic rule, shared by all examples using that operator.")
        add("A candidate is valid only if the same digit map and selected operator rules reproduce every visible output.")
        add("")

        add("Examples")
        for i, eq in enumerate(equations, 1):
            add(f"{i}. {eq.display()}")
        add(f"Target: {target_expr}")
        add("")

        add("Parsed equations")
        for i, eq in enumerate(equations, 1):
            expected = f"{eq.op}{eq.result}" if eq.has_sign else eq.result
            add(f"{i}. left={eq.left}, op={repr(eq.op)}, right={eq.right}, output={expected}, signed_output={eq.has_sign}")
        add("")

        add("Symbols")
        add("Digit symbols: " + ", ".join(repr(ch) for ch in symbols))
        add("Operator symbols: " + ", ".join(repr(op) for op in sorted(by_op)))
        add("AllDifferent: every listed digit symbol must take a different digit.")
        add("")

        add("Search axes")
        add("- read direction: std reads symbols left-to-right; rev reads symbols right-to-left")
        add("- arithmetic family: concat, add/add1/addm1, abs/sub, mul/mul1/mulm1")
        add("- digit map: injective assignment from visible digit symbols to 0..9")
        add("- output encoding: signed rules show a negative result by prefixing the operator symbol")
        add("Rule-combo order: " + ", ".join(repr(op) for op in ordered_ops))
        add("")

        lines.extend(self._render_structural_matching(by_op, candidates, combo))

        add("Local replay steps")
        add("Replay convention: a projection block is one local solver operation; replay it by intersecting the current domains with the shown supported projection, then applying the shown update.")
        add("Branch blocks explicitly assign one digit and then run the following propagation blocks.")
        add("Only local operations that change state, reject a branch/combo, or select a branch are shown; current domains are reconstructed from earlier updates unless audit mode enables them.")
        projection_count = 0
        for idx, event in enumerate(events, 1):
            if event.kind in {"COLUMN_PROJECT", "FULL_PROJECT", "PROJECTION"}:
                projection_count += 1
                if projection_count > self._cfg("max_projection_blocks", 10_000_000):
                    add(f"... remaining projection blocks omitted by max_projection_blocks={self._cfg('max_projection_blocks', 10_000_000)}")
                    break
            for line in self._format_replay_event(event, idx):
                add(line)
        add("")

        add("Selected digit map")
        add("; ".join(f"{repr(ch)}={assignment[ch]}" for ch in sorted(assignment)))
        add("")

        add("Verify selected rules")
        for op in sorted(by_op):
            rule = combo[op]
            add(f"Operator {repr(op)} uses {self._rule_name(rule)}")
            for eq in by_op[op]:
                a = self._number(eq.left, assignment, rule.reverse)
                b = self._number(eq.right, assignment, rule.reverse)
                op_text, value = self._operation_line(rule, a, b)
                encoded = None if value is None else self._encode_value(value, eq.op, rule, assignment)
                expected = f"{eq.op}{eq.result}" if eq.has_sign else eq.result
                status = "MATCH" if encoded == expected else "WRONG"
                add(f"- {eq.display()}")
                add(f"  decode {eq.left}->{a}, {eq.right}->{b}")
                add(f"  apply rule: {op_text}")
                add(f"  encode result: {encoded}")
                add(f"  compare visible output: produced {encoded} vs expected {expected} -> {status}")
        add("")

        add("Target")
        target_rule = combo[target_op]
        add(f"Target operator {repr(target_op)} was found in the examples.")
        add(f"Use selected rule: {self._rule_name(target_rule)}")
        add(f"Replay the rule on target {target_left}{target_op}{target_right}:")
        add(f"  decode {target_left}->{target_a}, {target_right}->{target_b}")
        if target_error or answer is None or target_value is None:
            add(f"  apply rule: {target_error or 'invalid target'}")
            add("Computed output: nan")
            add("Final answer: nan")
        else:
            operation, _ = self._operation_line(target_rule, target_a, target_b)
            add(f"  apply rule: {operation}")
            add(f"  encode {target_value}: {answer}")
            add(f"Computed output: {answer}")
            add(f"Final answer: {answer}")
            add(f"\\boxed{{{answer}}}")
        return lines

    def _failure(self, reason: str, started: float) -> Dict[str, Any]:
        lines = [
            "Search setup",
            "Each visible non-operator symbol is a digit variable; the digit map is injective; each operator has one selected arithmetic rule.",
            f"Solver failed: {reason}",
            "Final answer: nan",
        ]
        return {
            "answer": None,
            "solution": "\n".join(lines),
            "debug": lines,
            "trace": lines,
            "events": [],
            "stats": self.stats.__dict__,
            "elapsed_seconds": round(time.time() - started, 4),
        }


# Backward-compatible default name.  Existing evaluation code can still import
# CryptarithmSolver from this file.
CryptarithmSolver = CryptarithmReplaySolver


# ============================================================================
# v2.2 refinements
# ============================================================================
# v2.2 keeps the v2.1 search behaviour but makes the training CoT less
# solver-jargony and adds one safe real optimization: concat is projected as
# positional equalities instead of through generic support enumeration.

@dataclass
class ReplayConfigV22(ReplayConfig):
    # Renderer: support counts are useful for audit/debug but add little value to
    # training CoT.  The supported projection itself remains visible and is the
    # replay-complete local operation.
    show_support_counts: bool = False

    # Real optimization: cat/rev_cat is a deterministic positional constraint.
    # It should not enumerate assignments through the generic support projector.
    cat_as_positional_equalities: bool = True


_BaseCryptarithmReplaySolverV21 = CryptarithmReplaySolver


class CryptarithmReplaySolver(_BaseCryptarithmReplaySolverV21):
    """Replay solver v2.2.

    Changes from v2.1:
      * structural filtering is phrased as structural elimination/pre-checks,
        not as final rule rejection;
      * low-k modular formulas no longer mention noisy dominance/support-count
        internals in training CoT;
      * branch ordering stays deterministic and does not surface support-count
        heuristics;
      * cat/rev_cat uses direct positional constraints instead of generic
        support enumeration.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        if config is None:
            config = ReplayConfigV22()
        super().__init__(config)

    # ------------------------------------------------------------------
    # Real optimization: cat as positional equalities
    # ------------------------------------------------------------------

    def _cat_expected_output(self, eq: Equation, rule: Rule) -> str:
        # Under std/cat, encoded output must be left followed by right.
        # Under rev/cat, numbers are read right-to-left and the result is encoded
        # right-to-left, so the visible output must be right followed by left.
        return (eq.right + eq.left) if rule.reverse else (eq.left + eq.right)

    def _cat_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        before = self._snapshot_scope(domains, scope)
        expected = self._cat_expected_output(eq, rule)
        observed = eq.result
        label = f"positional concat {eq.display()} via {self._rule_name(rule)}"

        if len(expected) != len(observed):
            constraint = (
                f"{self._rule_name(rule)} requires visible output {repr(expected)} "
                f"with length {len(expected)}, but observed {repr(observed)} has length {len(observed)}"
            )
            trace.add_event(ReplayEvent(
                kind="FULL_PROJECT",
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={ch: tuple() for ch in scope},
                after=before,
                changes={},
                decision="reject",
                reason="concat positional length mismatch",
                count=0,
            ))
            return None

        mismatches = [(i, exp, obs) for i, (exp, obs) in enumerate(zip(expected, observed), 1) if exp != obs]
        if mismatches:
            i, exp, obs = mismatches[0]
            constraint = (
                f"{self._rule_name(rule)} requires visible output {repr(expected)}; "
                f"observed {repr(observed)}. At position {i}, expected symbol {repr(exp)} "
                f"but observed {repr(obs)}. Since the digit map is injective, two different "
                f"symbols cannot stand for the same digit."
            )
            trace.add_event(ReplayEvent(
                kind="FULL_PROJECT",
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={ch: tuple() for ch in scope},
                after=before,
                changes={},
                decision="reject",
                reason="concat positional mismatch",
                count=0,
            ))
            return None

        if self._cfg("include_no_change_events", False):
            constraint = f"{self._rule_name(rule)} requires visible output {repr(expected)}, matching observed output"
            trace.add_event(ReplayEvent(
                kind="FULL_PROJECT",
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported=before,
                after=before,
                changes={},
                decision="no-change",
                reason="concat positional equalities are already satisfied",
                count=None,
            ))
        return False

    def _exact_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        if rule.name == "cat" and self._cfg("cat_as_positional_equalities", True):
            return self._cat_project(eq, rule, domains, trace)
        return super()._exact_project(eq, rule, domains, trace)

    # ------------------------------------------------------------------
    # Cleaner modular projection wording
    # ------------------------------------------------------------------

    def _modular_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        if rule.name not in {"add", "add1", "addm1", "mul", "mul1", "mulm1"}:
            return False

        max_k = max(len(eq.left), len(eq.right), len(eq.result))
        full_scope = list(dict.fromkeys(eq.left + eq.right + eq.result))

        if self._cfg("projection_strategy", "adaptive") == "adaptive":
            if self._domain_space(domains, full_scope) <= int(self._cfg("exact_dominates_modular_space", 120_000)):
                if self._cfg("include_no_change_events", False):
                    trace.add_event(ReplayEvent(
                        kind="MODULAR_SKIP",
                        title="Modular cascade skipped",
                        decision="skip",
                        reason="full projection has small enough scope, so separate suffix projections are unnecessary",
                        meta={"equation": eq.display(), "rule": self._rule_name(rule), "space": self._domain_space(domains, full_scope)},
                    ))
                return False

            ks: List[int] = []
            for raw_k in self._cfg("modular_anchor_ks", (1, -1)):
                k = max_k if int(raw_k) == -1 else int(raw_k)
                if 1 <= k <= max_k and k not in ks:
                    ks.append(k)
        else:
            ks = list(range(1, max_k + 1))

        any_changed = False
        for k in ks:
            left = self._suffix(eq.left, k, rule.reverse)
            right = self._suffix(eq.right, k, rule.reverse)
            res = self._suffix(eq.result, k, rule.reverse)
            scope = list(dict.fromkeys(left + right + res))
            mod = 10**k

            def pred(local: Assignment, left=left, right=right, res=res, mod=mod) -> bool:
                value = rule.func(self._number(left, local, rule.reverse), self._number(right, local, rule.reverse))
                return value is not None and value % mod == self._number(res, local, rule.reverse) % mod

            label = f"suffix projection {eq.display()} low{k} via {self._rule_name(rule)}"
            if self._cfg("compact_projection_blocks", True):
                constraint = f"low{k} suffix constraint: rule({repr(left)}, {repr(right)}) mod {mod} = {repr(res)}"
            else:
                constraint = (
                    f"low{k} suffix: rule({repr(left)}, {repr(right)}) ≡ {repr(res)} (mod {mod}); "
                    f"full formula: {self._constraint_formula(eq, rule)}"
                )
            changed = self._project_replay(domains, scope, pred, label, trace, constraint=constraint, kind="COLUMN_PROJECT")
            if changed is None:
                return None
            any_changed = any_changed or changed
        return any_changed

    # ------------------------------------------------------------------
    # Structural matching phrased as grouped structural pre-checks
    # ------------------------------------------------------------------

    @staticmethod
    def _set_text(values: Iterable[Any]) -> str:
        vals = sorted(set(values))
        return "{" + ",".join(str(v) for v in vals) + "}"

    def _observed_signature_line(self, eqs: Sequence[Equation]) -> str:
        groups: Dict[Tuple[int, int], Dict[str, Set[Any]]] = {}
        for eq in eqs:
            key = (len(eq.left), len(eq.right))
            groups.setdefault(key, {"lengths": set(), "signed": set()})
            groups[key]["lengths"].add(len(eq.result))
            groups[key]["signed"].add(eq.has_sign)
        parts = []
        for (ll, rr), data in sorted(groups.items()):
            signed_vals = "{" + ",".join("True" if x else "False" for x in sorted(data["signed"])) + "}"
            parts.append(f"{ll}+{rr} -> output lengths {self._set_text(data['lengths'])}, signed outputs {signed_vals}")
        return "; ".join(parts)

    def _family_reason_details(self, reason: str, rules: Sequence[Rule], eqs: Sequence[Equation]) -> List[str]:
        rep = sorted(rules, key=self._rule_sort_key)[0]
        if reason == "sign":
            return ["signed output is present in the examples, but these rules are unsigned."]

        lines: List[str] = []
        grouped: Dict[Tuple[int, int], Set[int]] = {}
        for eq in eqs:
            grouped.setdefault((len(eq.left), len(eq.right)), set()).add(len(eq.result))
        for (ll, rr), observed in sorted(grouped.items()):
            # Build a small synthetic equation carrying the same operand lengths so
            # _possible_result_lengths can be used without depending on symbols.
            fake = Equation(raw="", left="A" * ll, right="B" * rr, op="?", result="R", has_sign=False)
            possible = self._possible_result_lengths(rep, fake)
            fam = rep.name
            if rep.name == "cat":
                capability = "concat"
            elif rep.name.startswith("add"):
                capability = "this add-offset rule"
            elif rep.name.startswith("mul"):
                capability = "this mul-offset rule"
            elif rep.name in {"abs", "sub"}:
                capability = "this abs/sub rule"
            else:
                capability = fam
            lines.append(
                f"for {ll}+{rr} operands, {capability} can produce output lengths {self._set_text(possible)}; observed lengths are {self._set_text(observed)}"
            )
        return lines

    def _family_structural_summary(self, rules: Sequence[Rule], eqs: Sequence[Equation]) -> Dict[str, Any]:
        kept: List[Rule] = []
        eliminated: Dict[str, List[Rule]] = {}
        reason_details: Dict[str, List[str]] = {}
        for rule in sorted(rules, key=self._rule_sort_key):
            ok, reason = self._structural_status(rule, eqs)
            if ok:
                kept.append(rule)
            else:
                eliminated.setdefault(reason, []).append(rule)
        for reason, rs in eliminated.items():
            reason_details[reason] = self._family_reason_details(reason, rs, eqs)
        return {"kept": kept, "eliminated": eliminated, "reason_details": reason_details}

    def _render_structural_matching(self, by_op: Dict[str, List[Equation]], candidates: Dict[str, List[Rule]], combo: Dict[str, Rule]) -> List[str]:
        lines: List[str] = ["Rule matching"]
        compact = self._cfg("compact_structural_matching", True)
        for op in sorted(by_op):
            eqs = by_op[op]
            lines.append(f"Operator {repr(op)}")
            lines.append("examples: " + "; ".join(eq.display() for eq in eqs))
            lines.append(f"structural signature: {self._observed_signature_line(eqs)}")
            lines.append("selection policy: these blocks perform only structural pre-checks; digit consistency is checked later by rule-combo search")
            block_idx = 1
            for family, names in self._rule_family_order():
                family_rules = [r for r in self.rules if r.name in names]
                if not family_rules:
                    continue
                lines.append(f"Family block {block_idx}: {family}")
                if compact:
                    summary = self._family_structural_summary(family_rules, eqs)
                    kept = summary["kept"]
                    eliminated = summary["eliminated"]
                    if kept:
                        lines.append("  kept after structural test: " + ", ".join(self._rule_name(r) for r in kept))
                    else:
                        lines.append("  kept after structural test: none")
                    if eliminated:
                        elim_parts = []
                        for reason, rules in sorted(eliminated.items()):
                            elim_parts.append(f"{reason}: " + ", ".join(self._rule_name(r) for r in rules))
                        lines.append("  eliminated by structural test: " + "; ".join(elim_parts))
                        for reason, details in sorted(summary["reason_details"].items()):
                            lines.append(f"  {reason} detail:")
                            for detail in details:
                                lines.append(f"    {detail}")
                    else:
                        lines.append("  eliminated by structural test: none")
                    lines.append(f"  decision: {'family remains' if kept else 'no candidate in this family remains'}")
                else:
                    kept_any = False
                    for rule in sorted(family_rules, key=self._rule_sort_key):
                        ok, reason = self._structural_status(rule, eqs)
                        row_details = [self._structural_reason(rule, eq) for eq in eqs]
                        verdict = "keep" if ok else f"eliminate by structural test ({reason})"
                        if ok:
                            kept_any = True
                        lines.append(f"  test {self._rule_name(rule)}: {verdict}")
                        for detail in row_details:
                            lines.append(f"    {detail}")
                    lines.append(f"  decision: {'family remains' if kept_any else 'no candidate in this family remains'}")
                block_idx += 1
            surviving = candidates.get(op, [])
            if surviving:
                lines.append("surviving after structural filtering: " + ", ".join(self._rule_name(r) for r in surviving))
                lines.append("final rule is selected later by rule-combo search")
            else:
                lines.append("surviving after structural filtering: none")
            lines.append("")
        return lines

    def _format_replay_event(self, event: ReplayEvent, block_num: int) -> List[str]:
        lines = super()._format_replay_event(event, block_num)
        if not self._cfg("show_support_counts", False):
            lines = [re.sub(r"supported projection \((?:unknown|\d+) supports\):", "supported projection:", line) for line in lines]
        return lines


# Backward-compatible alias used by some training scripts.
CryptarithmSolverReplay = CryptarithmReplaySolver

# Export v2.2 class under the historical solver name expected by harnesses.
CryptarithmSolver = CryptarithmReplaySolver

# ============================================================================
# v2.3 refinements
# ============================================================================
# v2.3 keeps v2.2 search behaviour, but makes rejected-candidate replay causal:
# candidate is tested first, the local projection/check is shown next, and the
# reject decision is printed only after the empty/contradictory result.

@dataclass
class ReplayConfigV23(ReplayConfigV22):
    # No search changes in v2.3; this flag only documents the renderer contract.
    causal_reject_blocks: bool = True


_BaseCryptarithmReplaySolverV22 = CryptarithmReplaySolver


class CryptarithmReplaySolver(_BaseCryptarithmReplaySolverV22):
    """Replay solver v2.3.

    Changes from v2.2:
      * rejected rule/combo/branch blocks are rendered in causal order:
        test candidate -> projection/check -> result -> reject decision;
      * no candidate is described as rejected before the shown local computation
        has produced the contradiction or empty supported projection.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        if config is None:
            config = ReplayConfigV23()
        super().__init__(config)

    @staticmethod
    def _split_decision_reason(reason: str) -> Tuple[str, str]:
        text = str(reason or "").strip()
        if ":" in text:
            head, tail = text.split(":", 1)
            return head.strip(), tail.strip()
        return text, text

    @staticmethod
    def _parse_operator_rule_fragment(text: str) -> Optional[Tuple[str, str]]:
        """Parse fragments like "operator '*' -> std/mul".

        This is used only by the renderer to distinguish a local candidate
        rejection from a prefix rejection caused by a downstream operator.
        It intentionally accepts quoted operator symbols, including escaped
        quotes/backslashes, because cryptarithm operators are arbitrary visible
        symbols.
        """
        m = re.search(
            r"operator\s+((?:'[^']*')|(?:\"[^\"]*\")|\S+)\s*->\s*([a-z]+/[a-z0-9]+)",
            str(text or ""),
        )
        if not m:
            return None
        return m.group(1).strip(), m.group(2).strip()

    @staticmethod
    def _strip_downstream_prefix(text: str) -> str:
        """Remove the leading "operator X -> rule:" part from a downstream failure."""
        return re.sub(
            r"^\s*operator\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s*->\s*[a-z]+/[a-z0-9]+\s*:\s*",
            "",
            str(text or "").strip(),
            count=1,
        )

    def _downstream_failure_info(
        self,
        event: ReplayEvent,
        candidate_text: str,
        failure_text: str,
    ) -> Optional[Tuple[str, str, str]]:
        """Return (failed_op, failed_rule, clean_failure) for downstream rejects.

        RULE_REJECT events can represent two different cases:
          * local candidate failure: the currently tested operator/rule fails;
          * prefix failure: the current prefix is tested, then a later operator
            fails during recursive rule-combo search.

        The search code records both as RULE_REJECT for backward compatibility.
        This renderer-only helper detects the second case from the textual
        witness and renders it as a prefix rejection instead of making it look
        like the current operator was directly disproved by another operator's
        projection.
        """
        tested = self._parse_operator_rule_fragment(candidate_text)
        failed = self._parse_operator_rule_fragment(failure_text)
        if not tested or not failed:
            return None
        tested_op, tested_rule = tested
        failed_op, failed_rule = failed
        if tested_op == failed_op and tested_rule == failed_rule:
            return None
        clean_failure = self._strip_downstream_prefix(failure_text) or failure_text
        return failed_op, failed_rule, clean_failure

    def _support_line(self, event: ReplayEvent, indent: str) -> Optional[str]:
        if event.supported is None:
            return None
        if self._cfg("show_support_counts", False):
            count = "unknown" if event.count is None else str(event.count)
            return f"{indent}supported projection ({count} supports): {self._domain_text_map(event.supported)}"
        return f"{indent}supported projection: {self._domain_text_map(event.supported)}"

    @staticmethod
    def _is_empty_supported_projection(event: ReplayEvent) -> bool:
        if event.supported is None:
            return False
        return all(len(tuple(vals)) == 0 for vals in event.supported.values())

    def _render_local_check(self, event: ReplayEvent, indent: str = "  ", result_override: Optional[str] = None) -> List[str]:
        lines: List[str] = []
        title = "Projection check" if event.supported is not None else "Local check"
        lines.append(f"{indent}{title}")
        if event.label:
            lines.append(f"{indent}  label: {event.label}")
        if event.constraint:
            lines.append(f"{indent}  formula: {event.constraint}")
        if self._cfg("show_current_domains", False):
            lines.append(f"{indent}  current local domains: {self._domain_text_map(event.before)}")
        support_line = self._support_line(event, indent + "  ")
        if support_line:
            lines.append(support_line)
        lines.append(f"{indent}  apply update: {self._changes_text(event.changes)}")
        if result_override is not None:
            result = result_override
        elif self._is_empty_supported_projection(event):
            result = event.reason or "no supported local digit assignment"
        else:
            result = event.reason or event.decision or "local check failed"
        if result:
            lines.append(f"{indent}  result: {result}")
        return lines

    def _format_replay_event(self, event: ReplayEvent, block_num: int) -> List[str]:
        if not self._cfg("causal_reject_blocks", True):
            return super()._format_replay_event(event, block_num)

        if event.kind == "RULE_REJECT":
            candidate, failure = self._split_decision_reason(event.reason)
            downstream = self._downstream_failure_info(event, candidate, failure)
            if downstream is not None:
                failed_op, failed_rule, clean_failure = downstream
                lines: List[str] = [f"Combo block {block_num}: test prefix assignment {candidate}"]
                lines.append("  Continue rule-combo search for remaining operators.")
                lines.append("  Downstream check:")
                lines.append(f"    operator {failed_op} -> {failed_rule}")
                if event.label or event.constraint or event.supported is not None:
                    lines.extend(self._render_local_check(event, indent="    ", result_override=clean_failure))
                else:
                    lines.append(f"    result: {clean_failure}")
                lines.append(f"  decision: reject prefix {candidate} because no globally consistent continuation remains")
                return lines

            lines = [f"Combo block {block_num}: test {candidate}"]
            if event.label or event.constraint or event.supported is not None:
                lines.extend(self._render_local_check(event, indent="  ", result_override=failure))
            else:
                lines.append(f"  result: {failure}")
            lines.append(f"  decision: reject {candidate}")
            return lines

        if event.kind == "COMBO_REJECT":
            combo = event.meta.get("combo", ()) if event.meta else ()
            combo_text = "; ".join(f"{repr(op)}->{rule}" for op, rule in combo)
            lines = [f"Combo block {block_num}: test combo {combo_text}"]
            if event.label or event.constraint or event.supported is not None:
                lines.extend(self._render_local_check(event, indent="  "))
            else:
                lines.append(f"  result: {event.reason}")
            lines.append(f"  decision: reject combo {combo_text}")
            return lines

        if event.kind == "DIGIT_REJECT":
            branch, failure = self._split_decision_reason(event.reason)
            lines = [f"Branch block {block_num}: test {branch}"]
            if event.label or event.constraint or event.supported is not None:
                lines.extend(self._render_local_check(event, indent="  ", result_override=failure))
            else:
                lines.append(f"  result: {failure}")
            lines.append(f"  decision: reject {branch}")
            return lines

        # For non-reject events, keep v2.2 behaviour, including removal of
        # support counts from training CoT by default.
        return super()._format_replay_event(event, block_num)


# Backward-compatible aliases.
CryptarithmSolverReplay = CryptarithmReplaySolver
CryptarithmSolver = CryptarithmReplaySolver

# ============================================================================
# v2.5 projection derivation refinements
# ============================================================================
# v2.5 keeps v2.4 search/accuracy behaviour, but makes projection blocks less
# oracle-like.  Each projection event now stores a compact derivation proof:
# local variables, checked domain space, satisfying tuple count, a few witness
# tuples, and the union-by-symbol step that yields the supported projection.
# This is intentionally bounded: it explains how the projection was obtained
# without dumping every satisfying tuple when the local support set is large.

@dataclass
class ReplayConfigV25(ReplayConfigV23):
    # Print a compact derivation section for projection blocks.
    show_projection_derivation: bool = True
    # Keep CoT bounded: show at most this many satisfying witness tuples.
    max_projection_witnesses: int = 8
    # If true, renderer still shows the full supported projection.  Keep this on
    # because the supported projection is the replay certificate used by domain
    # propagation blocks and augmentation.
    show_supported_projection: bool = True
    # If true, include local search-space size in every derivation.
    show_projection_search_space: bool = True


_BaseCryptarithmReplaySolverV24 = CryptarithmReplaySolver


class CryptarithmReplaySolver(_BaseCryptarithmReplaySolverV24):
    """Replay solver v2.5.

    Changes from v2.4:
      * projection events include a compact derivation proof instead of only a
        bare supported-projection certificate;
      * derivation proof is collected during the real local support computation,
        so the CoT describes the actual local computation that produced the
        projection;
      * rule-combo prefix-reject rendering from v2.4 is preserved.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        if config is None:
            config = ReplayConfigV25()
        super().__init__(config)
        self._support_cache_v25: Dict[
            Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]],
            Tuple[bool, Dict[str, Tuple[int, ...]], int, Tuple[Tuple[Tuple[str, int], ...], ...], int],
        ] = {}

    def solve(self, examples_text: Any, target_text: Optional[Any] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        self._support_cache_v25 = {}
        return super().solve(examples_text, target_text, timeout_seconds)

    def _supports_with_witnesses(
        self,
        scope: Sequence[str],
        domains: Domains,
        predicate: Callable[[Assignment], bool],
        max_witnesses: int,
    ) -> Tuple[bool, Dict[str, Set[int]], int, List[Dict[str, int]], int]:
        """Generic local support computation with bounded witness capture.

        This is the same semantic operation as _supports(), but it also captures
        the first few satisfying injective assignments.  The count is still the
        exact number of satisfying assignments, and supported[ch] is the exact
        union of digit values over all satisfying assignments.
        """
        self.stats.support_checks += 1
        ordered = sorted(dict.fromkeys(scope), key=lambda ch: (len(domains[ch]), ch))
        supported: Dict[str, Set[int]] = {ch: set() for ch in ordered}
        local: Assignment = {}
        used: Set[int] = set()
        witnesses: List[Dict[str, int]] = []
        count = 0
        space = 1
        for ch in ordered:
            space *= max(1, len(domains[ch]))

        def bt(i: int) -> None:
            nonlocal count
            if time.time() > self._deadline:
                raise TimeoutError("timeout")
            if i == len(ordered):
                if predicate(local):
                    count += 1
                    for ch, d in local.items():
                        supported[ch].add(d)
                    if len(witnesses) < max_witnesses:
                        witnesses.append({ch: local[ch] for ch in ordered})
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
        return count > 0, supported, count, witnesses, space

    def _projection_method_hint(self, kind: str, label: str, constraint: str) -> str:
        text = f"{kind} {label} {constraint}".lower()
        if "concat" in text or "/cat" in text:
            return "positional concat check: compare required output symbols position by position under injective digit mapping"
        if "low" in text and "mod" in text:
            return "column modular check: enumerate only the local suffix variables in this column constraint, respecting injective digits"
        if "abs(" in text:
            return "absolute-difference check: keep assignments satisfying the displayed abs(A-B)=C formula"
        if ")-(" in text or "-('" in text or "=-" in text:
            return "subtraction check: keep assignments satisfying the displayed signed/unsigned difference formula"
        if "*(" in text or ")*(" in text:
            return "multiplication check: keep assignments satisfying the displayed product formula"
        if "+1" in text:
            return "addition-with-offset check: keep assignments satisfying A+B+1=C"
        if "-1" in text:
            return "offset check: keep assignments satisfying the displayed -1 rule formula"
        if "+" in constraint:
            return "addition check: keep assignments satisfying the displayed sum formula"
        return "local support check: keep injective assignments satisfying the displayed formula"

    @staticmethod
    def _witnesses_to_tuple(witnesses: List[Dict[str, int]]) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
        return tuple(tuple(sorted(w.items())) for w in witnesses)

    @staticmethod
    def _witnesses_from_tuple(data: Tuple[Tuple[Tuple[str, int], ...], ...]) -> List[Dict[str, int]]:
        return [dict(items) for items in data]

    def _project_replay(
        self,
        domains: Domains,
        scope: Sequence[str],
        predicate: Callable[[Assignment], bool],
        label: str,
        trace: ReplayTrace,
        constraint: str,
        kind: str,
    ) -> Optional[bool]:
        scope = list(dict.fromkeys(scope))
        before = self._snapshot_scope(domains, scope)
        max_witnesses = int(self._cfg("max_projection_witnesses", 8))
        key = self._support_key(label, domains, scope)

        if self._cfg("use_support_cache", True) and key in self._support_cache_v25:
            ok, supported_tuple, count, witnesses_tuple, space = self._support_cache_v25[key]
            self.stats.support_cache_hits += 1
            supported = {ch: set(vals) for ch, vals in supported_tuple.items()}
            witnesses = self._witnesses_from_tuple(witnesses_tuple)
        else:
            ok, supported, count, witnesses, space = self._supports_with_witnesses(scope, domains, predicate, max_witnesses=max_witnesses)
            if self._cfg("use_support_cache", True):
                self._support_cache_v25[key] = (
                    ok,
                    {ch: tuple(sorted(vals)) for ch, vals in supported.items()},
                    count,
                    self._witnesses_to_tuple(witnesses),
                    space,
                )

        proof_meta = {
            "method": self._projection_method_hint(kind, label, constraint),
            "scope": tuple(scope),
            "space": space,
            "satisfying_count": count,
            "witnesses": tuple(tuple(sorted(w.items())) for w in witnesses),
            "witness_limit": max_witnesses,
            "union_supported": tuple((ch, tuple(sorted(vals))) for ch, vals in supported.items()),
        }

        if not ok:
            trace.add_event(ReplayEvent(
                kind=kind,
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={ch: tuple() for ch in scope},
                after=before,
                changes={},
                decision="reject",
                reason="no supported local digit assignment",
                count=0,
                meta={"projection_derivation": proof_meta},
            ))
            return None

        for ch, allowed in supported.items():
            domains[ch].intersection_update(allowed)
            if not domains[ch]:
                after = self._snapshot_scope(domains, scope)
                trace.add_event(ReplayEvent(
                    kind=kind,
                    title="Projection block",
                    label=label,
                    constraint=constraint,
                    before=before,
                    supported={c: tuple(sorted(v)) for c, v in supported.items()},
                    after=after,
                    changes=self._changes(before, after),
                    decision="reject",
                    reason=f"domain({repr(ch)}) became empty",
                    count=count,
                    meta={"projection_derivation": proof_meta},
                ))
                return None

        after = self._snapshot_scope(domains, scope)
        changes = self._changes(before, after)
        if changes:
            self.stats.domain_reductions += len(changes)
            trace.add_event(ReplayEvent(
                kind=kind,
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={c: tuple(sorted(v)) for c, v in supported.items()},
                after=after,
                changes=changes,
                decision="keep",
                reason="supported projection narrows at least one domain",
                count=count,
                meta={"projection_derivation": proof_meta},
            ))
            return True
        if self._cfg("include_no_change_events", False):
            trace.add_event(ReplayEvent(
                kind=kind,
                title="Projection block",
                label=label,
                constraint=constraint,
                before=before,
                supported={c: tuple(sorted(v)) for c, v in supported.items()},
                after=after,
                changes={},
                decision="no-change",
                reason="supported projection equals current domains",
                count=count,
                meta={"projection_derivation": proof_meta},
            ))
        return False

    @staticmethod
    def _tuple_domain_text(items: Tuple[Tuple[str, Tuple[int, ...]], ...]) -> str:
        pieces: List[str] = []
        for ch, vals in items:
            if tuple(vals) == tuple(range(10)):
                txt = "{0..9}"
            else:
                txt = "{" + ",".join(str(v) for v in vals) + "}"
            pieces.append(f"D[{repr(ch)}]={txt}")
        return "; ".join(pieces) if pieces else "none"

    @staticmethod
    def _witness_text(items: Tuple[Tuple[str, int], ...]) -> str:
        return ", ".join(f"{repr(ch)}={digit}" for ch, digit in items)

    def _render_projection_derivation(self, event: ReplayEvent, indent: str = "  ") -> List[str]:
        if not self._cfg("show_projection_derivation", True):
            return []
        meta = event.meta or {}
        proof = meta.get("projection_derivation") if isinstance(meta, dict) else None
        if not isinstance(proof, dict):
            return []

        lines: List[str] = []
        lines.append(f"{indent}projection derivation:")
        method = proof.get("method")
        if method:
            lines.append(f"{indent}  method: {method}")
        if self._cfg("show_projection_search_space", True):
            scope = proof.get("scope", ())
            space = proof.get("space", "unknown")
            if scope:
                scope_text = ", ".join(f"D[{repr(ch)}]" for ch in scope)
                lines.append(f"{indent}  local variables: {scope_text}")
            lines.append(f"{indent}  injective assignments checked: {space}")
        sat = proof.get("satisfying_count", event.count if event.count is not None else "unknown")
        lines.append(f"{indent}  satisfying assignments: {sat}")
        witnesses = proof.get("witnesses", ()) or ()
        if witnesses:
            limit = proof.get("witness_limit", len(witnesses))
            lines.append(f"{indent}  witness assignments shown: {len(witnesses)} of {sat}")
            for items in witnesses:
                lines.append(f"{indent}    {self._witness_text(tuple(items))}")
            if isinstance(sat, int) and sat > len(witnesses):
                lines.append(f"{indent}    ... {sat - len(witnesses)} more satisfying assignments omitted")
        else:
            lines.append(f"{indent}  witness assignments shown: none")
        union_supported = proof.get("union_supported", ()) or ()
        if union_supported:
            lines.append(f"{indent}  union by symbol gives: {self._tuple_domain_text(tuple(union_supported))}")
        return lines

    def _support_line(self, event: ReplayEvent, indent: str) -> Optional[str]:
        if not self._cfg("show_supported_projection", True):
            return None
        return super()._support_line(event, indent)

    def _render_local_check(self, event: ReplayEvent, indent: str = "  ", result_override: Optional[str] = None) -> List[str]:
        lines: List[str] = []
        title = "Projection check" if event.supported is not None else "Local check"
        lines.append(f"{indent}{title}")
        if event.label:
            lines.append(f"{indent}  label: {event.label}")
        if event.constraint:
            lines.append(f"{indent}  formula: {event.constraint}")
        if self._cfg("show_current_domains", False):
            lines.append(f"{indent}  current local domains: {self._domain_text_map(event.before)}")
        lines.extend(self._render_projection_derivation(event, indent + "  "))
        support_line = self._support_line(event, indent + "  ")
        if support_line:
            lines.append(support_line)
        lines.append(f"{indent}  apply update: {self._changes_text(event.changes)}")
        if result_override is not None:
            result = result_override
        elif self._is_empty_supported_projection(event):
            result = event.reason or "no supported local digit assignment"
        else:
            result = event.reason or event.decision or "local check failed"
        if result:
            lines.append(f"{indent}  result: {result}")
        return lines


# Backward-compatible aliases for v2.5.
CryptarithmSolverReplay = CryptarithmReplaySolver
CryptarithmSolver = CryptarithmReplaySolver

# v2.5.1 renderer hook: include derivation on successful projection blocks too.
_BaseCryptarithmReplaySolverV25 = CryptarithmReplaySolver


class CryptarithmReplaySolver(_BaseCryptarithmReplaySolverV25):
    def _format_replay_event(self, event: ReplayEvent, block_num: int) -> List[str]:
        if event.kind in {"COLUMN_PROJECT", "FULL_PROJECT", "PROJECTION"}:
            lines: List[str] = [f"Projection block {block_num}"]
            if event.label:
                lines.append(f"  label: {event.label}")
            if event.constraint:
                lines.append(f"  formula: {event.constraint}")
            if self._cfg("show_current_domains", False):
                lines.append(f"  current local domains: {self._domain_text_map(event.before)}")
            lines.extend(self._render_projection_derivation(event, "  "))
            support_line = self._support_line(event, "  ")
            if support_line:
                lines.append(support_line)
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            if event.reason and self._cfg("show_projection_reason", False):
                lines.append(f"  reason: {event.reason}")
            lines.append(f"  decision: {event.decision}")
            return lines
        return super()._format_replay_event(event, block_num)


CryptarithmSolverReplay = CryptarithmReplaySolver
CryptarithmSolver = CryptarithmReplaySolver
