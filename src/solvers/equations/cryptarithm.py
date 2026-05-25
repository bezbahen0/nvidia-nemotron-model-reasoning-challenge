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
    has_sign: bool

    def symbols(self) -> Set[str]:
        return set(self.left + self.right + self.result)

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
        return self.orientation == "little_endian"

    @property
    def desc(self) -> str:
        return f"{self.orientation}(A,B) \\mapsto {self.name}"


@dataclass
class TraceConfig:
    # This is not a debug trace. It is the final human-readable solution.
    # The defaults keep the proof compact enough for a 7500-token context.
    max_steps: int = 180
    max_solution_chars: int = 24000
    max_examples_per_step: int = 0
    include_rejected_hypotheses: bool = False
    include_search_branches: bool = True
    # Training CoT controls: keep the trace token-dense, but show real domain narrowing.
    max_domain_events: int = 80
    max_branch_rejects: int = 3
    include_global_support_pruning: bool = True
    max_global_support_space: int = 2_000_000


@dataclass
class Trace:
    config: TraceConfig = field(default_factory=TraceConfig)
    lines: List[str] = field(default_factory=list)
    omitted: int = 0
    char_count: int = 0

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        added_chars = len(text) + 1
        if len(self.lines) < self.config.max_steps and self.char_count + added_chars <= self.config.max_solution_chars:
            self.lines.append(text)
            self.char_count += added_chars
        else:
            self.omitted += 1

    def extend(self, other: "Trace") -> None:
        for line in other.lines:
            self.add(line)
        self.omitted += other.omitted

    def header(self, text: str) -> None:
        self.add(f"\n### {text}")

    def finish(self) -> List[str]:
        out = list(self.lines)
        if self.omitted:
            out.append("Some repetitive propagation steps were compressed to keep the solution within the token budget.")
        return out


@dataclass
class SearchStats:
    hypotheses_tested: int = 0
    branches_tested: int = 0
    domain_reductions: int = 0
    exact_support_checks: int = 0
    modular_support_checks: int = 0


@dataclass
class DomainEvent:
    kind: str
    label: str
    constraint: str = ""
    changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=dict)
    fixed: Dict[str, int] = field(default_factory=dict)
    rejects: List[str] = field(default_factory=list)
    keep: Optional[str] = None
    contradiction: Optional[str] = None


class DeductiveCryptarithmSolver:
    def __init__(self, trace_config: Optional[TraceConfig] = None):
        self.rules = self._build_rules()
        self.trace_config = trace_config or TraceConfig()
        self.stats = SearchStats()
        self._deadline = 0.0

    def _build_rules(self) -> List[Rule]:
        def nn(v: int) -> Optional[int]:
            return v if v >= 0 else None

        ops: List[Tuple[str, bool, Callable[[int, int], Optional[int]]]] = [
            ("add", False, lambda a, b: a + b),
            ("add_p1", False, lambda a, b: a + b + 1),
            ("add_m1", False, lambda a, b: nn(a + b - 1)),
            ("mul", False, lambda a, b: a * b),
            ("mul_p1", False, lambda a, b: a * b + 1),
            ("mul_m1", False, lambda a, b: nn(a * b - 1)),
            ("absdiff", False, lambda a, b: abs(a - b)),
            ("sub_signed", True, lambda a, b: a - b),
            ("concat_fwd", False, lambda a, b: int(f"{a}{b}")),
        ]
        rules: List[Rule] = []
        for orientation in ("standard", "little_endian"):
            for name, signed, func in ops:
                tier = 0 if orientation == "standard" else 1
                if name in {"concat_fwd", "sub_signed"}:
                    tier += 1
                rules.append(Rule(name, orientation, signed, tier, func))
        return rules

    @staticmethod
    def _preferred_operator_index(left: str) -> int:
        return 2 if len(left) >= 5 else len(left) // 2

    def _parse_equation(self, line: str, known_ops: Optional[Set[str]] = None) -> Optional[Equation]:
        raw = line.strip()
        compact = raw.replace(" ", "")
        if "=" not in compact:
            return None
        left_side, right_side = compact.split("=", 1)
        if not left_side or not right_side:
            return None

        idx: Optional[int] = None
        if known_ops:
            preferred = self._preferred_operator_index(left_side)
            if 0 < preferred < len(left_side) - 1 and left_side[preferred] in known_ops:
                idx = preferred
            else:
                idx = next((i for i, ch in enumerate(left_side) if 0 < i < len(left_side) - 1 and ch in known_ops), None)
        if idx is None:
            idx = self._preferred_operator_index(left_side)
            if not 0 < idx < len(left_side) - 1:
                return None

        op = left_side[idx]
        result_is_signed = len(right_side) > 1 and right_side[0] == op
        result = right_side[1:] if result_is_signed else right_side
        if not left_side[:idx] or not left_side[idx + 1 :] or not result:
            return None
        return Equation(raw, left_side[:idx], left_side[idx + 1 :], op, result, result_is_signed)

    def _parse_examples(self, text: str) -> Tuple[List[Equation], Set[str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip() and "=" in line]
        guessed = {eq.op for line in lines if (eq := self._parse_equation(line, None))}
        return [eq for line in lines if (eq := self._parse_equation(line, guessed))], guessed

    def _parse_target(self, text: str, known_ops: Set[str]) -> Optional[Tuple[str, str, str, str]]:
        expr = re.sub(r"(?is).*?now,\s*d[e]?termine\s*the\s*result\s*for:\s*", "", text)
        expr = expr.replace(" ", "").strip().rstrip(".")
        if "=" in expr:
            expr = expr.split("=", 1)[0]
        if not expr:
            return None
        preferred = self._preferred_operator_index(expr)
        idx: Optional[int] = preferred if 0 < preferred < len(expr) - 1 and expr[preferred] in known_ops else None
        if idx is None:
            idx = next((i for i, ch in enumerate(expr) if 0 < i < len(expr) - 1 and ch in known_ops), None)
        if idx is None:
            idx = preferred if 0 < preferred < len(expr) - 1 else None
        return None if idx is None else (expr, expr[:idx], expr[idx], expr[idx + 1 :])

    @staticmethod
    def _number(symbols: str, assignment: Assignment, reverse: bool) -> int:
        value = 0
        for ch in symbols[::-1] if reverse else symbols:
            value = value * 10 + assignment[ch]
        return value

    @staticmethod
    def _domain(values: Iterable[int]) -> str:
        return "\\{" + ",".join(str(v) for v in sorted(values)) + "\\}"

    @staticmethod
    def _bounds(length: int) -> Tuple[int, int]:
        return (0, 9) if length <= 1 else (10 ** (length - 1), 10**length - 1)

    @staticmethod
    def _lengths_between(lo: int, hi: int) -> Set[int]:
        if hi < 0:
            return set()
        lo = max(0, lo)
        lengths = {len(str(hi))}
        n = lo
        while n <= hi:
            lengths.add(len(str(n)))
            nxt = 10 ** len(str(n))
            n = nxt if nxt > n else nxt * 10
        return lengths

    def _possible_lengths(self, rule: Rule, eq: Equation) -> Set[int]:
        ll, lr = len(eq.left), len(eq.right)
        min_l, max_l = self._bounds(ll)
        min_r, max_r = self._bounds(lr)
        if rule.name == "concat_fwd":
            return {ll + lr}
        if rule.name.startswith("add"):
            delta = 1 if rule.name == "add_p1" else -1 if rule.name == "add_m1" else 0
            return self._lengths_between(min_l + min_r + delta, max_l + max_r + delta)
        if rule.name.startswith("mul"):
            delta = 1 if rule.name == "mul_p1" else -1 if rule.name == "mul_m1" else 0
            return self._lengths_between(min_l * min_r + delta, max_l * max_r + delta)
        high = max(abs(x - y) for x in (min_l, max_l) for y in (min_r, max_r))
        return self._lengths_between(0, high)

    def _structurally_possible(self, rule: Rule, equations: Sequence[Equation]) -> bool:
        for eq in equations:
            if eq.has_sign and not rule.signed:
                return False
            if len(eq.result) not in self._possible_lengths(rule, eq):
                return False
        return True

    def _matches(self, eq: Equation, rule: Rule, assignment: Assignment) -> bool:
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
        digits = [int(ch) for ch in str(value)]
        if rule.reverse:
            digits.reverse()
        return digits == [assignment[ch] for ch in eq.result]

    def _copy(self, domains: Domains) -> Domains:
        return {ch: set(values) for ch, values in domains.items()}

    def _all_symbols(self, equations: Sequence[Equation], target_left: str, target_right: str) -> List[str]:
        counts: Dict[str, int] = {}
        for eq in equations:
            for ch in eq.left + eq.right + eq.result:
                counts[ch] = counts.get(ch, 0) + 1
        for ch in target_left + target_right:
            counts[ch] = counts.get(ch, 0) + 1
        return sorted(counts, key=lambda ch: (-counts[ch], ch))

    def _leading_zero(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace) -> bool:
        reductions: List[str] = []
        for eq in equations:
            rule = combo.get(eq.op)
            if rule is None:
                continue
            for term in (eq.left, eq.right, eq.result):
                if len(term) <= 1:
                    continue
                lead = term[-1] if rule.reverse else term[0]
                if 0 not in domains[lead]:
                    continue
                domains[lead].remove(0)
                if not domains[lead]:
                    trace.add(f"Contradiction: ${lead}=0$ would make a multi-digit number start with zero in ${eq.display()}$.")
                    return False
                self.stats.domain_reductions += 1
                reductions.append(f"D_{{{lead}}} \\leftarrow {self._domain(domains[lead])}")
        if reductions:
            trace.add("Leading-zero restrictions give $" + ",\\;".join(reductions) + "$." )
        return True

    def _alldifferent(self, domains: Domains, trace: Trace) -> bool:
        singles = [next(iter(v)) for v in domains.values() if len(v) == 1]
        if len(singles) != len(set(singles)):
            trace.add("Contradiction: two symbols are forced to the same digit, violating $\\operatorname{AllDifferent}$.")
            return False
        fixed = set(singles)
        reductions: List[str] = []
        for ch, values in domains.items():
            if len(values) == 1:
                continue
            before = set(values)
            values.difference_update(fixed)
            if not values:
                trace.add(f"Contradiction: $\\operatorname{{AllDifferent}}$ empties $D_{{{ch}}}$.")
                return False
            if values != before:
                self.stats.domain_reductions += 1
                reductions.append(f"D_{{{ch}}} \\leftarrow {self._domain(values)}")
        if reductions:
            trace.add("Global $\\operatorname{AllDifferent}$ removes fixed digits: $" + ",\\;".join(reductions) + "$.")
        return True

    def _supports(
        self,
        scope: Sequence[str],
        domains: Domains,
        predicate: Callable[[Assignment], bool],
        max_examples: int,
    ) -> Tuple[bool, Dict[str, Set[int]], int, List[Assignment]]:
        ordered = sorted(dict.fromkeys(scope), key=lambda ch: (len(domains[ch]), ch))
        supported = {ch: set() for ch in ordered}
        examples: List[Assignment] = []
        assignment: Assignment = {}
        used: Set[int] = set()
        count = 0

        def bt(i: int) -> None:
            nonlocal count
            if time.time() > self._deadline:
                raise TimeoutError("timeout while enumerating supports")
            if i == len(ordered):
                if predicate(assignment):
                    count += 1
                    for c, d in assignment.items():
                        supported[c].add(d)
                    if len(examples) < max_examples:
                        examples.append(dict(assignment))
                return
            ch = ordered[i]
            for digit in sorted(domains[ch]):
                if digit in used:
                    continue
                assignment[ch] = digit
                used.add(digit)
                bt(i + 1)
                used.remove(digit)
                del assignment[ch]

        bt(0)
        return count > 0, supported, count, examples

    def _project(
        self,
        scope: Sequence[str],
        domains: Domains,
        predicate: Callable[[Assignment], bool],
        label: str,
        trace: Trace,
        exact: bool,
    ) -> Optional[bool]:
        ok, supported, count, examples = self._supports(scope, domains, predicate, self.trace_config.max_examples_per_step)
        if exact:
            self.stats.exact_support_checks += 1
        else:
            self.stats.modular_support_checks += 1
        if not ok:
            trace.add(f"Contradiction from {label}: no supported assignment remains.")
            return None

        changed = False
        reductions: List[str] = []
        for ch, allowed in supported.items():
            before = set(domains[ch])
            domains[ch].intersection_update(allowed)
            if not domains[ch]:
                trace.add(f"Contradiction from {label}: $D_{{{ch}}}$ becomes empty.")
                return None
            if domains[ch] != before:
                changed = True
                self.stats.domain_reductions += 1
                reductions.append(f"D_{{{ch}}} \\leftarrow {self._domain(domains[ch])}")
        if changed:
            trace.add(f"{label} gives $" + ",\\;".join(reductions) + "$." )
        return changed

    def _suffix(self, term: str, k: int, reverse: bool) -> str:
        return term[:k] if reverse else term[-k:]

    def _modular(self, eq: Equation, rule: Rule, domains: Domains, trace: Trace) -> Optional[bool]:
        if rule.name not in {"add", "add_p1", "add_m1", "mul", "mul_p1", "mul_m1"}:
            return False
        changed_any = False
        for k in range(1, max(len(eq.left), len(eq.right), len(eq.result)) + 1):
            lp, rp, sp = self._suffix(eq.left, k, rule.reverse), self._suffix(eq.right, k, rule.reverse), self._suffix(eq.result, k, rule.reverse)
            scope = list(dict.fromkeys(lp + rp + sp))
            mod = 10**k

            def pred(local: Assignment, l=lp, r=rp, s=sp, m=mod) -> bool:
                value = rule.func(self._number(l, local, rule.reverse), self._number(r, local, rule.reverse))
                return value is not None and value % m == self._number(s, local, rule.reverse) % m

            label = f"The lowest {k} digit(s) of ${eq.display()}$ under $\\mathcal{{H}}: {rule.desc}$"
            changed = self._project(scope, domains, pred, label, trace, exact=False)
            if changed is None:
                return None
            changed_any = changed_any or changed
        return changed_any

    def _exact(self, eq: Equation, rule: Rule, domains: Domains, trace: Trace) -> Optional[bool]:
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        label = f"The full equation ${eq.display()}$ under $\\mathcal{{H}}: {rule.desc}$"
        return self._project(scope, domains, lambda local: self._matches(eq, rule, local), label, trace, exact=True)

    def _propagate(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace) -> bool:
        while True:
            before = {ch: tuple(sorted(v)) for ch, v in domains.items()}
            if not self._leading_zero(equations, combo, domains, trace):
                return False
            if not self._alldifferent(domains, trace):
                return False
            for eq in equations:
                rule = combo.get(eq.op)
                if rule is None:
                    continue
                if self._modular(eq, rule, domains, trace) is None:
                    return False
                if self._exact(eq, rule, domains, trace) is None:
                    return False
            after = {ch: tuple(sorted(v)) for ch, v in domains.items()}
            if before == after:
                return True

    def _verify(self, equations: Sequence[Equation], combo: Dict[str, Rule], assignment: Assignment) -> bool:
        return all(self._matches(eq, combo[eq.op], assignment) for eq in equations)

    def _solve_digits(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace, depth: int) -> Optional[Assignment]:
        if time.time() > self._deadline:
            raise TimeoutError("solver timeout")
        unresolved = [ch for ch, values in domains.items() if len(values) > 1]
        if not unresolved:
            assignment = {ch: next(iter(values)) for ch, values in domains.items()}
            return assignment if self._verify(equations, combo, assignment) else None
        symbol = min(unresolved, key=lambda ch: (len(domains[ch]), ch))
        values = sorted(domains[symbol])
        for digit in values:
            self.stats.branches_tested += 1
            next_domains = self._copy(domains)
            next_domains[symbol] = {digit}
            branch_trace = Trace(trace.config)
            if self.trace_config.include_search_branches:
                branch_trace.add(f"The remaining ambiguity is resolved by the consistent branch ${symbol}={digit}$.")
            if self._propagate(equations, combo, next_domains, branch_trace):
                solved = self._solve_digits(equations, combo, next_domains, branch_trace, depth + 1)
                if solved is not None:
                    trace.extend(branch_trace)
                    return solved
        return None

    def _search_rules(
        self,
        ops: Sequence[str],
        candidates: Dict[str, List[Rule]],
        equations: Sequence[Equation],
        domains: Domains,
        combo: Dict[str, Rule],
        trace: Trace,
        op_index: int,
    ) -> Tuple[Optional[Assignment], Optional[Dict[str, Rule]]]:
        if time.time() > self._deadline:
            raise TimeoutError("solver timeout")
        if op_index == len(ops):
            next_domains = self._copy(domains)
            branch_trace = Trace(trace.config)
            if not self._propagate(equations, combo, next_domains, branch_trace):
                return None, None
            assignment = self._solve_digits(equations, combo, next_domains, branch_trace, 0)
            if assignment is not None:
                trace.extend(branch_trace)
                return assignment, dict(combo)
            return None, None

        op = ops[op_index]
        for rule in candidates[op]:
            self.stats.hypotheses_tested += 1
            next_combo = dict(combo)
            next_combo[op] = rule
            next_domains = self._copy(domains)
            branch_trace = Trace(trace.config)
            branch_trace.add(f"For operator {self._literal(op)}, the surviving rule is {rule.desc}.")
            if self._propagate(equations, next_combo, next_domains, branch_trace):
                assignment, solved_combo = self._search_rules(ops, candidates, equations, next_domains, next_combo, branch_trace, op_index + 1)
                if assignment is not None and solved_combo is not None:
                    trace.extend(branch_trace)
                    return assignment, solved_combo
            # Failed hypotheses are intentionally not printed: the final output is a solution, not a search log.
        return None, None

    def _encode_target(self, left: str, op: str, right: str, rule: Rule, assignment: Assignment) -> Tuple[Optional[str], Optional[str], int, int, Optional[int]]:
        a = self._number(left, assignment, rule.reverse)
        b = self._number(right, assignment, rule.reverse)
        value = rule.func(a, b)
        if value is None:
            return None, "target operation produced an invalid value", a, b, value
        sign = ""
        if rule.signed and value < 0:
            sign, value = op, abs(value)
        elif not rule.signed and value < 0:
            return None, "unsigned target operation produced a negative value", a, b, value
        reverse_map = {d: ch for ch, d in assignment.items()}
        chars: List[str] = []
        for digit in str(value):
            d = int(digit)
            if d not in reverse_map:
                return None, f"digit {d} is not mapped to any known symbol", a, b, value
            chars.append(reverse_map[d])
        if rule.reverse:
            chars.reverse()
        return sign + "".join(chars), None, a, b, value

    def _operation_text(self, rule: Rule, a: int, b: int) -> Tuple[str, Optional[int]]:
        value = rule.func(a, b)
        if value is None:
            return f"{a} {rule.name} {b}", None
        if rule.name == "add":
            return f"{a}+{b}={value}", value
        if rule.name == "add_p1":
            return f"{a}+{b}+1={value}", value
        if rule.name == "add_m1":
            return f"{a}+{b}-1={value}", value
        if rule.name == "mul":
            return f"{a}\\cdot {b}={value}", value
        if rule.name == "mul_p1":
            return f"{a}\\cdot {b}+1={value}", value
        if rule.name == "mul_m1":
            return f"{a}\\cdot {b}-1={value}", value
        if rule.name == "absdiff":
            return f"|{a}-{b}|={value}", value
        if rule.name == "sub_signed":
            return f"{a}-{b}={value}", value
        if rule.name == "concat_fwd":
            return f"\\operatorname{{concat}}({a},{b})={value}", value
        return f"f({a},{b})={value}", value

    def _rule_human(self, op: str, rule: Rule) -> str:
        orientation = "read each operand left-to-right" if not rule.reverse else "read each operand right-to-left"
        if rule.name == "add":
            meaning = "ordinary addition: $A+B$"
        elif rule.name == "add_p1":
            meaning = "addition plus one: $A+B+1$"
        elif rule.name == "add_m1":
            meaning = "addition minus one: $A+B-1$"
        elif rule.name == "mul":
            meaning = "ordinary multiplication: $A\\cdot B$"
        elif rule.name == "mul_p1":
            meaning = "multiplication plus one: $A\\cdot B+1$"
        elif rule.name == "mul_m1":
            meaning = "multiplication minus one: $A\\cdot B-1$"
        elif rule.name == "absdiff":
            meaning = "absolute difference: $|A-B|$"
        elif rule.name == "sub_signed":
            meaning = "signed subtraction: $A-B$; if the value is negative, the operator symbol is written as the sign"
        elif rule.name == "concat_fwd":
            meaning = "concatenation: write the decimal digits of $A$ followed by the decimal digits of $B$"
        else:
            meaning = rule.name
        return f"Operator {self._literal(op)} means {meaning}; {orientation}."

    @staticmethod
    def _literal(text: str) -> str:
        return repr(text)

    def _decode_term_text(self, term: str, rule: Rule, assignment: Assignment) -> str:
        order = term[::-1] if rule.reverse else term
        digits = "".join(str(assignment[ch]) for ch in order)
        return f"{self._literal(term)} decodes to $A={digits}$"

    def _encode_value(self, value: int, op: str, rule: Rule, assignment: Assignment) -> Optional[str]:
        sign = ""
        if rule.signed and value < 0:
            sign = op
            value = abs(value)
        elif value < 0:
            return None
        reverse_map = {digit: ch for ch, digit in assignment.items()}
        chars: List[str] = []
        for digit_char in str(value):
            digit = int(digit_char)
            if digit not in reverse_map:
                return None
            chars.append(reverse_map[digit])
        if rule.reverse:
            chars.reverse()
        return sign + "".join(chars)

    def _mapping_line(self, assignment: Assignment) -> str:
        parts = [f"${ch}={assignment[ch]}$" for ch in sorted(assignment)]
        return ", ".join(parts)

    def _rule_short(self, rule: Rule) -> str:
        names = {
            "add": "add",
            "add_p1": "add1",
            "add_m1": "addm1",
            "mul": "mul",
            "mul_p1": "mul1",
            "mul_m1": "mulm1",
            "absdiff": "abs",
            "sub_signed": "sub",
            "concat_fwd": "cat",
        }
        prefix = "rev" if rule.reverse else "std"
        return f"{prefix}/{names.get(rule.name, rule.name)}"

    def _rule_action_text(self, rule: Rule) -> str:
        read = "readR" if rule.reverse else "readL"
        enc = "encR" if rule.reverse else "encL"
        actions = {
            "add": "A+B",
            "add_p1": "A+B+1",
            "add_m1": "A+B-1",
            "mul": "A*B",
            "mul_p1": "A*B+1",
            "mul_m1": "A*B-1",
            "absdiff": "abs(A-B)",
            "sub_signed": "A-B, sign=op if negative",
            "concat_fwd": "concat(A,B)",
        }
        return f"{read}; {actions.get(rule.name, rule.name)}; {enc}"

    @staticmethod
    def _digits_dense(values: Iterable[int]) -> str:
        vals = sorted(set(values))
        if vals == list(range(10)):
            return "{0..9}"
        return "{" + ",".join(str(v) for v in vals) + "}"

    @staticmethod
    def _snapshot(domains: Domains) -> Dict[str, Tuple[int, ...]]:
        return {ch: tuple(sorted(values)) for ch, values in domains.items()}

    @staticmethod
    def _domain_space_upper_bound(domains: Domains, limit: int) -> int:
        product = 1
        for values in domains.values():
            product *= max(1, len(values))
            if product > limit:
                return product
        return product

    def _structural_reject_reason(self, rule: Rule, equations: Sequence[Equation]) -> Optional[str]:
        for eq in equations:
            if eq.has_sign and not rule.signed:
                return "sign"
            possible = self._possible_lengths(rule, eq)
            if len(eq.result) not in possible:
                if rule.name == "concat_fwd":
                    return "cat_len"
                if rule.name.startswith("mul"):
                    return "mul_len"
                return "len"
        return None

    def _term_formula(self, term: str, rule: Rule) -> str:
        ordered = term[::-1] if rule.reverse else term
        coeffs: Dict[str, int] = {}
        n = len(ordered)
        for i, ch in enumerate(ordered):
            coeffs[ch] = coeffs.get(ch, 0) + 10 ** (n - i - 1)
        parts: List[str] = []
        for ch in sorted(coeffs, key=lambda c: ordered.index(c)):
            coeff = coeffs[ch]
            lit = self._literal(ch)
            if coeff == 1:
                parts.append(lit)
            else:
                parts.append(f"{coeff}*{lit}")
        return "+".join(parts) if parts else "0"

    def _equation_constraint_text(self, eq: Equation, rule: Rule) -> str:
        left = self._term_formula(eq.left, rule)
        right = self._term_formula(eq.right, rule)
        result = self._term_formula(eq.result, rule)
        if rule.name == "add":
            expr = f"{left}+{right}={result}"
        elif rule.name == "add_p1":
            expr = f"{left}+{right}+1={result}"
        elif rule.name == "add_m1":
            expr = f"{left}+{right}-1={result}"
        elif rule.name == "mul":
            expr = f"({left})*({right})={result}"
        elif rule.name == "mul_p1":
            expr = f"({left})*({right})+1={result}"
        elif rule.name == "mul_m1":
            expr = f"({left})*({right})-1={result}"
        elif rule.name == "absdiff":
            expr = f"abs(({left})-({right}))={result}"
        elif rule.name == "sub_signed":
            rhs = f"-({result})" if eq.has_sign else result
            expr = f"({left})-({right})={rhs}"
        elif rule.name == "concat_fwd":
            expr = f"concat({left},{right})={result}"
        else:
            expr = f"{rule.name}({left},{right})={result}"
        return expr

    def _project_evidence(
        self,
        scope: Sequence[str],
        domains: Domains,
        predicate: Callable[[Assignment], bool],
        kind: str,
        label: str,
        constraint: str,
    ) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        ok, supported, _count, _examples = self._supports(scope, domains, predicate, 0)
        if not ok:
            return None, DomainEvent(kind=kind, label=label, constraint=constraint, contradiction="no_supported_assignment")

        changed = False
        changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        fixed: Dict[str, int] = {}
        for ch, allowed in supported.items():
            before = tuple(sorted(domains[ch]))
            domains[ch].intersection_update(allowed)
            after = tuple(sorted(domains[ch]))
            if not after:
                return None, DomainEvent(kind=kind, label=label, constraint=constraint, contradiction=f"D[{self._literal(ch)}] empty")
            if after != before:
                changed = True
                changes[ch] = (before, after)
                if len(after) == 1:
                    fixed[ch] = after[0]
        if not changed:
            return False, None
        return True, DomainEvent(kind=kind, label=label, constraint=constraint, changes=changes, fixed=fixed)

    def _leading_zero_evidence(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        fixed: Dict[str, int] = {}
        leads: List[str] = []
        for eq in equations:
            rule = combo.get(eq.op)
            if rule is None:
                continue
            for term in (eq.left, eq.right, eq.result):
                if len(term) <= 1:
                    continue
                lead = term[-1] if rule.reverse else term[0]
                leads.append(lead)
                if 0 not in domains[lead]:
                    continue
                before = tuple(sorted(domains[lead]))
                domains[lead].remove(0)
                after = tuple(sorted(domains[lead]))
                if not after:
                    return None, DomainEvent(kind="NO0", label="leading digits", constraint="multi-digit numbers cannot start with 0", contradiction=f"D[{self._literal(lead)}] empty")
                changes[lead] = (before, after)
                if len(after) == 1:
                    fixed[lead] = after[0]
        if not changes:
            return False, None
        unique_leads = ",".join(self._literal(ch) for ch in sorted(set(leads)))
        return True, DomainEvent(kind="NO0", label="leading digits", constraint=f"{unique_leads} != 0", changes=changes, fixed=fixed)

    def _alldifferent_evidence(self, domains: Domains) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        singles = [next(iter(v)) for v in domains.values() if len(v) == 1]
        if len(singles) != len(set(singles)):
            return None, DomainEvent(kind="ALLDIFF", label="injective digits", constraint="fixed digits must be unique", contradiction="duplicate fixed digit")
        fixed_digits = set(singles)
        if not fixed_digits:
            return False, None
        changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        fixed: Dict[str, int] = {}
        for ch, values in domains.items():
            if len(values) == 1:
                continue
            before = tuple(sorted(values))
            values.difference_update(fixed_digits)
            after = tuple(sorted(values))
            if not after:
                return None, DomainEvent(kind="ALLDIFF", label="injective digits", constraint="remove already fixed digits", contradiction=f"D[{self._literal(ch)}] empty")
            if after != before:
                changes[ch] = (before, after)
                if len(after) == 1:
                    fixed[ch] = after[0]
        if not changes:
            return False, None
        return True, DomainEvent(kind="ALLDIFF", label="injective digits", constraint="remove fixed digits from open domains", changes=changes, fixed=fixed)

    def _modular_evidence(self, eq: Equation, rule: Rule, domains: Domains) -> Tuple[Optional[bool], List[DomainEvent]]:
        if rule.name not in {"add", "add_p1", "add_m1", "mul", "mul_p1", "mul_m1"}:
            return False, []
        events: List[DomainEvent] = []
        changed_any = False
        for k in range(1, max(len(eq.left), len(eq.right), len(eq.result)) + 1):
            lp = self._suffix(eq.left, k, rule.reverse)
            rp = self._suffix(eq.right, k, rule.reverse)
            sp = self._suffix(eq.result, k, rule.reverse)
            scope = list(dict.fromkeys(lp + rp + sp))
            mod = 10**k

            def pred(local: Assignment, l=lp, r=rp, s=sp, m=mod) -> bool:
                value = rule.func(self._number(l, local, rule.reverse), self._number(r, local, rule.reverse))
                return value is not None and value % m == self._number(s, local, rule.reverse) % m

            label = eq.display()
            constraint = f"last{k}: {self._equation_constraint_text(Equation(eq.raw, lp, rp, eq.op, sp, eq.has_sign), rule)} mod {mod}"
            changed, event = self._project_evidence(scope, domains, pred, f"POS{k}", label, constraint)
            if changed is None:
                return None, [event] if event is not None else []
            if event is not None:
                events.append(event)
            changed_any = changed_any or bool(changed)
        return changed_any, events

    def _exact_evidence(self, eq: Equation, rule: Rule, domains: Domains) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        constraint = self._equation_constraint_text(eq, rule)
        return self._project_evidence(scope, domains, lambda local: self._matches(eq, rule, local), "EQ", eq.display(), constraint)

    def _global_exact_evidence(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        scope = list(dict.fromkeys(ch for eq in equations for ch in eq.left + eq.right + eq.result))
        limit = self.trace_config.max_global_support_space
        if self._domain_space_upper_bound({ch: domains[ch] for ch in scope}, limit) > limit:
            return False, None

        def pred(local: Assignment) -> bool:
            return all(self._matches(eq, combo[eq.op], local) for eq in equations)

        return self._project_evidence(scope, domains, pred, "GLOBAL", "GLOBAL all examples", "all equations together + injective digits")

    def _propagate_evidence(
        self,
        equations: Sequence[Equation],
        combo: Dict[str, Rule],
        domains: Domains,
        events: Optional[List[DomainEvent]],
    ) -> bool:
        while True:
            before = self._snapshot(domains)
            changed, event = self._leading_zero_evidence(equations, combo, domains)
            if changed is None:
                if events is not None and event is not None:
                    events.append(event)
                return False
            if events is not None and event is not None:
                events.append(event)

            changed, event = self._alldifferent_evidence(domains)
            if changed is None:
                if events is not None and event is not None:
                    events.append(event)
                return False
            if events is not None and event is not None:
                events.append(event)

            for eq in equations:
                rule = combo.get(eq.op)
                if rule is None:
                    continue
                changed_mod, mod_events = self._modular_evidence(eq, rule, domains)
                if changed_mod is None:
                    if events is not None:
                        events.extend(mod_events)
                    return False
                if events is not None:
                    events.extend(mod_events)

                changed_exact, exact_event = self._exact_evidence(eq, rule, domains)
                if changed_exact is None:
                    if events is not None and exact_event is not None:
                        events.append(exact_event)
                    return False
                if events is not None and exact_event is not None:
                    events.append(exact_event)

            after = self._snapshot(domains)
            if before == after:
                return True

    def _has_solution_from_domains(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains) -> bool:
        if time.time() > self._deadline:
            raise TimeoutError("solver timeout")
        local_domains = self._copy(domains)
        if not self._propagate_evidence(equations, combo, local_domains, None):
            return False
        unresolved = [ch for ch, values in local_domains.items() if len(values) > 1]
        if not unresolved:
            assignment = {ch: next(iter(values)) for ch, values in local_domains.items()}
            return self._verify(equations, combo, assignment)
        symbol = min(unresolved, key=lambda ch: (len(local_domains[ch]), ch))
        for digit in sorted(local_domains[symbol]):
            next_domains = self._copy(local_domains)
            next_domains[symbol] = {digit}
            if self._has_solution_from_domains(equations, combo, next_domains):
                return True
        return False

    def _build_mapping_proof(
        self,
        equations: Sequence[Equation],
        combo: Dict[str, Rule],
        symbols: Sequence[str],
        assignment: Assignment,
    ) -> Tuple[List[DomainEvent], Domains]:
        domains: Domains = {ch: set(range(10)) for ch in symbols}
        events: List[DomainEvent] = []

        if not self._propagate_evidence(equations, combo, domains, events):
            return events, domains

        if self.trace_config.include_global_support_pruning:
            changed_global, global_event = self._global_exact_evidence(equations, combo, domains)
            if changed_global is None:
                if global_event is not None:
                    events.append(global_event)
                return events, domains
            if global_event is not None:
                events.append(global_event)
                if not self._propagate_evidence(equations, combo, domains, events):
                    return events, domains

        branch_rounds = 0
        while any(len(values) > 1 for values in domains.values()) and branch_rounds < len(symbols):
            unresolved = [ch for ch, values in domains.items() if len(values) > 1]
            symbol = min(unresolved, key=lambda ch: (len(domains[ch]), ch))
            final_digit = assignment[symbol]
            before_values = tuple(sorted(domains[symbol]))
            if final_digit not in domains[symbol]:
                break

            rejects: List[str] = []
            for digit in before_values:
                if digit == final_digit:
                    continue
                test_domains = self._copy(domains)
                test_domains[symbol] = {digit}
                if not self._has_solution_from_domains(equations, combo, test_domains):
                    rejects.append(f"{self._literal(symbol)}={digit}")
                if len(rejects) >= self.trace_config.max_branch_rejects:
                    break

            domains[symbol] = {final_digit}
            event = DomainEvent(
                kind="BRANCH",
                label=f"choose {self._literal(symbol)}",
                constraint=f"{self._literal(symbol)} in {self._digits_dense(before_values)}",
                changes={symbol: (before_values, (final_digit,))},
                fixed={symbol: final_digit},
                rejects=rejects,
                keep=f"{self._literal(symbol)}={final_digit}",
            )
            events.append(event)
            if not self._propagate_evidence(equations, combo, domains, events):
                return events, domains
            branch_rounds += 1

        return events, domains

    def _format_domain_changes(self, changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]], fixed: Dict[str, int]) -> str:
        parts: List[str] = []
        for ch in sorted(changes):
            before, after = changes[ch]
            if ch in fixed:
                parts.append(f"{self._literal(ch)}={fixed[ch]}")
            else:
                parts.append(f"{self._literal(ch)}:{self._digits_dense(before)}->{self._digits_dense(after)}")
        return "; ".join(parts)

    def _format_domain_event(self, event: DomainEvent) -> str:
        if event.contradiction:
            return f"{event.kind} {event.label}: reject; {event.contradiction}."
        change_text = self._format_domain_changes(event.changes, event.fixed) if event.changes else "ok"
        if event.kind == "BRANCH":
            reject_text = "" if not event.rejects else "; reject " + ", ".join(event.rejects)
            keep_text = "" if event.keep is None else f"; keep {event.keep}"
            return f"BRANCH {event.constraint}{reject_text}{keep_text}; {change_text}."
        if event.constraint:
            return f"{event.kind} {event.label}: {event.constraint} => {change_text}."
        return f"{event.kind} {event.label}: {change_text}."

    def _select_domain_events(self, events: Sequence[DomainEvent], max_events: int) -> List[DomainEvent]:
        if len(events) <= max_events:
            return list(events)
        selected: List[DomainEvent] = []
        for event in events:
            important = event.kind in {"NO0", "GLOBAL", "BRANCH"} or bool(event.fixed) or bool(event.contradiction)
            if important:
                selected.append(event)
            if len(selected) >= max_events:
                return selected
        for event in events:
            if event not in selected:
                selected.append(event)
            if len(selected) >= max_events:
                break
        return selected

    def _operation_line_dense(self, rule: Rule, a: int, b: int) -> Tuple[str, Optional[int]]:
        value = rule.func(a, b)
        if value is None:
            return f"{self._rule_short(rule)}({a},{b}) invalid", None
        if rule.name == "add":
            return f"{a}+{b}={value}", value
        if rule.name == "add_p1":
            return f"{a}+{b}+1={value}", value
        if rule.name == "add_m1":
            return f"{a}+{b}-1={value}", value
        if rule.name == "mul":
            return f"{a}*{b}={value}", value
        if rule.name == "mul_p1":
            return f"{a}*{b}+1={value}", value
        if rule.name == "mul_m1":
            return f"{a}*{b}-1={value}", value
        if rule.name == "absdiff":
            return f"abs({a}-{b})={value}", value
        if rule.name == "sub_signed":
            return f"{a}-{b}={value}", value
        if rule.name == "concat_fwd":
            return f"cat({a},{b})={value}", value
        return f"{rule.name}({a},{b})={value}", value

    def _rule_signature_summary(self, op: str, equations: Sequence[Equation], candidates: Sequence[Rule], selected: Rule) -> str:
        len_sig = ",".join(f"{len(eq.left)}+{len(eq.right)}->{len(eq.result)}" for eq in equations)
        signed = "yes" if any(eq.has_sign for eq in equations) else "no"
        kept = ",".join(self._rule_short(rule) for rule in candidates)
        reject_groups: Dict[str, int] = {}
        for rule in self.rules:
            reason = self._structural_reject_reason(rule, equations)
            if reason is not None:
                reject_groups[reason] = reject_groups.get(reason, 0) + 1
        rejected = ",".join(f"{reason}:{count}" for reason, count in sorted(reject_groups.items())) or "none"
        return f"SIG {self._literal(op)} len[{len_sig}] signed={signed}; structural_keep={kept}; structural_reject={rejected}; select={self._rule_short(selected)}"

    def _render_training_solution(
        self,
        equations: Sequence[Equation],
        clean_target: str,
        target_left: str,
        target_op: str,
        target_right: str,
        combo: Dict[str, Rule],
        assignment: Assignment,
        answer: Optional[str],
        target_error: Optional[str],
        target_a: int,
        target_b: int,
        target_value: Optional[int],
        derivation_trace: Sequence[str],
        candidates: Dict[str, List[Rule]],
    ) -> List[str]:
        """Return token-dense CoT: signatures -> rule -> domain narrowing -> target."""
        del derivation_trace
        lines: List[str] = []
        add = lines.append

        by_op: Dict[str, List[Equation]] = {}
        for eq in equations:
            by_op.setdefault(eq.op, []).append(eq)

        symbols = sorted(assignment)
        map_events, proof_domains = self._build_mapping_proof(equations, combo, symbols, assignment)
        selected_events = self._select_domain_events(map_events, self.trace_config.max_domain_events)

        add("TYPE cryptarithm")
        add("M sym=digit; injective; no-leading-zero.")
        add("E " + " | ".join(self._literal(eq.display()) for eq in equations))
        add("T " + self._literal(clean_target))

        for op in sorted(combo):
            selected = combo[op]
            op_eqs = by_op.get(op, [])
            add(self._rule_signature_summary(op, op_eqs, candidates.get(op, []), selected))
            add(f"RULE {self._literal(op)}={self._rule_short(selected)}; {self._rule_action_text(selected)}.")
            for eq in op_eqs[:3]:
                add(f"C {eq.display()}: {self._equation_constraint_text(eq, selected)}.")

        add("MAP_SEARCH")
        add("D0 " + "; ".join(f"{self._literal(ch)}={self._digits_dense(range(10))}" for ch in symbols) + ".")
        for event in selected_events:
            add(self._format_domain_event(event))

        unresolved = [ch for ch in symbols if len(proof_domains.get(ch, set())) > 1]
        if unresolved:
            add("OPEN " + "; ".join(f"{self._literal(ch)}={self._digits_dense(proof_domains[ch])}" for ch in unresolved) + ".")
        add("MAP " + "; ".join(f"{self._literal(ch)}={assignment[ch]}" for ch in symbols) + ".")

        add("VERIFY")
        for eq in equations:
            rule = combo[eq.op]
            a = self._number(eq.left, assignment, rule.reverse)
            b = self._number(eq.right, assignment, rule.reverse)
            op_text, value = self._operation_line_dense(rule, a, b)
            encoded = "<invalid>" if value is None else self._encode_value(value, eq.op, rule, assignment)
            expected = f"{eq.op}{eq.result}" if eq.has_sign else eq.result
            status = "ok" if encoded == expected else "bad"
            add(f"V {self._literal(eq.display())}: A={a}; B={b}; {op_text}; enc={self._literal(encoded or '<invalid>')} expect={self._literal(expected)} {status}.")

        add("TARGET")
        target_rule = combo[target_op]
        if target_error or answer is None or target_value is None:
            add(f"TGT {self._literal(clean_target)}: A={target_a}; B={target_b}; error={target_error or 'invalid'}.")
            add("FINAL <no valid answer>")
        else:
            op_text, _ = self._operation_line_dense(target_rule, target_a, target_b)
            add(f"TGT {self._literal(clean_target)}: A={target_a}; B={target_b}; {op_text}; enc={self._literal(answer)}.")
            add(f"FINAL {answer}")

        text = "\n".join(lines)
        if len(text) <= self.trace_config.max_solution_chars:
            return lines

        compact: List[str] = []
        compact.append("TYPE cryptarithm")
        compact.append("M sym=digit; injective; no-leading-zero.")
        compact.append("E " + " | ".join(self._literal(eq.display()) for eq in equations))
        compact.append("T " + self._literal(clean_target))
        for op in sorted(combo):
            selected = combo[op]
            compact.append(f"RULE {self._literal(op)}={self._rule_short(selected)}; {self._rule_action_text(selected)}.")
        compact.append("MAP_SEARCH compressed: structural signatures + domain propagation + branch rejection produced single verified map.")
        for event in self._select_domain_events(map_events, 20):
            compact.append(self._format_domain_event(event))
        compact.append("MAP " + "; ".join(f"{self._literal(ch)}={assignment[ch]}" for ch in symbols) + ".")
        compact.append("TARGET")
        if target_error or answer is None or target_value is None:
            compact.append(f"TGT {self._literal(clean_target)}: A={target_a}; B={target_b}; error={target_error or 'invalid'}.")
            compact.append("FINAL <no valid answer>")
        else:
            op_text, _ = self._operation_line_dense(combo[target_op], target_a, target_b)
            compact.append(f"TGT {self._literal(clean_target)}: A={target_a}; B={target_b}; {op_text}; enc={self._literal(answer)}.")
            compact.append(f"FINAL {answer}")
        return compact

    def solve(self, examples_text: str, target_text: str, timeout_seconds: float = 30.0) -> Dict[str, Any]:
        self._deadline = time.time() + timeout_seconds
        self.stats = SearchStats()
        internal_trace = Trace(TraceConfig(max_steps=100000, max_solution_chars=10**9, max_examples_per_step=0, include_rejected_hypotheses=False, include_search_branches=False))
        started = time.time()
        try:
            equations, guessed_ops = self._parse_examples(examples_text)
            if not equations:
                return {"answer": None, "debug": ["No equations were parsed."], "trace": []}
            target = self._parse_target(target_text, guessed_ops)
            if target is None:
                return {"answer": None, "debug": ["Target expression could not be parsed."], "trace": []}
            clean_target, target_left, target_op, target_right = target

            symbols = self._all_symbols(equations, target_left, target_right)
            if len(symbols) > 10:
                return {"answer": None, "debug": [f"Too many distinct symbols ({len(symbols)} > 10)."], "trace": []}
            domains: Domains = {ch: set(range(10)) for ch in symbols}

            by_op: Dict[str, List[Equation]] = {}
            for eq in equations:
                by_op.setdefault(eq.op, []).append(eq)

            candidates: Dict[str, List[Rule]] = {}
            for op, op_eqs in by_op.items():
                kept = [rule for rule in self.rules if self._structurally_possible(rule, op_eqs)]
                kept.sort(key=lambda r: (r.tier, r.name, r.orientation))
                if not kept:
                    return {"answer": None, "debug": [f"No structural hypothesis remains for operator `{op}`."], "trace": []}
                candidates[op] = kept

            if target_op not in candidates:
                return {"answer": None, "debug": [f"Target operator `{target_op}` was not found among examples."], "trace": []}

            ordered_ops = sorted(candidates, key=lambda op: (len(candidates[op]), -len(by_op[op]), op))
            assignment, combo = self._search_rules(ordered_ops, candidates, equations, domains, {}, internal_trace, 0)
            elapsed = round(time.time() - started, 4)
            if assignment is None or combo is None:
                return {"answer": None, "debug": ["No solution survived all hypotheses and constraints."], "trace": [], "stats": self.stats.__dict__, "elapsed_seconds": elapsed}

            answer, error, a, b, value = self._encode_target(target_left, target_op, target_right, combo[target_op], assignment)
            solution = self._render_training_solution(equations, clean_target, target_left, target_op, target_right, combo, assignment, answer, error, a, b, value, internal_trace.finish(), candidates)
            return {
                "answer": answer,
                "debug": solution,
                "trace": solution,
                "solution": "\n".join(solution),
                "mapping": assignment,
                "rules": {op: r.desc for op, r in combo.items()},
                "rule_source": "direct_operator_rule_with_csp_assignment",
                "training_category": "equations_transformation.cryptarithm_direct_operator",
                "metadata": {
                    "target_operator_seen_in_examples": True,
                    "uses_fallback_inference": False,
                    "target_operator": target_op,
                    "symbols_count": len(symbols),
                },
                "stats": self.stats.__dict__,
                "elapsed_seconds": elapsed,
            }
        except TimeoutError:
            return {"answer": None, "debug": ["Solver timeout."], "trace": [], "stats": self.stats.__dict__, "elapsed_seconds": round(time.time() - started, 4)}
        except Exception as exc:
            return {"answer": None, "debug": [f"Execution error: {type(exc).__name__}: {exc}"], "trace": [], "stats": self.stats.__dict__, "elapsed_seconds": round(time.time() - started, 4)}


CryptarithmCSPSolver = DeductiveCryptarithmSolver


if __name__ == "__main__":
    examples = """
'[+'' = !``
|[*|( = ``!`
::+`" = "/`
""".strip()
    target = "Now, determine the result for: `!*/["
    result = DeductiveCryptarithmSolver().solve(examples, target, timeout_seconds=30)
    print("ANSWER:", result["answer"])
    print("\n".join(result["debug"][:80]))
