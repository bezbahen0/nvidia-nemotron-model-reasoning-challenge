from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from itertools import combinations
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



class _CryptarithmCore:

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
    def _bounds(length: int) -> Tuple[int, int]:
        return (0, 9) if length <= 1 else (10 ** (length - 1), 10**length - 1)

    @staticmethod
    def _domain(values: Iterable[int]) -> str:
        return "\\{" + ",".join(str(v) for v in sorted(values)) + "\\}"

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

    def _all_symbols(self, equations: Sequence[Equation], target_left: str, target_right: str) -> List[str]:
        counts: Dict[str, int] = {}
        for eq in equations:
            for ch in eq.left + eq.right + eq.result:
                counts[ch] = counts.get(ch, 0) + 1
        for ch in target_left + target_right:
            counts[ch] = counts.get(ch, 0) + 1
        return sorted(counts, key=lambda ch: (-counts[ch], ch))

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

    @staticmethod
    def _literal(text: str) -> str:
        return repr(text)

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
        """Propagate AllDifferent by fixed digits and small Hall sets.

        This is intentionally part of the training proof, not just validation:
        it shows the model that a map can be derived from set constraints before search.
        """
        singles = [next(iter(v)) for v in domains.values() if len(v) == 1]
        if len(singles) != len(set(singles)):
            return None, DomainEvent(kind="ALLDIFF", label="injective digits", constraint="fixed digits must be unique", contradiction="duplicate fixed digit")

        fixed_digits = set(singles)
        if fixed_digits:
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
            if changes:
                return True, DomainEvent(kind="ALLDIFF", label="fixed", constraint="fixed digits cannot appear in other domains", changes=changes, fixed=fixed)

        # Hall-set / naked-pair pruning. For <=10 symbols, checking small subsets is cheap.
        open_symbols = [ch for ch, values in domains.items() if 1 < len(values) <= 4]
        for size in range(2, min(4, len(open_symbols)) + 1):
            for subset in combinations(open_symbols, size):
                union: Set[int] = set()
                for ch in subset:
                    union.update(domains[ch])
                if len(union) != size:
                    continue
                subset_set = set(subset)
                changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
                fixed: Dict[str, int] = {}
                for ch, values in domains.items():
                    if ch in subset_set or len(values) == 1:
                        continue
                    if values.isdisjoint(union):
                        continue
                    before = tuple(sorted(values))
                    values.difference_update(union)
                    after = tuple(sorted(values))
                    if not after:
                        return None, DomainEvent(kind="HALL", label=",".join(self._literal(c) for c in subset), constraint=f"{','.join(self._literal(c) for c in subset)} reserve {self._digits_dense(union)}", contradiction=f"D[{self._literal(ch)}] empty")
                    if after != before:
                        changes[ch] = (before, after)
                        if len(after) == 1:
                            fixed[ch] = after[0]
                if changes:
                    return True, DomainEvent(kind="HALL", label=",".join(self._literal(c) for c in subset), constraint=f"{','.join(self._literal(c) for c in subset)} reserve {self._digits_dense(union)}", changes=changes, fixed=fixed)

        return False, None

    def _lsd_symbols(self, term: str, rule: Rule) -> List[str]:
        """Symbols from least-significant column to most-significant column."""
        return list(term if rule.reverse else term[::-1])

    def _symbol_at(self, symbols: Sequence[str], index: int) -> str:
        return self._literal(symbols[index]) if index < len(symbols) else "0"

    def _column_constraint_text(self, eq: Equation, rule: Rule, col: int) -> str:
        left = self._lsd_symbols(eq.left, rule)
        right = self._lsd_symbols(eq.right, rule)
        result = self._lsd_symbols(eq.result, rule)
        r = self._symbol_at(result, col)
        if rule.name in {"add", "add_p1", "add_m1"}:
            a = self._symbol_at(left, col)
            b = self._symbol_at(right, col)
            delta = "+1" if rule.name == "add_p1" and col == 0 else "-1" if rule.name == "add_m1" and col == 0 else ""
            c_in = "c0=0" if col == 0 and not delta else ""
            carry = f"c{col}"
            carry_next = f"c{col+1}"
            prefix = f"{a}+{b}{delta}"
            if col > 0:
                prefix += f"+{carry}"
            tail = f"={r}+10*{carry_next}"
            note = f", {c_in}" if c_in else ""
            return f"col{col}: {prefix}{tail}{note}"
        if rule.name in {"mul", "mul_p1", "mul_m1"}:
            terms: List[str] = []
            for i, a_sym in enumerate(left):
                j = col - i
                if 0 <= j < len(right):
                    terms.append(f"{self._literal(a_sym)}*{self._literal(right[j])}")
            if not terms:
                terms = ["0"]
            delta = "+1" if rule.name == "mul_p1" and col == 0 else "-1" if rule.name == "mul_m1" and col == 0 else ""
            carry = f"c{col}"
            carry_next = f"c{col+1}"
            lhs = "+".join(terms)
            if col > 0:
                lhs += f"+{carry}"
            lhs += delta
            return f"col{col}: {lhs}={r}+10*{carry_next}"
        if rule.name == "sub_signed":
            a = self._symbol_at(left, col)
            b = self._symbol_at(right, col)
            sign = "-" if eq.has_sign else ""
            # b_i are borrow variables; this text teaches the borrow recurrence.
            return f"col{col}: {a}-{b}-b{col}={sign}{r}-10*b{col+1}"
        return self._equation_constraint_text(eq, rule)

    def _concat_evidence(self, eq: Equation, rule: Rule, domains: Domains) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        expected = eq.right + eq.left if rule.reverse else eq.left + eq.right
        if eq.has_sign:
            return None, DomainEvent(kind="CAT", label=eq.display(), constraint="concat cannot have signed rhs", contradiction="signed concat")
        if eq.result != expected:
            return None, DomainEvent(kind="CAT", label=eq.display(), constraint=f"expected result pattern {self._literal(expected)}", contradiction=f"actual {self._literal(eq.result)}")
        return False, DomainEvent(kind="CAT", label=eq.display(), constraint=f"{self._rule_short(rule)} requires result={self._literal(expected)}", changes={})

    def _event_key(self, event: DomainEvent) -> Tuple[Any, ...]:
        changes_key = tuple((ch, before, after) for ch, (before, after) in sorted(event.changes.items()))
        fixed_key = tuple(sorted(event.fixed.items()))
        return (event.kind, event.label, event.constraint, changes_key, fixed_key, tuple(event.rejects), event.keep, event.contradiction)

    def _select_domain_events(self, events: Sequence[DomainEvent], max_events: int) -> List[DomainEvent]:
        deduped: List[DomainEvent] = []
        seen: Set[Tuple[Any, ...]] = set()
        for event in events:
            key = self._event_key(event)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)
        if len(deduped) <= max_events:
            return deduped
        selected: List[DomainEvent] = []
        selected_keys: Set[Tuple[Any, ...]] = set()
        for event in deduped:
            important = event.kind in {"NO0", "GLOBAL", "HALL", "CAT"} or event.kind.startswith("COL") or bool(event.fixed) or bool(event.contradiction)
            if important:
                selected.append(event)
                selected_keys.add(self._event_key(event))
            if len(selected) >= max_events:
                return selected
        for event in deduped:
            key = self._event_key(event)
            if key not in selected_keys:
                selected.append(event)
                selected_keys.add(key)
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

    def _verify(self, equations: Sequence[Equation], combo: Dict[str, Rule], assignment: Assignment) -> bool:
        return all(self._matches(eq, combo[eq.op], assignment) for eq in equations)

@dataclass
class _ColumnState:
    symbol_domains: Domains
    aux_domains: Dict[str, Set[int]] = field(default_factory=dict)



class _ColumnDeductiveSolver(_CryptarithmCore):

    def __init__(self, trace_config: Optional[TraceConfig] = None):
        cfg = trace_config or TraceConfig()
        cfg.include_global_support_pruning = False
        super().__init__(cfg)

    @staticmethod
    def _is_aux(name: str) -> bool:
        return name.startswith(("c:", "b:", "case:"))

    def _var_text(self, name: str) -> str:
        if self._is_aux(name):
            return name
        return self._literal(name)

    def _state_snapshot(self, state: _ColumnState) -> Dict[str, Tuple[int, ...]]:
        out = {ch: tuple(sorted(vals)) for ch, vals in state.symbol_domains.items()}
        out.update({name: tuple(sorted(vals)) for name, vals in state.aux_domains.items()})
        return out

    def _ensure_aux(self, state: _ColumnState, name: str, values: Iterable[int]) -> None:
        allowed = set(values)
        if name not in state.aux_domains:
            state.aux_domains[name] = allowed
        else:
            state.aux_domains[name].intersection_update(allowed)

    def _aux_values(self, state: _ColumnState, name: str) -> Set[int]:
        return state.aux_domains[name]

    def _carry_bounds_for_mul(self, eq: Equation, rule: Rule) -> range:
        # Conservative but still small enough for two-digit examples; covers +/-1 offset.
        max_terms = max(1, min(len(eq.left), len(eq.right)))
        max_sum = 81 * max_terms + 20
        return range(-1, max(10, max_sum // 10 + 3))

    def _init_aux_for_equation(self, eq_idx: int, eq: Equation, rule: Rule, state: _ColumnState) -> int:
        if rule.name in {"add", "add_p1", "add_m1"}:
            max_cols = max(len(eq.left), len(eq.right), len(eq.result))
            first = 1 if rule.name == "add_p1" else -1 if rule.name == "add_m1" else 0
            self._ensure_aux(state, f"c:{eq_idx}:0", {first})
            for col in range(1, max_cols):
                self._ensure_aux(state, f"c:{eq_idx}:{col}", {-1, 0, 1})
            self._ensure_aux(state, f"c:{eq_idx}:{max_cols}", {0})
            return max_cols
        if rule.name in {"mul", "mul_p1", "mul_m1"}:
            max_cols = max(len(eq.result), len(eq.left) + len(eq.right))
            first = 1 if rule.name == "mul_p1" else -1 if rule.name == "mul_m1" else 0
            self._ensure_aux(state, f"c:{eq_idx}:0", {first})
            carry_range = self._carry_bounds_for_mul(eq, rule)
            for col in range(1, max_cols):
                self._ensure_aux(state, f"c:{eq_idx}:{col}", carry_range)
            self._ensure_aux(state, f"c:{eq_idx}:{max_cols}", {0})
            return max_cols
        if rule.name == "sub_signed":
            max_cols = max(len(eq.left), len(eq.right), len(eq.result))
            self._ensure_aux(state, f"b:{eq_idx}:0", {0})
            for col in range(1, max_cols):
                self._ensure_aux(state, f"b:{eq_idx}:{col}", {0, 1})
            self._ensure_aux(state, f"b:{eq_idx}:{max_cols}", {0})
            return max_cols
        if rule.name == "absdiff":
            max_cols = max(len(eq.left), len(eq.right), len(eq.result))
            self._ensure_aux(state, f"case:{eq_idx}", {0, 1})  # 0: left-right; 1: right-left
            self._ensure_aux(state, f"b:{eq_idx}:0", {0})
            for col in range(1, max_cols):
                self._ensure_aux(state, f"b:{eq_idx}:{col}", {0, 1})
            self._ensure_aux(state, f"b:{eq_idx}:{max_cols}", {0})
            return max_cols
        return 0

    def _digit_expr_terms(self, symbols: Sequence[str], col: int) -> Optional[str]:
        return symbols[col] if col < len(symbols) else None

    def _enumerate_symbol_assignments(
        self,
        symbols: Sequence[str],
        domains: Domains,
    ) -> Iterable[Dict[str, int]]:
        ordered = sorted(dict.fromkeys(symbols), key=lambda ch: (len(domains[ch]), ch))
        assignment: Dict[str, int] = {}
        used: Set[int] = set()

        def bt(i: int):
            if time.time() > self._deadline:
                raise TimeoutError("solver timeout")
            if i == len(ordered):
                yield dict(assignment)
                return
            ch = ordered[i]
            for digit in sorted(domains[ch]):
                if digit in used:
                    continue
                assignment[ch] = digit
                used.add(digit)
                yield from bt(i + 1)
                used.remove(digit)
                del assignment[ch]

        yield from bt(0)

    def _project_column_real(
        self,
        eq_idx: int,
        eq: Equation,
        rule: Rule,
        col: int,
        state: _ColumnState,
    ) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        if rule.name == "concat_fwd":
            ok, event = self._concat_evidence(eq, rule, state.symbol_domains)
            return ok, event

        left = self._lsd_symbols(eq.left, rule)
        right = self._lsd_symbols(eq.right, rule)
        result = self._lsd_symbols(eq.result, rule)
        r_sym = result[col] if col < len(result) else None

        support_symbols: Dict[str, Set[int]] = {}
        support_aux: Dict[str, Set[int]] = {}
        symbol_scope: List[str] = []
        aux_scope: List[str] = []
        constraint_text = self._column_constraint_text(eq, rule, col)
        kind = f"COL{col}"

        def add_symbol(ch: Optional[str]) -> None:
            if ch is not None and ch not in symbol_scope:
                symbol_scope.append(ch)
                support_symbols.setdefault(ch, set())

        def add_aux(name: str) -> None:
            if name not in aux_scope:
                aux_scope.append(name)
                support_aux.setdefault(name, set())

        if rule.name in {"add", "add_p1", "add_m1"}:
            a = self._digit_expr_terms(left, col)
            b = self._digit_expr_terms(right, col)
            add_symbol(a); add_symbol(b); add_symbol(r_sym)
            c_in, c_out = f"c:{eq_idx}:{col}", f"c:{eq_idx}:{col+1}"
            add_aux(c_in); add_aux(c_out)
            for local in self._enumerate_symbol_assignments(symbol_scope, state.symbol_domains):
                av = 0 if a is None else local[a]
                bv = 0 if b is None else local[b]
                rv = 0 if r_sym is None else local[r_sym]
                for cin in sorted(self._aux_values(state, c_in)):
                    cout_num = av + bv + cin - rv
                    if cout_num % 10 != 0:
                        continue
                    cout = cout_num // 10
                    if cout not in state.aux_domains[c_out]:
                        continue
                    for ch in symbol_scope:
                        support_symbols[ch].add(local[ch])
                    support_aux[c_in].add(cin)
                    support_aux[c_out].add(cout)
        elif rule.name in {"mul", "mul_p1", "mul_m1"}:
            terms: List[Tuple[str, str]] = []
            for i, a_ch in enumerate(left):
                j = col - i
                if 0 <= j < len(right):
                    terms.append((a_ch, right[j]))
                    add_symbol(a_ch); add_symbol(right[j])
            add_symbol(r_sym)
            c_in, c_out = f"c:{eq_idx}:{col}", f"c:{eq_idx}:{col+1}"
            add_aux(c_in); add_aux(c_out)
            for local in self._enumerate_symbol_assignments(symbol_scope, state.symbol_domains):
                product_sum = sum(local[a_ch] * local[b_ch] for a_ch, b_ch in terms)
                rv = 0 if r_sym is None else local[r_sym]
                for cin in sorted(self._aux_values(state, c_in)):
                    cout_num = product_sum + cin - rv
                    if cout_num % 10 != 0:
                        continue
                    cout = cout_num // 10
                    if cout not in state.aux_domains[c_out]:
                        continue
                    for ch in symbol_scope:
                        support_symbols[ch].add(local[ch])
                    support_aux[c_in].add(cin)
                    support_aux[c_out].add(cout)
        elif rule.name == "sub_signed":
            # Convert signed negative result to ordinary positive subtraction by swapping sides.
            minuend = right if eq.has_sign else left
            subtrahend = left if eq.has_sign else right
            m = self._digit_expr_terms(minuend, col)
            s = self._digit_expr_terms(subtrahend, col)
            add_symbol(m); add_symbol(s); add_symbol(r_sym)
            b_in, b_out = f"b:{eq_idx}:{col}", f"b:{eq_idx}:{col+1}"
            add_aux(b_in); add_aux(b_out)
            for local in self._enumerate_symbol_assignments(symbol_scope, state.symbol_domains):
                mv = 0 if m is None else local[m]
                sv = 0 if s is None else local[s]
                rv = 0 if r_sym is None else local[r_sym]
                for bin_ in sorted(self._aux_values(state, b_in)):
                    # mv - sv - bin = rv - 10*bout
                    bout_num = rv - mv + sv + bin_
                    if bout_num % 10 != 0:
                        continue
                    bout = bout_num // 10
                    if bout not in state.aux_domains[b_out]:
                        continue
                    for ch in symbol_scope:
                        support_symbols[ch].add(local[ch])
                    support_aux[b_in].add(bin_)
                    support_aux[b_out].add(bout)
        elif rule.name == "absdiff":
            add_symbol(self._digit_expr_terms(left, col))
            add_symbol(self._digit_expr_terms(right, col))
            add_symbol(r_sym)
            b_in, b_out, case = f"b:{eq_idx}:{col}", f"b:{eq_idx}:{col+1}", f"case:{eq_idx}"
            add_aux(b_in); add_aux(b_out); add_aux(case)
            for local in self._enumerate_symbol_assignments(symbol_scope, state.symbol_domains):
                rv = 0 if r_sym is None else local[r_sym]
                for case_v in sorted(state.aux_domains[case]):
                    minuend = left if case_v == 0 else right
                    subtrahend = right if case_v == 0 else left
                    m = self._digit_expr_terms(minuend, col)
                    s = self._digit_expr_terms(subtrahend, col)
                    mv = 0 if m is None else local[m]
                    sv = 0 if s is None else local[s]
                    for bin_ in sorted(self._aux_values(state, b_in)):
                        bout_num = rv - mv + sv + bin_
                        if bout_num % 10 != 0:
                            continue
                        bout = bout_num // 10
                        if bout not in state.aux_domains[b_out]:
                            continue
                        for ch in symbol_scope:
                            support_symbols[ch].add(local[ch])
                        support_aux[b_in].add(bin_)
                        support_aux[b_out].add(bout)
                        support_aux[case].add(case_v)
        else:
            return False, None

        if any(not support_symbols[ch] for ch in symbol_scope) or any(not support_aux[name] for name in aux_scope):
            return None, DomainEvent(kind=kind, label=eq.display(), constraint=constraint_text, contradiction="no_supported_column_assignment")

        changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        fixed: Dict[str, int] = {}
        for ch in symbol_scope:
            before = tuple(sorted(state.symbol_domains[ch]))
            state.symbol_domains[ch].intersection_update(support_symbols[ch])
            after = tuple(sorted(state.symbol_domains[ch]))
            if not after:
                return None, DomainEvent(kind=kind, label=eq.display(), constraint=constraint_text, contradiction=f"D[{self._literal(ch)}] empty")
            if after != before:
                changes[ch] = (before, after)
                if len(after) == 1:
                    fixed[ch] = after[0]
        for name in aux_scope:
            before = tuple(sorted(state.aux_domains[name]))
            state.aux_domains[name].intersection_update(support_aux[name])
            after = tuple(sorted(state.aux_domains[name]))
            if not after:
                return None, DomainEvent(kind=kind, label=eq.display(), constraint=constraint_text, contradiction=f"D[{name}] empty")
            if after != before:
                changes[name] = (before, after)
                if len(after) == 1:
                    fixed[name] = after[0]
        if not changes:
            return False, None
        return True, DomainEvent(kind=kind, label=eq.display(), constraint=constraint_text, changes=changes, fixed=fixed)

    def _project_exact_symbols_only(
        self,
        eq: Equation,
        rule: Rule,
        state: _ColumnState,
    ) -> Tuple[Optional[bool], Optional[DomainEvent]]:
        # Use exact equation as a final local proof check.  This is still a projection over
        # the equation's symbols, not a global digit search.
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        constraint = self._equation_constraint_text(eq, rule)
        return self._project_evidence(scope, state.symbol_domains, lambda local: self._matches(eq, rule, local), "EQ", eq.display(), constraint)

    def _propagate_deductive(
        self,
        equations: Sequence[Equation],
        combo: Dict[str, Rule],
        state: _ColumnState,
        events: Optional[List[DomainEvent]],
        use_exact_projection: bool = True,
    ) -> bool:
        # Initialize aux domains once.
        for idx, eq in enumerate(equations):
            rule = combo[eq.op]
            self._init_aux_for_equation(idx, eq, rule, state)

        while True:
            before = self._state_snapshot(state)
            changed, event = self._leading_zero_evidence(equations, combo, state.symbol_domains)
            if changed is None:
                if events is not None and event is not None: events.append(event)
                return False
            if events is not None and event is not None: events.append(event)

            changed, event = self._alldifferent_evidence(state.symbol_domains)
            if changed is None:
                if events is not None and event is not None: events.append(event)
                return False
            if events is not None and event is not None: events.append(event)

            for idx, eq in enumerate(equations):
                rule = combo[eq.op]
                if rule.name == "concat_fwd":
                    changed_col, event = self._concat_evidence(eq, rule, state.symbol_domains)
                    if changed_col is None:
                        if events is not None and event is not None: events.append(event)
                        return False
                    if events is not None and event is not None: events.append(event)
                    continue
                max_cols = self._init_aux_for_equation(idx, eq, rule, state)
                for col in range(max_cols):
                    changed_col, event = self._project_column_real(idx, eq, rule, col, state)
                    if changed_col is None:
                        if events is not None and event is not None: events.append(event)
                        return False
                    if events is not None and event is not None: events.append(event)
                if use_exact_projection and rule.name not in {"concat_fwd"}:
                    changed_exact, event = self._project_exact_symbols_only(eq, rule, state)
                    if changed_exact is None:
                        if events is not None and event is not None: events.append(event)
                        return False
                    if events is not None and event is not None: events.append(event)

            after = self._state_snapshot(state)
            if before == after:
                return True

    def _assignment_from_domains(self, domains: Domains) -> Optional[Assignment]:
        if any(len(values) != 1 for values in domains.values()):
            return None
        assignment = {ch: next(iter(values)) for ch, values in domains.items()}
        if len(set(assignment.values())) != len(assignment):
            return None
        return assignment

    def _enumerate_rule_combos(self, ordered_ops: Sequence[str], candidates: Dict[str, List[Rule]]) -> Iterable[Dict[str, Rule]]:
        combo: Dict[str, Rule] = {}
        def bt(i: int):
            if time.time() > self._deadline:
                raise TimeoutError("solver timeout")
            if i == len(ordered_ops):
                yield dict(combo)
                return
            op = ordered_ops[i]
            for rule in candidates[op]:
                combo[op] = rule
                yield from bt(i + 1)
                del combo[op]
        yield from bt(0)

    def solve(self, examples_text: str, target_text: str, timeout_seconds: float = 30.0) -> Dict[str, Any]:
        self._deadline = time.time() + timeout_seconds
        self.stats = SearchStats()
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

            solved: List[Tuple[str, Assignment, Dict[str, Rule], List[DomainEvent], _ColumnState, Tuple[Any, ...]]] = []
            for combo in self._enumerate_rule_combos(ordered_ops, candidates):
                state = _ColumnState({ch: set(range(10)) for ch in symbols})
                events: List[DomainEvent] = []
                if not self._propagate_deductive(equations, combo, state, events, use_exact_projection=True):
                    continue
                assignment = self._assignment_from_domains(state.symbol_domains)
                if assignment is None:
                    continue
                if not self._verify(equations, combo, assignment):
                    continue
                answer, error, a, b, value = self._encode_target(target_left, target_op, target_right, combo[target_op], assignment)
                if answer is None or error is not None:
                    continue
                key = (answer, tuple(sorted(assignment.items())), tuple(sorted((op, self._rule_short(rule)) for op, rule in combo.items())))
                solved.append((answer, assignment, combo, events, state, (a, b, value, key)))
                if len(solved) > 1:
                    break

            elapsed = round(time.time() - started, 4)
            if not solved:
                return {
                    "answer": None,
                    "debug": ["REJECT_DEDUCTIVE: no rule combo produced a singleton digit map by column propagation only."],
                    "trace": ["REJECT_DEDUCTIVE: no rule combo produced a singleton digit map by column propagation only."],
                    "solution": "REJECT_DEDUCTIVE: no rule combo produced a singleton digit map by column propagation only.",
                    "metadata": {"deductive_only": True, "uses_digit_search": False, "rejected_for_training": True},
                    "stats": self.stats.__dict__,
                    "elapsed_seconds": elapsed,
                }
            if len(solved) > 1:
                answers = {item[0] for item in solved}
                return {
                    "answer": None,
                    "debug": [f"REJECT_DEDUCTIVE_AMBIGUOUS: multiple singleton rule/map solutions survived; answers={sorted(answers)}."],
                    "trace": [f"REJECT_DEDUCTIVE_AMBIGUOUS: multiple singleton rule/map solutions survived; answers={sorted(answers)}."],
                    "solution": f"REJECT_DEDUCTIVE_AMBIGUOUS: multiple singleton rule/map solutions survived; answers={sorted(answers)}.",
                    "metadata": {"deductive_only": True, "uses_digit_search": False, "rejected_for_training": True},
                    "stats": self.stats.__dict__,
                    "elapsed_seconds": elapsed,
                }

            answer, assignment, combo, events, state, extra = solved[0]
            a, b, value, _key = extra
            lines = self._render_deductive_solution(equations, clean_target, target_left, target_op, target_right, combo, assignment, answer, None, a, b, value, candidates, events, state)
            return {
                "answer": answer,
                "debug": lines,
                "trace": lines,
                "solution": "\n".join(lines),
                "mapping": assignment,
                "rules": {op: r.desc for op, r in combo.items()},
                "rule_source": "deductive_column_propagation_only",
                "training_category": "equations_transformation.cryptarithm_deductive_column_only",
                "metadata": {
                    "deductive_only": True,
                    "uses_digit_search": False,
                    "uses_global_projection": False,
                    "uses_search_fallback": False,
                    "target_operator_seen_in_examples": True,
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



class CryptarithmSolver(_ColumnDeductiveSolver):

    def _var_text(self, name: str) -> str:
        if name.startswith("c:"):
            _, eq, col = name.split(":")
            return f"C{eq}_{col}"
        if name.startswith("b:"):
            _, eq, col = name.split(":")
            return f"B{eq}_{col}"
        if name.startswith("case:"):
            _, eq = name.split(":")
            return f"K{eq}"
        return self._literal(name)

    @staticmethod
    def _digits_dense(values: Iterable[int]) -> str:
        vals = sorted(set(values))
        if not vals:
            return "{}"
        ranges = []
        start = prev = vals[0]
        for v in vals[1:]:
            if v == prev + 1:
                prev = v
            else:
                ranges.append((start, prev))
                start = prev = v
        ranges.append((start, prev))
        parts = []
        for a, b in ranges:
            if a == b:
                parts.append(str(a))
            elif b == a + 1:
                parts.append(f"{a},{b}")
            else:
                parts.append(f"{a}..{b}")
        return "{" + ",".join(parts) + "}"

    def _compact_constraint(self, text: str) -> str:
        # Keep equations mathematical but reduce prose/token overhead.
        text = text.replace("col", "k")
        text = re.sub(r"c(\d+)", r"c\1", text)
        text = re.sub(r"b(\d+)", r"b\1", text)
        return text

    def _format_domain_changes(self, changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]], fixed: Dict[str, int]) -> str:
        parts: List[str] = []
        for ch in sorted(changes):
            before, after = changes[ch]
            label = self._var_text(ch)
            before_set = set(before)
            if ch in fixed:
                parts.append(f"{label}={fixed[ch]}")
            elif len(before) == 10 and before_set == set(range(10)):
                parts.append(f"{label}∈{self._digits_dense(after)}")
            elif len(before) > 12:
                parts.append(f"{label}:|{len(before)}|→{self._digits_dense(after)}")
            else:
                parts.append(f"{label}:{self._digits_dense(before)}→{self._digits_dense(after)}")
        return "; ".join(parts)

    def _format_domain_event(self, event: DomainEvent) -> str:
        label = event.label
        if event.contradiction:
            return f"{event.kind} {label}: ⊥ {event.contradiction}."
        change_text = self._format_domain_changes(event.changes, event.fixed) if event.changes else "ok"
        if event.constraint:
            constraint = self._compact_constraint(event.constraint)
            return f"{event.kind} {label}: {constraint} ⇒ {change_text}."
        return f"{event.kind} {label}: {change_text}."

    def _compact_rule_signature_summary(self, op: str, equations: Sequence[Equation], candidates: Sequence[Rule], selected: Rule) -> str:
        lens = sorted({f"{len(eq.left)}+{len(eq.right)}→{len(eq.result)}" for eq in equations})
        signed = "Y" if any(eq.has_sign for eq in equations) else "N"
        # Keep the selected rule and only aggregate rejects by reason.
        rejects: Dict[str, int] = {}
        for rule in self.rules:
            if rule in candidates:
                continue
            reason = self._structural_reject_reason(rule, equations)
            if reason:
                rejects[reason] = rejects.get(reason, 0) + 1
        rej = ",".join(f"{k}:{v}" for k, v in sorted(rejects.items())) or "none"
        return f"SIG {self._literal(op)} len={','.join(lens)} sign={signed}; keep={len(candidates)}; rej={rej}; choose={self._rule_short(selected)}."

    def _operation_line_dense(self, rule: Rule, a: int, b: int) -> Tuple[str, Optional[int]]:
        text, value = super()._operation_line_dense(rule, a, b)
        text = text.replace("cat", "concat")
        return text, value

    def _render_deductive_solution(
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
        candidates: Dict[str, List[Rule]],
        events: Sequence[DomainEvent],
        proof_state: _ColumnState,
    ) -> List[str]:
        by_op: Dict[str, List[Equation]] = {}
        for eq in equations:
            by_op.setdefault(eq.op, []).append(eq)
        symbols = sorted(assignment)
        lines: List[str] = []
        add = lines.append
        add("TYPE cryptarithm")
        add("M: sym∈0..9; injective; no-leading-zero; columns/carry/borrow; no digit search.")
        add("E: " + " | ".join(self._literal(eq.display()) for eq in equations))
        add("T: " + self._literal(clean_target))
        add("RULES")
        for op in sorted(combo):
            selected = combo[op]
            op_eqs = by_op.get(op, [])
            add(self._compact_rule_signature_summary(op, op_eqs, candidates.get(op, []), selected))
            add(f"R {self._literal(op)}={self._rule_short(selected)} [{self._rule_action_text(selected)}].")
        add("MAP_SEARCH")
        add("D0: " + ",".join(self._literal(ch) for ch in symbols) + "∈{0..9}; all-different.")
        selected_events = self._select_domain_events(events, self.trace_config.max_domain_events)
        for event in selected_events:
            add(self._format_domain_event(event))
        open_symbols = [ch for ch in sorted(proof_state.symbol_domains) if len(proof_state.symbol_domains[ch]) > 1]
        open_aux = [name for name in sorted(proof_state.aux_domains) if len(proof_state.aux_domains[name]) > 1]
        if open_symbols:
            add("REJECT_OPEN_MAP: " + "; ".join(f"{self._literal(ch)}∈{self._digits_dense(proof_state.symbol_domains[ch])}" for ch in open_symbols) + ".")
        elif open_aux:
            add("AUX_OPEN: " + "; ".join(f"{self._var_text(name)}∈{self._digits_dense(proof_state.aux_domains[name])}" for name in open_aux[:12]) + ".")
            add("MAP_SOLVED: digit map fixed; aux ambiguity irrelevant.")
        else:
            add("MAP_SOLVED: deterministic columns + AllDiff/Hall; no search.")
        add("MAP: " + "; ".join(f"{self._literal(ch)}={assignment[ch]}" for ch in symbols) + ".")
        add("VERIFY")
        for eq in equations:
            rule = combo[eq.op]
            a = self._number(eq.left, assignment, rule.reverse)
            b = self._number(eq.right, assignment, rule.reverse)
            op_text, value = self._operation_line_dense(rule, a, b)
            encoded = "<invalid>" if value is None else self._encode_value(value, eq.op, rule, assignment)
            expected = f"{eq.op}{eq.result}" if eq.has_sign else eq.result
            status = "✓" if encoded == expected else "✗"
            add(f"V {self._literal(eq.display())}: {a},{b}; {op_text}; enc={self._literal(encoded or '<invalid>')} {status}.")
        add("TARGET")
        if target_error or answer is None or target_value is None:
            add(f"TGT {self._literal(clean_target)}: {target_a},{target_b}; error={target_error or 'invalid'}.")
            add("FINAL <no valid answer>")
        else:
            op_text, _ = self._operation_line_dense(combo[target_op], target_a, target_b)
            add(f"TGT {self._literal(clean_target)}: {target_a},{target_b}; {op_text}; enc={self._literal(answer)}.")
            add(f"FINAL {answer}")
        return lines



# Backward-compatible public alias for existing imports.
DeductiveCryptarithmSolver = CryptarithmSolver


if __name__ == "__main__":
    examples = '>>}($ = ^^\n`/}>[ = "!\n><:^[ = :$"\n"[+(/ = $(`[\n`[:/! = /('
    target = "Now, determine the result for: $$}<$"
    solver = CryptarithmSolver(TraceConfig(max_solution_chars=40000, max_domain_events=120))
    result = solver.solve(examples, target)
    print(result.get("solution") or "\n".join(result.get("debug", [])))
