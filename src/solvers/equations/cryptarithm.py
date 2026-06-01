from __future__ import annotations

import re
import time
import sys
import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple, Iterable, Any

Digit = int
Domains = Dict[str, Set[int]]
Assignment = Dict[str, int]

# ----------------------------- data model -----------------------------

@dataclass(frozen=True)
class Equation:
    raw: str
    left: str
    op: str
    right: str
    result: str
    has_sign: bool = False

    def display(self) -> str:
        return f"{self.left}{self.op}{self.right}={'-' if False else ''}{self.op + self.result if self.has_sign else self.result}"

@dataclass(frozen=True)
class Variant:
    family: str              # cat, add_offset, sub_family, mul_offset
    orientation: str          # std/rev
    delta: int = 0            # add/mul offset
    mode: str = ""           # abs/sub for sub_family

    @property
    def reverse(self) -> bool:
        return self.orientation == "rev"

    @property
    def raw_name(self) -> str:
        if self.family == "cat":
            return f"{self.orientation}/cat"
        if self.family == "add_offset":
            name = "add" if self.delta == 0 else "add1" if self.delta == 1 else "addm1"
            return f"{self.orientation}/{name}"
        if self.family == "mul_offset":
            name = "mul" if self.delta == 0 else "mul1" if self.delta == 1 else "mulm1"
            return f"{self.orientation}/{name}"
        if self.family == "sub_family":
            return f"{self.orientation}/{self.mode}"
        return f"{self.orientation}/{self.family}"

    @property
    def compact(self) -> str:
        if self.family in {"add_offset", "mul_offset"}:
            return f"{self.family}({self.orientation},δ={self.delta:+d})"
        if self.family == "sub_family":
            return f"sub_family({self.orientation},{self.mode})"
        return f"cat({self.orientation})"

@dataclass
class Relation:
    name: str
    formula: str
    variables: Tuple[str, ...]
    rows: List[Tuple[int, ...]]
    eq: Equation
    variant: Variant

@dataclass
class TraceBlock:
    title: str
    lines: List[str] = field(default_factory=list)

    def render(self) -> List[str]:
        if not self.lines:
            return [self.title]
        return [self.title] + ["  " + l for l in self.lines]

@dataclass
class SolveResult:
    assignment: Assignment
    combo: Dict[str, Variant]
    trace: List[TraceBlock]
    target_output: str

# ----------------------------- solver -----------------------------

class CryptarithmSolver:
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self.deadline = 0.0
        self.stats: Dict[str, int] = {}
        self._rel_cache: Dict[Tuple[str, str, Tuple[Tuple[str, Tuple[int, ...]], ...]], Relation] = {}

    # public API: one normal launch, no trace modes
    def solve(self, examples_text: Any, target_text: Optional[Any] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        self.deadline = time.time() + float(timeout_seconds if timeout_seconds is not None else self.timeout_seconds)
        self.stats = {"family_candidates": 0, "early_pruned_variants": 0, "relation_rows": 0, "branches": 0, "combos_tried": 0}
        started = time.time()
        try:
            text = str(examples_text)
            if target_text is not None:
                text = text + "\n" + str(target_text)
            equations, known_ops = self._parse_examples(text)
            target = self._parse_target(text, known_ops)
            if not equations or target is None:
                return self._failure("could not parse examples or target", started)
            target_expr, target_left, target_op, target_right = target
            all_symbols = self._symbols(equations, target_left, target_right)

            op_domains, domain_trace = self._build_operation_domains(equations)
            op_domains, prune_trace = self._early_family_prune(equations, op_domains)
            if any(not variants for variants in op_domains.values()):
                return self._failure("an operator has no viable family variant after pruning", started, domain_trace + prune_trace)

            op_order = self._operator_order(equations, op_domains, target_op)
            found = self._search_operator_combos(equations, target_left, target_op, target_right, all_symbols, op_domains, op_order)
            if found is None:
                return self._failure("no globally consistent operation/digit assignment found", started, domain_trace + prune_trace)

            answer = found.target_output
            solution = self._render_solution(equations, target_expr, target_op, all_symbols, domain_trace, prune_trace, op_order, found)
            elapsed = time.time() - started
            self.stats["replay_blocks"] = solution.count("\n")
            return {
                "answer": answer,
                "final_answer": answer,
                "target_output": answer,
                "output": answer,
                "result": answer,
                "solution": solution,
                "trace": solution.splitlines(),
                "rules": {op: v.raw_name for op, v in found.combo.items()},
                "assignment": dict(sorted(found.assignment.items())),
                "stats": dict(self.stats),
                "elapsed_seconds": elapsed,
                "training_category": "cryptarithm",
            }
        except TimeoutError:
            return self._failure("timeout", started)
        except Exception as e:
            return self._failure(f"error: {type(e).__name__}: {e}", started)

    def _failure(self, reason: str, started: float, trace: Optional[List[TraceBlock]] = None) -> Dict[str, Any]:
        lines = ["Search failed", f"reason: {reason}"]
        if trace:
            for b in trace:
                lines.extend(b.render())
        return {"answer": None, "solution": "\n".join(lines), "trace": lines, "error": reason, "stats": dict(self.stats), "elapsed_seconds": time.time() - started, "training_category": "cryptarithm"}

    # ----------------------------- parsing -----------------------------

    @staticmethod
    def _preferred_operator_index(left_side: str) -> int:
        return 2 if len(left_side) >= 5 else len(left_side) // 2

    def _parse_equation(self, line: str, known_ops: Optional[Set[str]] = None) -> Optional[Equation]:
        compact = str(line).strip().replace(" ", "")
        if "=" not in compact:
            return None
        lhs, rhs = compact.split("=", 1)
        if not lhs or not rhs:
            return None
        idx: Optional[int] = None
        preferred = self._preferred_operator_index(lhs)
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
        has_sign = len(rhs) > 1 and rhs[0] == op
        result = rhs[1:] if has_sign else rhs
        left, right = lhs[:idx], lhs[idx + 1:]
        if not left or not right or not result:
            return None
        return Equation(raw=compact, left=left, op=op, right=right, result=result, has_sign=has_sign)

    def _parse_examples(self, text: str) -> Tuple[List[Equation], Set[str]]:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip() and "=" in ln and "==" not in ln]
        guessed = {eq.op for ln in lines if (eq := self._parse_equation(ln, None))}
        equations = [eq for ln in lines if (eq := self._parse_equation(ln, guessed))]
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
        for pat in patterns:
            new = re.sub(pat, "", raw).strip()
            if new != raw.strip():
                expr = new
                break
        expr = expr.splitlines()[-1].strip().replace(" ", "").rstrip(".")
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
        return expr, expr[:idx], expr[idx], expr[idx+1:]

    def _symbols(self, equations: Sequence[Equation], target_left: str, target_right: str) -> List[str]:
        ops = {eq.op for eq in equations}
        out: List[str] = []
        for eq in equations:
            for ch in eq.left + eq.right + eq.result:
                if ch not in ops and ch not in out:
                    out.append(ch)
        for ch in target_left + target_right:
            if ch not in ops and ch not in out:
                out.append(ch)
        return sorted(out)

    # ----------------------------- family domains -----------------------------

    def _all_variants(self) -> List[Variant]:
        variants: List[Variant] = []
        for ori in ("std", "rev"):
            variants.append(Variant("cat", ori))
            for d in (-1, 0, 1):
                variants.append(Variant("add_offset", ori, delta=d))
                variants.append(Variant("mul_offset", ori, delta=d))
            variants.append(Variant("sub_family", ori, mode="abs"))
            variants.append(Variant("sub_family", ori, mode="sub"))
        return variants

    def _build_operation_domains(self, equations: Sequence[Equation]) -> Tuple[Dict[str, List[Variant]], List[TraceBlock]]:
        by_op: Dict[str, List[Equation]] = {}
        for eq in equations:
            by_op.setdefault(eq.op, []).append(eq)
        op_domains: Dict[str, List[Variant]] = {}
        blocks: List[TraceBlock] = [TraceBlock("Operation-domain structural pruning")]
        for op, eqs in sorted(by_op.items()):
            kept: List[Variant] = []
            removed: Dict[str, List[str]] = {}
            for v in self._all_variants():
                ok, reason = self._structural_ok(eqs, v)
                if ok:
                    kept.append(v)
                else:
                    removed.setdefault(reason, []).append(v.raw_name)
            self.stats["family_candidates"] += len(kept)
            op_domains[op] = self._sort_variants(kept)
            lines = [f"op {repr(op)} examples: " + "; ".join(eq.raw for eq in eqs)]
            fam_summary: Dict[str, List[str]] = {}
            for v in kept:
                fam_summary.setdefault(v.family, []).append(v.raw_name)
            lines.append("survive structural: " + ("; ".join(f"{fam}:{{{','.join(vals)}}}" for fam, vals in fam_summary.items()) or "none"))
            if removed:
                parts = []
                for reason, vals in sorted(removed.items()):
                    parts.append(f"{reason}: {len(vals)}")
                lines.append("removed by structural family tests: " + "; ".join(parts))
            blocks.append(TraceBlock(f"Operator {repr(op)}", lines))
        return op_domains, blocks

    def _structural_ok(self, eqs: Sequence[Equation], v: Variant) -> Tuple[bool, str]:
        for eq in eqs:
            if eq.has_sign:
                if not (v.family == "sub_family" and v.mode == "sub"):
                    return False, "sign"
            if v.family == "cat":
                if eq.has_sign:
                    return False, "sign"
                expected = eq.left + eq.right if v.orientation == "std" else eq.right + eq.left
                if expected != eq.result:
                    return False, "concat pattern"
                continue
            possible = self._possible_lengths(eq, v)
            if len(eq.result) not in possible:
                return False, "length"
            if v.family == "sub_family" and v.mode == "abs" and eq.has_sign:
                return False, "sign"
        return True, ""

    @staticmethod
    def _bounds(length: int) -> Tuple[int, int]:
        if length <= 1:
            return 0, 9
        return 10 ** (length - 1), 10 ** length - 1

    @staticmethod
    def _lengths_between(lo: int, hi: int) -> Set[int]:
        if hi < 0:
            return set()
        lo = max(0, lo)
        out = set()
        for n in (0, 1, 2, 3, 4, 5, 6):
            a = 0 if n == 1 else 10 ** (n - 1)
            b = 9 if n == 1 else 10 ** n - 1
            if n == 0:
                a = b = 0
            if max(lo, a) <= min(hi, b):
                out.add(1 if n == 0 else n)
        return out or {1}

    def _possible_lengths(self, eq: Equation, v: Variant) -> Set[int]:
        la, lb = len(eq.left), len(eq.right)
        alo, ahi = self._bounds(la)
        blo, bhi = self._bounds(lb)
        if v.family == "add_offset":
            return self._lengths_between(alo + blo + v.delta, ahi + bhi + v.delta)
        if v.family == "mul_offset":
            return self._lengths_between(alo * blo + v.delta, ahi * bhi + v.delta)
        if v.family == "sub_family":
            hi = max(abs(x - y) for x in (alo, ahi) for y in (blo, bhi))
            return self._lengths_between(0, hi)
        if v.family == "cat":
            return {la + lb}
        return {len(eq.result)}

    def _sort_variants(self, variants: Sequence[Variant]) -> List[Variant]:
        fam_order = {"cat": 0, "add_offset": 1, "sub_family": 2, "mul_offset": 3}
        ori_order = {"std": 0, "rev": 1}
        # Prefer base rules before offsets, and std before rev, matching the old solver's deterministic tie-breaking.
        delta_order = {0: 0, 1: 1, -1: 2}
        mode_order = {"abs": 0, "sub": 1, "": 0}
        return sorted(variants, key=lambda v: (fam_order.get(v.family, 9), ori_order.get(v.orientation, 9), delta_order.get(v.delta, 9), mode_order.get(v.mode, 9), v.raw_name))

    # ----------------------------- early pruning -----------------------------

    def _early_family_prune(self, equations: Sequence[Equation], domains_by_op: Dict[str, List[Variant]]) -> Tuple[Dict[str, List[Variant]], List[TraceBlock]]:
        by_op: Dict[str, List[Equation]] = {}
        for eq in equations:
            by_op.setdefault(eq.op, []).append(eq)
        blocks: List[TraceBlock] = [TraceBlock("Early family-domain pruning")]
        new_domains: Dict[str, List[Variant]] = {}
        for op, variants in sorted(domains_by_op.items()):
            kept: List[Variant] = []
            rejected_by_family: Dict[str, List[Variant]] = {}
            for v in variants:
                if self._variant_locally_viable(by_op.get(op, ()), v):
                    kept.append(v)
                else:
                    rejected_by_family.setdefault(self._family_key(v), []).append(v)
            self.stats["early_pruned_variants"] += len(variants) - len(kept)
            new_domains[op] = self._sort_variants(kept)
            lines: List[str] = []
            if rejected_by_family:
                for fam, vals in sorted(rejected_by_family.items()):
                    lines.append(f"reject {fam}: {{{','.join(v.raw_name for v in vals)}}} have zero local rows")
            lines.append("survive early: " + (", ".join(v.raw_name for v in new_domains[op]) or "none"))
            blocks.append(TraceBlock(f"Operator {repr(op)} early domain", lines))
        return new_domains, blocks

    def _family_key(self, v: Variant) -> str:
        if v.family in {"add_offset", "mul_offset"}:
            return f"{v.family}({v.orientation},δ-domain)"
        if v.family == "sub_family":
            return f"sub_family({v.orientation},mode-domain)"
        return f"cat({v.orientation})"

    def _variant_locally_viable(self, eqs: Sequence[Equation], v: Variant) -> bool:
        if not eqs:
            return True
        symbols = sorted(set("".join(eq.left + eq.right + eq.result for eq in eqs)))
        domains: Domains = {s: set(range(10)) for s in symbols}
        self._apply_leading_domains(domains, eqs, {eq.op: v for eq in eqs})
        relations = []
        for eq in eqs:
            rel = self._relation_for(eq, v, domains)
            if rel is not None:
                if not rel.rows:
                    return False
                relations.append(rel)
        ok, _trace = self._propagate(domains, relations, record=False)
        return ok

    # ----------------------------- relations / arithmetic -----------------------------

    def _word_digits(self, word: str, assign: Assignment, reverse: bool) -> List[int]:
        chars = list(reversed(word)) if reverse else list(word)
        return [assign[ch] for ch in chars]

    def _word_value(self, word: str, assign: Assignment, reverse: bool) -> int:
        val = 0
        for d in self._word_digits(word, assign, reverse):
            val = val * 10 + d
        return val

    def _leading_symbol(self, word: str, reverse: bool) -> str:
        return word[-1] if reverse else word[0]

    def _apply_leading_domains(self, domains: Domains, eqs: Sequence[Equation], combo: Dict[str, Variant], target: Optional[Tuple[str, str, str]] = None) -> None:
        for eq in eqs:
            v = combo[eq.op]
            if v.family == "cat":
                continue
            for word in (eq.left, eq.right, eq.result):
                if len(word) > 1:
                    domains[self._leading_symbol(word, v.reverse)].discard(0)
        if target is not None:
            left, op, right = target
            if op in combo and combo[op].family != "cat":
                v = combo[op]
                for word in (left, right):
                    if len(word) > 1 and self._leading_symbol(word, v.reverse) in domains:
                        domains[self._leading_symbol(word, v.reverse)].discard(0)

    def _match_equation(self, eq: Equation, v: Variant, assign: Assignment) -> bool:
        if v.family == "cat":
            if eq.has_sign:
                return False
            expected = eq.left + eq.right if v.orientation == "std" else eq.right + eq.left
            return expected == eq.result
        # leading zeros are invalid for encoded words
        for w in (eq.left, eq.right, eq.result):
            if len(w) > 1 and assign[self._leading_symbol(w, v.reverse)] == 0:
                return False
        a = self._word_value(eq.left, assign, v.reverse)
        b = self._word_value(eq.right, assign, v.reverse)
        r = self._word_value(eq.result, assign, v.reverse)
        if v.family == "add_offset":
            if eq.has_sign:
                return False
            return a + b + v.delta == r
        if v.family == "mul_offset":
            if eq.has_sign:
                return False
            return a * b + v.delta == r
        if v.family == "sub_family":
            value = abs(a - b) if v.mode == "abs" else a - b
            if v.mode == "abs":
                return (not eq.has_sign) and value == r
            if eq.has_sign:
                return value < 0 and -value == r
            return value >= 0 and value == r
        return False

    def _relation_for(self, eq: Equation, v: Variant, domains: Domains) -> Optional[Relation]:
        if v.family == "cat":
            expected = eq.left + eq.right if v.orientation == "std" else eq.right + eq.left
            if (not eq.has_sign) and expected == eq.result:
                return None
            return Relation(f"{eq.raw} via {v.raw_name}", f"concat pattern {expected!r}={eq.result!r}", tuple(), [], eq, v)
        scope = tuple(sorted(set(eq.left + eq.right + eq.result)))
        key = (eq.raw, v.raw_name, tuple((ch, tuple(sorted(domains[ch]))) for ch in scope))
        if key in self._rel_cache:
            return self._rel_cache[key]
        rows: List[Tuple[int, ...]] = []
        assign: Assignment = {}
        used: Set[int] = set()
        ordered = sorted(scope, key=lambda ch: (len(domains[ch]), ch))
        def bt(i: int) -> None:
            if time.time() > self.deadline:
                raise TimeoutError()
            if i == len(ordered):
                if self._match_equation(eq, v, assign):
                    rows.append(tuple(assign[ch] for ch in scope))
                return
            ch = ordered[i]
            for d in sorted(domains[ch]):
                if d in used:
                    continue
                assign[ch] = d
                used.add(d)
                bt(i + 1)
                used.remove(d)
                del assign[ch]
        bt(0)
        rel = Relation(f"{eq.raw} via {v.raw_name}", self._formula(eq, v), scope, rows, eq, v)
        self.stats["relation_rows"] += len(rows)
        self._rel_cache[key] = rel
        return rel

    def _formula(self, eq: Equation, v: Variant) -> str:
        def term(w: str) -> str:
            chars = list(reversed(w)) if v.reverse else list(w)
            parts = []
            n = len(chars)
            for i, ch in enumerate(chars):
                pow10 = n - i - 1
                coef = 10 ** pow10
                parts.append(repr(ch) if coef == 1 else f"{coef}*{repr(ch)}")
            return "+".join(parts) if parts else "0"
        a, b, r = term(eq.left), term(eq.right), term(eq.result)
        if v.family == "add_offset":
            off = "" if v.delta == 0 else f"{v.delta:+d}"
            return f"{a}+{b}{off}={r}"
        if v.family == "mul_offset":
            off = "" if v.delta == 0 else f"{v.delta:+d}"
            return f"({a})*({b}){off}={r}"
        if v.family == "sub_family" and v.mode == "abs":
            return f"abs(({a})-({b}))={r}"
        if v.family == "sub_family":
            return f"({a})-({b})={'-(' + r + ')' if eq.has_sign else r}"
        return f"concat({eq.left},{eq.right})={eq.result}"

    # ----------------------------- propagation/search -----------------------------

    def _propagate(self, domains: Domains, relations: Sequence[Relation], record: bool = True) -> Tuple[bool, List[TraceBlock]]:
        trace: List[TraceBlock] = []
        changed = True
        filtered_rows: Dict[str, List[Tuple[int, ...]]] = {rel.name: list(rel.rows) for rel in relations}
        while changed:
            changed = False
            # relation projections
            for rel in relations:
                rows = [row for row in filtered_rows[rel.name] if all(row[i] in domains[ch] for i, ch in enumerate(rel.variables))]
                filtered_rows[rel.name] = rows
                if not rows:
                    if record:
                        trace.append(TraceBlock(f"Reject by {rel.name}", [f"generator: {rel.formula}", "local result: zero rows under current deltas"]))
                    return False, trace
                updates: Dict[str, Tuple[Set[int], Set[int]]] = {}
                for i, ch in enumerate(rel.variables):
                    sup = {row[i] for row in rows}
                    new = domains[ch] & sup
                    if not new:
                        return False, trace
                    if new != domains[ch]:
                        updates[ch] = (set(domains[ch]), set(new))
                        domains[ch] = new
                        changed = True
                if record and updates:
                    lines = [f"generator: {rel.formula}"]
                    if len(rows) <= 8:
                        lines.append("rows: " + "; ".join("{" + ",".join(f"{c}={d}" for c, d in zip(rel.variables, row)) + "}" for row in rows))
                    lines.append("Δ: " + self._delta_text(updates))
                    trace.append(TraceBlock(f"Relation {rel.name}", lines))
            # fixed-digit all-different
            fixed = {next(iter(v)) for v in domains.values() if len(v) == 1}
            for ch, vals in list(domains.items()):
                if len(vals) > 1:
                    new = vals - fixed
                    if not new:
                        return False, trace
                    if new != vals:
                        before = set(vals); domains[ch] = new; changed = True
                        if record:
                            trace.append(TraceBlock("AllDifferent fixed-digit propagation", ["Δ: " + self._delta_text({ch: (before, set(new))})]))
            # small Hall sets
            symbols = list(domains)
            for k in range(2, min(4, len(symbols)) + 1):
                for subset in combinations(symbols, k):
                    union = set().union(*(domains[s] for s in subset))
                    if len(union) == k:
                        outside_updates: Dict[str, Tuple[Set[int], Set[int]]] = {}
                        for ch in symbols:
                            if ch in subset:
                                continue
                            new = domains[ch] - union
                            if not new:
                                return False, trace
                            if new != domains[ch]:
                                outside_updates[ch] = (set(domains[ch]), set(new))
                                domains[ch] = new
                                changed = True
                        if record and outside_updates:
                            trace.append(TraceBlock("Hall AllDifferent", [f"S={{{','.join(repr(x) for x in subset)}}}, U={{{','.join(map(str, sorted(union)))}}}", "Δ: " + self._delta_text(outside_updates)]))
                        if changed:
                            break
                if changed:
                    break
        return True, trace

    def _delta_text(self, updates: Dict[str, Tuple[Set[int], Set[int]]]) -> str:
        def fmt(vals: Set[int]) -> str:
            if vals == set(range(10)):
                return "{0..9}"
            return "{" + ",".join(map(str, sorted(vals))) + "}"
        return "; ".join(f"D[{repr(ch)}]: {fmt(a)}->{fmt(b)}" for ch, (a, b) in sorted(updates.items()))

    def _operator_order(self, equations: Sequence[Equation], domains: Dict[str, List[Variant]], target_op: str) -> List[str]:
        by_op: Dict[str, int] = {}
        for eq in equations:
            by_op[eq.op] = by_op.get(eq.op, 0) + 1
        return sorted(domains, key=lambda op: (len(domains[op]), by_op.get(op, 1), 0 if op == target_op else 1, op))

    def _search_operator_combos(self, equations: Sequence[Equation], target_left: str, target_op: str, target_right: str, symbols: List[str], op_domains: Dict[str, List[Variant]], op_order: List[str]) -> Optional[SolveResult]:
        combo: Dict[str, Variant] = {}
        def rec(i: int) -> Optional[SolveResult]:
            if time.time() > self.deadline:
                raise TimeoutError()
            if i == len(op_order):
                self.stats["combos_tried"] += 1
                return self._solve_digits_for_combo(equations, target_left, target_op, target_right, symbols, combo)
            op = op_order[i]
            for v in op_domains[op]:
                combo[op] = v
                # cheap partial: all assigned op equations viable under initial domains
                out = rec(i + 1)
                if out is not None:
                    return out
                del combo[op]
            return None
        return rec(0)

    def _solve_digits_for_combo(self, equations: Sequence[Equation], target_left: str, target_op: str, target_right: str, symbols: List[str], combo: Dict[str, Variant]) -> Optional[SolveResult]:
        domains: Domains = {s: set(range(10)) for s in symbols}
        self._apply_leading_domains(domains, equations, combo, (target_left, target_op, target_right))
        relations: List[Relation] = []
        for eq in equations:
            rel = self._relation_for(eq, combo[eq.op], domains)
            if rel is not None:
                if not rel.rows:
                    return None
                relations.append(rel)
        ok, trace = self._propagate(domains, relations, record=True)
        if not ok:
            return None
        result = self._backtrack_domains(domains, relations, equations, combo, target_left, target_op, target_right, trace)
        if result is not None:
            return SolveResult(result, dict(combo), trace, self._target_output(target_left, target_op, target_right, combo[target_op], result))
        return None

    def _backtrack_domains(self, domains: Domains, relations: Sequence[Relation], equations: Sequence[Equation], combo: Dict[str, Variant], target_left: str, target_op: str, target_right: str, trace: List[TraceBlock]) -> Optional[Assignment]:
        if time.time() > self.deadline:
            raise TimeoutError()
        if all(len(v) == 1 for v in domains.values()):
            assign = {ch: next(iter(vals)) for ch, vals in domains.items()}
            if len(set(assign.values())) != len(assign):
                return None
            if all(self._match_equation(eq, combo[eq.op], assign) for eq in equations):
                return assign
            return None
        ch = min((c for c in domains if len(domains[c]) > 1), key=lambda c: (len(domains[c]), c))
        for d in sorted(domains[ch]):
            self.stats["branches"] += 1
            new_domains = {c: set(v) for c, v in domains.items()}
            new_domains[ch] = {d}
            ok, subtrace = self._propagate(new_domains, relations, record=True)
            if not ok:
                continue
            found = self._backtrack_domains(new_domains, relations, equations, combo, target_left, target_op, target_right, trace + subtrace)
            if found is not None:
                if subtrace:
                    trace.append(TraceBlock("Branch commit", [f"set {repr(ch)}={d}", "propagation after branch accepted"]))
                    trace.extend(subtrace)
                return found
        return None

    def _target_output(self, left: str, op: str, right: str, v: Variant, assign: Assignment) -> str:
        if v.family == "cat":
            return left + right if v.orientation == "std" else right + left
        a = self._word_value(left, assign, v.reverse)
        b = self._word_value(right, assign, v.reverse)
        if v.family == "add_offset":
            value = a + b + v.delta
        elif v.family == "mul_offset":
            value = a * b + v.delta
        elif v.family == "sub_family" and v.mode == "abs":
            value = abs(a - b)
        elif v.family == "sub_family":
            value = a - b
        else:
            value = 0
        sign = ""
        if value < 0:
            sign = op
            value = -value
        digits = str(value)
        if v.reverse:
            digits = digits[::-1]
        inverse = {d: ch for ch, d in assign.items()}
        # The puzzle expects the encoded visible-symbol output, not raw decimal digits.
        # If a digit is not present in the current map, keep the raw digit as a safe fallback.
        return sign + "".join(inverse.get(int(d), d) for d in digits)

    # ----------------------------- rendering -----------------------------

    def _render_solution(self, equations: Sequence[Equation], target_expr: str, target_op: str, symbols: Sequence[str], domain_trace: List[TraceBlock], prune_trace: List[TraceBlock], op_order: List[str], found: SolveResult) -> str:
        out: List[str] = []
        out.append("Search setup")
        out.append("Digit symbols: " + ", ".join(repr(s) for s in symbols))
        out.append("Invariant: different symbols take different digits; current domains are reconstructed only from printed Δ updates.")
        out.append("Operation hypotheses are compressed as family domains: add/mul use offset δ, sub uses mode {abs,sub}, orientation is {std,rev}.")
        out.append("")
        for block in domain_trace + prune_trace:
            out.extend(block.render())
        out.append("Operator search order: " + ", ".join(repr(op) for op in op_order))
        out.append("Selected operation rules: " + "; ".join(f"{repr(op)}->{v.raw_name}" for op, v in sorted(found.combo.items())))
        out.append("")
        out.append("Replay certificate")
        if found.trace:
            for block in found.trace:
                out.extend(block.render())
        else:
            out.append("No digit-domain deltas were needed after operation-domain pruning.")
        out.append("")
        out.append("Final digit map")
        out.append("; ".join(f"{repr(ch)}={d}" for ch, d in sorted(found.assignment.items())))
        out.append("")
        out.append("Compact rule check")
        for eq in equations:
            v = found.combo[eq.op]
            a = self._word_value(eq.left, found.assignment, v.reverse) if v.family != "cat" else None
            b = self._word_value(eq.right, found.assignment, v.reverse) if v.family != "cat" else None
            if v.family == "cat":
                line = f"{eq.raw}: {v.raw_name} pattern ✓"
            else:
                produced = self._target_output(eq.left, eq.op, eq.right, v, found.assignment)
                line = f"{eq.raw}: {v.raw_name}, {a},{b} -> {produced} ✓"
            out.append("  " + line)
        out.append("")
        out.append("Target")
        out.append(f"{target_expr} via {found.combo[target_op].raw_name}: {found.target_output}")
        # Compatibility for benchmark/extractors: keep a simple, stable final-answer marker.
        out.append(f"Answer: {found.target_output}")
        out.append(f"Final answer: {found.target_output}")
        return "\n".join(out)


def extract_answer(obj: Any) -> Optional[str]:
    """Backward-compatible answer extractor.

    Accepts the dict returned by solve(), a raw solution string, or any object that
    stringifies to the solver output.  This is intentionally permissive because
    external benchmarks often changed between reading result['answer'] and regexing
    the CoT text.
    """
    if isinstance(obj, dict):
        for key in ("answer", "final_answer", "target_output", "output", "result"):
            val = obj.get(key)
            if val is not None and str(val) != "":
                return str(val)
        obj = obj.get("solution", "")
    text = str(obj)
    patterns = [
        r"(?m)^Final answer:\s*(\S+)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None


# convenience alias
Solver = CryptarithmSolver


if __name__ == "__main__":
    # CLI compatibility: python solver.py < input.txt prints CoT with Answer markers.
    # Use --json to print the full result dict for harnesses that expect structured output.
    data = sys.stdin.read()
    solver = CryptarithmSolver()
    result = solver.solve(data)
    if "--json" in sys.argv:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result.get("solution", ""))
