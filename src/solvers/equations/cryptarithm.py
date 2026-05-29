from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Digit = int
Assignment = Dict[str, Digit]
Domains = Dict[str, Set[Digit]]

@dataclass
class SolverConfig:
    timeout_seconds: Any = 30.0
    max_trace_lines: Any = 10000000
    max_trace_chars: Any = 1000000000
    include_digit_matching_lines: Any = True
    include_rule_reject_summary: Any = True
    include_search_branches: Any = True
    max_kept_rules_shown: Any = 16
    max_digit_lines_shown: Any = 10000000
    include_no_change_events: Any = False
    include_failed_attempt_events: Any = False
    max_projection_blocks: Any = 10000000
    max_structural_tests_per_operator: Any = 10000000
    use_support_cache: Any = True
    projection_strategy: Any = 'rule_specific_exact_first'
    exact_dominates_modular_space: Any = 1000
    modular_anchor_ks: Any = (1, 2, -1)
    compact_structural_matching: Any = True
    compact_projection_blocks: Any = True
    show_current_domains: Any = True
    show_projection_reason: Any = False
    show_support_counts: Any = True
    cat_as_positional_equalities: Any = True
    causal_reject_blocks: Any = True
    show_projection_derivation: Any = True
    max_projection_witnesses: Any = 0
    show_supported_projection: Any = True
    show_projection_search_space: Any = True
    max_exact_witness_count: Any = 6
    use_rule_specific_exact_projection: Any = True
    max_rule_specific_variables: Any = 8
    max_transition_values_shown: Any = 10000000
    skip_modular_when_rule_specific_exact: Any = True
    show_exact_first_skip_events: Any = False
    include_ambiguity_audit: Any = True
    max_audit_solutions: Any = 100000
    max_audit_rule_combos: Any = 100000
    max_audit_target_outputs_shown: Any = 16
    audit_stop_after_distinct_target_outputs: Any = 2
    # Full CoT/audit is always enabled in this build; these are pruning knobs, not modes.
    use_bounds_sign_structural_pruning: Any = True
    use_target_no_leading_zero: Any = True
    use_column_tuple_branching: Any = True
    max_column_branch_tuples: Any = 24

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
class Stats:
    rule_hypotheses: int = 0
    digit_branches: int = 0
    domain_reductions: int = 0
    support_checks: int = 0
    tuple_branches: int = 0


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
    config: SolverConfig
    events: List[ReplayEvent] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    last_reject: str = ""

    def add(self, line: str) -> None:
        
        line = str(line).strip()
        if line:
            self.notes.append(line)

    def add_event(self, event: ReplayEvent) -> None:
        self.events.append(event)
        if event.decision == "reject" or "contradiction" in event.kind.lower() or "reject" in event.kind.lower():
            self.last_reject = event.reason or event.title or event.label

    def extend(self, other: "ReplayTrace") -> None:
        self.events.extend(other.events)
        self.notes.extend(other.notes)
        if other.last_reject:
            self.last_reject = other.last_reject

    def finish(self) -> List[ReplayEvent]:
        return list(self.events)

    def reject_reason(self) -> str:
        return self.last_reject or "no supported continuation"


class SilentReplayTrace:
    """Trace sink for exhaustive audits.

    It accepts the same minimal calls used by propagation, but does not retain
    event bodies.  This keeps ambiguity audits from consuming memory or
    contaminating the train-style replay trace.
    """

    def __init__(self) -> None:
        self.events: List[ReplayEvent] = []
        self.notes: List[str] = []
        self.last_reject = ""

    def add(self, line: str) -> None:
        return None

    def add_event(self, event: ReplayEvent) -> None:
        if event.decision == "reject" or "contradiction" in event.kind.lower() or "reject" in event.kind.lower():
            self.last_reject = event.reason or event.title or event.label

    def extend(self, other: Any) -> None:
        if getattr(other, "last_reject", ""):
            self.last_reject = other.last_reject

    def finish(self) -> List[ReplayEvent]:
        return []

    def reject_reason(self) -> str:
        return self.last_reject or "no supported continuation"


class CryptarithmSolver:
    _EXACT_FIRST_RULES: Set[str] = {"add", "add1", "addm1", "mul", "mul1", "mulm1"}

    """Final cryptarithm solver with replay-reproducible CoT.

    This file intentionally contains only the final solver path. The CoT is a structured
    replay trace: from the prompt plus the printed trace, the selected rules,
    digit map, verified examples, and target answer can be recomputed without
    hidden solver state.
    """


    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.rules = self._build_rules()
        self.stats = Stats()
        self._deadline = 0.0
        self._support_cache: Dict[Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]], Tuple[bool, Dict[str, Tuple[int, ...]], int]] = {}
        self._support_cache_v25: Dict[
            Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]],
            Tuple[bool, Dict[str, Tuple[int, ...]], int, Tuple[Tuple[Tuple[str, int], ...], ...], int],
        ] = {}

    def solve(self, examples_text: Any, target_text: Optional[Any] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        self.stats = Stats()
        self.stats.support_cache_hits = 0
        self.stats.replay_events = 0
        self._support_cache = {}
        self._support_cache_v25 = {}
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
            for op in list(candidates):
                candidates[op] = sorted(candidates[op], key=self._rule_sort_key)

            if any(not cands for cands in candidates.values()):
                bad = [op for op, cands in candidates.items() if not cands][0]
                return self._failure(f"no structural rule remains for operator {repr(bad)}", started)
            if target_op not in candidates:
                return self._failure(f"target operator {repr(target_op)} was not seen in examples", started)

            domains: Domains = {s: set(range(10)) for s in symbols}
            ordered_ops = sorted(candidates, key=lambda op: (0 if op == target_op else 1, len(candidates[op]), -len(by_op[op]), op))
            trace = ReplayTrace(self.config)
            trace.add_event(
                ReplayEvent(
                    kind="RULE_ORDER",
                    title="Rule-combo search order",
                    decision="order",
                    meta={"ordered_ops": tuple(ordered_ops), "target_op": target_op},
                )
            )
            assignment, combo = self._search_rules(ordered_ops, candidates, equations, domains, {}, trace, 0, target_context=(target_left, target_op, target_right))

            if assignment is None or combo is None:
                return self._failure("no rule/map combination satisfies all examples", started)

            answer, target_error, target_a, target_b, target_value = self._encode_target(
                target_left, target_op, target_right, combo[target_op], assignment
            )
            audit_result = self._audit_target_outputs(
                ordered_ops=ordered_ops,
                candidates=candidates,
                equations=equations,
                initial_domains=domains,
                target_left=target_left,
                target_op=target_op,
                target_right=target_right,
                selected_answer=answer,
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
                audit_result=audit_result,
            )
            return {
                "answer": answer,
                "solution": "\n".join(solution_lines),
                "debug": solution_lines,
                "trace": solution_lines,
                "events": [asdict(e) for e in events],
                "mapping": assignment,
                "rules": {op: self._rule_name(rule) for op, rule in combo.items()},
                "audit": audit_result,
                "stats": self.stats.__dict__,
                "elapsed_seconds": round(time.time() - started, 4),
                "training_category": "cryptarithm.final_replay",
            }
        except TimeoutError:
            return self._failure("solver timeout", started)
        except Exception as exc:
            return self._failure(f"execution error: {type(exc).__name__}: {exc}", started)

    def _support_line(self, event: ReplayEvent, indent: str) -> Optional[str]:
        if not self._cfg("show_supported_projection", True):
            return None
        if event.supported is None:
            return None
        if self._cfg("show_support_counts", False):
            count = "unknown" if event.count is None else str(event.count)
            return f"{indent}supported projection ({count} supports): {self._domain_text_map(event.supported)}"
        return f"{indent}supported projection: {self._domain_text_map(event.supported)}"

    def _rule_specific_project_replay(
        self,
        eq: Equation,
        rule: Rule,
        domains: Domains,
        trace: ReplayTrace,
        label: str,
        constraint: str,
        kind: str,
    ) -> Optional[bool]:
        deriv = self._derive_rule_specific_support(eq, rule, domains)
        if deriv is None:
            scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
            return self._project_replay(
                domains,
                scope,
                lambda local: self._equation_matches(eq, rule, local),
                label,
                trace,
                constraint=constraint,
                kind=kind,
            )
        scope = list(deriv["scope"])
        before = self._snapshot_scope(domains, scope)
        supported: Dict[str, Set[int]] = deriv["supported"]
        count = int(deriv["count"])
        union_supported = tuple((ch, tuple(sorted(supported.get(ch, set())))) for ch in scope)
        proof_meta = {
            "method": deriv.get("method", "rule-specific column DP"),
            "scope": tuple(scope),
            "satisfying_count": count,
            "union_supported": union_supported,
            "derivation_lines": tuple(deriv.get("lines", ())),
            "rule_specific": True,
        }

        if count <= 0:
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
                reason="rule-specific projection narrows at least one domain",
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
                reason="rule-specific projection equals current domains",
                count=count,
                meta={"projection_derivation": proof_meta},
            ))
        return False

    def _exact_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        label = f"full match {eq.display()} via {self._rule_name(rule)}"
        if rule.name == "cat" and self._cfg("cat_as_positional_equalities", True):
            return self._cat_project(eq, rule, domains, trace)
        if self._cfg("use_rule_specific_exact_projection", True) and rule.name in {"add", "add1", "addm1", "sub", "abs", "mul", "mul1", "mulm1"}:
            return self._rule_specific_project_replay(
                eq, rule, domains, trace, label, self._constraint_formula(eq, rule), "FULL_PROJECT"
            )
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        return self._project_replay(
            domains,
            scope,
            lambda local: self._equation_matches(eq, rule, local),
            label,
            trace,
            constraint=self._constraint_formula(eq, rule),
            kind="FULL_PROJECT",
        )

    def _use_exact_first_for_rule(self, rule: Rule) -> bool:
        return (
            self._cfg("projection_strategy", "rule_specific_exact_first") == "rule_specific_exact_first"
            and self._cfg("skip_modular_when_rule_specific_exact", True)
            and rule.name in {"add", "add1", "addm1", "mul", "mul1", "mulm1"}
            and self._cfg("use_rule_specific_exact_projection", True)
        )

    def _modular_project(self, eq: Equation, rule: Rule, domains: Domains, trace: ReplayTrace) -> Optional[bool]:
        # The exact full projection below is rule-specific DP and dominates all low-k suffix projections.
        # Returning False means “no state changed here; continue to _exact_project().”
        if self._use_exact_first_for_rule(rule):
            if self._cfg("show_exact_first_skip_events", False) or self._cfg("include_no_change_events", False):
                full_scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
                trace.add_event(ReplayEvent(
                    kind="MODULAR_SKIP",
                    title="Modular cascade skipped",
                    decision="skip",
                    reason="rule-specific full projection is exact and dominates all low-k suffix projections",
                    meta={
                        "equation": eq.display(),
                        "rule": self._rule_name(rule),
                        "scope": tuple(full_scope),
                        "strategy": "rule_specific_exact_first",
                    },
                ))
            return False
        # Final solver is exact-first.  If a non-exact-first config is forced, do no modular pruning rather than emit weaker suffix blocks.
        return False

    def _format_replay_event_default(self, event: ReplayEvent, block_num: int) -> List[str]:
        lines: List[str] = []
        kind = event.kind
        if kind == "RULE_ORDER":
            ordered = event.meta.get("ordered_ops", ()) if event.meta else ()
            lines.append(f"Rule-combo order: " + ", ".join(repr(op) for op in ordered))
            return lines
        if kind == "RULE_TRY":
            lines.append(f"Combo block {block_num}: try {event.reason}")
            return lines
        if kind == "COMBO_PROPAGATE":
            combo = event.meta.get("combo", ()) if event.meta else ()
            combo_text = "; ".join(f"{repr(op)}->{rule}" for op, rule in combo)
            lines.append(f"Combo block {block_num}: all operators assigned; start digit-domain solving")
            lines.append(f"  combo: {combo_text}")
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
            if kind == "ALLDIFFERENT" and event.meta:
                hall_symbols = event.meta.get("hall_symbols")
                hall_digits = event.meta.get("hall_digits")
                if hall_symbols is not None and hall_digits is not None:
                    sym_text = "{" + ",".join(repr(ch) for ch in hall_symbols) + "}"
                    digit_text = "{" + ",".join(str(d) for d in hall_digits) + "}"
                    lines.append(f"  Hall set: symbols {sym_text}; union digits {digit_text}")
            if event.reason:
                lines.append(f"  reason: {event.reason}")
            if self._cfg("show_current_domains", False):
                lines.append(f"  current domains: {self._domain_text_map(event.before)}")
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            lines.append(f"  decision: {event.decision}")
            return lines
        if kind == "BRANCH_SELECT":
            lines.append(f"Branch block {block_num}: choose symbol")
            lines.append(f"  reason: {event.reason}")
            return lines
        if kind == "TUPLE_BRANCH_SELECT":
            lines.append(f"Branch block {block_num}: choose column tuple relation")
            if event.label:
                lines.append(f"  label: {event.label}")
            if event.constraint:
                lines.append(f"  formula: {event.constraint}")
            lines.append(f"  reason: {event.reason}")
            if event.meta:
                tuples = event.meta.get("tuples", ())
                if tuples:
                    rendered = []
                    for tup in tuples:
                        rendered.append("{" + ", ".join(f"{repr(ch)}={d}" for ch, d in tup) + "}")
                    lines.append("  branch tuples: " + "; ".join(rendered))
            return lines
        if kind == "TUPLE_TRY":
            lines.append(f"Branch block {block_num}: try tuple {event.reason}")
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            lines.append("  decision: propagate this tuple assignment")
            return lines
        if kind == "DIGIT_TRY":
            lines.append(f"Branch block {block_num}: try {event.reason}")
            lines.append(f"  apply update: {self._changes_text(event.changes)}")
            lines.append("  decision: propagate this assignment")
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
        if kind == "MODULAR_SKIP":
            if self._cfg("include_no_change_events", False):
                lines.append(f"Projection block {block_num}: modular suffix checks skipped")
                lines.append(f"  reason: {event.reason}")
                lines.append("  decision: skip")
            return lines
        lines.append(f"Replay block {block_num}: {event.kind}: {event.title}")
        if event.reason:
            lines.append(f"  reason: {event.reason}")
        if event.decision:
            lines.append(f"  decision: {event.decision}")
        return lines

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

        if self._cfg("causal_reject_blocks", True):
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
                local_failure = f"{candidate}: {failure}" if failure else candidate
                if event.label or event.constraint or event.supported is not None:
                    lines.extend(self._render_local_check(event, indent="  ", result_override=local_failure))
                else:
                    lines.append(f"  result: {local_failure}")
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

            if event.kind in {"DIGIT_REJECT", "TUPLE_REJECT"}:
                branch, failure = self._split_decision_reason(event.reason)
                lines = [f"Branch block {block_num}: test {branch}"]
                if event.label or event.constraint or event.supported is not None:
                    lines.extend(self._render_local_check(event, indent="  ", result_override=failure))
                else:
                    lines.append(f"  result: {failure}")
                lines.append(f"  decision: reject {branch}")
                return lines

        return self._format_replay_event_default(event, block_num)

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
        audit_result: Optional[Dict[str, Any]] = None,
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
        add("Replay convention: a projection block is one local solver operation; rule-specific exact full projections are used instead of low-k suffix projection cascades.")
        add("Branch blocks explicitly assign one digit and then run the following propagation blocks.")
        if self._cfg("show_current_domains", False):
            add("Only local operations that change state, reject a branch/combo, or select a branch are shown; each local pruning block prints its current local domains before the update.")
        else:
            add("Only local operations that change state, reject a branch/combo, or select a branch are shown; current domains are reconstructed from earlier updates.")
        projection_count = 0
        for idx, event in enumerate(events, 1):
            if event.kind in {"COLUMN_PROJECT", "FULL_PROJECT", "PROJECTION"}:
                projection_count += 1
                if projection_count > self._cfg("max_projection_blocks", 10_000_000):
                    add(f"... remaining projection blocks omitted by max_projection_blocks={self._cfg('max_projection_blocks', 10_000_000)}")
                    break
            for line in self._format_replay_event(event, idx):
                if line:
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

        if audit_result and audit_result.get("enabled"):
            add("Ambiguity audit")
            add("Audit scope: all structurally kept rule combos and all digit maps satisfying the training examples.")
            add(f"Rule combos tested: {audit_result.get('rule_combos_tested', 0)}")
            add(f"Valid rule combos found: {audit_result.get('valid_rule_combos', 0)}")
            add(f"Valid complete solutions found: {audit_result.get('valid_complete_solutions', 0)}")
            outputs = audit_result.get("target_outputs", ())
            if outputs:
                max_shown = int(self._cfg("max_audit_target_outputs_shown", 16))
                shown = tuple(outputs[:max_shown])
                shown_outputs = ", ".join(repr(x) for x in shown)
                suffix = "" if len(outputs) <= max_shown else f", ... {len(outputs) - max_shown} more"
                add(f"Target outputs found: {{{shown_outputs}{suffix}}}")
            else:
                add("Target outputs found: none")
            selected_output = audit_result.get("selected_output", "nan")
            add(f"Selected solution output: {repr(selected_output)}")
            outputs_count = len(outputs)
            if audit_result.get("complete") or outputs_count > 1:
                add(f"decision: {audit_result.get('decision', 'audit complete')}")
                if not audit_result.get("complete"):
                    add(f"audit stopped early: {audit_result.get('stopped_reason', 'unknown stop reason')}")
            else:
                add(f"decision: audit incomplete — {audit_result.get('stopped_reason', 'unknown stop reason')}")
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

    @staticmethod
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

    def generate_cot(self, prompt: Any) -> str:
        result = self.solve(str(prompt), None, timeout_seconds=self.config.timeout_seconds)
        return result.get("solution", "")

    def _alldifferent(self, domains: Domains, trace: ReplayTrace) -> bool:
        """Enforce injective digit mapping with singleton and Hall-set pruning.

        Hall rule: for any symbol subset S, let U be the union of their current
        domains.  If |U| < |S|, injectivity is impossible.  If |U| == |S|,
        digits in U are reserved for S and can be removed from all symbols
        outside S.  With at most 10 symbols this exhaustive subset scan is cheap,
        deterministic, and gives replayable pruning blocks.
        """
        from itertools import combinations

        scope = sorted(domains)

        # 1) Singleton conflict: two already-fixed symbols cannot share a digit.
        before = self._snapshot_scope(domains, scope)
        fixed_by_digit: Dict[int, List[str]] = {}
        for ch in scope:
            vals = domains[ch]
            if len(vals) == 1:
                fixed_by_digit.setdefault(next(iter(vals)), []).append(ch)
        duplicate_fixed = {d: syms for d, syms in fixed_by_digit.items() if len(syms) > 1}
        if duplicate_fixed:
            witness_digit = sorted(duplicate_fixed)[0]
            witness_symbols = tuple(sorted(duplicate_fixed[witness_digit]))
            trace.add_event(ReplayEvent(
                kind="ALLDIFFERENT",
                title="AllDifferent block",
                constraint="fixed digits must be unique",
                before=before,
                after=before,
                changes={},
                decision="reject",
                reason=(
                    f"fixed digit {witness_digit} is assigned to multiple symbols "
                    f"{', '.join(repr(ch) for ch in witness_symbols)}"
                ),
                meta={"duplicate_digit": witness_digit, "duplicate_symbols": witness_symbols},
            ))
            return False

        # 2) Standard singleton elimination.
        fixed = set(fixed_by_digit)
        if fixed:
            before = self._snapshot_scope(domains, scope)
            for ch in scope:
                vals = domains[ch]
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
                        meta={"fixed_digits": tuple(sorted(fixed))},
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
                    meta={"fixed_digits": tuple(sorted(fixed))},
                ))

        # 3) Hall-set propagation.  Repeat inside this call because one Hall
        #    pruning may expose another before the next equation projection.
        emitted_no_change = False
        while True:
            hall_changed = False
            n = len(scope)
            for size in range(2, n + 1):
                for hall_symbols in combinations(scope, size):
                    union_digits: Set[int] = set()
                    for ch in hall_symbols:
                        union_digits.update(domains[ch])

                    # Hall violation: too few distinct digits remain for this subset.
                    if len(union_digits) < size:
                        before = self._snapshot_scope(domains, scope)
                        trace.add_event(ReplayEvent(
                            kind="ALLDIFFERENT",
                            title="AllDifferent Hall block",
                            constraint="Hall set under injective digit map",
                            before=before,
                            after=before,
                            changes={},
                            decision="reject",
                            reason=(
                                f"{size} symbols have only {len(union_digits)} possible distinct digits; "
                                "injective assignment is impossible"
                            ),
                            meta={
                                "hall_symbols": tuple(hall_symbols),
                                "hall_digits": tuple(sorted(union_digits)),
                                "hall_symbol_count": size,
                                "hall_digit_count": len(union_digits),
                            },
                        ))
                        return False

                    # Proper Hall set: these digits are reserved for this subset.
                    # size == n has no outside symbols to prune, but is still useful
                    # for contradiction detection above.
                    if len(union_digits) != size or size == n:
                        continue

                    outside = [ch for ch in scope if ch not in hall_symbols]
                    before = self._snapshot_scope(domains, scope)
                    for ch in outside:
                        domains[ch].difference_update(union_digits)
                        if not domains[ch]:
                            after = self._snapshot_scope(domains, scope)
                            trace.add_event(ReplayEvent(
                                kind="ALLDIFFERENT",
                                title="AllDifferent Hall block",
                                constraint="Hall set under injective digit map",
                                before=before,
                                after=after,
                                changes=self._changes(before, after),
                                decision="reject",
                                reason=(
                                    f"Hall digits {self._dom(union_digits)} are reserved for "
                                    f"{tuple(hall_symbols)}, leaving domain({repr(ch)}) empty"
                                ),
                                meta={
                                    "hall_symbols": tuple(hall_symbols),
                                    "hall_digits": tuple(sorted(union_digits)),
                                    "hall_symbol_count": size,
                                    "hall_digit_count": len(union_digits),
                                },
                            ))
                            return False
                    after = self._snapshot_scope(domains, scope)
                    changes = self._changes(before, after)
                    if changes:
                        self.stats.domain_reductions += len(changes)
                        trace.add_event(ReplayEvent(
                            kind="ALLDIFFERENT",
                            title="AllDifferent Hall block",
                            constraint="Hall set under injective digit map",
                            before=before,
                            after=after,
                            changes=changes,
                            decision="keep",
                            reason=(
                                f"{size} symbols have exactly {size} possible digits, so "
                                "those digits are reserved for that symbol set"
                            ),
                            meta={
                                "hall_symbols": tuple(hall_symbols),
                                "hall_digits": tuple(sorted(union_digits)),
                                "hall_symbol_count": size,
                                "hall_digit_count": len(union_digits),
                            },
                        ))
                        hall_changed = True
                        break
                if hall_changed:
                    break
            if not hall_changed:
                if self._cfg("include_no_change_events", False) and not emitted_no_change:
                    before = self._snapshot_scope(domains, scope)
                    trace.add_event(ReplayEvent(
                        kind="ALLDIFFERENT",
                        title="AllDifferent block",
                        constraint="singleton and Hall-set pruning under injective digit map",
                        before=before,
                        after=before,
                        changes={},
                        decision="no-change",
                        reason="no fixed digit or Hall set can prune another domain",
                    ))
                    emitted_no_change = True
                return True

    @staticmethod
    @staticmethod
    def _bounds(length: int) -> Tuple[int, int]:
        return (0, 9) if length <= 1 else (10 ** (length - 1), 10**length - 1)

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

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    @staticmethod
    @staticmethod
    def _changes(before: Dict[str, Tuple[int, ...]], after: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        return {ch: (before[ch], after[ch]) for ch in before if before[ch] != after.get(ch, ())}

    @staticmethod
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

    def _choose_column_assignments(
        self,
        symbols: Sequence[str],
        domains: Domains,
        assign: Assignment,
        used: Set[int],
        callback: Callable[[], None],
    ) -> None:
        needed = [s for s in dict.fromkeys(symbols) if s not in assign]
        needed.sort(key=lambda ch: (len(domains[ch]), ch))

        def bt(i: int) -> None:
            if time.time() > self._deadline:
                raise TimeoutError("timeout")
            if i == len(needed):
                callback()
                return
            ch = needed[i]
            for d in sorted(domains[ch]):
                if d in used:
                    continue
                assign[ch] = d
                used.add(d)
                bt(i + 1)
                used.remove(d)
                del assign[ch]

        bt(0)

    def _collect_solution(self, assign: Assignment, scope: Sequence[str], supported: Dict[str, Set[int]]) -> None:
        for ch in dict.fromkeys(scope):
            supported[ch].add(assign[ch])

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

    @staticmethod
    @staticmethod
    def _copy_domains(domains: Domains) -> Domains:
        return {ch: set(vals) for ch, vals in domains.items()}

    def _derive_abs_support(self, eq: Equation, rule: Rule, domains: Domains) -> Optional[Dict[str, Any]]:
        if rule.name != "abs":
            return None
        a = self._low_to_high_symbols(eq.left, rule)
        b = self._low_to_high_symbols(eq.right, rule)
        r = self._low_to_high_symbols(eq.result, rule)
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        if len(scope) > int(self._cfg("max_rule_specific_variables", 8)):
            return None
        case1 = self._derive_sub_positive_support(a, b, r, domains, scope, "case A-B=R: borrow propagation")
        case2 = self._derive_sub_positive_support(b, a, r, domains, scope, "case B-A=R: borrow propagation")
        return self._merge_support_results(
            [case1, case2], scope,
            [f"method: abs case split for {self._rule_name(rule)}: abs(A-B)=R means A-B=R or B-A=R"],
        )

    def _derive_add_support(self, eq: Equation, rule: Rule, domains: Domains) -> Optional[Dict[str, Any]]:
        if rule.name not in {"add", "add1", "addm1"}:
            return None
        delta = 1 if rule.name == "add1" else -1 if rule.name == "addm1" else 0
        a_syms = self._low_to_high_symbols(eq.left, rule)
        b_syms = self._low_to_high_symbols(eq.right, rule)
        r_syms = self._low_to_high_symbols(eq.result, rule)
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        if len(scope) > int(self._cfg("max_rule_specific_variables", 8)):
            return None

        supported: Dict[str, Set[int]] = {ch: set() for ch in scope}
        count = 0
        max_col = max(len(a_syms), len(b_syms), len(r_syms))
        transitions: List[Set[Tuple[int, int]]] = [set() for _ in range(max_col)]

        def rec(col: int, carry: int, assign: Assignment, used: Set[int]) -> None:
            nonlocal count
            if time.time() > self._deadline:
                raise TimeoutError("timeout")
            if col == max_col:
                if carry == 0:
                    # Exact encoding has the displayed result length.  A multi-symbol
                    # result cannot have a zero most-significant encoded digit; otherwise
                    # str(value) would be shorter and _equation_matches() would reject it.
                    if len(r_syms) > 1 and assign.get(r_syms[-1], 0) == 0:
                        return
                    count += 1
                    self._collect_solution(assign, scope, supported)
                return
            a = a_syms[col] if col < len(a_syms) else None
            b = b_syms[col] if col < len(b_syms) else None
            r = r_syms[col] if col < len(r_syms) else None
            syms = [s for s in (a, b, r) if s is not None]

            def check() -> None:
                av = 0 if a is None else assign[a]
                bv = 0 if b is None else assign[b]
                rv = 0 if r is None else assign[r]
                total = av + bv + carry + (delta if col == 0 else 0)
                if total % 10 != rv:
                    return
                next_carry = total // 10
                transitions[col].add((carry, next_carry))
                rec(col + 1, next_carry, assign, used)

            self._choose_column_assignments(syms, domains, assign, used, check)

        rec(0, 0, {}, set())
        limit = int(self._cfg("max_transition_values_shown", 12))
        lines = [
            f"method: addition carry propagation for {self._rule_name(rule)}",
            f"offset at column 0: {delta}",
        ]
        for i in range(max_col):
            a = self._fmt_symbol_digit(a_syms[i] if i < len(a_syms) else None)
            b = self._fmt_symbol_digit(b_syms[i] if i < len(b_syms) else None)
            r = self._fmt_symbol_digit(r_syms[i] if i < len(r_syms) else None)
            extra = f" + {delta}" if i == 0 and delta else ""
            lines.append(f"column {i}: {a}+{b}+carry{extra} = {r} + 10*next_carry")
            lines.append(self._transition_line("carry", i, transitions[i], limit))
        return {"scope": scope, "supported": supported, "count": count, "lines": lines, "method": "addition carry DP"}

    def _derive_mul_support(self, eq: Equation, rule: Rule, domains: Domains) -> Optional[Dict[str, Any]]:
        if rule.name not in {"mul", "mul1", "mulm1"}:
            return None
        delta = 1 if rule.name == "mul1" else -1 if rule.name == "mulm1" else 0
        a_syms = self._low_to_high_symbols(eq.left, rule)
        b_syms = self._low_to_high_symbols(eq.right, rule)
        r_syms = self._low_to_high_symbols(eq.result, rule)
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        if len(scope) > int(self._cfg("max_rule_specific_variables", 8)):
            return None
        # This solver is tuned for the benchmark's 2x2 cryptarithms.  Fallback
        # for wider multiplication to avoid accidental slowdowns.
        if len(a_syms) > 3 or len(b_syms) > 3 or len(r_syms) > 6:
            return None

        supported: Dict[str, Set[int]] = {ch: set() for ch in scope}
        count = 0
        max_col = max(len(r_syms), len(a_syms) + len(b_syms))
        transitions: List[Set[Tuple[int, int]]] = [set() for _ in range(max_col)]

        def rec(col: int, carry: int, assign: Assignment, used: Set[int]) -> None:
            nonlocal count
            if time.time() > self._deadline:
                raise TimeoutError("timeout")
            if col == max_col:
                if carry == 0:
                    if len(r_syms) > 1 and assign.get(r_syms[-1], 0) == 0:
                        return
                    count += 1
                    self._collect_solution(assign, scope, supported)
                return
            involved: List[str] = []
            for i, a in enumerate(a_syms):
                j = col - i
                if 0 <= j < len(b_syms):
                    involved.extend([a, b_syms[j]])
            if col < len(r_syms):
                involved.append(r_syms[col])
            involved = [s for s in dict.fromkeys(involved) if s is not None]

            def check() -> None:
                total = carry + (delta if col == 0 else 0)
                for i, a in enumerate(a_syms):
                    j = col - i
                    if 0 <= j < len(b_syms):
                        total += assign[a] * assign[b_syms[j]]
                rv = assign[r_syms[col]] if col < len(r_syms) else 0
                if total % 10 != rv:
                    return
                next_carry = total // 10
                transitions[col].add((carry, next_carry))
                rec(col + 1, next_carry, assign, used)

            self._choose_column_assignments(involved, domains, assign, used, check)

        rec(0, 0, {}, set())
        limit = int(self._cfg("max_transition_values_shown", 12))
        lines = [
            f"method: grade-school multiplication carry propagation for {self._rule_name(rule)}",
            f"offset at column 0: {delta}",
        ]
        for col in range(max_col):
            terms = []
            for i, a in enumerate(a_syms):
                j = col - i
                if 0 <= j < len(b_syms):
                    terms.append(f"{repr(a)}*{repr(b_syms[j])}")
            if not terms:
                terms = ["0"]
            r = self._fmt_symbol_digit(r_syms[col] if col < len(r_syms) else None)
            extra = f" + {delta}" if col == 0 and delta else ""
            lines.append(f"column {col}: {' + '.join(terms)} + carry{extra} = {r} + 10*next_carry")
            lines.append(self._transition_line("carry", col, transitions[col], limit))
        return {"scope": scope, "supported": supported, "count": count, "lines": lines, "method": "multiplication carry DP"}

    def _derive_rule_specific_support(self, eq: Equation, rule: Rule, domains: Domains) -> Optional[Dict[str, Any]]:
        if rule.name in {"add", "add1", "addm1"}:
            return self._derive_add_support(eq, rule, domains)
        if rule.name == "sub":
            return self._derive_sub_support(eq, rule, domains)
        if rule.name == "abs":
            return self._derive_abs_support(eq, rule, domains)
        if rule.name in {"mul", "mul1", "mulm1"}:
            return self._derive_mul_support(eq, rule, domains)
        return None

    def _derive_sub_positive_support(
        self,
        minuend: List[str],
        subtrahend: List[str],
        result: List[str],
        domains: Domains,
        scope: Sequence[str],
        heading: str,
    ) -> Dict[str, Any]:
        supported: Dict[str, Set[int]] = {ch: set() for ch in scope}
        count = 0
        max_col = max(len(minuend), len(subtrahend), len(result))
        transitions: List[Set[Tuple[int, int]]] = [set() for _ in range(max_col)]

        def rec(col: int, borrow: int, assign: Assignment, used: Set[int]) -> None:
            nonlocal count
            if time.time() > self._deadline:
                raise TimeoutError("timeout")
            if col == max_col:
                if borrow == 0:
                    if len(result) > 1 and assign.get(result[-1], 0) == 0:
                        return
                    count += 1
                    self._collect_solution(assign, scope, supported)
                return
            m = minuend[col] if col < len(minuend) else None
            s = subtrahend[col] if col < len(subtrahend) else None
            r = result[col] if col < len(result) else None
            syms = [x for x in (m, s, r) if x is not None]

            def check() -> None:
                mv = 0 if m is None else assign[m]
                sv = 0 if s is None else assign[s]
                rv = 0 if r is None else assign[r]
                raw = mv - sv - borrow
                if raw < 0:
                    digit = raw + 10
                    next_borrow = 1
                else:
                    digit = raw
                    next_borrow = 0
                if digit != rv:
                    return
                transitions[col].add((borrow, next_borrow))
                rec(col + 1, next_borrow, assign, used)

            self._choose_column_assignments(syms, domains, assign, used, check)

        rec(0, 0, {}, set())
        limit = int(self._cfg("max_transition_values_shown", 12))
        lines = [heading]
        for i in range(max_col):
            m = self._fmt_symbol_digit(minuend[i] if i < len(minuend) else None)
            s = self._fmt_symbol_digit(subtrahend[i] if i < len(subtrahend) else None)
            r = self._fmt_symbol_digit(result[i] if i < len(result) else None)
            lines.append(f"column {i}: {m}-{s}-borrow = {r} modulo 10")
            lines.append(self._transition_line("borrow", i, transitions[i], limit))
        return {"supported": supported, "count": count, "lines": lines}

    def _derive_sub_support(self, eq: Equation, rule: Rule, domains: Domains) -> Optional[Dict[str, Any]]:
        if rule.name != "sub":
            return None
        a = self._low_to_high_symbols(eq.left, rule)
        b = self._low_to_high_symbols(eq.right, rule)
        r = self._low_to_high_symbols(eq.result, rule)
        scope = list(dict.fromkeys(eq.left + eq.right + eq.result))
        if len(scope) > int(self._cfg("max_rule_specific_variables", 8)):
            return None
        if eq.has_sign:
            # A-B=-R, so use B-A=R.
            res = self._derive_sub_positive_support(
                b, a, r, domains, scope,
                f"method: signed subtraction rewrite for {self._rule_name(rule)}: A-B=-R becomes B-A=R; then borrow propagation",
            )
        else:
            res = self._derive_sub_positive_support(
                a, b, r, domains, scope,
                f"method: unsigned subtraction borrow propagation for {self._rule_name(rule)}",
            )
        res["scope"] = scope
        res["method"] = "subtraction borrow DP"
        return res

    @staticmethod
    @staticmethod
    def _dom(values: Iterable[int]) -> str:
        vals = sorted(set(values))
        if vals == list(range(10)):
            return "{0..9}"
        return "{" + ",".join(str(v) for v in vals) + "}"

    @staticmethod
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

    @staticmethod
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

        Rule rejections can be local failures or downstream prefix failures.
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

    def _family_reason_details(self, reason: str, rules: Sequence[Rule], eqs: Sequence[Equation]) -> List[str]:
        rep = sorted(rules, key=self._rule_sort_key)[0]
        if reason == "sign":
            return ["signed output is present in the examples, but these rules are unsigned."]
        if reason == "concat positional":
            lines: List[str] = []
            for rule in sorted(rules, key=self._rule_sort_key):
                for eq in eqs:
                    expected = self._cat_expected_output(eq, rule)
                    observed = eq.result
                    if len(expected) == len(observed) and expected != observed:
                        mismatch = next(
                            (idx, exp, obs)
                            for idx, (exp, obs) in enumerate(zip(expected, observed), 1)
                            if exp != obs
                        )
                        idx, exp, obs = mismatch
                        lines.append(
                            f"{self._rule_name(rule)} on {eq.display()} requires visible output {expected!r}; "
                            f"observed {observed!r}; first mismatch at position {idx}: expected {exp!r}, observed {obs!r}. "
                            "Because the digit map is injective, different visible symbols cannot be identified."
                        )
                        break
            return lines or ["concat visible output does not match the positional symbol pattern required by these rules."]
        if reason in {"value range", "signed value range", "unsigned value range"}:
            lines: List[str] = []
            for rule in sorted(rules, key=self._rule_sort_key):
                for eq in eqs:
                    ok, why = self._bounds_sign_status(rule, eq)
                    if not ok and why == reason:
                        lines.append(self._bounds_sign_reason(rule, eq))
                        break
            return lines or ["numeric value range cannot overlap the visible output range with the required sign."]

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

    @staticmethod
    @staticmethod
    def _fmt_symbol_digit(sym: Optional[str]) -> str:
        return "0" if sym is None else repr(sym)

    @staticmethod
    @staticmethod
    def _fmt_value_set(values: Iterable[int], limit: int = 12) -> str:
        vals = sorted(set(values))
        if not vals:
            return "{}"
        if len(vals) <= limit:
            return "{" + ",".join(str(v) for v in vals) + "}"
        head = ",".join(str(v) for v in vals[:limit])
        return "{" + head + f", ... {len(vals) - limit} more" + "}"

    @staticmethod
    @staticmethod
    def _is_empty_supported_projection(event: ReplayEvent) -> bool:
        if event.supported is None:
            return False
        return all(len(tuple(vals)) == 0 for vals in event.supported.values())

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

    def _target_leading_zero(
        self,
        target_left: str,
        target_op: str,
        target_right: str,
        combo: Dict[str, Rule],
        domains: Domains,
        trace: ReplayTrace,
    ) -> bool:
        """Apply the same no-leading-zero convention to target operands.

        This is rule-orientation dependent: std reads the visible first symbol as
        the leading digit, while rev reads the visible last symbol as the leading
        digit.  The target has no known result yet, so only target operands are
        constrained here.
        """
        if not self._cfg("use_target_no_leading_zero", True):
            return True
        rule = combo.get(target_op)
        if rule is None:
            return True
        scope: List[str] = []
        terms: List[str] = []
        for term in (target_left, target_right):
            if len(term) <= 1:
                continue
            lead = term[-1] if rule.reverse else term[0]
            scope.append(lead)
            terms.append(f"{repr(lead)} is leading digit of target operand {repr(term)} under {self._rule_name(rule)}")
        scope = list(dict.fromkeys(scope))
        if not scope:
            return True
        before = self._snapshot_scope(domains, scope)
        for lead in scope:
            if 0 in domains[lead]:
                domains[lead].remove(0)
                if not domains[lead]:
                    after = self._snapshot_scope(domains, scope)
                    trace.add_event(ReplayEvent(
                        kind="NO_LEADING_ZERO",
                        title="Target no-leading-zero block",
                        constraint="target operands use the same no-leading-zero convention as training operands",
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
                title="Target no-leading-zero block",
                constraint="target operands use the same no-leading-zero convention as training operands",
                before=before,
                after=after,
                changes=changes,
                decision="keep",
                reason="; ".join(terms),
                meta={"target_terms": (target_left, target_right), "target_op": target_op, "rule": self._rule_name(rule)},
            ))
        elif self._cfg("include_no_change_events", False):
            trace.add_event(ReplayEvent(
                kind="NO_LEADING_ZERO",
                title="Target no-leading-zero block",
                constraint="target operands use the same no-leading-zero convention as training operands",
                before=before,
                after=after,
                changes={},
                decision="no-change",
                reason="; ".join(terms),
                meta={"target_terms": (target_left, target_right), "target_op": target_op, "rule": self._rule_name(rule)},
            ))
        return True

    @staticmethod
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

    @staticmethod
    @staticmethod
    def _literal(s: str) -> str:
        return repr(s)

    def _low_to_high_symbols(self, term: str, rule: Rule) -> List[str]:
        # _number() reads term[::-1] for rev and term for std as high-to-low.
        # Therefore low-to-high is term for rev and reversed(term) for std.
        return list(term if rule.reverse else term[::-1])

    def _merge_support_results(self, results: Sequence[Dict[str, Any]], scope: Sequence[str], intro: List[str]) -> Dict[str, Any]:
        supported: Dict[str, Set[int]] = {ch: set() for ch in scope}
        count = 0
        lines = list(intro)
        for idx, res in enumerate(results, 1):
            count += int(res.get("count", 0))
            for ch, vals in res.get("supported", {}).items():
                supported[ch].update(vals)
            lines.append(f"case {idx}: satisfying assignments = {res.get('count', 0)}")
            for line in res.get("lines", [])[:8]:
                lines.append("  " + line)
        return {"scope": list(scope), "supported": supported, "count": count, "lines": lines, "method": "case split DP"}

    @staticmethod
    @staticmethod
    def _number(text: str, assignment: Assignment, reverse: bool) -> int:
        digits = text[::-1] if reverse else text
        value = 0
        for ch in digits:
            value = value * 10 + assignment[ch]
        return value

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

    @staticmethod
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

    @staticmethod
    @staticmethod
    def _preferred_operator_index(left_side: str) -> int:
        return 2 if len(left_side) >= 5 else len(left_side) // 2

    def _project(
        self,
        domains: Domains,
        scope: Sequence[str],
        predicate: Callable[[Assignment], bool],
        label: str,
        trace: ReplayTrace,
    ) -> Optional[bool]:
        return self._project_replay(domains, scope, predicate, label, trace, constraint=label, kind="PROJECTION")

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

    def _propagate(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: Trace, target_context: Optional[Tuple[str, str, str]] = None) -> bool:
        while True:
            before = {ch: tuple(sorted(vals)) for ch, vals in domains.items()}
            if not self._leading_zero(equations, combo, domains, trace):
                return False
            if target_context is not None:
                target_left, target_op, target_right = target_context
                if not self._target_leading_zero(target_left, target_op, target_right, combo, domains, trace):
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

    def _render_projection_derivation(self, event: ReplayEvent, indent: str = "  ") -> List[str]:
        if not self._cfg("show_projection_derivation", True):
            return []
        meta = event.meta or {}
        proof = meta.get("projection_derivation") if isinstance(meta, dict) else None
        if not isinstance(proof, dict):
            return []
        lines: List[str] = [f"{indent}projection derivation:"]
        deriv_lines = proof.get("derivation_lines") or ()
        if deriv_lines:
            for line in deriv_lines:
                lines.append(f"{indent}  {line}")
        else:
            # Fallback to the v2.5 generic renderer, but without large witness lists.
            method = proof.get("method")
            if method:
                lines.append(f"{indent}  method: {method}")
            scope = proof.get("scope", ())
            if scope:
                scope_text = ", ".join(f"D[{repr(ch)}]" for ch in scope)
                lines.append(f"{indent}  local variables: {scope_text}")
            sat = proof.get("satisfying_count", event.count if event.count is not None else "unknown")
            lines.append(f"{indent}  satisfying assignments: {sat}")
        sat = proof.get("satisfying_count", event.count if event.count is not None else "unknown")
        lines.append(f"{indent}  satisfying assignments after column propagation: {sat}")
        union_supported = proof.get("union_supported", ()) or ()
        if union_supported:
            lines.append(f"{indent}  union by symbol gives: {self._tuple_domain_text(tuple(union_supported))}")
        return lines

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

    def _rule_family_order(self) -> List[Tuple[str, List[str]]]:
        return [
            ("concat", ["cat"]),
            ("add_offset", ["add", "add1", "addm1"]),
            ("abs_sub", ["abs", "sub"]),
            ("mul_offset", ["mul", "mul1", "mulm1"]),
        ]

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

    @staticmethod
    @staticmethod
    def _rule_name(rule: Rule) -> str:
        return f"{rule.orientation}/{rule.name}"

    def _rule_sort_key(self, rule: Rule) -> Tuple[int, int, str]:
        family_rank = {name: i for i, (_, names) in enumerate(self._rule_family_order()) for name in names}
        return (family_rank.get(rule.name, 99), 0 if rule.orientation == "std" else 1, rule.name)

    def _audit_target_outputs(
        self,
        ordered_ops: Sequence[str],
        candidates: Dict[str, List[Rule]],
        equations: Sequence[Equation],
        initial_domains: Domains,
        target_left: str,
        target_op: str,
        target_right: str,
        selected_answer: Optional[str],
    ) -> Dict[str, Any]:
        """Exhaustively audit whether the target output is invariant.

        The main solver deliberately stops at the first valid solution.  This
        audit is a separate silent search over every structurally kept rule
        combo and every digit map that satisfies the training examples.  It is
        allowed to stop early only when it can already prove ambiguity, or when
        explicit audit limits/timeouts are hit; incomplete audits are rendered
        as incomplete rather than as uniqueness proofs.
        """
        selected_output = selected_answer if selected_answer is not None else "nan"
        result: Dict[str, Any] = {
            "enabled": bool(self._cfg("include_ambiguity_audit", True)),
            "complete": False,
            "stopped_reason": "not run",
            "rule_combos_tested": 0,
            "valid_rule_combos": 0,
            "valid_complete_solutions": 0,
            "target_outputs": tuple(),
            "selected_output": selected_output,
            "decision": "audit not run",
        }
        if not result["enabled"]:
            return result

        max_solutions = int(self._cfg("max_audit_solutions", 100000))
        max_rule_combos = int(self._cfg("max_audit_rule_combos", 100000))
        stop_after_outputs = int(self._cfg("audit_stop_after_distinct_target_outputs", 2))
        target_outputs: Set[str] = set()
        valid_rule_combos: Set[Tuple[Tuple[str, str], ...]] = set()
        previous_stats = dict(self.stats.__dict__)
        stopped = {"reason": "exhausted"}

        def finish(complete: bool, reason: str) -> Dict[str, Any]:
            # Restore main-solve stats: the audit is intentionally silent and
            # should not make replay statistics look larger than the printed
            # replay path.  Then add explicit audit counters.
            self.stats.__dict__.clear()
            self.stats.__dict__.update(previous_stats)
            self.stats.audit_rule_combos_tested = result["rule_combos_tested"]
            self.stats.audit_valid_rule_combos = len(valid_rule_combos)
            self.stats.audit_valid_complete_solutions = result["valid_complete_solutions"]
            self.stats.audit_distinct_target_outputs = len(target_outputs)

            outputs = tuple(sorted(target_outputs))
            result.update({
                "complete": complete,
                "stopped_reason": reason,
                "valid_rule_combos": len(valid_rule_combos),
                "target_outputs": outputs,
            })
            if len(outputs) > 1:
                result["decision"] = "ambiguous item: multiple target outputs are consistent with the training examples"
            elif not complete:
                result["decision"] = "target-output uniqueness was not proven"
            elif len(outputs) == 0:
                result["decision"] = "no valid complete solution was found during audit"
            else:
                only = outputs[0]
                if only == selected_output:
                    result["decision"] = "target output is unique over all valid complete solutions"
                else:
                    result["decision"] = "audit found one target output, but it differs from the selected solution output"
            return result

        def should_stop_after_output() -> bool:
            if stop_after_outputs <= 0:
                return False
            if len(target_outputs) >= stop_after_outputs:
                stopped["reason"] = f"found {len(target_outputs)} distinct target outputs; ambiguity is already proven"
                return True
            return False

        def record_solution(combo: Dict[str, Rule], assignment: Assignment) -> bool:
            if not self._verify_assignment(equations, combo, assignment):
                return True
            answer, _, _, _, _ = self._encode_target(target_left, target_op, target_right, combo[target_op], assignment)
            target_outputs.add(answer if answer is not None else "nan")
            result["valid_complete_solutions"] += 1
            combo_key = tuple(sorted((op, self._rule_name(rule)) for op, rule in combo.items()))
            valid_rule_combos.add(combo_key)
            if result["valid_complete_solutions"] >= max_solutions:
                stopped["reason"] = f"reached max_audit_solutions={max_solutions} before exhaustive audit"
                return False
            if should_stop_after_output():
                return False
            return True

        def enumerate_digit_maps(combo: Dict[str, Rule], domains: Domains) -> bool:
            if time.time() > self._deadline:
                raise TimeoutError("audit timeout")
            unresolved = [ch for ch, vals in domains.items() if len(vals) > 1]
            if not unresolved:
                assignment = {ch: next(iter(vals)) for ch, vals in domains.items()}
                return record_solution(combo, assignment)

            tuple_branch = self._select_column_tuple_branch(equations, combo, domains)
            if tuple_branch is not None:
                for tuple_assignment in tuple_branch["tuples"]:
                    applied = self._apply_tuple_to_domains(domains, tuple_assignment)
                    if applied is None:
                        continue
                    branch_domains, _changes = applied
                    if self._propagate(equations, combo, branch_domains, SilentReplayTrace(), (target_left, target_op, target_right)):
                        if not enumerate_digit_maps(combo, branch_domains):
                            return False
                return True

            symbol = min(unresolved, key=lambda ch: (len(domains[ch]), ch))
            for digit in sorted(domains[symbol]):
                branch_domains = self._copy_domains(domains)
                branch_domains[symbol] = {digit}
                if self._propagate(equations, combo, branch_domains, SilentReplayTrace(), (target_left, target_op, target_right)):
                    if not enumerate_digit_maps(combo, branch_domains):
                        return False
            return True

        def search_rule_combos(index: int, combo: Dict[str, Rule], domains: Domains) -> bool:
            if time.time() > self._deadline:
                raise TimeoutError("audit timeout")
            if index == len(ordered_ops):
                result["rule_combos_tested"] += 1
                if result["rule_combos_tested"] > max_rule_combos:
                    stopped["reason"] = f"reached max_audit_rule_combos={max_rule_combos} before exhaustive audit"
                    return False
                combo_domains = self._copy_domains(domains)
                if not self._propagate(equations, combo, combo_domains, SilentReplayTrace(), (target_left, target_op, target_right)):
                    return True
                return enumerate_digit_maps(combo, combo_domains)

            op = ordered_ops[index]
            for rule in candidates[op]:
                combo2 = dict(combo)
                combo2[op] = rule
                domains2 = self._copy_domains(domains)
                if self._propagate(equations, combo2, domains2, SilentReplayTrace(), (target_left, target_op, target_right)):
                    if not search_rule_combos(index + 1, combo2, domains2):
                        return False
            return True

        try:
            completed = search_rule_combos(0, {}, self._copy_domains(initial_domains))
            if completed:
                return finish(True, "exhausted all structurally kept rule combos and digit maps")
            return finish(False, stopped["reason"])
        except TimeoutError:
            return finish(False, "audit timeout before exhaustive proof")
        except Exception as exc:
            return finish(False, f"audit error: {type(exc).__name__}: {exc}")

    def _search_rules(
        self,
        ordered_ops: Sequence[str],
        candidates: Dict[str, List[Rule]],
        equations: Sequence[Equation],
        domains: Domains,
        combo: Dict[str, Rule],
        trace: ReplayTrace,
        index: int,
        target_context: Optional[Tuple[str, str, str]] = None,
    ) -> Tuple[Optional[Assignment], Optional[Dict[str, Rule]]]:
        if time.time() > self._deadline:
            raise TimeoutError("timeout")
        if index == len(ordered_ops):
            domains2 = self._copy_domains(domains)
            trace2 = ReplayTrace(self.config)
            trace2.add_event(ReplayEvent(
                kind="COMBO_PROPAGATE",
                title="Rule combo propagation",
                decision="try",
                reason="all operators have candidate rules; start digit-domain solving",
                meta={"combo": tuple(sorted((op, self._rule_name(rule)) for op, rule in combo.items()))},
            ))
            if not self._propagate(equations, combo, domains2, trace2, target_context):
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
            assignment = self._solve_digits(equations, combo, domains2, trace2, target_context)
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
            trace2 = ReplayTrace(self.config)
            trace2.add_event(ReplayEvent(
                kind="RULE_TRY",
                title="Rule candidate",
                decision="try",
                reason=f"operator {repr(op)} -> {self._rule_name(rule)}",
                meta={"op": op, "rule": self._rule_name(rule), "depth": index},
            ))
            if self._propagate(equations, combo2, domains2, trace2, target_context):
                assignment, solved_combo = self._search_rules(ordered_ops, candidates, equations, domains2, combo2, trace2, index + 1, target_context)
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

    @staticmethod
    @staticmethod
    def _set_text(values: Iterable[Any]) -> str:
        vals = sorted(set(values))
        return "{" + ",".join(str(v) for v in vals) + "}"

    @staticmethod
    @staticmethod
    def _snapshot_scope(domains: Domains, scope: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
        return {ch: tuple(sorted(domains[ch])) for ch in dict.fromkeys(scope)}

    def _enumerate_local_digit_tuples(
        self,
        symbols: Sequence[str],
        domains: Domains,
        predicate: Callable[[Assignment], bool],
        max_tuples: int,
    ) -> List[Tuple[Tuple[str, int], ...]]:
        ordered = list(dict.fromkeys(symbols))
        out: List[Tuple[Tuple[str, int], ...]] = []
        local: Assignment = {}
        used: Set[int] = set()

        def bt(i: int) -> None:
            if len(out) > max_tuples:
                return
            if i == len(ordered):
                if predicate(local):
                    out.append(tuple((ch, local[ch]) for ch in ordered))
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
        # De-duplicate; a local assignment can be supported by multiple hidden carry values/cases.
        seen: Set[Tuple[Tuple[str, int], ...]] = set()
        uniq: List[Tuple[Tuple[str, int], ...]] = []
        for tup in out:
            if tup not in seen:
                seen.add(tup)
                uniq.append(tup)
        return uniq

    def _column_tuple_candidates_for_eq(
        self,
        eq: Equation,
        rule: Rule,
        domains: Domains,
        max_tuples: int,
    ) -> List[Dict[str, Any]]:
        if not self._cfg("use_column_tuple_branching", True):
            return []
        a_syms = self._low_to_high_symbols(eq.left, rule)
        b_syms = self._low_to_high_symbols(eq.right, rule)
        r_syms = self._low_to_high_symbols(eq.result, rule)
        candidates: List[Dict[str, Any]] = []

        def add_candidate(label: str, constraint: str, symbols: Sequence[str], pred: Callable[[Assignment], bool]) -> None:
            scope = [s for s in dict.fromkeys(symbols) if s is not None]
            if not scope or not any(len(domains[s]) > 1 for s in scope):
                return
            product_space = 1
            for s in scope:
                product_space *= len(domains[s])
            tuples = self._enumerate_local_digit_tuples(scope, domains, pred, max_tuples + 1)
            if not tuples or len(tuples) > max_tuples:
                return
            # Branching is useful only if the relation is stricter than the raw domain product.
            if len(tuples) >= product_space:
                return
            candidates.append({
                "label": label,
                "constraint": constraint,
                "symbols": tuple(scope),
                "tuples": tuple(tuples),
                "tuple_count": len(tuples),
                "product_space": product_space,
            })

        # Addition columns.  Column 0 has fixed carry-in 0; later columns allow both carry states.
        if rule.name in {"add", "add1", "addm1"}:
            delta = 1 if rule.name == "add1" else -1 if rule.name == "addm1" else 0
            max_col = max(len(a_syms), len(b_syms), len(r_syms))
            for col in range(max_col):
                a = a_syms[col] if col < len(a_syms) else None
                b = b_syms[col] if col < len(b_syms) else None
                r = r_syms[col] if col < len(r_syms) else None
                syms = [x for x in (a, b, r) if x is not None]
                carry_values = (0,) if col == 0 else (0, 1)
                def pred(local: Assignment, a=a, b=b, r=r, col=col, carry_values=carry_values) -> bool:
                    av = 0 if a is None else local[a]
                    bv = 0 if b is None else local[b]
                    rv = 0 if r is None else local[r]
                    for carry in carry_values:
                        total = av + bv + carry + (delta if col == 0 else 0)
                        if total >= 0 and total % 10 == rv:
                            return True
                    return False
                extra = f" + {delta}" if col == 0 and delta else ""
                add_candidate(
                    f"column {col} tuple relation for {eq.display()} via {self._rule_name(rule)}",
                    f"column {col}: a_digit+b_digit+carry{extra} = result_digit mod 10",
                    syms,
                    pred,
                )
            return candidates

        # Multiplication: use low column only, where carry-in is definitely 0.
        if rule.name in {"mul", "mul1", "mulm1"}:
            delta = 1 if rule.name == "mul1" else -1 if rule.name == "mulm1" else 0
            if a_syms and b_syms:
                a, b = a_syms[0], b_syms[0]
                r = r_syms[0] if r_syms else None
                syms = [x for x in (a, b, r) if x is not None]
                def pred(local: Assignment, a=a, b=b, r=r) -> bool:
                    total = local[a] * local[b] + delta
                    if total < 0:
                        return False
                    rv = 0 if r is None else local[r]
                    return total % 10 == rv
                extra = f" + {delta}" if delta else ""
                add_candidate(
                    f"low-column tuple relation for {eq.display()} via {self._rule_name(rule)}",
                    f"column 0: a0*b0{extra} = result0 mod 10",
                    syms,
                    pred,
                )
            return candidates

        # Subtraction: low column has borrow-in 0.  For signed outputs A-B=-R, use B-A=R.
        if rule.name == "sub":
            minuend = b_syms if eq.has_sign else a_syms
            subtrahend = a_syms if eq.has_sign else b_syms
            if minuend or subtrahend or r_syms:
                m = minuend[0] if minuend else None
                sub = subtrahend[0] if subtrahend else None
                r = r_syms[0] if r_syms else None
                syms = [x for x in (m, sub, r) if x is not None]
                def pred(local: Assignment, m=m, sub=sub, r=r) -> bool:
                    mv = 0 if m is None else local[m]
                    sv = 0 if sub is None else local[sub]
                    rv = 0 if r is None else local[r]
                    raw = mv - sv
                    digit = raw + 10 if raw < 0 else raw
                    return digit == rv
                rewrite = "B-A=R" if eq.has_sign else "A-B=R"
                add_candidate(
                    f"low-column tuple relation for {eq.display()} via {self._rule_name(rule)}",
                    f"column 0 subtraction after rewrite {rewrite}: minuend0-subtrahend0 = result0 mod 10",
                    syms,
                    pred,
                )
            return candidates

        # Absolute difference: low column can come from A-B=R or B-A=R.
        if rule.name == "abs":
            if a_syms or b_syms or r_syms:
                a = a_syms[0] if a_syms else None
                b = b_syms[0] if b_syms else None
                r = r_syms[0] if r_syms else None
                syms = [x for x in (a, b, r) if x is not None]
                def pred(local: Assignment, a=a, b=b, r=r) -> bool:
                    av = 0 if a is None else local[a]
                    bv = 0 if b is None else local[b]
                    rv = 0 if r is None else local[r]
                    raw1 = av - bv
                    raw2 = bv - av
                    d1 = raw1 + 10 if raw1 < 0 else raw1
                    d2 = raw2 + 10 if raw2 < 0 else raw2
                    return d1 == rv or d2 == rv
                add_candidate(
                    f"low-column tuple relation for {eq.display()} via {self._rule_name(rule)}",
                    "column 0 abs split: either A-B=R or B-A=R modulo 10",
                    syms,
                    pred,
                )
            return candidates

        return candidates

    def _select_column_tuple_branch(
        self,
        equations: Sequence[Equation],
        combo: Dict[str, Rule],
        domains: Domains,
    ) -> Optional[Dict[str, Any]]:
        max_tuples = int(self._cfg("max_column_branch_tuples", 24))
        if max_tuples <= 0:
            return None
        candidates: List[Dict[str, Any]] = []
        for eq in equations:
            rule = combo.get(eq.op)
            if rule is None:
                continue
            candidates.extend(self._column_tuple_candidates_for_eq(eq, rule, domains, max_tuples))
        if not candidates:
            return None
        # Prefer the smallest relation, then the relation that fixes more unresolved symbols.
        candidates.sort(key=lambda c: (c["tuple_count"], -sum(1 for s in c["symbols"] if len(domains[s]) > 1), c["label"]))
        return candidates[0]

    def _apply_tuple_to_domains(
        self,
        domains: Domains,
        tuple_assignment: Sequence[Tuple[str, int]],
    ) -> Optional[Tuple[Domains, Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]]:
        branch_domains = self._copy_domains(domains)
        changes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        for ch, digit in tuple_assignment:
            if digit not in branch_domains[ch]:
                return None
            before = tuple(sorted(branch_domains[ch]))
            after = (digit,)
            branch_domains[ch] = {digit}
            if before != after:
                changes[ch] = (before, after)
        if not changes:
            return None
        return branch_domains, changes

    def _solve_digits(self, equations: Sequence[Equation], combo: Dict[str, Rule], domains: Domains, trace: ReplayTrace, target_context: Optional[Tuple[str, str, str]] = None) -> Optional[Assignment]:
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

        tuple_branch = self._select_column_tuple_branch(equations, combo, domains)
        if tuple_branch is not None:
            tuple_items = tuple(tuple(t) for t in tuple_branch["tuples"])
            trace.add_event(ReplayEvent(
                kind="TUPLE_BRANCH_SELECT",
                title="Column tuple branch selection",
                label=tuple_branch["label"],
                constraint=tuple_branch["constraint"],
                decision="select",
                reason=(
                    f"choose smallest local column relation: {tuple_branch['tuple_count']} tuples "
                    f"instead of raw domain product {tuple_branch['product_space']}"
                ),
                meta={
                    "symbols": tuple_branch["symbols"],
                    "tuple_count": tuple_branch["tuple_count"],
                    "product_space": tuple_branch["product_space"],
                    "tuples": tuple_items,
                },
            ))
            for idx, tuple_assignment in enumerate(tuple_branch["tuples"], 1):
                applied = self._apply_tuple_to_domains(domains, tuple_assignment)
                if applied is None:
                    continue
                self.stats.tuple_branches += 1
                branch_domains, changes = applied
                tuple_text = "{" + ", ".join(f"{repr(ch)}={digit}" for ch, digit in tuple_assignment) + "}"
                branch_trace = ReplayTrace(self.config)
                branch_trace.add_event(ReplayEvent(
                    kind="TUPLE_TRY",
                    title="Column tuple branch",
                    label=tuple_branch["label"],
                    constraint=tuple_branch["constraint"],
                    before={ch: before for ch, (before, _after) in changes.items()},
                    after={ch: after for ch, (_before, after) in changes.items()},
                    changes=changes,
                    decision="try",
                    reason=tuple_text,
                    meta={"tuple_index": idx, "tuple": tuple(tuple_assignment)},
                ))
                if self._propagate(equations, combo, branch_domains, branch_trace, target_context):
                    found = self._solve_digits(equations, combo, branch_domains, branch_trace, target_context)
                    if found is not None:
                        trace.extend(branch_trace)
                        return found
                    reject_reason = branch_trace.reject_reason()
                else:
                    reject_reason = branch_trace.reject_reason()
                failure_event = branch_trace.events[-1] if branch_trace.events else None
                trace.add_event(ReplayEvent(
                    kind="TUPLE_REJECT",
                    title="Column tuple branch rejected",
                    label=(failure_event.label if failure_event else tuple_branch["label"]),
                    constraint=(failure_event.constraint if failure_event else tuple_branch["constraint"]),
                    before=(failure_event.before if failure_event else None),
                    supported=(failure_event.supported if failure_event else None),
                    after=(failure_event.after if failure_event else None),
                    changes=(failure_event.changes if failure_event else None),
                    count=(failure_event.count if failure_event else None),
                    decision="reject",
                    reason=f"{tuple_text}: {reject_reason}",
                    meta={"tuple_index": idx, "tuple": tuple(tuple_assignment), "failure_kind": failure_event.kind if failure_event else None},
                ))
                if self._cfg("include_failed_attempt_events", False):
                    trace.extend(branch_trace)
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
            branch_trace = ReplayTrace(self.config)
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
            if self._propagate(equations, combo, branch_domains, branch_trace, target_context):
                found = self._solve_digits(equations, combo, branch_domains, branch_trace, target_context)
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

    @staticmethod
    @staticmethod
    def _split_decision_reason(reason: str) -> Tuple[str, str]:
        """Split a renderer reason into ``candidate`` and ``failure`` safely.

        Old code used ``text.split(":", 1)``.  That breaks when the operator
        itself is a colon, because reasons are rendered as, for example:

            operator ':' -> std/add: no supported local digit assignment

        Splitting at the first colon turns the candidate into ``operator '``.
        This helper first recognizes the structured rule-candidate prefix and
        splits only after the ``std/add``-style rule name.  It also supports
        quoted arbitrary operator symbols such as ``":"``, ``"'"`` and
        ``"\"``.
        """
        text = str(reason or "").strip()
        if not text:
            return "", ""

        op_lit = r"(?:\'[^\']*\'|\"[^\"]*\"|\\S+)"
        rule_name = r"[a-z]+/[a-z0-9]+"

        # Most RULE_REJECT reasons have this exact shape:
        #   operator <op> -> <rule>: <failure>
        m = re.match(rf"^(operator\s+{op_lit}\s*->\s*{rule_name})\s*:\s*(.*)$", text)
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # Combo strings can appear in COMBO_REJECT-like contexts.
        # Avoid splitting inside quoted symbols and prefer the separator after
        # a rule name whenever possible.
        m = re.match(rf"^(.+?->\s*{rule_name})\s*:\s*(.*)$", text)
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # Digit branch reasons still use a simple form such as "'x'=3: ...";
        # for those, a plain first-colon split is safe enough because the
        # candidate prefix is not an operator literal expression.
        if ":" in text:
            head, tail = text.split(":", 1)
            return head.strip(), tail.strip()
        return text, text

    @staticmethod
    @staticmethod
    def _strip_downstream_prefix(text: str) -> str:
        """Remove the leading "operator X -> rule:" part from a downstream failure."""
        return re.sub(
            r"^\s*operator\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s*->\s*[a-z]+/[a-z0-9]+\s*:\s*",
            "",
            str(text or "").strip(),
            count=1,
        )

    def _result_value_interval(self, eq: Equation) -> Tuple[int, int]:
        lo, hi = self._bounds(len(eq.result))
        if eq.has_sign:
            return -hi, -lo
        return lo, hi

    @staticmethod
    def _intervals_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return max(a[0], b[0]) <= min(a[1], b[1])

    def _rule_value_interval(self, rule: Rule, eq: Equation) -> Tuple[int, int]:
        lo_l, hi_l = self._bounds(len(eq.left))
        lo_r, hi_r = self._bounds(len(eq.right))
        if rule.name == "cat":
            return self._bounds(len(eq.left) + len(eq.right))
        if rule.name in {"add", "add1", "addm1"}:
            delta = 1 if rule.name == "add1" else -1 if rule.name == "addm1" else 0
            lo, hi = lo_l + lo_r + delta, hi_l + hi_r + delta
            if rule.name == "addm1":
                lo = max(0, lo)
            return lo, hi
        if rule.name in {"mul", "mul1", "mulm1"}:
            delta = 1 if rule.name == "mul1" else -1 if rule.name == "mulm1" else 0
            lo, hi = lo_l * lo_r + delta, hi_l * hi_r + delta
            if rule.name == "mulm1":
                lo = max(0, lo)
            return lo, hi
        if rule.name == "sub":
            return lo_l - hi_r, hi_l - lo_r
        # abs(A-B): the lower bound is 0 iff the operand intervals overlap;
        # otherwise it is the distance between the closest endpoints.
        if self._intervals_overlap((lo_l, hi_l), (lo_r, hi_r)):
            lo = 0
        else:
            lo = min(abs(lo_l - hi_r), abs(lo_r - hi_l))
        hi = max(abs(a - b) for a in (lo_l, hi_l) for b in (lo_r, hi_r))
        return lo, hi

    def _bounds_sign_status(self, rule: Rule, eq: Equation) -> Tuple[bool, str]:
        """Reject structurally impossible numeric ranges/signs before digit search."""
        if not self._cfg("use_bounds_sign_structural_pruning", True):
            return True, "keep"
        if rule.name == "cat":
            return True, "keep"
        value_interval = self._rule_value_interval(rule, eq)
        output_interval = self._result_value_interval(eq)
        if rule.signed:
            if eq.has_sign:
                feasible = (value_interval[0] < 0) and self._intervals_overlap(value_interval, output_interval)
                return (True, "keep") if feasible else (False, "signed value range")
            feasible = (value_interval[1] >= 0) and self._intervals_overlap(value_interval, output_interval)
            return (True, "keep") if feasible else (False, "unsigned value range")
        feasible = self._intervals_overlap(value_interval, output_interval)
        return (True, "keep") if feasible else (False, "value range")

    def _bounds_sign_reason(self, rule: Rule, eq: Equation) -> str:
        value_interval = self._rule_value_interval(rule, eq)
        output_interval = self._result_value_interval(eq)
        sign_text = "signed" if eq.has_sign else "unsigned"
        return (
            f"bounds/sign check for {self._rule_name(rule)} on {eq.display()}: "
            f"rule value range [{value_interval[0]},{value_interval[1]}], "
            f"visible {sign_text} output range [{output_interval[0]},{output_interval[1]}]; "
            "ranges do not overlap with the required sign -> no"
        )

    def _structural_reason(self, rule: Rule, eq: Equation) -> str:
        if eq.has_sign and not rule.signed:
            return f"signed output is present, but {self._rule_name(rule)} is unsigned -> no"
        possible = self._possible_result_lengths(rule, eq)
        actual = len(eq.result)
        lengths = "{" + ",".join(str(x) for x in sorted(possible)) + "}"
        if actual not in possible:
            return f"lengths {len(eq.left)} and {len(eq.right)} can produce result length in {lengths}; actual length {actual} -> no"
        if rule.name == "cat":
            expected = self._cat_expected_output(eq, rule)
            observed = eq.result
            if expected == observed:
                return f"concat positional check requires visible output {expected!r}; observed {observed!r} -> possible"
            mismatch = next(
                (idx, exp, obs)
                for idx, (exp, obs) in enumerate(zip(expected, observed), 1)
                if exp != obs
            )
            idx, exp, obs = mismatch
            return (
                f"concat positional check requires visible output {expected!r}; observed {observed!r}; "
                f"first mismatch at position {idx}: expected {exp!r}, observed {obs!r}; "
                "injective digit map forbids identifying different symbols -> no"
            )
        bounds_ok, _ = self._bounds_sign_status(rule, eq)
        if not bounds_ok:
            return self._bounds_sign_reason(rule, eq)
        return f"lengths {len(eq.left)} and {len(eq.right)} can produce result length in {lengths}; actual length {actual}; bounds/sign ranges overlap -> possible"

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
            if rule.name == "cat" and self._cat_expected_output(eq, rule) != eq.result:
                return False, "concat positional"
            bounds_ok, bounds_reason = self._bounds_sign_status(rule, eq)
            if not bounds_ok:
                return False, bounds_reason
        return True, "keep"

    def _suffix(self, term: str, k: int, reverse: bool) -> str:
        return term[:k] if reverse else term[-k:]

    def _support_key(self, label: str, domains: Domains, scope: Sequence[str]) -> Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]]:
        ordered = tuple((ch, tuple(sorted(domains[ch]))) for ch in dict.fromkeys(scope))
        return (label, ordered)

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

    @staticmethod
    @staticmethod
    def _symbols(equations: Sequence[Equation], target_left: str, target_right: str) -> List[str]:
        counts: Dict[str, int] = {}
        for eq in equations:
            for ch in eq.left + eq.right + eq.result:
                counts[ch] = counts.get(ch, 0) + 1
        for ch in target_left + target_right:
            counts[ch] = counts.get(ch, 0) + 1
        return sorted(counts, key=lambda ch: (-counts[ch], ch))

    def _term_formula(self, term: str, rule: Rule) -> str:
        ordered = term[::-1] if rule.reverse else term
        parts: List[str] = []
        for i, ch in enumerate(ordered):
            power = len(ordered) - i - 1
            coeff = 10**power
            parts.append(repr(ch) if coeff == 1 else f"{coeff}*{repr(ch)}")
        return "+".join(parts) if parts else "0"

    def _transition_line(self, name: str, idx: int, transitions: Set[Tuple[int, int]], limit: int) -> str:
        if not transitions:
            return f"column {idx}: no surviving {name} transition"
        pairs = sorted(transitions)
        shown = pairs[:limit]
        text = ", ".join(f"{a}->{b}" for a, b in shown)
        if len(pairs) > limit:
            text += f", ... {len(pairs) - limit} more"
        return f"column {idx}: surviving {name} transitions {{{text}}}"

    @staticmethod
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

    def _use_exact_first_for_rule(self, rule: Rule) -> bool:
        return (
            bool(self._cfg("skip_modular_when_rule_specific_exact", True))
            and bool(self._cfg("use_rule_specific_exact_projection", True))
            and str(self._cfg("projection_strategy", "rule_specific_exact_first")) in {
                "rule_specific_exact_first",
                "exact_first",
            }
            and rule.name in self._EXACT_FIRST_RULES
        )

    def _verify_assignment(self, equations: Sequence[Equation], combo: Dict[str, Rule], assignment: Assignment) -> bool:
        if len(set(assignment.values())) != len(assignment):
            return False
        return all(self._equation_matches(eq, combo[eq.op], assignment) for eq in equations)

    @staticmethod
    @staticmethod
    def _witness_text(items: Tuple[Tuple[str, int], ...]) -> str:
        return ", ".join(f"{repr(ch)}={digit}" for ch, digit in items)

    @staticmethod
    @staticmethod
    def _witnesses_from_tuple(data: Tuple[Tuple[Tuple[str, int], ...], ...]) -> List[Dict[str, int]]:
        return [dict(items) for items in data]

    @staticmethod
    @staticmethod
    def _witnesses_to_tuple(witnesses: List[Dict[str, int]]) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
        return tuple(tuple(sorted(w.items())) for w in witnesses)

