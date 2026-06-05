from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

N_BITS = 8

SYM_FAMILIES = ("XOR", "OR", "AND")
ASYM_FAMILIES = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES
TERNARY_FAMILIES = ("MAJ", "CH")
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")
DEFAULT_FAMILY = "DEFAULT"

SECTION_ORDER = (
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
)

RuleFamily = Literal[
    "I",
    "NOT",
    "0",
    "1",
    "XOR",
    "OR",
    "AND",
    "AND-NOT",
    "XOR-NOT",
    "OR-NOT",
    "MAJ",
    "CH",
    "DEFAULT",
]


@dataclass(frozen=True)
class Example:
    input_value: str
    output_value: str


@dataclass(frozen=True)
class Problem:
    examples: List[Example]
    question: str
    answer: Optional[str] = None


@dataclass(frozen=True)
class RuleCandidate:
    family: RuleFamily
    primary: Optional[int]
    secondary: Optional[int]
    expr: str
    tertiary: Optional[int] = None

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


@dataclass(frozen=True)
class Record:
    label: str
    col: str
    hash_: str
    matches: Tuple[int, ...]


@dataclass
class Analysis:
    inputs: List[str]
    outputs: List[str]
    question_bits: str
    output_columns: List[str]
    input_columns: List[str]
    records: Dict[str, List[Record]]
    matches: Dict[str, List[List[RuleCandidate]]]
    section_lefts: List[Tuple[str, str]]
    section_rights: List[Tuple[str, str]]
    selected: List[RuleCandidate]
    answer: str
    trace: str


def normalize_bits(value: Any) -> str:
    raw = "" if value is None else str(value)
    bits = "".join(ch for ch in raw if ch in {"0", "1"})
    if len(bits) == N_BITS:
        return bits
    if 0 < len(bits) < N_BITS:
        return bits.zfill(N_BITS)
    if len(bits) > N_BITS:
        return bits[-N_BITS:]
    return ""


def parse_prompt(prompt: Any, answer: Optional[Any] = None) -> Problem:
    text = "" if prompt is None else str(prompt)
    examples: List[Example] = []

    for line in text.splitlines():
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        left_bits = re.findall(r"(?<![01])([01]{1,8})(?![01])", left)
        right_bits = re.findall(r"(?<![01])([01]{1,8})(?![01])", right)
        if not left_bits or not right_bits:
            continue
        x = normalize_bits(left_bits[-1])
        y = normalize_bits(right_bits[0])
        if x and y:
            examples.append(Example(x, y))

    target_match = re.search(
        r"(?:output\s+for|target)\s*:?\s*([01]{1,8})(?![01])",
        text,
        re.IGNORECASE,
    )
    question = normalize_bits(target_match.group(1)) if target_match else ""
    gold = normalize_bits(answer) if answer is not None else None
    return Problem(examples=examples, question=question, answer=gold)


def column_bits(values: Sequence[str], bit: int) -> str:
    return "".join(v[bit] for v in values)


def bit_not(bit: str) -> str:
    return "1" if bit == "0" else "0"


def invert(bits: str) -> str:
    return "".join(bit_not(b) for b in bits)


def column_hash(bits: str, total_examples: int) -> str:
    ones = bits.count("1")
    if ones == 0 or ones == total_examples:
        return "a"
    return format(ones, "x")


def evaluate_binary(a: str, b: str, family: str) -> str:
    if family in ("AND", "AND-NOT"):
        return "1" if a == "1" and b == "1" else "0"
    if family in ("OR", "OR-NOT"):
        return "1" if a == "1" or b == "1" else "0"
    if family in ("XOR", "XOR-NOT"):
        return "1" if a != b else "0"
    raise ValueError(f"Unsupported family: {family}")


def evaluate_ternary(a: str, b: str, c: str, family: str) -> str:
    if family == "MAJ":
        return "1" if ((a == "1") + (b == "1") + (c == "1")) >= 2 else "0"
    if family == "CH":
        return b if a == "1" else c
    raise ValueError(f"Unsupported family: {family}")


def apply_ternary(a_bits: str, b_bits: str, c_bits: str, family: str) -> str:
    return "".join(evaluate_ternary(a, b, c, family) for a, b, c in zip(a_bits, b_bits, c_bits))


def apply_family(a_bits: str, b_bits: str, family: str, invert_second: bool = False) -> str:
    b_eff = invert(b_bits) if invert_second else b_bits
    return "".join(evaluate_binary(a, b, family) for a, b in zip(a_bits, b_eff))


def evaluate_rule(bits: str, rule: RuleCandidate) -> str:
    if rule.family == DEFAULT_FAMILY:
        return "1"
    if rule.family == "0":
        return "0"
    if rule.family == "1":
        return "1"
    if rule.family == "I":
        assert rule.primary is not None
        return bits[rule.primary]
    if rule.family == "NOT":
        assert rule.primary is not None
        return bit_not(bits[rule.primary])
    if rule.family in PAIR_FAMILIES:
        assert rule.primary is not None and rule.secondary is not None
        a = bits[rule.primary]
        b = bits[rule.secondary]
        if "-NOT" in rule.family:
            b = bit_not(b)
        return evaluate_binary(a, b, rule.family)
    if rule.family in TERNARY_FAMILIES:
        assert rule.primary is not None and rule.secondary is not None and rule.tertiary is not None
        a = bits[rule.primary]
        b = bits[rule.secondary]
        c = bits[rule.tertiary]
        return evaluate_ternary(a, b, c, rule.family)
    raise ValueError(rule.family)


def compact_rule(c: RuleCandidate) -> str:
    if c.primary is not None and c.secondary is not None and c.tertiary is not None:
        return f"{c.primary}{c.secondary}{c.tertiary}"
    if c.primary is not None and c.secondary is not None:
        return f"{c.primary}{c.secondary}"
    if c.primary is not None:
        return str(c.primary)
    return c.expr


def format_list(cands: List[RuleCandidate], with_count: bool = False, failed: Optional[str] = None) -> str:
    if not cands:
        return "none"
    if with_count:
        parts = []
        for i, c in enumerate(cands):
            parts.append(c.expr if i == 0 else compact_rule(c))
        return " ".join(parts) + f": {len(cands)}"
    parts = [compact_rule(c) for c in cands]
    if failed:
        parts.append(failed)
    return " ".join(parts)


def find_match(candidates: List[RuleCandidate], fam: str, ep: Optional[int], es: Optional[int]) -> Optional[RuleCandidate]:
    for c in candidates:
        if c.family != fam:
            continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None


def exists_anywhere(all_matches: List[List[RuleCandidate]], fam: str, ep: Optional[int], es: Optional[int]) -> bool:
    return any(find_match(bit_cands, fam, ep, es) is not None for bit_cands in all_matches)


def fail_suffix(all_matches: List[List[RuleCandidate]], fam: str, ep: Optional[int], es: Optional[int]) -> str:
    return "y" if exists_anywhere(all_matches, fam, ep, es) else "x"


def find_all_left_runs(all_matches: List[List[RuleCandidate]]) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    if not all_matches or not all_matches[0]:
        return []

    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for start_cand in all_matches[0]:
        fam = start_cand.family
        chain = [start_cand]
        cur_p, cur_s = start_cand.primary, start_cand.secondary
        failed_next: Optional[str] = None

        for out_idx in range(1, len(all_matches)):
            ep = (cur_p + 1) % N_BITS if cur_p is not None else None
            es = (cur_s + 1) % N_BITS if cur_s is not None else None
            found = find_match(all_matches[out_idx], fam, ep, es)
            if found is None:
                suffix = fail_suffix(all_matches, fam, ep, es)
                if ep is not None and es is not None:
                    failed_next = f"{ep}{es}{suffix}"
                elif ep is not None:
                    failed_next = f"{ep}{suffix}"
                break
            chain.append(found)
            cur_p, cur_s = ep, es
        runs.append((chain, failed_next))
    return runs


def find_all_right_runs(all_matches: List[List[RuleCandidate]]) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []

    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        chain = [end_cand]
        cur_p, cur_s = end_cand.primary, end_cand.secondary
        failed_next: Optional[str] = None

        for step in range(1, n):
            out_idx = n - 1 - step
            pp = (cur_p - 1) % N_BITS if cur_p is not None else None
            ps = (cur_s - 1) % N_BITS if cur_s is not None else None
            found = find_match(all_matches[out_idx], fam, pp, ps)
            if found is None:
                suffix = fail_suffix(all_matches, fam, pp, ps)
                if pp is not None and ps is not None:
                    failed_next = f"{pp}{ps}{suffix}"
                elif pp is not None:
                    failed_next = f"{pp}{suffix}"
                break
            chain.insert(0, found)
            cur_p, cur_s = pp, ps
        runs.append((chain, failed_next))
    return runs


def lr_from_matches(all_matches: List[List[RuleCandidate]]) -> Tuple[List[str], str, List[str], str]:
    left_runs = find_all_left_runs(all_matches)
    right_runs = find_all_right_runs(all_matches)
    left_run = max(left_runs, key=lambda t: len(t[0])) if left_runs else ([], None)
    right_run = max(right_runs, key=lambda t: len(t[0])) if right_runs else ([], None)

    left_lines = [format_list(chain, failed=failed) for chain, failed in left_runs] if left_runs else ["none"]
    left_best = format_list(left_run[0], with_count=True)

    right_lines = [format_list(list(reversed(chain)), failed=failed) for chain, failed in right_runs] if right_runs else ["none"]
    right_best = format_list(list(reversed(right_run[0])), with_count=True)
    return left_lines, left_best, right_lines, right_best


def parse_count(val: str) -> int:
    if val == "none":
        return 0
    try:
        return int(val.rsplit(": ", 1)[-1])
    except Exception:
        return 0


def pick_winner(entries: List[Tuple[str, str]]) -> Tuple[Optional[str], str, int]:
    best_name: Optional[str] = None
    best_text = "none"
    best_count = 0
    for name, val in entries:
        count = parse_count(val)
        if count > best_count:
            best_count = count
            best_name = name
            best_text = val
    return best_name, best_text, best_count


class BitManipulationSolver:
    """
    Tonghuikang-style bit manipulation solver.

    Public interface expected by your generator:
        solver = BitManipulationSolver()
        cot = solver.generate_cot(prompt)
        answer = solver.extract_answer(cot)

    The trace is intentionally a column-matching reasoning trace, not a brute-force
    global transform search ledger.
    """

    def __init__(self, include_ternary_completion: bool = True, max_ternary_candidates_to_show: int = 6):
        self.include_ternary_completion = include_ternary_completion
        self.max_ternary_candidates_to_show = max_ternary_candidates_to_show

    @staticmethod
    def _preferred_digits(pref: str) -> List[int]:
        if not pref.startswith("?") or pref == "?":
            return []
        return [int(d) for d in pref[1:] if d != "?"]

    def _ternary_candidates_for_bit(
        self,
        input_columns: Sequence[str],
        output_column: str,
    ) -> List[RuleCandidate]:
        """Return exact MAJ/CH candidates for one unresolved output bit.

        These candidates are intentionally not added to the main section tables.
        They are only used as a compact completion step for unresolved bits, so
        the normal unary/binary CoT stays short and stable.
        """
        candidates: List[RuleCandidate] = []

        # MAJ is symmetric, so keep each unordered triple once.
        for a in range(N_BITS):
            for b in range(a + 1, N_BITS):
                for c in range(b + 1, N_BITS):
                    col = apply_ternary(input_columns[a], input_columns[b], input_columns[c], "MAJ")
                    if col == output_column:
                        candidates.append(RuleCandidate("MAJ", a, b, f"MAJ{a}{b}{c}", c))

        # CH(a,b,c) is ordered: if a then b else c.  Require distinct operands
        # to avoid degenerating into unary/binary rules already checked above.
        for a in range(N_BITS):
            for b in range(N_BITS):
                if b == a:
                    continue
                for c in range(N_BITS):
                    if c == a or c == b:
                        continue
                    col = apply_ternary(input_columns[a], input_columns[b], input_columns[c], "CH")
                    if col == output_column:
                        candidates.append(RuleCandidate("CH", a, b, f"CH{a}{b}{c}", c))

        return candidates

    def _score_ternary_candidate(self, cand: RuleCandidate, pref: str) -> Tuple[int, int, int, int]:
        digits = self._preferred_digits(pref)
        operands = tuple(
            x for x in (cand.primary, cand.secondary, cand.tertiary)
            if x is not None
        )

        score = 0
        if digits:
            if operands[: len(digits)] == tuple(digits):
                score += 20
            if all(d in operands for d in digits):
                score += 10
            score -= sum(1 for d in digits if d not in operands) * 5

        # Prefer candidates whose operands form a compact circular neighborhood;
        # this keeps the completion aligned with the bit-chain style.
        circular_span = max(operands) - min(operands) if operands else N_BITS
        family_bonus = 1 if cand.family == "MAJ" else 0
        return (score, -circular_span, family_bonus, -sum(operands))

    def _choose_ternary_candidate(self, cands: List[RuleCandidate], pref: str) -> RuleCandidate:
        return max(cands, key=lambda c: self._score_ternary_candidate(c, pref))

    @staticmethod
    def _preferred_display(pref: str) -> str:
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            return f"{pref} ?{pref[2]}{pref[1]}"
        return pref

    @staticmethod
    def _candidate_operand_text(cand: RuleCandidate) -> str:
        operands = [
            str(x) for x in (cand.primary, cand.secondary, cand.tertiary)
            if x is not None
        ]
        return "".join(operands) if operands else "none"

    def _ternary_choice_reason(self, cand: RuleCandidate, pref: str) -> str:
        digits = self._preferred_digits(pref)
        operands = [
            x for x in (cand.primary, cand.secondary, cand.tertiary)
            if x is not None
        ]
        if digits:
            if operands[: len(digits)] == digits:
                return (
                    f"use {cand.expr} because it exactly matches the output column "
                    f"and its leading operands {self._candidate_operand_text(cand)} follow the preferred pattern "
                    f"{self._preferred_display(pref)}"
                )
            if all(d in operands for d in digits):
                return (
                    f"use {cand.expr} because it exactly matches the output column "
                    f"and contains the preferred operands from {self._preferred_display(pref)}"
                )
        return f"use {cand.expr} because it exactly matches the output column"

    def _format_all_ternary_candidates(self, cands: List[RuleCandidate]) -> str:
        return " ".join(c.expr for c in cands) if cands else "none"

    def analyze(self, prompt: Any, answer: Optional[Any] = None) -> Optional[Analysis]:
        problem = parse_prompt(prompt, answer)
        if not problem.examples or not problem.question:
            return None

        inputs = [normalize_bits(ex.input_value) for ex in problem.examples]
        outputs = [normalize_bits(ex.output_value) for ex in problem.examples]
        question_bits = normalize_bits(problem.question)
        if any(not b for b in inputs + outputs) or not question_bits:
            return None

        n_examples = len(outputs)
        output_columns = [column_bits(outputs, i) for i in range(N_BITS)]
        input_columns = [column_bits(inputs, i) for i in range(N_BITS)]
        input_inverted = [invert(c) for c in input_columns]

        records: Dict[str, List[Record]] = {name: [] for name in SECTION_ORDER}
        matches: Dict[str, List[List[RuleCandidate]]] = {name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER}

        for out_idx, out_col in enumerate(output_columns):
            for i_col, in_col in enumerate(input_columns):
                if in_col == out_col:
                    matches["Identity"][out_idx].append(RuleCandidate("I", i_col, None, f"I{i_col}"))
                if input_inverted[i_col] == out_col:
                    matches["NOT"][out_idx].append(RuleCandidate("NOT", i_col, None, f"NOT{i_col}"))
            if out_col.count("1") == 0:
                matches["Constant"][out_idx].append(RuleCandidate("0", None, None, "C0"))
            if out_col.count("1") == n_examples:
                matches["Constant"][out_idx].append(RuleCandidate("1", None, None, "C1"))

        for label, col in zip([str(i) for i in range(N_BITS)], input_columns):
            m = tuple(i for i, oc in enumerate(output_columns) if col == oc)
            records["Identity"].append(Record(label, col, column_hash(col, n_examples), m))

        for label, col in zip([str(i) for i in range(N_BITS)], input_inverted):
            m = tuple(i for i, oc in enumerate(output_columns) if col == oc)
            records["NOT"].append(Record(label, col, column_hash(col, n_examples), m))

        for val in ("0", "1"):
            col = val * n_examples
            m = tuple(i for i, oc in enumerate(output_columns) if col == oc)
            records["Constant"].append(Record(val, col, column_hash(col, n_examples), m))

        for fam in ("XOR", "OR", "AND"):
            for circ_diff in range(1, N_BITS // 2 + 1):
                n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
                for a in range(n_pairs):
                    b = (a + circ_diff) % N_BITS
                    lo, hi = min(a, b), max(a, b)
                    col = apply_family(input_columns[lo], input_columns[hi], fam)
                    m = tuple(i for i, oc in enumerate(output_columns) if col == oc)
                    records[fam].append(Record(f"{a}{b} {b}{a}", col, column_hash(col, n_examples), m))
                    for out_idx in m:
                        matches[fam][out_idx].append(RuleCandidate(fam, a, b, f"{fam}{a}{b}"))
                        matches[fam][out_idx].append(RuleCandidate(fam, b, a, f"{fam}{b}{a}"))

        for fam in ("AND-NOT", "XOR-NOT", "OR-NOT"):
            for diff in range(1, N_BITS):
                for a in range(N_BITS):
                    b = (a + diff) % N_BITS
                    col = apply_family(input_columns[a], input_columns[b], fam, invert_second=True)
                    m = tuple(i for i, oc in enumerate(output_columns) if col == oc)
                    records[fam].append(Record(f"{a}{b}", col, column_hash(col, n_examples), m))
                    for out_idx in m:
                        matches[fam][out_idx].append(RuleCandidate(fam, a, b, f"{fam}{a}{b}"))

        for name in ("Identity", "NOT", "Constant"):
            records[name].sort(key=lambda r: r.label)

        lines: List[str] = []
        lines.append("We need to deduce the transformation by matching the example outputs.")
        lines.append("I will put my final answer inside \\boxed{}.")
        lines.append("")

        for i, out in enumerate(outputs):
            lines.append(f"Output {i}: {out}")
            for bit in range(N_BITS):
                lines.append(f"{bit} {out[bit]}")
            lines.append("")

        lines.append("Output bit columns (with bitsum as hash)")
        for bit in range(N_BITS):
            lines.append(f"{bit} {output_columns[bit]} {column_hash(output_columns[bit], n_examples)}")
        lines.append("")

        for i, inp in enumerate(inputs):
            lines.append(f"Input {i}: {inp}")
            for bit in range(N_BITS):
                lines.append(f"{bit} {inp[bit]}")
            lines.append("")

        lines.append("When matching output")
        lines.append("x: not in operator")
        lines.append("y: wrong position")
        lines.append("")

        section_lefts: List[Tuple[str, str]] = []
        section_rights: List[Tuple[str, str]] = []

        def add_section(name: str) -> None:
            section_records = records[name]
            per_bit = matches[name]
            lines.append(name)
            prev_diff: Optional[int] = None
            for rec in section_records:
                if len(rec.label) >= 2 and rec.label[0].isdigit() and rec.label[1].isdigit():
                    diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                    if prev_diff is not None and diff != prev_diff:
                        lines.append("")
                    prev_diff = diff
                line = f"{rec.label} {rec.col} {rec.hash_}"
                if rec.matches:
                    line += " match " + " ".join(str(i) for i in rec.matches)
                lines.append(line)

            lines.append("")
            lines.append("Matching output")
            for i in range(N_BITS):
                cands = per_bit[i]
                if cands:
                    lines.append(f"{i} " + " ".join(compact_rule(c) for c in cands))
                else:
                    lines.append(f"{i} absent")
            lines.append("")

            left_lines, left_best, right_lines, right_best = lr_from_matches(per_bit)
            section_lefts.append((name, left_best))
            section_rights.append((name, right_best))

            lines.append("Left")
            lines.extend(left_lines)
            lines.append(f"Best: {left_best}")
            lines.append("")
            lines.append("Right")
            lines.extend(right_lines)
            lines.append(f"Best: {right_best}")
            lines.append("")

        for name in SECTION_ORDER:
            add_section(name)

        lines.append("Selecting")
        lines.append("")
        left_name, left_text, left_count = pick_winner(section_lefts)
        right_name, right_text, right_count = pick_winner(section_rights)

        def get_section_run(winner: Optional[str], direction: str) -> List[RuleCandidate]:
            if winner is None:
                return []
            per_bit = matches[winner]
            runs = find_all_left_runs(per_bit) if direction == "left" else find_all_right_runs(per_bit)
            if not runs:
                return []
            chain, _ = max(runs, key=lambda t: len(t[0]))
            return chain

        left_run = get_section_run(left_name, "left")
        right_run = get_section_run(right_name, "right")

        lines.append("Lefts")
        for name, lb in section_lefts:
            lines.append(f"{name} {lb}")
        lines.append("")
        lines.append("Rights")
        for name, rb in section_rights:
            lines.append(f"{name} {rb}")
        lines.append("")
        lines.append(f"Left longest: {left_count}")
        lines.append(f"Right longest: {right_count}")
        lines.append("")

        def matching_line(label: str, winner_name: Optional[str], entries: List[Tuple[str, str]]) -> str:
            parts = [f"{name} {'yes' if name == winner_name else 'no'}" for name, _ in entries]
            return f"{label} winner: {', '.join(parts)}"

        if right_count > left_count:
            lines.append(matching_line("Right", right_name, section_rights))
            lines.append(matching_line("Left", left_name, section_lefts))
            lines.append("")
            lines.append(f"Best right: {right_text}")
            lines.append(f"Best left: {left_text}")
        else:
            lines.append(matching_line("Left", left_name, section_lefts))
            lines.append(matching_line("Right", right_name, section_rights))
            lines.append("")
            lines.append(f"Best left: {left_text}")
            lines.append(f"Best right: {right_text}")
        lines.append("")

        left_len = left_count
        right_len = right_count
        if left_len + right_len > N_BITS:
            if right_len > left_len:
                left_len = N_BITS - right_len
                left_run = left_run[:left_len]
            else:
                right_len = N_BITS - left_len
                right_run = right_run[-right_len:] if right_len else []

        left_was_trunc = left_len < left_count
        right_was_trunc = right_len < right_count
        trunc_left = f"Truncated left: {format_list(left_run, with_count=True)}" + (" truncated" if left_was_trunc else "")
        trunc_right = f"Truncated right: {format_list(list(reversed(right_run)), with_count=True)}" + (" truncated" if right_was_trunc else "")
        if right_count > left_count:
            lines.append(trunc_right)
            lines.append(trunc_left)
        else:
            lines.append(trunc_left)
            lines.append(trunc_right)
        lines.append("")

        right_start = N_BITS - right_len
        best: List[RuleCandidate] = [RuleCandidate(DEFAULT_FAMILY, None, None, "default 1") for _ in range(N_BITS)]
        for i, rc in enumerate(left_run):
            best[i] = rc
        for i, rc in enumerate(right_run):
            best[right_start + i] = rc

        lines.append("Tentative from right")
        for i in range(N_BITS - 1, -1, -1):
            if i >= right_start and right_run:
                lines.append(f"{i} {right_run[i - right_start].expr}")
            else:
                lines.append(f"{i} pending")
        lines.append("")

        lines.append("Tentative")
        for i in range(N_BITS):
            if i < left_len:
                lines.append(f"{i} {left_run[i].expr}")
            elif i >= right_start and right_run:
                lines.append(f"{i} {right_run[i - right_start].expr}")
            else:
                lines.append(f"{i} pending")
        lines.append("")

        def extrap_from(run: List[RuleCandidate], bit: int, run_start: int, side: str) -> Optional[str]:
            if not run:
                return None
            r = run[0]
            if r.primary is not None:
                p_off = (r.primary - run_start) % N_BITS
                ep = (p_off + bit) % N_BITS
            else:
                ep = None
            if r.secondary is not None:
                s_off = (r.secondary - run_start) % N_BITS
                es = (s_off + bit) % N_BITS
            else:
                es = None
            if ep is not None and es is not None:
                return f"?{ep}{es}"
            if ep is not None:
                return f"?{ep}?" if side == "left" else f"??{ep}"
            return None

        left_fam = left_run[0].family if left_run else None
        right_fam = right_run[0].family if right_run else None
        left_binary_or_unary = bool(left_fam and (left_fam in PAIR_FAMILIES or left_fam in UNARY_FAMILIES))
        right_binary_or_unary = bool(right_fam and (right_fam in PAIR_FAMILIES or right_fam in UNARY_FAMILIES))
        left_unary = bool(left_fam and left_fam in UNARY_FAMILIES)
        right_unary = bool(right_fam and right_fam in UNARY_FAMILIES)

        if right_count > left_count:
            preferred: List[str] = []
            for i in range(N_BITS):
                if i >= right_start and right_run:
                    preferred.append(right_run[i - right_start].expr)
                elif i < left_len:
                    preferred.append(left_run[i].expr)
                elif right_binary_or_unary:
                    preferred.append(extrap_from(right_run, i, right_start, "right") or "pending")
                else:
                    preferred.append("pending")

            lines.append("Preferred from right")
            for i in range(N_BITS - 1, -1, -1):
                lines.append(f"{i} {preferred[i]}")
            lines.append("")

            for i in range(N_BITS):
                if preferred[i] == "pending":
                    preferred[i] = extrap_from(left_run, i, 0, "left") if left_binary_or_unary else "?"
                    preferred[i] = preferred[i] or "?"
                elif "?" in preferred[i][1:] and left_unary:
                    el = extrap_from(left_run, i, 0, "left")
                    if el:
                        merged = list(preferred[i])
                        for j, ch in enumerate(el[: len(merged)]):
                            if j > 0 and merged[j] == "?" and ch != "?":
                                merged[j] = ch
                        preferred[i] = "".join(merged)

            lines.append("Preferred from left")
            for i in range(N_BITS):
                lines.append(f"{i} {preferred[i]}")
            lines.append("")
        else:
            preferred = []
            for i in range(N_BITS):
                if i < left_len:
                    preferred.append(left_run[i].expr)
                elif i >= right_start and right_run:
                    preferred.append(right_run[i - right_start].expr)
                elif left_binary_or_unary:
                    preferred.append(extrap_from(left_run, i, 0, "left") or "pending")
                else:
                    preferred.append("pending")

            lines.append("Preferred from left")
            for i in range(N_BITS):
                lines.append(f"{i} {preferred[i]}")
            lines.append("")

            for i in range(N_BITS):
                if preferred[i] == "pending":
                    preferred[i] = extrap_from(right_run, i, right_start, "right") if right_binary_or_unary else "?"
                    preferred[i] = preferred[i] or "?"
                elif "?" in preferred[i][1:] and right_unary:
                    er = extrap_from(right_run, i, right_start, "right")
                    if er:
                        merged = list(preferred[i])
                        for j, ch in enumerate(er[: len(merged)]):
                            if j > 0 and merged[j] == "?" and ch != "?":
                                merged[j] = ch
                        preferred[i] = "".join(merged)

            lines.append("Preferred from right")
            for i in range(N_BITS - 1, -1, -1):
                lines.append(f"{i} {preferred[i]}")
            lines.append("")

        lines.append("Preferred")
        for i, pref in enumerate(preferred):
            if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
                lines.append(f"{i} {pref} ?{pref[2]}{pref[1]}")
            else:
                lines.append(f"{i} {pref}")
        lines.append("")

        lines.append("Matching")
        pending_indices: List[int] = []
        per_bit_cat: Dict[str, Dict[int, List[RuleCandidate]]] = {name: {} for name in SECTION_ORDER}

        for i in range(N_BITS):
            pref = preferred[i]
            if not pref.startswith("?") or pref == "?":
                lines.append(f"{i} {best[i].expr}")
                continue

            pending_indices.append(i)
            pref_digits = [int(d) for d in pref[1:] if d != "?"]
            checks: List[str] = []

            for section_name in SECTION_ORDER:
                cands = matches[section_name][i]
                if section_name in ("Identity", "NOT"):
                    found = [c for c in cands if c.primary in pref_digits]
                    if found:
                        checks.append(section_name + " " + " ".join(c.expr for c in found))
                        per_bit_cat[section_name][i] = found
                    else:
                        checks.append(f"{section_name} absent")
                elif section_name == "Constant":
                    if cands:
                        checks.append("Constant " + " ".join(c.expr for c in cands))
                        per_bit_cat[section_name][i] = list(cands)
                    else:
                        checks.append("Constant absent")
                else:
                    found_c: Optional[RuleCandidate] = None
                    want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                    want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                    orderings = [(want_p, want_s)]
                    if want_p is not None and want_s is not None and want_p != want_s:
                        orderings.append((want_s, want_p))
                    for wp, ws in orderings:
                        for c in cands:
                            if (wp is None or c.primary == wp) and (ws is None or c.secondary == ws):
                                found_c = c
                                break
                        if found_c is not None:
                            break
                    if found_c is not None:
                        checks.append(found_c.expr)
                        per_bit_cat[section_name][i] = [found_c]
                    else:
                        checks.append(f"{section_name} absent")

            pref_display = f"{pref} ?{pref[2]}{pref[1]}" if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?" else pref
            lines.append(f"{i} {pref_display} - {', '.join(checks)}")
        lines.append("")

        lines.append("Perfect match")
        chosen_cat: Optional[str] = None
        for cat in SECTION_ORDER:
            is_perfect = chosen_cat is None and bool(pending_indices) and all(i in per_bit_cat[cat] for i in pending_indices)
            lines.append(f"{cat} {'yes' if is_perfect else 'no'}")
            if is_perfect:
                chosen_cat = cat
        lines.append("")

        lines.append("Matched")
        pending_set = set(pending_indices)
        for i in range(N_BITS):
            if i in pending_set:
                if chosen_cat and i in per_bit_cat[chosen_cat]:
                    best[i] = per_bit_cat[chosen_cat][i][0]
                    lines.append(f"{i} {best[i].expr}")
                else:
                    all_cands: List[RuleCandidate] = []
                    for name in SECTION_ORDER:
                        all_cands.extend(per_bit_cat[name].get(i, []))
                    if all_cands:
                        best[i] = all_cands[0]
                        lines.append(f"{i} " + " ".join(c.expr for c in all_cands))
                    else:
                        best[i] = RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
                        lines.append(f"{i} none")
            else:
                lines.append(f"{i} {best[i].expr}")
        lines.append("")

        if self.include_ternary_completion:
            unresolved_indices = [i for i, rule in enumerate(best) if rule.family == DEFAULT_FAMILY]
            if unresolved_indices:
                lines.append("Ternary completion")
                lines.append(
                    "Some positions remain unresolved after unary/binary matching. "
                    "For those positions, I check exact majority/choice column matches."
                )
                lines.append(
                    "The preferred pattern comes from the left/right chain above; "
                    "when several exact ternary matches exist, I choose the one most aligned with that pattern."
                )
                ternary_by_bit = {
                    i: self._ternary_candidates_for_bit(input_columns, output_columns[i])
                    for i in unresolved_indices
                }
                can_complete_all = all(ternary_by_bit[i] for i in unresolved_indices)

                for i in unresolved_indices:
                    cands = ternary_by_bit[i]
                    pref_i = preferred[i] if i < len(preferred) else "?"
                    pref_display = self._preferred_display(pref_i)
                    if cands and can_complete_all:
                        chosen = self._choose_ternary_candidate(cands, pref_i)
                        best[i] = chosen
                        shown = self._format_all_ternary_candidates(cands)
                        reason = self._ternary_choice_reason(chosen, pref_i)
                        lines.append(f"{i} preferred {pref_display}")
                        lines.append(f"{i} exact ternary candidates: {shown}")
                        lines.append(f"{i} {reason}")
                    elif cands:
                        shown = self._format_all_ternary_candidates(cands)
                        lines.append(f"{i} preferred {pref_display}")
                        lines.append(f"{i} exact ternary candidates: {shown}")
                        lines.append(f"{i} keep {best[i].expr} because not every unresolved position has a ternary completion")
                    else:
                        lines.append(f"{i} preferred {pref_display}")
                        lines.append(f"{i} exact ternary candidates: none")
                        lines.append(f"{i} keep {best[i].expr}")
                lines.append("")

        lines.append("Selected")
        for i, rule in enumerate(best):
            lines.append(f"{i} {rule.expr}")
        lines.append("")

        answer_bits = self._emit_apply(lines, question_bits, best)
        trace = "\n".join(lines)
        return Analysis(
            inputs=inputs,
            outputs=outputs,
            question_bits=question_bits,
            output_columns=output_columns,
            input_columns=input_columns,
            records=records,
            matches=matches,
            section_lefts=section_lefts,
            section_rights=section_rights,
            selected=best,
            answer=answer_bits,
            trace=trace,
        )

    def _emit_apply(self, lines: List[str], question_bits: str, vector: List[RuleCandidate]) -> str:
        lines.append(f"Applying to {question_bits}")
        lines.append("Input")
        for i, bit in enumerate(question_bits):
            lines.append(f"{i} {bit}")
        lines.append("Output")

        answer_bits: List[str] = []
        for i, rule in enumerate(vector):
            if rule.family == DEFAULT_FAMILY:
                lines.append(f"{i} default 1 = 1")
                answer_bits.append("1")
            elif rule.family in CONSTANT_FAMILIES:
                lines.append(f"{i} {rule.expr} = {rule.family}")
                answer_bits.append(rule.family)
            elif rule.family == "I":
                assert rule.primary is not None
                val = question_bits[rule.primary]
                lines.append(f"{i} {rule.expr} = {val}")
                answer_bits.append(val)
            elif rule.family == "NOT":
                assert rule.primary is not None
                val = question_bits[rule.primary]
                nval = bit_not(val)
                lines.append(f"{i} {rule.expr} = NOT({val}) = {nval}")
                answer_bits.append(nval)
            elif rule.family in TERNARY_FAMILIES:
                assert rule.primary is not None and rule.secondary is not None and rule.tertiary is not None
                a = question_bits[rule.primary]
                b = question_bits[rule.secondary]
                c = question_bits[rule.tertiary]
                result = evaluate_rule(question_bits, rule)
                lines.append(f"{i} {rule.expr} = {rule.family}({a},{b},{c}) = {result}")
                answer_bits.append(result)
            else:
                assert rule.primary is not None and rule.secondary is not None
                a = question_bits[rule.primary]
                b = question_bits[rule.secondary]
                result = evaluate_rule(question_bits, rule)
                if rule.family in SYM_FAMILIES:
                    lines.append(f"{i} {rule.expr} = {rule.family}({a},{b}) = {result}")
                else:
                    base = rule.family.split("-")[0]
                    lines.append(f"{i} {rule.expr} = {base}({a},NOT({b})) = {result}")
                answer_bits.append(result)

        answer = "".join(answer_bits)
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
        return answer

    def generate_cot(self, prompt: Any) -> str:
        analysis = self.analyze(prompt)
        if analysis is None:
            return ""
        return analysis.trace

    @staticmethod
    def extract_answer(cot: Any) -> str:
        text = "" if cot is None else str(cot)
        patterns = [
            r"\\boxed\{([01]{8})\}",
            r"(?im)^\s*Answer\s*[:=]\s*([01]{8})\s*$",
            r"(?i)final\s+answer\s*[:=]\s*([01]{8})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        return ""

    def solve(self, prompt: Any) -> str:
        return self.extract_answer(self.generate_cot(prompt))
