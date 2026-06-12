from __future__ import annotations

import re
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

N_BITS = 8

SYM_FAMILIES = ("XOR", "OR", "AND")
ASYM_FAMILIES = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")
DEFAULT_FAMILY = "DEFAULT"
BYTE_MASK = (1 << N_BITS) - 1

# Operands in globally selected per-bit rules are either input bit indices or
# explicit constants. Existing local matching rules leave this tuple empty.
BitOperand = Union[int, Literal["C0", "C1"]]

TERNARY_FAMILIES = ("MAJ", "MIN", "CHOICE", "NCHOICE")

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
    "MAJ",
    "MIN",
    "CHOICE",
    "NCHOICE",
)

BASE_SECTION_ORDER = SECTION_ORDER[:9]

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
    "MIN",
    "CHOICE",
    "NCHOICE",
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
    operands: Tuple[BitOperand, ...] = ()
    program_key: Optional[str] = None

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


@dataclass(frozen=True)
class GlobalTransform:
    """A deterministic whole-byte transform used by the exact selector."""

    kind: Literal["ROT", "SHL", "SHR"]
    amount: int

    def apply_int(self, value: int) -> int:
        value &= BYTE_MASK
        if self.kind == "ROT":
            k = self.amount % N_BITS
            if k == 0:
                return value
            return ((value << k) & BYTE_MASK) | (value >> (N_BITS - k))
        if self.kind == "SHL":
            return (value << self.amount) & BYTE_MASK
        if self.kind == "SHR":
            return value >> self.amount
        raise ValueError(self.kind)

    def source_at(self, output_bit: int) -> BitOperand:
        # Bit strings are indexed from the most-significant bit at position 0.
        if self.kind == "ROT":
            return (output_bit + self.amount) % N_BITS
        if self.kind == "SHL":
            source = output_bit + self.amount
            return source if source < N_BITS else "C0"
        if self.kind == "SHR":
            source = output_bit - self.amount
            return source if source >= 0 else "C0"
        raise ValueError(self.kind)

    def rank(self) -> Tuple[int, int, int]:
        # Rank is used only to render one canonical proof after every exact
        # candidate has already agreed on the target answer.
        kind_rank = {"ROT": 0, "SHL": 1, "SHR": 2}
        non_identity = 0 if self.kind == "ROT" and self.amount == 0 else 1
        return non_identity, kind_rank[self.kind], self.amount

    @property
    def expr(self) -> str:
        return f"{self.kind}{self.amount}"


@dataclass(frozen=True)
class GlobalProgram:
    family: str
    transforms: Tuple[GlobalTransform, ...]

    def apply_int(self, value: int) -> int:
        vals = tuple(t.apply_int(value) for t in self.transforms)
        return evaluate_global_family(self.family, vals)

    def rank(self) -> Tuple[Any, ...]:
        family_rank = {
            "I": 0,
            "NOT": 1,
            "AND": 2,
            "OR": 3,
            "XOR": 4,
            "AND-NOT": 5,
            "OR-NOT": 6,
            "XOR-NOT": 7,
            "MAJ": 8,
            "MIN": 9,
            "CHOICE": 10,
            "NCHOICE": 11,
        }
        return (
            len(self.transforms),
            family_rank[self.family],
            tuple(t.rank() for t in self.transforms),
        )

    @property
    def expr(self) -> str:
        return " ".join([self.family] + [t.expr for t in self.transforms])

    def to_bit_rules(self) -> List[RuleCandidate]:
        rules: List[RuleCandidate] = []
        for output_bit in range(N_BITS):
            operands = tuple(t.source_at(output_bit) for t in self.transforms)

            labels = "".join(str(op) for op in operands)
            expr = f"{self.family}{labels}"
            primary = operands[0] if operands and isinstance(operands[0], int) else None
            secondary = operands[1] if len(operands) > 1 and isinstance(operands[1], int) else None
            rules.append(
                RuleCandidate(
                    self.family,  # type: ignore[arg-type]
                    primary,
                    secondary,
                    expr,
                    operands=operands,
                    program_key=self.expr,
                )
            )
        return rules


@dataclass(frozen=True)
class GlobalConsensus:
    program: GlobalProgram
    answer: str
    exact_program_count: int


GLOBAL_TRANSFORMS: Tuple[GlobalTransform, ...] = tuple(
    [GlobalTransform("ROT", k) for k in range(N_BITS)]
    + [GlobalTransform("SHL", k) for k in range(1, N_BITS)]
    + [GlobalTransform("SHR", k) for k in range(1, N_BITS)]
)


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


def pack_bytes(values: Sequence[int]) -> int:
    packed = 0
    for i, value in enumerate(values):
        packed |= (value & BYTE_MASK) << (N_BITS * i)
    return packed


def evaluate_global_family(family: str, vals: Sequence[int], mask: int = BYTE_MASK) -> int:
    if family == "I":
        return vals[0] & mask
    if family == "NOT":
        return (~vals[0]) & mask
    if family == "AND":
        return vals[0] & vals[1]
    if family == "OR":
        return vals[0] | vals[1]
    if family == "XOR":
        return vals[0] ^ vals[1]
    if family == "AND-NOT":
        return vals[0] & ((~vals[1]) & mask)
    if family == "OR-NOT":
        return vals[0] | ((~vals[1]) & mask)
    if family == "XOR-NOT":
        return vals[0] ^ ((~vals[1]) & mask)
    if family == "MAJ":
        a, b, c = vals
        return (a & b) | (a & c) | (b & c)
    if family == "MIN":
        a, b, c = vals
        return (~((a & b) | (a & c) | (b & c))) & mask
    if family == "CHOICE":
        selector, when_one, when_zero = vals
        return (selector & when_one) | (((~selector) & mask) & when_zero)
    if family == "NCHOICE":
        selector, when_one, when_zero = vals
        choice = (selector & when_one) | (((~selector) & mask) & when_zero)
        return (~choice) & mask
    raise ValueError(f"Unsupported global family: {family}")


def find_global_consensus(
    inputs: Sequence[str],
    outputs: Sequence[str],
    question_bits: str,
) -> Optional[GlobalConsensus]:
    """Find an exact ternary program when every exact program agrees.

    MAJ/MIN share one computation and CHOICE/NCHOICE share one computation.
    Unary and binary families are not recomputed here; the original matcher
    already handles them. Gold target data is never used.
    """
    xs = [int(x, 2) for x in inputs]
    ys = [int(y, 2) for y in outputs]
    target = int(question_bits, 2)
    packed_target = pack_bytes(ys)
    packed_mask = (1 << (N_BITS * len(xs))) - 1

    transformed: List[Tuple[GlobalTransform, int, int]] = []
    for transform in GLOBAL_TRANSFORMS:
        transformed.append((
            transform,
            pack_bytes([transform.apply_int(x) for x in xs]),
            transform.apply_int(target),
        ))

    by_answer: Dict[int, GlobalProgram] = {}
    exact_program_count = 0

    def add_if_exact(value: int, target_value: int, family: str, transforms: Tuple[GlobalTransform, ...]) -> None:
        nonlocal exact_program_count
        if value != packed_target:
            return
        exact_program_count += 1
        program = GlobalProgram(family, transforms)
        old = by_answer.get(target_value)
        if old is None or program.rank() < old.rank():
            by_answer[target_value] = program

    # One traversal for MAJ and MIN.
    for i, (ta, a, target_a) in enumerate(transformed):
        for j in range(i, len(transformed)):
            tb, b, target_b = transformed[j]
            ab = a & b
            target_ab = target_a & target_b
            for tc, c, target_c in transformed[j:]:
                transforms = (ta, tb, tc)
                majority = ab | (a & c) | (b & c)
                target_majority = target_ab | (target_a & target_c) | (target_b & target_c)
                add_if_exact(majority, target_majority, "MAJ", transforms)
                add_if_exact((~majority) & packed_mask, (~target_majority) & BYTE_MASK, "MIN", transforms)

    # One traversal for CHOICE and NCHOICE.
    for ts, selector, target_selector in transformed:
        not_selector = (~selector) & packed_mask
        not_target_selector = (~target_selector) & BYTE_MASK
        for ta, when_one, target_one in transformed:
            selected_one = selector & when_one
            target_selected_one = target_selector & target_one
            for tb, when_zero, target_zero in transformed:
                transforms = (ts, ta, tb)
                choice = selected_one | (not_selector & when_zero)
                target_choice = target_selected_one | (not_target_selector & target_zero)
                add_if_exact(choice, target_choice, "CHOICE", transforms)
                add_if_exact((~choice) & packed_mask, (~target_choice) & BYTE_MASK, "NCHOICE", transforms)

    if len(by_answer) != 1 or exact_program_count == 0:
        return None
    answer_int, program = next(iter(by_answer.items()))
    return GlobalConsensus(program, f"{answer_int:08b}", exact_program_count)


def populate_ternary_matching(
    records: Dict[str, List[Record]],
    matches: Dict[str, List[List[RuleCandidate]]],
    input_columns: Sequence[str],
    output_columns: Sequence[str],
) -> None:
    """Populate ternary sections with one pass per complementary pair.

    MAJ is evaluated once and MIN reuses its inverse. CHOICE is evaluated once
    and NCHOICE reuses its inverse. Only positive output-column matches are
    stored.
    """
    n_examples = len(output_columns[0])
    source_labels: List[BitOperand] = list(range(N_BITS)) + ["C0"]
    source_columns = list(input_columns) + ["0" * n_examples]

    def add_column(family: str, operands: Tuple[BitOperand, BitOperand, BitOperand], col: str) -> None:
        matched_outputs = tuple(i for i, out_col in enumerate(output_columns) if col == out_col)
        if not matched_outputs:
            return
        label = "".join(str(op) for op in operands)
        records[family].append(Record(label, col, column_hash(col, n_examples), matched_outputs))
        primary = operands[0] if isinstance(operands[0], int) else None
        secondary = operands[1] if isinstance(operands[1], int) else None
        rule = RuleCandidate(family, primary, secondary, f"{family}{label}", operands=operands)  # type: ignore[arg-type]
        for out_idx in matched_outputs:
            matches[family][out_idx].append(rule)

    source_range = range(len(source_labels))
    for ia, ib, ic in itertools.combinations_with_replacement(source_range, 3):
        operands = (source_labels[ia], source_labels[ib], source_labels[ic])
        a_col, b_col, c_col = source_columns[ia], source_columns[ib], source_columns[ic]
        maj_col = "".join(
            "1" if (a == "1") + (b == "1") + (c == "1") >= 2 else "0"
            for a, b, c in zip(a_col, b_col, c_col)
        )
        add_column("MAJ", operands, maj_col)
        add_column("MIN", operands, invert(maj_col))

    for isel, ione, izero in itertools.product(source_range, repeat=3):
        operands = (source_labels[isel], source_labels[ione], source_labels[izero])
        selector_col = source_columns[isel]
        one_col = source_columns[ione]
        zero_col = source_columns[izero]
        choice_col = "".join(
            one if selector == "1" else zero
            for selector, one, zero in zip(selector_col, one_col, zero_col)
        )
        add_column("CHOICE", operands, choice_col)
        add_column("NCHOICE", operands, invert(choice_col))


def inject_global_program_into_matching(
    records: Dict[str, List[Record]],
    matches: Dict[str, List[List[RuleCandidate]]],
    program: GlobalProgram,
    inputs: Sequence[str],
    output_columns: Sequence[str],
) -> None:
    """Insert one exact byte-level program into the ordinary matching tables.

    The program is rendered as eight normal per-output-bit records in its own
    family section. The shared program_key lets Left/Right treat the records
    as one deterministic chain, including zero-filled shift positions.
    """
    family = program.family
    if family not in records or family not in matches:
        return

    rules = program.to_bit_rules()
    for out_bit, rule in enumerate(rules):
        col = "".join(evaluate_rule(bits, rule) for bits in inputs)
        matched_outputs = tuple(i for i, out_col in enumerate(output_columns) if col == out_col)
        label = compact_rule(rule)
        if not any(rec.label == label and rec.col == col for rec in records[family]):
            records[family].append(
                Record(label, col, column_hash(col, len(inputs)), matched_outputs)
            )
        if out_bit in matched_outputs:
            matches[family][out_bit].append(rule)



def resolve_operand(bits: str, operand: BitOperand) -> str:
    if operand == "C0":
        return "0"
    if operand == "C1":
        return "1"
    return bits[operand]



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

    if rule.operands:
        values = tuple(int(resolve_operand(bits, op)) for op in rule.operands)
        return str(evaluate_global_family(rule.family, values) & 1)

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
    raise ValueError(rule.family)

def compact_rule(c: RuleCandidate) -> str:
    if c.operands:
        return "".join(str(op) for op in c.operands)
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
        failed_next: Optional[str] = None

        if start_cand.family in TERNARY_FAMILIES and start_cand.program_key is None:
            runs.append((chain, None))
            continue

        if start_cand.program_key is not None:
            for out_idx in range(1, len(all_matches)):
                found = next(
                    (c for c in all_matches[out_idx] if c.program_key == start_cand.program_key),
                    None,
                )
                if found is None:
                    failed_next = "x"
                    break
                chain.append(found)
            runs.append((chain, failed_next))
            continue

        cur_p, cur_s = start_cand.primary, start_cand.secondary
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
        failed_next: Optional[str] = None

        if end_cand.family in TERNARY_FAMILIES and end_cand.program_key is None:
            runs.append((chain, None))
            continue

        if end_cand.program_key is not None:
            for step in range(1, n):
                out_idx = n - 1 - step
                found = next(
                    (c for c in all_matches[out_idx] if c.program_key == end_cand.program_key),
                    None,
                )
                if found is None:
                    failed_next = "x"
                    break
                chain.insert(0, found)
            runs.append((chain, failed_next))
            continue

        cur_p, cur_s = end_cand.primary, end_cand.secondary
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

    is_ternary = bool(
        (left_run[0] and left_run[0][0].family in TERNARY_FAMILIES)
        or (right_run[0] and right_run[0][0].family in TERNARY_FAMILIES)
    )
    if is_ternary:
        left_lines = [format_list(left_run[0], failed=left_run[1])] if left_run[0] else ["none"]
        right_lines = [format_list(list(reversed(right_run[0])), failed=right_run[1])] if right_run[0] else ["none"]
    else:
        left_lines = [format_list(chain, failed=failed) for chain, failed in left_runs] if left_runs else ["none"]
        right_lines = [format_list(list(reversed(chain)), failed=failed) for chain, failed in right_runs] if right_runs else ["none"]
    left_best = format_list(left_run[0], with_count=True)
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


def select_rules_without_trace(
    matches: Dict[str, List[List[RuleCandidate]]],
    section_order: Sequence[str],
) -> List[RuleCandidate]:
    """Run the original selection logic using already-computed matches.

    This is used only to decide whether the new ternary exact answer changes
    the old solver answer. It does not recompute any columns or call analyze().
    """
    section_lefts: List[Tuple[str, str]] = []
    section_rights: List[Tuple[str, str]] = []
    for name in section_order:
        _, left_best, _, right_best = lr_from_matches(matches[name])
        section_lefts.append((name, left_best))
        section_rights.append((name, right_best))

    left_name, _left_text, left_count = pick_winner(section_lefts)
    right_name, _right_text, right_count = pick_winner(section_rights)

    def get_run(winner: Optional[str], direction: str) -> List[RuleCandidate]:
        if winner is None:
            return []
        per_bit = matches[winner]
        runs = find_all_left_runs(per_bit) if direction == "left" else find_all_right_runs(per_bit)
        if not runs:
            return []
        return list(max(runs, key=lambda item: len(item[0]))[0])

    left_run = get_run(left_name, "left")
    right_run = get_run(right_name, "right")
    left_len = left_count
    right_len = right_count
    if left_len + right_len > N_BITS:
        if right_len > left_len:
            left_len = N_BITS - right_len
            left_run = left_run[:left_len]
        else:
            right_len = N_BITS - left_len
            right_run = right_run[-right_len:] if right_len else []

    right_start = N_BITS - right_len
    best: List[RuleCandidate] = [
        RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
        for _ in range(N_BITS)
    ]
    for i, rule in enumerate(left_run):
        best[i] = rule
    for i, rule in enumerate(right_run):
        best[right_start + i] = rule

    def extrap_from(
        run: List[RuleCandidate], bit: int, run_start: int, side: str
    ) -> Optional[str]:
        if not run:
            return None
        rule = run[0]
        if rule.primary is not None:
            p_off = (rule.primary - run_start) % N_BITS
            ep = (p_off + bit) % N_BITS
        else:
            ep = None
        if rule.secondary is not None:
            s_off = (rule.secondary - run_start) % N_BITS
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
    left_binary_or_unary = bool(
        left_fam and (left_fam in PAIR_FAMILIES or left_fam in UNARY_FAMILIES)
    )
    right_binary_or_unary = bool(
        right_fam and (right_fam in PAIR_FAMILIES or right_fam in UNARY_FAMILIES)
    )
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
        for i in range(N_BITS):
            if preferred[i] == "pending":
                preferred[i] = (
                    extrap_from(left_run, i, 0, "left")
                    if left_binary_or_unary
                    else "?"
                ) or "?"
            elif "?" in preferred[i][1:] and left_unary:
                candidate = extrap_from(left_run, i, 0, "left")
                if candidate:
                    merged = list(preferred[i])
                    for j, ch in enumerate(candidate[: len(merged)]):
                        if j > 0 and merged[j] == "?" and ch != "?":
                            merged[j] = ch
                    preferred[i] = "".join(merged)
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
        for i in range(N_BITS):
            if preferred[i] == "pending":
                preferred[i] = (
                    extrap_from(right_run, i, right_start, "right")
                    if right_binary_or_unary
                    else "?"
                ) or "?"
            elif "?" in preferred[i][1:] and right_unary:
                candidate = extrap_from(right_run, i, right_start, "right")
                if candidate:
                    merged = list(preferred[i])
                    for j, ch in enumerate(candidate[: len(merged)]):
                        if j > 0 and merged[j] == "?" and ch != "?":
                            merged[j] = ch
                    preferred[i] = "".join(merged)

    pending_indices: List[int] = []
    per_bit_cat: Dict[str, Dict[int, List[RuleCandidate]]] = {
        name: {} for name in section_order
    }
    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith("?") or pref == "?":
            continue
        pending_indices.append(i)
        pref_digits = [int(d) for d in pref[1:] if d != "?"]
        for section_name in section_order:
            cands = matches[section_name][i]
            if section_name in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    per_bit_cat[section_name][i] = found
            elif section_name == "Constant":
                if cands:
                    per_bit_cat[section_name][i] = list(cands)
            else:
                found_c: Optional[RuleCandidate] = None
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                orderings = [(want_p, want_s)]
                if want_p is not None and want_s is not None and want_p != want_s:
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for cand in cands:
                        if (wp is None or cand.primary == wp) and (
                            ws is None or cand.secondary == ws
                        ):
                            found_c = cand
                            break
                    if found_c is not None:
                        break
                if found_c is not None:
                    per_bit_cat[section_name][i] = [found_c]

    chosen_cat: Optional[str] = None
    for cat in section_order:
        if (
            chosen_cat is None
            and bool(pending_indices)
            and all(i in per_bit_cat[cat] for i in pending_indices)
        ):
            chosen_cat = cat

    pending_set = set(pending_indices)
    for i in range(N_BITS):
        if i not in pending_set:
            continue
        if chosen_cat and i in per_bit_cat[chosen_cat]:
            best[i] = per_bit_cat[chosen_cat][i][0]
            continue
        all_cands: List[RuleCandidate] = []
        for name in section_order:
            all_cands.extend(per_bit_cat[name].get(i, []))
        best[i] = (
            all_cands[0]
            if all_cands
            else RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
        )
    return best


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

    def __init__(self, enable_global_consensus: bool = True):
        self.enable_global_consensus = enable_global_consensus

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

        baseline_rules = select_rules_without_trace(matches, BASE_SECTION_ORDER)
        baseline_answer = "".join(
            evaluate_rule(question_bits, rule) for rule in baseline_rules
        )

        global_consensus = (
            find_global_consensus(inputs, outputs, question_bits)
            if self.enable_global_consensus
            else None
        )

        # New functions are checked last. Preserve a substantially simpler
        # seven-bit Identity plus one-bit XOR explanation.
        identity_missing = [bit for bit in range(N_BITS) if not matches["Identity"][bit]]
        simple_near_identity = (
            len(identity_missing) == 1
            and all(matches["Identity"][bit] for bit in range(N_BITS) if bit not in identity_missing)
            and bool(matches["XOR"][identity_missing[0]])
            and not matches["Constant"][identity_missing[0]]
        )

        use_extended_matching = bool(
            global_consensus is not None
            and global_consensus.answer != baseline_answer
            and not simple_near_identity
        )
        if use_extended_matching and global_consensus is not None:
            populate_ternary_matching(records, matches, input_columns, output_columns)
            inject_global_program_into_matching(records, matches, global_consensus.program, inputs, output_columns)

        active_section_order = SECTION_ORDER if use_extended_matching else BASE_SECTION_ORDER

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

        def compact_ternary_view(
            name: str,
            section_records: List[Record],
            per_bit: List[List[RuleCandidate]],
        ) -> Tuple[List[Record], List[List[RuleCandidate]]]:
            # All candidates were evaluated. Render the exact chain plus a
            # deterministic high-coverage subset of partial matches.
            exact_exprs: set[str] = set()
            if global_consensus is not None and global_consensus.program.family == name:
                exact_exprs.update(rule.expr for rule in global_consensus.program.to_bit_rules())
            chosen_labels: set[str] = {expr[len(name):] for expr in exact_exprs if expr.startswith(name)}
            covered: set[int] = set()
            for rec in section_records:
                if rec.label in chosen_labels:
                    covered.update(rec.matches)
            for rec in sorted(section_records, key=lambda r: (-len(r.matches), r.label, r.col)):
                if len(chosen_labels) >= 16:
                    break
                if rec.label in chosen_labels:
                    continue
                if any(bit not in covered for bit in rec.matches):
                    chosen_labels.add(rec.label)
                    covered.update(rec.matches)
            filtered_records = [rec for rec in section_records if rec.label in chosen_labels]
            allowed_exprs = {f"{name}{label}" for label in chosen_labels}
            filtered_matches = [
                [cand for cand in bit_cands if cand.expr in allowed_exprs or cand.program_key is not None]
                for bit_cands in per_bit
            ]
            return filtered_records, filtered_matches

        def add_section(name: str) -> None:
            section_records = records[name]
            per_bit = matches[name]
            if name in TERNARY_FAMILIES:
                section_records, per_bit = compact_ternary_view(name, section_records, per_bit)
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
                    seen: set[str] = set()
                    labels: List[str] = []
                    for cand in cands:
                        label = compact_rule(cand)
                        if label not in seen:
                            seen.add(label)
                            labels.append(label)
                    lines.append(f"{i} " + " ".join(labels))
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

        for name in active_section_order:
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
        per_bit_cat: Dict[str, Dict[int, List[RuleCandidate]]] = {name: {} for name in active_section_order}

        for i in range(N_BITS):
            pref = preferred[i]
            if not pref.startswith("?") or pref == "?":
                lines.append(f"{i} {best[i].expr}")
                continue

            pending_indices.append(i)
            pref_digits = [int(d) for d in pref[1:] if d != "?"]
            checks: List[str] = []

            for section_name in active_section_order:
                cands = matches[section_name][i]
                program_cands = [c for c in cands if c.program_key is not None]
                if program_cands:
                    checks.append(section_name + " " + " ".join(c.expr for c in program_cands))
                    per_bit_cat[section_name][i] = program_cands
                elif section_name in ("Identity", "NOT"):
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
        full_run_cat: Optional[str] = None
        if not pending_indices:
            if left_count == N_BITS and left_name is not None:
                full_run_cat = left_name
            elif right_count == N_BITS and right_name is not None:
                full_run_cat = right_name

        for cat in active_section_order:
            if full_run_cat is not None:
                is_perfect = chosen_cat is None and cat == full_run_cat
            else:
                is_perfect = (
                    chosen_cat is None
                    and bool(pending_indices)
                    and all(i in per_bit_cat[cat] for i in pending_indices)
                )
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
                    for name in active_section_order:
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
            elif rule.operands:
                values = tuple(resolve_operand(question_bits, op) for op in rule.operands)
                result = evaluate_rule(question_bits, rule)
                joined = ",".join(values)
                if rule.family == "NOT":
                    detail = f"NOT({joined})"
                elif rule.family == "AND-NOT":
                    detail = f"AND({values[0]},NOT({values[1]}))"
                elif rule.family == "OR-NOT":
                    detail = f"OR({values[0]},NOT({values[1]}))"
                elif rule.family == "XOR-NOT":
                    detail = f"XOR({values[0]},NOT({values[1]}))"
                elif rule.family == "MIN":
                    detail = f"NOT(MAJ({joined}))"
                elif rule.family == "NCHOICE":
                    detail = f"NOT(CHOICE({joined}))"
                else:
                    detail = f"{rule.family}({joined})"
                lines.append(f"{i} {rule.expr} = {detail} = {result}")
                answer_bits.append(result)
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