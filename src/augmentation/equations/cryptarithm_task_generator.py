from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


RULE_LEGEND = """Rule notation:
std: read each visible number left-to-right; rev: read each visible number right-to-left.
cat: concatenate the two visible operands; concat normally does not constrain digit values.
add/add1/addm1: compute A+B, A+B+1, or A+B-1.
abs/sub: compute abs(A-B) or signed A-B.
mul/mul1/mulm1: compute A*B, A*B+1, or A*B-1.
Digit map is injective: different visible digit symbols must use different decimal digits.
No-leading-zero: a multi-digit decoded number cannot start with 0."""


@dataclass
class MappingSubtaskConfig:
    include_build_mapping: bool = True
    include_rule_selection: bool = True
    include_projection_step: bool = True
    include_domain_update: bool = True
    include_target_application: bool = True
    include_concat_direct: bool = True
    include_rule_legend: bool = True
    skip_fixed_concat_build_mapping: bool = True
    # Projection blocks with huge carry transition text are still emitted; no truncation.


class CryptarithmMappingSubtaskGenerator:
    """Generate focused cryptarithm subtasks from v3 training CoTs.

    This intentionally targets the new v3 format:
      - Rule and digit-domain search
      - Family/domain block N: operator ... tests candidate families
      - projection by ...
      - accepted digit/tuple decisions
      - fixed concat direct traces

    The main task is `cryptarithm_build_mapping`: reconstruct selected rules
    and digit map from the domain reasoning, without target application.
    """

    def __init__(self, config: Optional[MappingSubtaskConfig] = None):
        self.config = config or MappingSubtaskConfig()

    # ----------------------------- parsing -----------------------------

    @staticmethod
    def parse_text_dump(text: str) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        parts = re.split(r"(?=^=== Категория:)", text or "", flags=re.M)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            mid = re.search(r"ID:\s*([^\s=]+)", part)
            source_id = mid.group(1) if mid else hashlib.sha256(part.encode()).hexdigest()[:8]
            mp = re.search(r"--- Промпт \(начало\) ---\n(.*?)\n\n--- Решение ---", part, flags=re.S)
            ms = re.search(r"--- Решение ---\n(.*?)(?:\n\n--- Истинный ответ ---|\Z)", part, flags=re.S)
            ma = re.search(r"--- Истинный ответ ---\n(.*?)(?:\n=== Категория:|\Z)", part, flags=re.S)
            if not ms:
                continue
            prompt = mp.group(1).strip() if mp else ""
            cot = ms.group(1).strip()
            truth = ma.group(1).strip() if ma else ""
            records.append({"id": source_id, "source_prompt": prompt, "cot": cot, "truth_answer": truth})
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _stable_id(source_id: str, task_type: str, key: str) -> str:
        return hashlib.sha256(f"{source_id}|{task_type}|{key}".encode()).hexdigest()[:16]

    @staticmethod
    def _section(text: str, start_header: str, end_headers: Sequence[str]) -> str:
        # start_header is literal header without anchors, e.g. "Examples".
        m = re.search(rf"(?m)^{re.escape(start_header)}\s*$", text or "")
        if not m:
            return ""
        tail = text[m.end():]
        if not end_headers:
            return tail.strip("\n")
        end_re = r"(?m)^" + r"(?:" + "|".join(re.escape(h) for h in end_headers) + r")" + r"\s*$"
        e = re.search(end_re, tail)
        return tail[: e.start()].strip("\n") if e else tail.strip("\n")

    @staticmethod
    def _extract_final_answer(cot: str) -> str:
        m = re.findall(r"(?m)^Final answer:\s*(\S+)\s*$", cot or "")
        return m[-1] if m else ""

    @staticmethod
    def _is_fixed_concat_direct(cot: str) -> bool:
        return "The target operator is a transparent positional concat" in (cot or "")

    @staticmethod
    def _split_family_blocks(domain_search: str) -> List[Tuple[int, str, str]]:
        # returns (number, header, full_block)
        starts = list(re.finditer(r"(?m)^Family/domain block\s+(\d+):\s*(.+)$", domain_search or ""))
        out: List[Tuple[int, str, str]] = []
        for i, m in enumerate(starts):
            start = m.start()
            end = starts[i + 1].start() if i + 1 < len(starts) else len(domain_search)
            out.append((int(m.group(1)), m.group(2).strip(), domain_search[start:end].strip()))
        return out

    @staticmethod
    def _strip_lines(block: str, prefixes: Sequence[str]) -> str:
        out: List[str] = []
        for ln in str(block or "").splitlines():
            s = ln.strip()
            if any(s.startswith(p) for p in prefixes):
                continue
            out.append(ln.rstrip())
        return "\n".join(out).strip()

    @staticmethod
    def _domain_update_pattern() -> re.Pattern[str]:
        return re.compile(
            r"(D\[(?:'[^']*'|\"[^\"]*\"|[^\]]+)\])\s*:\s*(\{[^{}]*\})\s*->\s*(\{[^{}]*\})"
        )

    @classmethod
    def _extract_apply_projection_updates(cls, block: str) -> str:
        m = re.search(r"(?m)^\s*apply projection:\s*(.+)$", block or "")
        if m:
            return m.group(1).strip()
        m = re.search(r"(?m)^\s*update:\s*(.+)$", block or "")
        if m:
            return m.group(1).strip()
        m = re.search(r"(?m)^\s*apply update:\s*(.+)$", block or "")
        return m.group(1).strip() if m else ""

    @classmethod
    def _current_domains_from_updates(cls, update_text: str) -> List[str]:
        rows = []
        for m in cls._domain_update_pattern().finditer(update_text or ""):
            rows.append(f"{m.group(1)}={m.group(2)}")
        return rows

    @staticmethod
    def _extract_selected_rules(cot: str) -> str:
        return CryptarithmMappingSubtaskGenerator._section(cot, "Selected rules", ["Selected digit map", "Verify selected rules", "Target"])

    @staticmethod
    def _extract_selected_map(cot: str) -> str:
        return CryptarithmMappingSubtaskGenerator._section(cot, "Selected digit map", ["Verify selected rules", "Target"])

    @staticmethod
    def _extract_target_section(cot: str) -> str:
        return CryptarithmMappingSubtaskGenerator._section(cot, "Target", [])

    @staticmethod
    def _context_sections(cot: str, include_domain_search: bool = False) -> str:
        names = [
            ("Examples", ["Parsed equations", "Symbols", "Transparent concat", "Search space", "Structural family filtering", "Rule and digit-domain search", "Selected rules", "Target"]),
            ("Parsed equations", ["Symbols", "Transparent concat", "Search space", "Structural family filtering", "Rule and digit-domain search", "Selected rules", "Target"]),
            ("Symbols", ["Transparent concat", "Search space", "Structural family filtering", "Rule and digit-domain search", "Selected rules", "Target"]),
            ("Transparent concat", ["Search space", "Structural family filtering", "Rule and digit-domain search", "Selected rules", "Target"]),
            ("Search space", ["Structural family filtering", "Rule and digit-domain search", "Selected rules", "Target"]),
            ("Structural family filtering", ["Rule and digit-domain search", "Selected rules", "Target"]),
        ]
        parts: List[str] = []
        for name, ends in names:
            sec = CryptarithmMappingSubtaskGenerator._section(cot, name, ends)
            if sec:
                parts.append(f"{name}\n{sec}")
        if include_domain_search:
            sec = CryptarithmMappingSubtaskGenerator._section(cot, "Rule and digit-domain search", ["Selected rules", "Selected digit map", "Verify selected rules", "Target"])
            if sec:
                parts.append(f"Rule and digit-domain search\n{sec}")
        return "\n\n".join(parts).strip()

    # ----------------------------- row helpers -----------------------------

    def _prompt_prefix(self, task: str) -> List[str]:
        lines = ["In Alice's Wonderland, solve only the focused cryptarithm subtask below.", task]
        if self.config.include_rule_legend:
            lines.extend(["", RULE_LEGEND])
        return lines

    def _row(self, source_id: str, source_prompt: str, source_answer: str, task_type: str, key: str,
             prompt: str, completion: str, answer: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "id": self._stable_id(source_id, task_type, key),
            "source_task_id": source_id,
            "source_prompt": source_prompt,
            "source_answer": source_answer,
            "task_mode": task_type,
            "category": task_type,
            "label": task_type,
            "prompt": prompt.strip(),
            "completion": completion.strip(),
            "answer": answer.strip(),
            "computed_answer": answer.strip(),
            "generated_cot": completion.strip(),
            "source_solver_correct": True,
        }
        if extra:
            row.update(extra)
        return row

    # ----------------------------- task makers -----------------------------

    def _make_build_mapping(self, source_id: str, source_prompt: str, source_answer: str, cot: str) -> Optional[Dict[str, Any]]:
        selected_rules = self._extract_selected_rules(cot)
        selected_map = self._extract_selected_map(cot)
        domain_search = self._section(cot, "Rule and digit-domain search", ["Selected rules", "Selected digit map", "Verify selected rules", "Target"])
        if not selected_map or not domain_search:
            return None
        context = self._context_sections(cot, include_domain_search=True)
        prompt_lines = self._prompt_prefix(
            "Task: build the digit mapping. Use the family support tables, projections, AllDifferent, and accepted decisions. Do not verify examples and do not solve the target."
        )
        prompt_lines.extend(["", "Source reasoning context:", context, "", "Return the selected rules and the selected digit map. Finish with:", "Answer: <semicolon-separated digit map>"])
        answer = selected_map.replace("\n", " ").strip()
        completion = f"Selected rules\n{selected_rules}\n\nSelected digit map\n{selected_map}\nAnswer: {answer}"
        return self._row(source_id, source_prompt, source_answer, "cryptarithm_build_mapping", "build_mapping", "\n".join(prompt_lines), completion, answer)

    def _make_rule_selection_tasks(self, source_id: str, source_prompt: str, source_answer: str, cot: str) -> List[Dict[str, Any]]:
        domain_search = self._section(cot, "Rule and digit-domain search", ["Selected rules", "Selected digit map", "Verify selected rules", "Target"])
        if not domain_search:
            return []
        context = self._context_sections(cot, include_domain_search=False)
        out: List[Dict[str, Any]] = []
        for num, header, block in self._split_family_blocks(domain_search):
            if "tests candidate families" not in header:
                continue
            selected = re.search(r"(?m)^\s*selected successful candidate:\s*(.+)$", block)
            family = re.search(r"(?m)^\s*selected family:\s*(.+)$", block)
            if not selected:
                continue
            selected_rule = selected.group(1).strip()
            # Prompt version: remove direct selected lines and remove selected marker in support table.
            prompt_block = self._strip_lines(block, ["selected successful candidate:", "selected family:", "decision:"])
            prompt_block = re.sub(r"\s*<- selected full-continuation rule", "", prompt_block)
            # Keep explanation, but hide selected candidate token in the first sentence where possible.
            prompt_block = prompt_block.replace(f"candidates tried before {selected_rule}:", "candidates failed before the successful continuation:")
            prompt_block = prompt_block.replace(f"decision: select {selected_rule}", "decision: select <hidden>")
            prompt_lines = self._prompt_prefix(
                "Task: choose the operator rule. Local support is not enough; select the first rule whose downstream continuation reaches a verifying global digit map."
            )
            prompt_lines.extend(["", "Context:", context, "", "Rule-selection block:", prompt_block, "", "Return the selected full-continuation rule and explain briefly. Finish with:", "Answer: <rule>"])
            completion = block.rstrip() + f"\nAnswer: {selected_rule}"
            out.append(self._row(
                source_id, source_prompt, source_answer, "cryptarithm_rule_selection", f"rule_{num}", "\n".join(prompt_lines), completion, selected_rule,
                extra={"block_number": num, "selected_family": family.group(1).strip() if family else ""}
            ))
        return out

    def _make_projection_tasks(self, source_id: str, source_prompt: str, source_answer: str, cot: str) -> List[Dict[str, Any]]:
        domain_search = self._section(cot, "Rule and digit-domain search", ["Selected rules", "Selected digit map", "Verify selected rules", "Target"])
        if not domain_search:
            return []
        context = self._context_sections(cot, include_domain_search=False)
        out: List[Dict[str, Any]] = []
        for num, header, block in self._split_family_blocks(domain_search):
            if not header.startswith("projection by"):
                continue
            update_text = self._extract_apply_projection_updates(block)
            if not update_text:
                continue
            # Remove target outputs from prompt: supported projection and apply projection/update.
            prompt_block = self._strip_lines(block, ["supported digit projection:", "apply projection:", "apply update:", "update:"])
            current_domains = self._current_domains_from_updates(update_text)
            prompt_lines = self._prompt_prefix(
                "Task: perform this mapping projection step. From the formula, column reasoning, and current domains, compute the supported digit projection and the domain update."
            )
            prompt_lines.extend(["", "Context:", context, "", "Projection block input:", prompt_block])
            if current_domains:
                prompt_lines.extend(["", "Current domains before this update:"])
                prompt_lines.extend(current_domains)
            prompt_lines.extend(["", "Return the supported digit projection and apply projection. Finish with:", "Answer: <domain updates>"])
            # Completion keeps the original full block.
            completion = block.rstrip() + f"\nAnswer: {update_text}"
            out.append(self._row(
                source_id, source_prompt, source_answer, "cryptarithm_mapping_projection", f"projection_{num}", "\n".join(prompt_lines), completion, update_text,
                extra={"block_number": num}
            ))
        return out

    def _make_domain_update_tasks(self, source_id: str, source_prompt: str, source_answer: str, cot: str) -> List[Dict[str, Any]]:
        domain_search = self._section(cot, "Rule and digit-domain search", ["Selected rules", "Selected digit map", "Verify selected rules", "Target"])
        if not domain_search:
            return []
        context = self._context_sections(cot, include_domain_search=False)
        out: List[Dict[str, Any]] = []
        update_headers = ("No-leading-zero", "AllDifferent", "accepted digit assignment", "accepted tuple assignment")
        for num, header, block in self._split_family_blocks(domain_search):
            if not any(header.startswith(h) for h in update_headers):
                continue
            update_text = self._extract_apply_projection_updates(block)
            if not update_text:
                continue
            prompt_block = self._strip_lines(block, ["update:", "apply update:", "decision:"])
            current_domains = self._current_domains_from_updates(update_text)
            prompt_lines = self._prompt_prefix(
                "Task: apply this local domain update while building the digit map."
            )
            prompt_lines.extend(["", "Context:", context, "", "Local domain operation:", prompt_block])
            if current_domains:
                prompt_lines.extend(["", "Current domains before this update:"])
                prompt_lines.extend(current_domains)
            prompt_lines.extend(["", "Return the domain update. Finish with:", "Answer: <domain updates>"])
            completion = block.rstrip() + f"\nAnswer: {update_text}"
            out.append(self._row(
                source_id, source_prompt, source_answer, "cryptarithm_mapping_domain_update", f"domain_{num}", "\n".join(prompt_lines), completion, update_text,
                extra={"block_number": num, "domain_header": header}
            ))
        return out

    def _make_target_application(self, source_id: str, source_prompt: str, source_answer: str, cot: str) -> Optional[Dict[str, Any]]:
        target = self._extract_target_section(cot)
        if not target:
            return None
        selected_rules = self._extract_selected_rules(cot)
        selected_map = self._extract_selected_map(cot)
        answer = self._extract_final_answer(cot)
        # For fixed concat direct, no map is needed.
        prompt_lines = self._prompt_prefix("Task: apply the selected rule and digit map to the target only. Do not redo the mapping search.")
        if selected_rules:
            prompt_lines.extend(["", "Selected rules:", selected_rules])
        if selected_map:
            prompt_lines.extend(["", "Selected digit map:", selected_map])
        else:
            prompt_lines.extend(["", "Digit map:", "not needed for a fixed concat target"])
        prompt_lines.extend(["", "Target section input:", self._strip_lines(target, ["Computed output:", "Final answer:", "\\boxed"]), "", "Return the target calculation. Finish with:", "Answer: <visible output>"])
        completion = target.rstrip() + f"\nAnswer: {answer}"
        mode = "cryptarithm_concat_direct_target" if self._is_fixed_concat_direct(cot) else "cryptarithm_target_application"
        return self._row(source_id, source_prompt, source_answer, mode, "target", "\n".join(prompt_lines), completion, answer)

    def _make_concat_direct(self, source_id: str, source_prompt: str, source_answer: str, cot: str) -> Optional[Dict[str, Any]]:
        if not self._is_fixed_concat_direct(cot):
            return None
        transparent = self._section(cot, "Transparent concat", ["Structural family filtering", "Target"])
        structural = self._section(cot, "Structural family filtering", ["Target"])
        target = self._extract_target_section(cot)
        answer = self._extract_final_answer(cot)
        if not target:
            return None
        prompt_lines = self._prompt_prefix("Task: solve a fixed positional concat target. Do not assign digit values.")
        prompt_lines.extend(["", "Transparent concat evidence:", transparent, "", "Structural filtering:", structural, "", "Target input:", self._strip_lines(target, ["Computed output:", "Final answer:", "\\boxed"]), "", "Return the concat output. Finish with:", "Answer: <visible output>"])
        completion = target.rstrip() + f"\nAnswer: {answer}"
        return self._row(source_id, source_prompt, source_answer, "cryptarithm_concat_direct", "concat_direct", "\n".join(prompt_lines), completion, answer)

    # ----------------------------- public -----------------------------

    def make_tasks_for_record(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_id = str(rec.get("id", ""))
        source_prompt = str(rec.get("source_prompt", ""))
        cot = str(rec.get("cot", ""))
        source_answer = str(rec.get("truth_answer", "")) or self._extract_final_answer(cot)
        fixed_concat = self._is_fixed_concat_direct(cot)
        rows: List[Dict[str, Any]] = []

        if fixed_concat:
            if self.config.include_concat_direct:
                row = self._make_concat_direct(source_id, source_prompt, source_answer, cot)
                if row:
                    rows.append(row)
            if self.config.include_target_application:
                row = self._make_target_application(source_id, source_prompt, source_answer, cot)
                if row:
                    rows.append(row)
            # Important: no build_mapping task for fixed concat target.
            if self.config.skip_fixed_concat_build_mapping:
                return rows

        if self.config.include_build_mapping:
            row = self._make_build_mapping(source_id, source_prompt, source_answer, cot)
            if row:
                rows.append(row)
        if self.config.include_rule_selection:
            rows.extend(self._make_rule_selection_tasks(source_id, source_prompt, source_answer, cot))
        if self.config.include_projection_step:
            rows.extend(self._make_projection_tasks(source_id, source_prompt, source_answer, cot))
        if self.config.include_domain_update:
            rows.extend(self._make_domain_update_tasks(source_id, source_prompt, source_answer, cot))
        if self.config.include_target_application:
            row = self._make_target_application(source_id, source_prompt, source_answer, cot)
            if row:
                rows.append(row)
        return rows

    def generate_from_text(self, text: str) -> pd.DataFrame:
        df = self.parse_text_dump(text)
        rows: List[Dict[str, Any]] = []
        for rec in df.to_dict("records"):
            rows.extend(self.make_tasks_for_record(rec))
        return pd.DataFrame.from_records(rows)

    def generate_dataset(self, source_data: pd.DataFrame) -> pd.DataFrame:
        """Generate mapping-focused subtasks from a normal solver dataframe.

        Expected source columns are the same as prepare_dataset.py uses:
        id, prompt, answer, generated_cot/computed_answer, and label.
        Rows that do not contain the v3 cryptarithm training-trace format simply
        emit zero subtasks.
        """
        if source_data is None or len(source_data) == 0:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for idx, row in source_data.iterrows():
            label = str(row.get("label", ""))
            if label and label != "cryptarithm":
                continue
            cot = str(row.get("generated_cot", row.get("cot", row.get("completion", ""))) or "")
            if not cot or "Cryptarithm training trace" not in cot:
                continue
            source_id = str(row.get("id", idx))
            prompt = str(row.get("prompt", "") or "")
            answer = str(row.get("answer", "") or self._extract_final_answer(cot))
            rec = {
                "id": source_id,
                "source_prompt": prompt,
                "cot": cot,
                "truth_answer": answer,
            }
            rows.extend(self.make_tasks_for_record(rec))
        return pd.DataFrame.from_records(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mapping-focused cryptarithm subtasks from v3 CoT dumps.")
    ap.add_argument("input", help="Input pasted text dump produced by the v3 cryptarithm CoT solver")
    ap.add_argument("--output", "-o", default="cryptarithm_mapping_subtasks_v3.jsonl", help="Output .jsonl/.csv/.parquet path")
    ap.add_argument("--csv", default="", help="Optional extra CSV output path")
    ap.add_argument("--no-target", action="store_true", help="Do not emit target application subtasks")
    ap.add_argument("--no-rule-selection", action="store_true", help="Do not emit rule-selection subtasks")
    ap.add_argument("--no-projection", action="store_true", help="Do not emit projection subtasks")
    ap.add_argument("--no-domain-update", action="store_true", help="Do not emit local domain update subtasks")
    ap.add_argument("--no-concat-direct", action="store_true", help="Do not emit fixed-concat direct subtasks")
    ap.add_argument("--no-legend", action="store_true", help="Do not include rule notation legend in prompts")
    args = ap.parse_args()

    config = MappingSubtaskConfig(
        include_rule_selection=not args.no_rule_selection,
        include_projection_step=not args.no_projection,
        include_domain_update=not args.no_domain_update,
        include_target_application=not args.no_target,
        include_concat_direct=not args.no_concat_direct,
        include_rule_legend=not args.no_legend,
    )
    text = Path(args.input).read_text(encoding="utf-8")
    gen = CryptarithmMappingSubtaskGenerator(config)
    df = gen.generate_from_text(text)

    out = Path(args.output)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    elif out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        with out.open("w", encoding="utf-8") as f:
            for row in df.to_dict("records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.csv:
        df.to_csv(args.csv, index=False)

    print(f"wrote {len(df)} rows to {out}")
    if len(df):
        print(df["task_mode"].value_counts().to_string())


if __name__ == "__main__":
    main()
