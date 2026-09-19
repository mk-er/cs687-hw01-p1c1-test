"""Check whether a Homework 1 notebook has the structure required for grading.

This public preflight tool checks format and structure only. It does not grade
answers, run hidden tests, or guarantee a particular mark. Staff grading uses
an independent trusted checker.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


ASSIGNMENT_ID = "cs687-hw01-p1c1"
ASSIGNMENT_VERSION = 3
FILENAME_PATTERN = re.compile(r"^homework01_colab_(\d+)\.ipynb$")
REPORT_PLACEHOLDER = "_Replace this line with your response._"
REPORT_RESPONSE_MARKER = "**Response:**"
CODE_START_MARKER = "# ================= YOUR CODE STARTS HERE ================="
CODE_END_MARKER = "# ================= YOUR CODE ENDS HERE ==================="


@dataclass(frozen=True)
class AnswerSpec:
    key: str
    tag: str
    cell_id: str
    cell_type: str


ANSWER_SPECS = (
    AnswerSpec("guided_probability", "answer-guided-probability", "hw01-probability-to-loss-code", "code"),
    AnswerSpec("task_1", "answer-task-1", "ea939615", "code"),
    AnswerSpec("guided_fertility", "answer-guided-fertility", "0fa1fe8e", "code"),
    AnswerSpec("task_2", "answer-task-2", "8592ac6b", "code"),
    AnswerSpec("guided_embedding_components", "answer-guided-embedding-components", "hw01-embedding-components-code", "code"),
    AnswerSpec("guided_unit_conversion", "answer-guided-unit-conversion", "d2a186d2", "code"),
    AnswerSpec("report_1", "answer-report-1", "hw01-report-bpe", "markdown"),
    AnswerSpec("report_2", "answer-report-2", "hw01-report-inductive-bias", "markdown"),
    AnswerSpec("report_3", "answer-report-3", "hw01-report-transformer-inputs", "markdown"),
    AnswerSpec("report_4", "answer-report-4", "hw01-report-evaluation", "markdown"),
)


@dataclass
class Finding:
    level: str
    message: str


def _read_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        notebook = json.load(handle)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("the file is JSON but not a Jupyter notebook")
    if not isinstance(notebook.get("metadata", {}), dict):
        raise ValueError("the notebook metadata is malformed")
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            raise ValueError(f"notebook cell {index} is malformed")
        if not isinstance(cell.get("metadata", {}), dict):
            raise ValueError(f"metadata for notebook cell {index} is malformed")
        tags = cell.get("metadata", {}).get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"tags for notebook cell {index} are malformed")
        source = cell.get("source", "")
        if not isinstance(source, (str, list)) or (
            isinstance(source, list)
            and not all(isinstance(part, str) for part in source)
        ):
            raise ValueError(f"source for notebook cell {index} is malformed")
    return notebook


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _normalized(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _same_source(left: str, right: str) -> bool:
    return _normalized(left).rstrip("\n") == _normalized(right).rstrip("\n")


def _tags(cell: dict[str, Any]) -> list[str]:
    return list(cell.get("metadata", {}).get("tags", []))


def _code_scaffold(source: str) -> tuple[str, str] | None:
    source = _normalized(source)
    if source.count(CODE_START_MARKER) != 1 or source.count(CODE_END_MARKER) != 1:
        return None
    before, remainder = source.split(CODE_START_MARKER, 1)
    _, after = remainder.split(CODE_END_MARKER, 1)
    return before, after.rstrip("\n")


def _report_response(source: str) -> str:
    if REPORT_RESPONSE_MARKER not in source:
        return ""
    return source.split(REPORT_RESPONSE_MARKER, 1)[1].strip()


def check_submission(
    submission: dict[str, Any], template: dict[str, Any], filename: str
) -> tuple[list[Finding], bool]:
    """Return findings and whether an older version needs staff inspection."""
    findings: list[Finding] = []

    if FILENAME_PATTERN.fullmatch(filename) is None:
        findings.append(Finding(
            "warning",
            "Rename the file to homework01_colab_STUDENTNUMBER.ipynb, using your own student number.",
        ))

    metadata = submission.get("metadata", {}).get("cs687_assignment", {})
    if metadata.get("assignment_id") != ASSIGNMENT_ID:
        findings.append(Finding(
            "error",
            "This is not the CS 687 Homework 1 notebook. Start from the official Homework 1 release.",
        ))
        return findings, False

    version = metadata.get("version")
    if version != ASSIGNMENT_VERSION:
        findings.append(Finding(
            "inspection",
            f"This notebook is assignment version {version!r}; the current version is {ASSIGNMENT_VERSION}. "
            "Do not edit the version number yourself. Contact the course staff so they can inspect the notebook.",
        ))
        return findings, True

    cells = submission["cells"]
    template_cells = template["cells"]
    submitted_by_id = {cell.get("id"): cell for cell in cells}
    template_ids = [cell.get("id") for cell in template_cells]
    template_id_set = set(template_ids)

    submitted_ids = [cell.get("id") for cell in cells]
    duplicate_ids = sorted({
        cell_id
        for cell_id in submitted_ids
        if cell_id is not None and submitted_ids.count(cell_id) > 1
    })
    for cell_id in duplicate_ids:
        findings.append(Finding(
            "error",
            f"Notebook cell ID {cell_id!r} appears more than once. Restore unique cell IDs from a fresh notebook.",
        ))

    for cell_id in template_ids:
        if cell_id not in submitted_by_id:
            findings.append(Finding(
                "error",
                f"Required notebook cell {cell_id!r} is missing. Restore it from a fresh released notebook.",
            ))

    submitted_order = [cell.get("id") for cell in cells if cell.get("id") in template_id_set]
    expected_order = [cell_id for cell_id in template_ids if cell_id in submitted_by_id]
    if submitted_order != expected_order:
        findings.append(Finding(
            "error",
            "Required notebook cells were reordered. Restore their original order; extra scratch cells may remain.",
        ))

    declared = metadata.get("answer_cells", {})
    by_tag = {spec.tag: [] for spec in ANSWER_SPECS}
    for cell in cells:
        for tag in _tags(cell):
            if tag in by_tag:
                by_tag[tag].append(cell)

    for spec in ANSWER_SPECS:
        matches = by_tag[spec.tag]
        if len(matches) != 1:
            findings.append(Finding(
                "error",
                f"Expected exactly one cell tagged {spec.tag!r}, but found {len(matches)}. Restore the tag from a fresh notebook.",
            ))
            continue
        cell = matches[0]
        if cell.get("id") != spec.cell_id:
            findings.append(Finding(
                "error",
                f"The stable ID of {spec.tag!r} changed. Restore that cell and copy only your answer back.",
            ))
        if cell.get("cell_type") != spec.cell_type:
            findings.append(Finding("error", f"Cell {spec.tag!r} must remain a {spec.cell_type} cell."))
        if declared.get(spec.key) != spec.cell_id:
            findings.append(Finding(
                "error",
                f"The answer-cell metadata for {spec.tag!r} changed. Restore it from a fresh notebook.",
            ))

        source = _source(cell)
        if spec.cell_type == "code" and "NotImplementedError" in source:
            findings.append(Finding(
                "error",
                f"Cell {spec.tag!r} is incomplete because it still contains NotImplementedError.",
            ))
        if spec.cell_type == "markdown":
            response = _report_response(source)
            if not response or REPORT_PLACEHOLDER in response:
                findings.append(Finding("error", f"Cell {spec.tag!r} still has a blank report response."))

    template_by_id = {cell.get("id"): cell for cell in template_cells}
    spec_by_id = {spec.cell_id: spec for spec in ANSWER_SPECS}
    for cell_id, template_cell in template_by_id.items():
        submitted_cell = submitted_by_id.get(cell_id)
        if submitted_cell is None:
            continue
        if submitted_cell.get("cell_type") != template_cell.get("cell_type"):
            findings.append(Finding(
                "error",
                f"Supplied cell {cell_id!r} changed cell type. Restore it from a fresh notebook.",
            ))
            continue
        spec = spec_by_id.get(cell_id)
        if spec is None:
            if not _same_source(_source(submitted_cell), _source(template_cell)):
                findings.append(Finding(
                    "error",
                    f"Supplied cell {cell_id!r} was modified. Restore it; put personal work only in answer regions or scratch cells.",
                ))
            continue

        if spec.cell_type == "code":
            if _code_scaffold(_source(submitted_cell)) != _code_scaffold(_source(template_cell)):
                findings.append(Finding(
                    "error",
                    f"Supplied code outside the answer region in {spec.tag!r} was modified. Restore it and copy only your answer between the marked lines.",
                ))
        else:
            submitted_source = _normalized(_source(submitted_cell))
            template_source = _normalized(_source(template_cell))
            submitted_prompt = submitted_source.split(REPORT_RESPONSE_MARKER, 1)[0]
            template_prompt = template_source.split(REPORT_RESPONSE_MARKER, 1)[0]
            if REPORT_RESPONSE_MARKER not in submitted_source or submitted_prompt != template_prompt:
                findings.append(Finding(
                    "error",
                    f"The supplied prompt in {spec.tag!r} was modified. Restore it and change only the response.",
                ))

    return findings, False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="completed notebook to check")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent / "notebooks" / "homework01_colab.ipynb",
        help="untouched released Homework 1 notebook",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.notebook.suffix.lower() != ".ipynb":
        print("SUBMISSION CHECK FAILED")
        print("[ERROR] The submitted file does not have the .ipynb extension.")
        print("A non-.ipynb submission cannot be graded and receives zero.")
        return 2
    try:
        submission = _read_notebook(args.notebook)
    except Exception as exc:
        print("SUBMISSION CHECK FAILED")
        print(f"[ERROR] The submitted file is not a readable .ipynb notebook: {exc}")
        print("An unreadable or non-notebook submission cannot be graded and receives zero.")
        return 2

    try:
        template = _read_notebook(args.template)
    except Exception as exc:
        print(f"Could not read the released notebook template: {exc}", file=sys.stderr)
        print("Restore a fresh homework repository and try again.", file=sys.stderr)
        return 2

    if args.notebook.resolve() == args.template.resolve():
        print(
            "[WARNING] This is also the reference-template path. If you edited it locally, "
            "use an untouched template from a fresh repository copy."
        )

    findings, inspection_required = check_submission(submission, template, args.notebook.name)
    for finding in findings:
        print(f"[{finding.level.upper()}] {finding.message}")

    errors = [finding for finding in findings if finding.level == "error"]
    if errors:
        print(f"\nNOT READY: fix the {len(errors)} error(s) above before submitting.")
        return 1
    if inspection_required:
        print("\nSTAFF INSPECTION REQUIRED: contact the course staff before submitting.")
        return 3

    print("\nREADY: the notebook has the structure required by the grading system.")
    print("This does not grade correctness; run all notebook checks as well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
