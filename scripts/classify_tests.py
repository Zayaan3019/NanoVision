"""
Classify every test function in tests/ as "shape-only", "numeric/value", or
"structural/behavioral", to make an auditable claim about what the test
suite actually verifies (see AUDIT.md §1).

This is a transparent, keyword-based heuristic over each test function's
source text — not a semantic prover. It is intentionally conservative: a
test is only labeled "shape-only" if EVERY assertion in its body matches a
shape/len pattern and nothing else. The full source is printed alongside
each verdict in --verbose mode so the classification can be spot-checked
against the actual assertions by eye.

Usage:
    python scripts/classify_tests.py            # summary table
    python scripts/classify_tests.py --verbose   # + per-test source excerpt
    python scripts/classify_tests.py --json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

# Evidence patterns, checked against each individual `assert` statement's
# source text (not the whole function), so a function is classified by the
# UNION of what its assertions actually do.
SHAPE_PATTERN      = re.compile(r"\.shape\b|^\s*assert\s+len\(")
GRADIENT_PATTERN   = re.compile(r"gradcheck|\.grad\b|requires_grad")
NUMERIC_PATTERNS = [
    re.compile(r"allclose"),
    re.compile(r"isfinite"),
    re.compile(r"pytest\.approx"),
    re.compile(r"\.item\(\)"),
    re.compile(r"\bstd\(\)|\bmean\(\)|\.norm\("),
    re.compile(r"==\s*-?\d"),                 # equality against a literal number
    re.compile(r"[<>]=?"),                     # any ordering comparison (ranges, monotonicity)
    re.compile(r"torch\.eye|torch\.arange|np\.pad|np\.array_equal"),
    re.compile(r"argmax|argmin|topk|sort\("),
    re.compile(r"==\s*[\"'\w]"),                # equality against a literal/variable value
    re.compile(r"\.tolist\(\)"),
]
STRUCTURAL_PATTERNS = [
    re.compile(r"isinstance"),
    re.compile(r"pytest\.raises"),
    re.compile(r"is not None|is None"),
    re.compile(r"\bin\s+\w"),
]
# Behavioral evidence that lives OUTSIDE any `assert` statement, e.g.
#   with pytest.raises(ValueError):
#       ...
RAISES_CONTEXT_PATTERN = re.compile(r"with\s+pytest\.raises")


@dataclass
class TestVerdict:
    file: str
    cls: str
    name: str
    category: str
    evidence: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def qualname(self) -> str:
        prefix = f"{self.cls}." if self.cls else ""
        return f"{self.file}::{prefix}{self.name}"


def _iter_asserts(node: ast.AST):
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            yield child


def _classify_function(fn: ast.FunctionDef, src_lines: list[str]) -> TestVerdict | None:
    if not fn.name.startswith("test_"):
        return None

    fn_src = ast.get_source_segment("\n".join(src_lines), fn) or ""
    asserts = list(_iter_asserts(fn))

    if not asserts:
        if RAISES_CONTEXT_PATTERN.search(fn_src):
            return TestVerdict(
                file="", cls="", name=fn.name, category="structural",
                evidence=["raises"], source=fn_src.strip(),
            )
        return TestVerdict(file="", cls="", name=fn.name, category="no-assertions")

    assert_texts = []
    for a in asserts:
        seg = ast.get_source_segment("\n".join(src_lines), a) or ""
        assert_texts.append(seg)

    has_shape      = any(SHAPE_PATTERN.search(t) for t in assert_texts)
    has_gradient   = any(GRADIENT_PATTERN.search(t) for t in assert_texts)
    has_numeric    = any(p.search(t) for t in assert_texts for p in NUMERIC_PATTERNS)
    has_structural = any(p.search(t) for t in assert_texts for p in STRUCTURAL_PATTERNS)

    non_shape = has_gradient or has_numeric or has_structural
    if has_shape and not non_shape:
        category = "shape-only"
    elif has_gradient:
        category = "gradient"
    elif has_numeric:
        category = "numeric"
    elif has_structural:
        category = "structural"
    else:
        category = "other"

    evidence = []
    if has_shape:      evidence.append("shape")
    if has_gradient:   evidence.append("gradient")
    if has_numeric:    evidence.append("numeric")
    if has_structural: evidence.append("structural")

    return TestVerdict(
        file="", cls="", name=fn.name, category=category,
        evidence=evidence, source="\n".join(assert_texts),
    )


def classify_file(path: Path) -> list[TestVerdict]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    src_lines = src.splitlines()
    out: list[TestVerdict] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    v = _classify_function(item, src_lines)
                    if v:
                        v.file, v.cls = path.name, node.name
                        out.append(v)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Module-level test function (not inside a Test* class)
            parent_classes = {
                n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
            }
            is_nested = any(node in ast.walk(c) for c in parent_classes)
            if not is_nested:
                v = _classify_function(node, src_lines)
                if v:
                    v.file = path.name
                    out.append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    all_verdicts: list[TestVerdict] = []
    for f in sorted(TESTS_DIR.glob("test_*.py")):
        all_verdicts.extend(classify_file(f))

    counts: dict[str, int] = {}
    for v in all_verdicts:
        counts[v.category] = counts.get(v.category, 0) + 1

    total = len(all_verdicts)
    print(f"Total test functions: {total}\n")
    print(f"{'Category':<14} {'Count':>6} {'%':>7}")
    print("-" * 30)
    for cat in ("shape-only", "gradient", "numeric", "structural", "other", "no-assertions"):
        n = counts.get(cat, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"{cat:<14} {n:>6} {pct:>6.1f}%")

    non_shape = total - counts.get("shape-only", 0)
    print(
        f"\n{non_shape}/{total} tests ({100.0*non_shape/total:.1f}%) assert something "
        f"beyond tensor shape (numeric values, gradients, or explicit "
        f"structural/behavioral invariants)."
    )
    print(f"Of those, {counts.get('gradient', 0)} directly touch gradients "
          f"(.grad / gradcheck / requires_grad).")

    if args.verbose:
        print("\n" + "=" * 70)
        for v in all_verdicts:
            print(f"\n[{v.category}] {v.qualname}")
            for line in v.source.splitlines():
                print(f"    {line.strip()}")

    if args.json:
        payload = [
            {"file": v.file, "class": v.cls, "name": v.name,
             "category": v.category, "evidence": v.evidence}
            for v in all_verdicts
        ]
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
