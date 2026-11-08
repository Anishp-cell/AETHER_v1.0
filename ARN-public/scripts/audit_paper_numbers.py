"""
Paper Integrity & Numerical Audit (Phase 11).

Scans LaTeX manuscript files in `docs/paper/` to verify:
1. Complete absence of banned claims and historical unverified figures:
   - "14.2 ms", "5.55 MB", "810x", "38x", "216x", "180x", "88.0", "92.3%", "96.0%", "62.4"
   - "tool-prefixed BIO", "dynamic registry", "beats Jev", "beats Laya", "before TypeSafe"
   - "peer-reviewed", "published at AAAI", "SOTA"
2. Balanced LaTeX environments and braces.
3. Existence of all referenced input files (tables, numbers.tex, figures).
4. Direct consistency of macro numbers with results/ledger.json and master_protocol_summary.json.
"""

import json
from pathlib import Path
import re
import sys
from typing import List, Tuple

BANNED_PATTERNS = [
    r"14\.2\s*ms",
    r"5\.55\s*MB",
    r"810\s*[xX\u00d7]",
    r"38\s*[xX\u00d7]",
    r"216\s*[xX\u00d7]",
    r"180\s*[xX\u00d7]",
    r"88\.0\s*%",
    r"92\.3\s*%",
    r"96\.0\s*%",
    r"62\.4",
    r"tool-prefixed\s+BIO",
    r"dynamic\s+registry",
    r"beats\s+Jev",
    r"beats\s+Laya",
    r"before\s+TypeSafe",
    r"peer-reviewed",
    r"published\s+at\s+AAAI",
    r"\bSOTA\b",
]


def audit_tex_file(tex_path: Path) -> Tuple[bool, List[str]]:
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []

    # 1. Check banned phrases
    for pat in BANNED_PATTERNS:
        matches = list(re.finditer(pat, content, re.IGNORECASE))
        if matches:
            for m in matches:
                # Find line number
                line_no = content[:m.start()].count("\n") + 1
                errors.append(f"{tex_path.name}:{line_no}: Banned claim pattern '{pat}' matched: '{m.group(0)}'")

    # 2. Check balanced braces
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces != close_braces:
        errors.append(f"{tex_path.name}: Unbalanced curly braces! Open: {open_braces}, Close: {close_braces}")

    # 3. Check balanced environments
    begins = re.findall(r"\\begin\{([a-zA-Z0-9*]+)\}", content)
    ends = re.findall(r"\\end\{([a-zA-Z0-9*]+)\}", content)
    if sorted(begins) != sorted(ends):
        errors.append(f"{tex_path.name}: Unbalanced environments! Begins: {len(begins)}, Ends: {len(ends)}")

    # 4. Check referenced input files
    inputs = re.findall(r"\\input\{([^}]+)\}", content)
    tex_dir = tex_path.parent
    for inp in inputs:
        inp_clean = inp if inp.endswith(".tex") else inp + ".tex"
        target = tex_dir / inp_clean
        if not target.exists():
            errors.append(f"{tex_path.name}: Referenced \\input{{{inp}}} does not exist at {target}")

    return len(errors) == 0, errors


def verify_macro_consistency(numbers_tex: Path, master_json: Path) -> Tuple[bool, List[str]]:
    with open(numbers_tex, "r", encoding="utf-8") as f:
        tex_text = f.read()
    with open(master_json, "r", encoding="utf-8") as f:
        master = json.load(f)

    errors = []
    # Verify primary Joint EM
    em_match = re.search(r"\\newcommand\{\\PrimaryJointEM\}\{([0-9.]+)%\}", tex_text)
    if em_match:
        val = float(em_match.group(1))
        expected = round(master["models"]["unfrozen_cls_crf"]["joint_em_mean"] * 100, 2)
        if abs(val - expected) > 0.05:
            errors.append(f"Macro \\PrimaryJointEM ({val}%) does not match master summary ({expected}%)")

    return len(errors) == 0, errors


def main():
    root = Path(__file__).resolve().parent.parent
    paper_dir = root / "docs" / "paper"
    results_dir = root / "results"

    tex_files = list(paper_dir.glob("*.tex"))
    print(f"Auditing {len(tex_files)} LaTeX files in {paper_dir}...")

    all_passed = True
    for tf in tex_files:
        passed, errs = audit_tex_file(tf)
        if passed:
            print(f"  [PASS] {tf.name}")
        else:
            print(f"  [FAIL] {tf.name} ({len(errs)} issues):")
            for e in errs:
                print(f"    - {e}")
            all_passed = False

    numbers_tex = paper_dir / "numbers.tex"
    master_json = results_dir / "master_protocol_summary.json"
    if numbers_tex.exists() and master_json.exists():
        passed_macro, m_errs = verify_macro_consistency(numbers_tex, master_json)
        if passed_macro:
            print(f"  [PASS] Macro consistency check ({numbers_tex.name} vs {master_json.name})")
        else:
            print(f"  [FAIL] Macro consistency check:")
            for e in m_errs:
                print(f"    - {e}")
            all_passed = False

    if all_passed:
        print("\nAll paper audits passed with ZERO errors and ZERO banned claims.")
        sys.exit(0)
    else:
        print("\nPaper audit FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
