"""
Unit test for Paper Number Consistency and Banned Claim Absence (Phase 11).
"""

from pathlib import Path
from scripts.audit_paper_numbers import audit_tex_file, verify_macro_consistency


def test_paper_tex_integrity_and_banned_claims():
    paper_dir = Path("docs/paper")
    if not paper_dir.exists():
        paper_dir = Path("ARN/docs/paper")
    assert paper_dir.exists(), f"Paper dir not found at {paper_dir}"

    tex_files = list(paper_dir.glob("*.tex"))
    assert len(tex_files) >= 4, f"Expected at least 4 .tex files, found {len(tex_files)}"

    all_errors = []
    for tf in tex_files:
        passed, errs = audit_tex_file(tf)
        if not passed:
            all_errors.extend(errs)

    assert len(all_errors) == 0, f"Paper audit found violations:\n" + "\n".join(all_errors)


def test_paper_macro_consistency():
    paper_dir = Path("docs/paper")
    results_dir = Path("results")
    if not paper_dir.exists():
        paper_dir = Path("ARN/docs/paper")
        results_dir = Path("ARN/results")

    numbers_tex = paper_dir / "numbers.tex"
    master_json = results_dir / "master_protocol_summary.json"

    assert numbers_tex.exists()
    assert master_json.exists()

    passed, errs = verify_macro_consistency(numbers_tex, master_json)
    assert passed, f"Macro consistency check failed:\n" + "\n".join(errs)
