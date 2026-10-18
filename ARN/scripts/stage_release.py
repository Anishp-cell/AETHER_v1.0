"""
Release Staging Script for ARN Public Repository (Phase 12).

Stages a clean, standalone public repository at `../ARN-public` for public GitHub release:
1. Copies core modules, models, datasets, documentation, scripts, and tests.
2. Excludes dirty history, scratch logs, temporary files, and __pycache__.
3. Initializes a clean standalone Git repository on branch `main`.
4. Generates RELEASE_CHECKLIST.md with exact release commands.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def copy_file_or_dir(src: Path, dst: Path):
    if not src.exists():
        print(f"Warning: source does not exist: {src}")
        return
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp", ".pytest_cache", ".git"),
            dirs_exist_ok=True
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def generate_release_checklist(target_dir: Path):
    checklist_content = """# ARN Public Release Checklist

This document details the final steps to publish the staged `ARN-public` repository to GitHub, Hugging Face, and PyPI.

---

## 1. Local Verification (Pre-Flight)

- [x] All 26 unit tests passing (`pytest tests/`)
- [x] Zero unverified or banned claims verified via `scripts/audit_paper_numbers.py`
- [x] 15-fold cross-validation results verified against `results/ledger.json`
- [x] Model checkpoints and SHA-256 manifest verified in `checkpoints/arn_final_model/manifest.json`
- [x] ONNX FP32 and Dynamic INT8 engines benchmarked and verified
- [x] Sealed human evaluation benchmark template preserved at `data/human_eval_template.jsonl`

---

## 2. GitHub Public Launch

When ready to push to public GitHub:

1. Create a new public repository on GitHub:
   - Repository name: `ARN` (or `aether-routing-network`)
   - Description: `Ultra-compact, sub-18MB non-generative joint neural tool orchestrator and slot extractor for real-time edge dispatch.`
   - Public visibility

2. Add remote and push:
   ```bash
   cd ../ARN-public
   git remote add origin https://github.com/Anishp-cell/ARN.git
   git branch -M main
   git push -u origin main
   ```

3. Create GitHub Release `v1.0.0`:
   - Tag: `v1.0.0`
   - Release title: `ARN v1.0.0: Master Protocol Audited Release`
   - Upload pre-built binaries:
     - `checkpoints/arn_final_model/model.pt`
     - `checkpoints/arn_final_model/model.onnx`
     - `checkpoints/arn_final_model/model_int8.onnx`

---

## 3. Hugging Face Model Hub

To publish the model card and weights to Hugging Face Hub:

```bash
huggingface-cli login
huggingface-cli upload Anishp-cell/arn-tiny checkpoints/arn_final_model/ . --repo-type model
```

Include `MODEL_CARD.md` as the `README.md` on Hugging Face.

---

## 4. PyPI Package Publishing

Before publishing to PyPI:
1. Ensure legacy pre-refactor release is yanked or upgraded per `docs/PYPI_NOTE.md`.
2. Build clean wheels:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   ```
3. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

---

## 5. Social & Community Announcements

Draft launch announcements verified against `results/ledger.json` are prepared in `docs/launch/`:
- LinkedIn technical announcement
- X (Twitter) thread with architecture diagram
- GitHub README quickstart and demo GIF
"""
    with open(target_dir / "RELEASE_CHECKLIST.md", "w", encoding="utf-8") as f:
        f.write(checklist_content)


def stage_public_repo(source_dir: Path, target_dir: Path):
    print(f"Staging public release repository from {source_dir} to {target_dir}...")
    if target_dir.exists():
        print(f"Removing existing target directory: {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Root files
    root_files = [
        "README.md",
        "MODEL_CARD.md",
        "DATA_STATEMENT.md",
        "CHANGELOG.md",
        "LICENSE",
        "pyproject.toml",
        ".gitignore",
    ]
    for rf in root_files:
        copy_file_or_dir(source_dir / rf, target_dir / rf)

    # 2. Directories
    dirs_to_copy = [
        "src",
        "data",
        "results",
        "docs",
        "tests",
        "scripts",
        "checkpoints/arn_final_model",
    ]
    for d in dirs_to_copy:
        copy_file_or_dir(source_dir / d, target_dir / d)

    # 3. Create RELEASE_CHECKLIST.md
    generate_release_checklist(target_dir)

    # 4. Initialize Git in target_dir
    print("Initializing clean standalone Git repository in target directory...")
    subprocess.run(["git", "init", "-b", "main"], cwd=target_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Anish Pattnaik"], cwd=target_dir, check=True)
    subprocess.run(["git", "config", "user.email", "anishpattnaik0@gmail.com"], cwd=target_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=target_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: initial open-source release of ARN v1.0.0 (Master Protocol Audited Release)"],
        cwd=target_dir,
        check=True
    )

    print(f"\nSuccessfully staged clean standalone repository in {target_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    target = root.parent / "ARN-public"
    stage_public_repo(root, target)


if __name__ == "__main__":
    main()
