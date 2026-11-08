# arXiv submission package: Aether Routing Network v2

Checked against arXiv's own help pages on 2026-11-08: submission overview (info.arxiv.org/help/submit),
TeX rules (help/submit_tex), metadata rules (help/prep), moderation (help/moderation), endorsement
(help/endorsement).

## 1. Metadata to paste into the arXiv form (ASCII only, no TeX font commands)

**Title**
Copy, Don't Generate: Aether Routing Network v2, a 4.4M-Parameter Extractive Function-Calling Router

**Authors** (form "First Last", affiliation in parentheses, city and country only)
<YOUR FULL NAME> (<Institution>, <City>, <Country>)

**Abstract** (arXiv limit: 1,920 characters; this text: 1,419 characters)
On-device assistants must map a user request to function calls with exact arguments, usually with generative language models of hundreds of millions of parameters. We observe that on Google's Mobile Actions benchmark 94.8% of rows have every non-date argument appearing verbatim in the request, and ask how far a non-generative model that copies argument spans can go. We present the Aether Routing Network v2 (ARN-v2), a 4.4M-parameter encoder with a multi-label tool router, a linear-chain CRF over tool-qualified BIO tags, registry-constrained joint decoding, and a train-fitted rule-based datetime canonicalizer. On Mobile Actions (961 test requests, strict exact match) ARN-v2 reaches 89.32 +/- 0.47 (3 seeds), outperforming FunctionGemma-270M fine-tuned on the same split (85.12; paired difference +3.75 to +4.68, all 95% CIs above zero) and all evaluated base models, while trailing an LFM2.5-230M fine-tuned and tuned on the same split (92.51). End to end it runs at 11.4 ms median on one CPU thread versus 4.3 s for LFM2.5-230M. In a pre-registered evaluation on multi-intent SLU benchmarks without any rules, ARN-v2 obtains 40.94 (MixATIS) and 70.29 (MixSNIPS) overall accuracy, on par with or below small task-specific BiLSTM models. We document every development choice that touched the test split, the contribution of rule-based components (about 34 points), and the scope limits of a closed tool registry.

**Comments**
<N> pages (fill after compiling), 1 figure, 2 tables. Code, predictions and logs: https://github.com/Anishp-cell/AETHER_v1.0

**Primary category:** cs.CL (Computation and Language). **Cross-list:** cs.LG.
**License:** CC BY 4.0 (recommended; the dataset is CC-BY-4.0 and it allows reuse with attribution).
**Report-no / Journal-ref / DOI:** leave empty.

## 2. Files to upload (one .zip or .tar.gz, flat)

| File | Why |
|---|---|
| `main.tex` | top-level file (only file with `\documentclass`) |
| `main.bbl` | generated bibliography; name must match `main.tex`. arXiv can also run BibTeX from `refs.bib`; uploading both is safest |
| `refs.bib` | source bibliography |
| `fig_pareto.pdf` | figure; pdfLaTeX accepts .pdf/.png/.jpg only, do not mix with .eps |

Do NOT upload: the compiled `main.pdf`, `.aux`, `.log`, `.out`, hidden files, `ARXIV_SUBMISSION.md`.

## 3. Compliance checklist (status)

| Rule (source) | Status |
|---|---|
| Source, not a PDF made from TeX (submit) | OK: upload TeX source |
| Only standard TeX Live packages (submit_tex) | OK: geometry, fontenc, lmodern, microtype, booktabs, amsmath, amssymb, xcolor, hyperref, graphicx |
| Figures in pdfLaTeX formats, referenced by relative path, exact case (submit_tex) | OK: `fig_pareto.pdf` |
| No `\today` in `\date` (submit_tex) | OK: `\date{}` |
| No `\pdfoutput` override, no JavaScript/animation (submit_tex) | OK |
| `.bbl` matches main file name (submit_tex) | Generate `main.bbl` (step 4) |
| Abstract <= 1,920 chars, no word "Abstract", ASCII (prep) | OK (1,419 chars, ASCII) |
| Authors as "First Last", no honorifics (prep) | **TODO: fill your name** in `main.tex` line 10 and the form |
| Significant generative-AI use disclosed; AI not an author; authors responsible (moderation) | OK: section "Reproducibility and AI Assistance" + `docs/AI_ASSISTANCE.md` |
| Refereeable original research, topical for category (moderation) | OK: new method + experiments, cs.CL |
| No `[PENDING]` markers left | **TODO: only the author line remains** |

## 4. Your steps

1. Edit `main.tex` line 10: replace `\author{\PENDING{...}}` with
   `\author{Your Name\\Your Institution\\\texttt{you@institution.edu}}`.
2. Compile on Overleaf (free): New Project -> Upload Project -> zip of `main.tex`, `refs.bib`,
   `fig_pareto.pdf`. Set compiler to pdfLaTeX. Recompile, check the log has no errors and no "??" references.
3. Read the whole PDF yourself. You are responsible for every claim (arXiv policy).
4. Overleaf -> Submit -> "Submit to arXiv" (or Logs and output files -> download `main.bbl`), zip
   `main.tex main.bbl refs.bib fig_pareto.pdf`.
5. Register at arxiv.org with an institutional email if you have one (arXiv says it speeds endorsement).
6. Start a new submission, choose cs.CL. If arXiv asks for endorsement it shows a 6-character
   **endorsement code**. Send that code, the PDF, and the abstract to your endorser. The endorser must have
   enough cs papers submitted between 3 months and 5 years ago; they enter the code at arxiv.org/auth/endorse.
7. Upload the zip, check the generated PDF arXiv shows you, paste the metadata from section 1, submit.

**Note on the endorser:** an endorsement only says the paper is appropriate for the category, not that it
is correct. Ask them to read it; do not ask anyone to endorse a paper they have not looked at.
