# Use of AI assistance

Following arXiv's policy that authors are fully responsible for all content and must report significant use of
generative AI tools (AI systems cannot be authors):

- **Code and experiments.** Claude (Anthropic, Claude Code) wrote most of the Aether Routing Network v2 code in
  `src/arn/v2/`, `src/arn/canon/`, `src/arn/eval/` and `scripts/` (v2 training, decoding, scorer, cross-model
  harness, latency benchmark), ran the experiments on the author's hardware, and drafted documentation.
- **Independent review.** A different model family, Gemini 3.8 Flash via the Antigravity CLI, reviewed the
  evaluation code for leakage and scoring bugs; its valid findings were fixed (see `docs/RESULTS_v2.md` §4).
- **Literature and fact checks.** Claude-driven research agents checked published numbers and citations against
  primary sources.
- **Human responsibility.** The author designed ARN v1, chose the research direction, approved every experiment
  and is responsible for verifying all claims before submission.
