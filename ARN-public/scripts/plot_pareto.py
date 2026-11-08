"""Figure 1 of docs/paper_v2: strict EM vs parameters on Mobile Actions (all points measured in our harness).

Values are copied from docs/RESULTS_v2.md / results/key_results_v2.json. Writes docs/paper_v2/fig_pareto.pdf.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# (label, params in millions, strict EM, sd or None, fine-tuned on train split?, is ours)
PTS = [
    ("ARN-v2 (BERT-Tiny)", 4.44, 89.32, 0.47, True, True),
    ("ARN-v2 (BERT-Mini)", 11.38, 89.91, 0.46, True, True),
    ("LFM2.5-230M FT + sweep", 230, 92.51, None, True, False),
    ("FunctionGemma-270M FT", 268, 85.12, None, True, False),
    ("LFM2.5-230M base", 230, 68.05, None, False, False),
    ("Qwen3-0.6B base", 596, 65.04, None, False, False),
    ("FunctionGemma-270M base", 268, 60.15, None, False, False),
]
OFF = {"ARN-v2 (BERT-Tiny)": (6, -12), "ARN-v2 (BERT-Mini)": (6, 4), "LFM2.5-230M FT + sweep": (-8, 6),
       "FunctionGemma-270M FT": (6, -10), "LFM2.5-230M base": (-8, 6), "Qwen3-0.6B base": (-8, -12),
       "FunctionGemma-270M base": (8, -3)}

fig, ax = plt.subplots(figsize=(5.2, 3.3))
for lab, n, em, sd, ft, ours in PTS:
    color = "#1f5fbf" if ours else ("#555555" if ft else "#aaaaaa")
    ax.errorbar(n, em, yerr=sd, fmt="o" if ours else ("s" if ft else "^"), ms=6, color=color,
                ecolor=color, elinewidth=1, capsize=2, mec="white", mew=0.8, zorder=3)
    dx, dy = OFF[lab]
    ax.annotate(lab, (n, em), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                ha="left" if dx > 0 else "right", color="#222222")
ax.set_xscale("log")
ax.set_xlim(2, 2000)
ax.set_ylim(55, 96)
ax.set_xlabel("Parameters (millions, log scale)", fontsize=8)
ax.set_ylabel("Strict exact match (%)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, which="major", lw=0.4, color="#dddddd")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.plot([], [], "o", color="#1f5fbf", label="ours (extractive)")
ax.plot([], [], "s", color="#555555", label="generative, fine-tuned on same split")
ax.plot([], [], "^", color="#aaaaaa", label="generative, base")
ax.legend(fontsize=6.5, loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "docs/paper_v2/fig_pareto.pdf")
print("wrote docs/paper_v2/fig_pareto.pdf")
