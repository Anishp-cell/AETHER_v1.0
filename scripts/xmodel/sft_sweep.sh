#!/usr/bin/env bash
# Equal-budget tuning for the fine-tuned LFM2.5-230M baseline (fairness control).
# Each config: fine-tune on the fit split -> score on the seeded VAL slice of train.
# Only the best-val config is then decoded on the eval split (once).
set -u
cd "$(dirname "$0")/../.."
PY=../.venv/Scripts/python.exe
D='C:\Users\Lenovo\.cache\huggingface\hub\datasets--google--mobile-actions\snapshots\e920309bc2acbc2e99a5e3201cf37df2b9fd9151\dataset.jsonl'
DL=/mnt/c/Users/Lenovo/.cache/huggingface/hub/datasets--google--mobile-actions/snapshots/e920309bc2acbc2e99a5e3201cf37df2b9fd9151/dataset.jsonl
: > results/sft_sweep_val.tsv
for cfg in "5e-5 2" "2e-5 2" "1e-4 2" "5e-5 3" "1e-4 3"; do
  set -- $cfg; lr=$1; ep=$2; name="lfm_sweep_lr${lr}_ep${ep}"
  $PY -u scripts/xmodel/sft_baseline.py --data "$D" --model LiquidAI/LFM2.5-230M --out checkpoints/sft/$name \
      --lr "$lr" --epochs "$ep" --bs 2 --accum 8 > results/$name.train.log 2>&1 || { echo "FAIL $name"; continue; }
  $PY -u scripts/xmodel/run_hf_fc.py --data "$D" --model checkpoints/sft/$name --split val --batch 16 \
      --name ${name}_val > results/$name.val.log 2>&1 || { echo "FAIL val $name"; continue; }
  uv run --no-project python -I - "$DL" "$name" <<'EOF' >> results/sft_sweep_val.tsv
import json, random, sys, importlib.util
spec = importlib.util.spec_from_file_location("S", "src/arn/eval/mobile_actions_scorer.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
tr = [r for r in rows if r["metadata"] == "train"]; order = list(range(len(tr))); random.Random(42).shuffle(order)
val = [tr[i] for i in order[: len(order) // 10]]
P = json.load(open(f"results/xmodel/{sys.argv[2]}_val.json"))["predictions"]
print(f"{sys.argv[2]}\t{S.score(val, P)['strict_em']}")
EOF
done
best=$(sort -t$'\t' -k2 -nr results/sft_sweep_val.tsv | head -1 | cut -f1)
echo "BEST_ON_VAL $best"; cat results/sft_sweep_val.tsv
$PY -u scripts/xmodel/run_hf_fc.py --data "$D" --model checkpoints/sft/$best --batch 16 --name lfm2.5_230m_sft_tuned > results/sft_sweep_eval.log 2>&1
uv run --no-project python -I src/arn/eval/mobile_actions_scorer.py --data "$DL" results/xmodel/lfm2.5_230m_sft_tuned.json \
    | grep -E '"(strict_em|multi_call_em|datetime_arg_acc)"'
