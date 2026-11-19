# Benchmark: paper methods vs. + TPR without class balance (ablation) vs. + full TPR (--refine).
# Usage (from this folder):  bash run_benchmark.sh  [python executable]   -> results/benchmark.csv
PY=${1:-python}
B=ViT-B/32

# dataset : high-level concept prompt used by the paper ("" = paper has none)
for entry in "dtd:" "eurosat:A photo of a land use: a " "pets:A photo of a breed: a " "cub:A photo of a bird: a "; do
  d=${entry%%:*}; concept=${entry#*:}
  for R in "" "--refine --refine_balance=0" "--refine"; do
    $PY base_main.py --savename=benchmark --dataset=$d --model_size=$B --mode=clip $R
    $PY base_main.py --savename=benchmark --dataset=$d --model_size=$B --mode=gpt_descriptions $R
    $PY base_main.py --savename=benchmark --dataset=$d --model_size=$B --mode=waffle --waffle_count=15 --reps=7 $R
    if [ -n "$concept" ]; then
      $PY base_main.py --savename=benchmark --dataset=$d --model_size=$B --mode=waffle --waffle_count=15 --reps=7 --label_before_text="$concept" $R
    fi
  done
done
