#!/bin/bash
# x86 (Xeon E5-2620 v4, AVX2, idle): BirdNET v3.0 on ORT and OpenVINO CPU at 1/2/4 threads (H005)
cd ~/bird-model-eval/v3-older
PY=~/bird-model-eval/v24-runtime/venv/bin/python; W=~/bird-model-eval/v24-runtime/win/win5s.npy
M=~/bird-model-eval/models_dl
df -h ~ | tail -1; uptime
cpus() { case $1 in 1) echo 9;; 2) echo 9,10;; 4) echo 9-12;; esac; }
for t in 1 2 4; do
  for mm in north-america-east-fp32-b1 fp32-b1; do
    taskset -c $(cpus $t) $PY -I scripts/stock_threads_bench.py $M/birdnet-v3.0-preview3.1-$mm.onnx --threads $t --npy $W --runs 20 --out results/H005_ort_${mm}_t$t.json
    taskset -c $(cpus $t) $PY -I scripts/stock_threads_bench.py $M/birdnet-v3.0-preview3.1-$mm.onnx --ov --threads $t --npy $W --runs 20 --out results/H005_ov_${mm}_t$t.json
  done
done
uptime
