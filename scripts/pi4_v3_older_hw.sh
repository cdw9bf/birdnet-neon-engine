#!/bin/bash
# Pi 4 (Cortex-A72): BirdNET v3.0 on the stock runtimes at 1 / 2 / 4 threads (H001), and the A72 conv micro-benchmark
# (H002). Run ON the Pi 4 from ~/v3-older, under the shared bench lock:
#   flock -w 3600 ~/pi-bench.lock bash scripts/pi4_v3_older_hw.sh PHASE
# PHASE: stock-na | na-t4 | stock-global | contended (stock-na's 2/4-thread rows with a 25 %-duty single-thread background
# load, standing in for BirdNET-Go's capture / resampling / UI work) | cadence (sustained runs at the overlap-2.2
# cadence, one window every 4/3 s) | micro (bench/bench_a72_conv.c)
# Threads are pinned: 1 -> core 3, 2 -> cores 2-3, 4 -> cores 0-3. One fresh process per (model, threads).
# Models: ~/v3-older/models (NA-east files symlinked from an earlier bench directory; the global files copied in for
# the run and removed afterwards by the caller). Results: ~/v3-older/results.
set -u
cd ~/v3-older
PY=~/bird-bench/v24-runtime/venv/bin/python
R=results; M=models; W=win/win5s.npy
mkdir -p $R
log() { echo "== $(date '+%F %T') $* | load $(cut -d' ' -f1-3 /proc/loadavg) | temp $(($(cat /sys/class/thermal/thermal_zone0/temp)/1000)) C | $(vcgencmd get_throttled) | $(grep MemAvailable /proc/meminfo | tr -s ' ')"; }
cpus() { case $1 in 1) echo 3;; 2) echo 2,3;; 4) echo 0-3;; esac; }
bench() { # model threads tag
  local m=$1 t=$2 tag=$3
  taskset -c "$(cpus $t)" $PY -I scripts/stock_threads_bench.py $m --threads $t --npy $W --runs ${RUNS:-10} --warm 2 \
    --out $R/H001_${tag}_t${t}.json 2> >(grep -v -E "device_discovery|GetGpuDevices" >&2)
}
phase=${1:-stock-na}
exec > >(tee -a $R/H00x_log.txt) 2>&1
log "start $phase"
case $phase in
stock-na)
  sha256sum $M/*north-america-east* $M/v3na_*
  for round in 1 2; do
    for t in 1 2 4; do
      bench $M/birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx $t na_ortfp32_r$round
      bench $M/birdnet-v3.0-preview3.1-north-america-east-fp32-fft3.onnx $t na_fft3_r$round
      bench $M/v3na_fft_i8dyn_s0se.tflite $t na_tfli8_r$round
    done
    log "round $round done"
  done
  for t in 1 2 4; do bench $M/v3na_fft_fp32.tflite $t na_tflfp32_r1; done
  ;;
stock-global)
  sha256sum $M/birdnet-v3.0-preview3.1-fp32-*.onnx
  for t in 1 2 4; do
    RUNS=8 bench $M/birdnet-v3.0-preview3.1-fp32-b1.onnx $t gl_ortfp32_r1
    RUNS=8 bench $M/birdnet-v3.0-preview3.1-fp32-fft3.onnx $t gl_fft3_r1
  done
  ;;
contended)
  $PY -I -c "
import time
while True:
    t = time.perf_counter()
    while time.perf_counter() - t < 0.025: pass
    time.sleep(0.075)" &
  BG=$!
  sleep 2
  for t in 2 4; do
    bench $M/birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx $t na_ortfp32_bg25
    bench $M/birdnet-v3.0-preview3.1-north-america-east-fp32-fft3.onnx $t na_fft3_bg25
    bench $M/v3na_fft_i8dyn_s0se.tflite $t na_tfli8_bg25
  done
  kill $BG
  ;;
na-t4)  # a third round of stock-na's 4-thread rows (rounds 2 and 3 both overlapped another job's unlocked work)
  for t in 4; do
    bench $M/birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx $t na_ortfp32_r3
    bench $M/birdnet-v3.0-preview3.1-north-america-east-fp32-fft3.onnx $t na_fft3_r3
    bench $M/v3na_fft_i8dyn_s0se.tflite $t na_tfli8_r3
  done
  ;;
cadence)
  taskset -c 0-3 $PY -I scripts/stock_cadence_bench.py $M/birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx \
    --threads 4 --stride 1.33333 --duration 600 --npy $W --out $R/H003_cad_ortfp32_t4.json
  log "cooldown"; sleep 120; log "after cooldown"
  taskset -c 2,3 $PY -I scripts/stock_cadence_bench.py $M/v3na_fft_i8dyn_s0se.tflite \
    --threads 2 --stride 1.33333 --duration 600 --npy $W --out $R/H003_cad_tfli8_t2.json
  ;;
micro)
  gcc -O3 -mcpu=cortex-a72 -std=gnu11 -Wall -o bench_a72_conv bench/bench_a72_conv.c src/gemm_f32.c -lm -lpthread
  taskset -c 3 ./bench_a72_conv 1 | tee $R/H002_a72_conv_t1.txt
  ./bench_a72_conv 4 | tee $R/H002_a72_conv_t4.txt
  ;;
esac
log "end $phase"
