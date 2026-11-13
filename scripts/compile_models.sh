#!/bin/bash
# Compile the shipped model packages (one file per CPU path) into weights/:
#   birdnet_v2.4-neon-fp16.bnpk / -neon-fp32.bnpk, battybirdnet_<region>_256khz-neon-fp16|32.bnpk (11 regions),
#   perch_v2-neon-fp16|32.bnpk, perch_v2-neon-fp16-s8.bnpk, perch_v2-neon-int8.bnpk (the Pi 4's), perch_v2-neon-fp32-h.bnpk,
#   perch_v2_nae-neon-fp16-s8|fp16|fp32.bnpk (the north-america-east slice),
#   birdnet_v3.0[_north-america-east]-neon-fp16|int8|fp32.bnpk.
#   Prints name, size, sha256.
# usage: scripts/compile_models.sh [v24|bats|perch|v3|all]
#   e.g. PY=cenv/bin/python V24=model.onnx V24L=labels.txt scripts/compile_models.sh v24   (defaults: files under data/)
set -e
cd "$(dirname "$0")/.."
# inputs (override by environment): PY = a Python with onnx + onnxruntime + numpy; V24 = BirdNET v2.4 fp32 ONNX
# (the BattyBirdNET catalog's 2-output copy of it by default: same graph and constants, logits + embedding outputs);
# V24L = the BirdNET v2.4 label file (its sha256 goes into the package); B = BattyBirdNET catalog; P = Perch catalog;
# PNAE = the directory with the Perch v2 catalog's north-america-east slice files (perch_v2_north-america-east_indices.txt,
# _labels.txt; default data/perch/regional_nae);
# V3 = the directory with the BirdNET v3.0 preview3.1 fp32 ONNX files and label files (global and north-america-east);
# V3INT8_NAE / V3INT8_GLOB = the int8 layer classes of each v3.0 int8 package (the gated choice, results/MENU.txt)
# The defaults are paths under data/ (git-ignored): put or link the files there, or set the variables. Layout:
#   data/battybirdnet/{birdnet-v2.4-embeddings-fp32.onnx, birdnet-v2.4-embeddings-fp32-dfttrunc.onnx, fp32/, labels/}
#   data/birdnet/birdnet_labels_v2.4.txt   data/perch/{perch_v2_fp32.onnx, perch_v2_labels.txt, regional_nae/}
#   data/birdnet_v3/ (the four v3.0 files below). Where to download each file: README.md, "Getting the models".
PY=${PY:-python3}
B=${B:-data/battybirdnet}
V24=${V24:-$B/birdnet-v2.4-embeddings-fp32.onnx}
V24L=${V24L:-data/birdnet/birdnet_labels_v2.4.txt}
P=${P:-data/perch}
PNAE=${PNAE:-data/perch/regional_nae}
V3=${V3:-data/birdnet_v3}
V3INT8_NAE=${V3INT8_NAE:-pw,pwl,head,sed,lin}
V3INT8_GLOB=${V3INT8_GLOB:-pw,pwl,head,sed,lin}
what=${1:-all}
# The compiles run JOBS at a time (default 3; JOBS=1 is the old sequential run), the BirdNET v3.0 global ones after
# the others and one at a time (each needs ~5 GB). c / c1 queue one compile (c1: on its own); run_queued runs them,
# then prints each compile's summary lines in queue order (the order of the old sequential run). A failed compile
# fails the script with its exit code, after the others have finished.
JOBS=${JOBS:-3}
Q=$(mktemp -d "${TMPDIR:-/tmp}/compile_models.XXXXXX")
trap 'rm -rf "$Q"' EXIT
nq=0
queue() {  # queue LIST ARGS...: one compile of this checkout's compiler (python/bnneon), installed or not
  local list=$1; shift
  nq=$((nq + 1))
  local job=$Q/$(printf '%03d' $nq)
  # -P -s keep the current directory and user site off sys.path
  { printf 'cd %q && PYTHONPATH=%q %q -P -s compiler/bnc.py compile' "$PWD" "$PWD/python" "$PY"; printf ' %q' "$@"
    echo; } > $job.sh
  printf '%s\0' $job >> $Q/$list
}
c() { queue parallel "$@"; }
c1() { queue serial "$@"; }
run_one() { "$BASH" "$1.sh" > "$1.log" 2>&1; echo $? > "$1.rc"; }
export -f run_one
run_queued() {
  [ -f $Q/parallel ] && xargs -0 -n1 -P "$JOBS" "$BASH" -c 'run_one "$0"' < $Q/parallel
  [ -f $Q/serial ] && xargs -0 -n1 -P 1 "$BASH" -c 'run_one "$0"' < $Q/serial
  local rc=0 r
  for job in $(ls $Q/*.sh 2>/dev/null | sed 's/\.sh$//'); do
    grep -v VerifyOutputSizes $job.log | grep -E 'wrote|path|unsupported|Error|error|Traceback' || true
    r=$(cat $job.rc 2>/dev/null || echo 1)  # no .rc: the job was killed
    [ $rc = 0 ] && [ "$r" != 0 ] && rc=$r
  done
  return $rc
}
for path in neon-fp16 neon-fp32; do
  if [ $what = v24 -o $what = all ]; then
    c $V24 --logits output --labels $V24L --model-id birdnet_v2.4 --family 24 --post sigmoid --sample-rate 48000 --paths $path \
      --license-file weights/LICENSE-birdnet-v24.txt \
      --attribution "BirdNET v2.4, K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology" \
      -o weights/birdnet_v2.4-$path.bnpk
  fi
  if [ $what = bats -o $what = all ]; then
    for h in $B/fp32/BattyBirdNET-*_fp32.onnx; do
      r=$(basename $h _fp32.onnx); id=$(echo ${r#BattyBirdNET-} | tr 'A-Z-' 'a-z_')
      c $B/birdnet-v2.4-embeddings-fp32-dfttrunc.onnx --append $h --at model/GLOBAL_AVG_POOL/Mean_reduced_0 \
        --labels $B/labels/${r}_Labels.txt --model-id battybirdnet_$id --family 224 --post sigmoid --sample-rate 256000 \
        --paths $path --license-file weights/LICENSE-battybirdnet.txt \
        --attribution "BattyBirdNET (R. D. Zinck) on BirdNET v2.4 (K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology); powered by BirdNET" \
        -o weights/battybirdnet_$id-$path.bnpk
    done
  fi
  if [ $what = perch -o $what = all ]; then
    c $P/perch_v2_fp32.onnx --labels $P/perch_v2_labels.txt --model-id perch_v2 --family 120 --post softmax \
      --sample-rate 32000 --paths $path --license-file weights/LICENSE-perch-v2.txt \
      --attribution "Perch v2, Google Research (bird-vocalization-classifier)" -o weights/perch_v2-$path.bnpk
    if [ $path = neon-fp16 ]; then  # candidate: int8 SDOT prototype head (needs dotprod)
      c $P/perch_v2_fp32.onnx --labels $P/perch_v2_labels.txt --model-id perch_v2 --family 120 --post softmax \
        --sample-rate 32000 --paths $path --proto-s8 --license-file weights/LICENSE-perch-v2.txt \
        --attribution "Perch v2, Google Research (bird-vocalization-classifier)" -o weights/perch_v2-$path-s8.bnpk
    fi
  fi
done
if [ $what = perch -o $what = all ]; then  # Perch v2 for CPUs without dotprod (the Pi 4): fp32 body, int8 prototype head
  c $P/perch_v2_fp32.onnx --labels $P/perch_v2_labels.txt --model-id perch_v2 --family 120 --post softmax \
    --sample-rate 32000 --paths neon-int8 --int8 proto --license-file weights/LICENSE-perch-v2.txt \
    --attribution "Perch v2, Google Research (bird-vocalization-classifier)" -o weights/perch_v2-neon-int8.bnpk
  # option: fp32 math with fp16-stored prototypes (half the head's memory)
  c $P/perch_v2_fp32.onnx --labels $P/perch_v2_labels.txt --model-id perch_v2 --family 120 --post softmax \
    --sample-rate 32000 --paths neon-fp32 --proto32-h16 --license-file weights/LICENSE-perch-v2.txt \
    --attribution "Perch v2, Google Research (bird-vocalization-classifier)" -o weights/perch_v2-neon-fp32-h.bnpk
  # the north-america-east slice (999 classes of the same head; the catalog's index and label files)
  n=perch_v2_north-america-east
  if [ -f $PNAE/${n}_indices.txt -a -f $PNAE/${n}_labels.txt ]; then
    for v in neon-fp16:--proto-s8:-s8 neon-fp16:: neon-fp32::; do
      path=${v%%:*}; x=${v#*:}; x=${x%:*}; sfx=${v##*:}
      c $P/perch_v2_fp32.onnx --labels $PNAE/${n}_labels.txt --classes $PNAE/${n}_indices.txt --model-id $n --family 120 \
        --post softmax --sample-rate 32000 --paths $path $x --license-file weights/LICENSE-perch-v2.txt \
        --attribution "Perch v2, Google Research (bird-vocalization-classifier)" -o weights/perch_v2_nae-$path$sfx.bnpk
    done
  else
    echo "no ${n}_indices.txt / _labels.txt in PNAE=$PNAE: skipping the Perch v2 north-america-east slice"
  fi
fi
if [ $what = v3 -o $what = all ]; then  # BirdNET v3.0: neon-fp16 (fp16 + dotprod: the Pi 5's path), neon-int8 (the Pi 4's) and neon-fp32
  v3ok=1
  for f in birdnet-v3.0-preview3.1-fp32-b1.onnx birdnet-v3.0-preview3.1-labels-b1.txt \
           birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx birdnet-v3.0-preview3.1-north-america-east-labels-b1.txt; do
    [ -f $V3/$f ] || { echo "no $f in V3=$V3: set V3 to the directory with the BirdNET v3.0 ONNX and label files"; v3ok=0; }
  done
  [ $v3ok = 1 ] || { [ $what = v3 ] && exit 3; echo "skipping BirdNET v3.0"; }
fi
if [ \( $what = v3 -o $what = all \) -a "${v3ok:-0}" = 1 ]; then
  for m in north-america-east global; do
    if [ $m = global ]; then s=""; id=birdnet_v3.0; i8=$V3INT8_GLOB; else s="-$m"; id=birdnet_v3.0_$m; i8=$V3INT8_NAE; fi
    for path in neon-fp16 neon-int8 neon-fp32; do
      x="--wino 24"; [ $path = neon-int8 ] && x="--wino 24 --int8 $i8"; [ $path = neon-fp16 ] && x=""  # --wino: fp32 lowerings only
      lic=weights/LICENSE-birdnet-v3.txt; [ $path = neon-fp16 ] && lic=weights/LICENSE-birdnet-v3-fp16.txt  # each states its path's changes
      cq=c; [ $m = global ] && cq=c1  # the global compiles one at a time (memory)
      $cq $V3/birdnet-v3.0-preview3.1$s-fp32-b1.onnx --labels $V3/birdnet-v3.0-preview3.1$s-labels-b1.txt --model-id $id \
        --family 30 --post sigmoid --sample-rate 32000 --emb embeddings --paths $path $x \
        --license-file $lic \
        --attribution "BirdNET v3.0, K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology" \
        -o weights/$id-$path.bnpk
    done
  done
fi
run_queued
ls weights/*.bnpk | while read f; do echo "$(basename "$f") $(wc -c < "$f" | tr -d ' ') $(shasum -a 256 "$f" | cut -c1-64)"; done
