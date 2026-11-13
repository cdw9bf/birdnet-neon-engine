#!/bin/bash
# Perch precision screen: the oracle (build-mac/perch_oracle) with PERCH_Q=<config> on devP (256 windows) in 8 parallel processes,
# compared against the fp32 oracle and ORT fp32.  usage: scripts/perch_qsweep.sh <label> '<PERCH_Q>' [flush]
set -e
cd "$(dirname "$0")/.."
. scripts/lib_chunks.sh
S=${SCRATCH:?set SCRATCH to a work dir (not /tmp: it is cleaned)}
PY=${PY:-$(cd .. && pwd)/v24-runtime/venv/bin/python}
mkdir -p $S/q
split8 data/perch/devP.f32 $S/q/chunk $((32*160000*4))
for i in 0 1 2 3 4 5 6 7; do
  PERCH_Q="$2" PERCH_FLUSH=${3:-128} ./build-mac/perch_oracle weights/perch_v2.bnp2 $S/q/chunk$i.f32 $S/q/$1.$i.f32 2>/dev/null &
done
wait
cat8 $S/q/$1 $S/q/$1.f32 && rm $S/q/$1.?.f32
echo "== $1  [$2] flush ${3:-128}"
echo -n "vs oracle: "; $PY scripts/perch_cmp.py data/perch/devP.ref.f32 $S/q/$1.f32
echo -n "vs ORT:    "; $PY scripts/perch_cmp.py data/perch/devP.ort.logits.f32 $S/q/$1.f32
