#!/bin/bash
# v2.4 oracle fake-quant screen on the bat diagnosis windows (data/bats/batdev.f32): embedding SQNR vs the exact
# oracle (BN24_Q=fe=fft) for the BN24_Q config given.  usage: scripts/bat_qsweep.sh <label> '<BN24_Q>'
set -e
cd "$(dirname "$0")/.."
. scripts/lib_chunks.sh
PY=${PY:-$(cd .. && pwd)/v24-runtime/venv/bin/python}
S=${SCRATCH:?set SCRATCH to a work dir (not /tmp: it is cleaned)}/bq
mkdir -p $S
split8 data/bats/batdev.f32 $S/c $((32*144000*4))
for i in 0 1 2 3 4 5 6 7; do
  BN24_Q="$2" $PY scripts/pipe_run.py build-mac/cli weights/v24.bn24 1 $S/c$i.f32 144000 $S/$1.$i.f32 --emb >/dev/null &
done
wait
cat8 $S/$1 $S/$1.f32
$PY - "$S/$1.f32" "$1" "$2" <<'PYEOF'
import sys, numpy as np
r = np.fromfile("data/bats/batdev.embref.f32", np.float32).reshape(-1, 1024).astype(np.float64)
c = np.fromfile(sys.argv[1], np.float32).reshape(-1, 1024)
d = c - r
per = 10 * np.log10((r ** 2).sum(1) / np.maximum((d ** 2).sum(1), 1e-30))
print(f"{sys.argv[2]:16s} [{sys.argv[3]}] emb SQNR {10*np.log10((r**2).sum()/(d**2).sum()):.1f} dB p5 {np.percentile(per,5):.1f} min {per.min():.1f}")
PYEOF
