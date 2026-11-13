#!/bin/bash
# fake-quant sweep with the oracle: scripts/qsweep.sh <windows.f32> <ref logits.f32> <label=BN24_Q> ...
W=$1; R=$2; shift 2
PY=${BN24_PY:-python3}  # needs numpy
cd "$(dirname "$0")/.."
for spec in "$@"; do
  lab=${spec%%=*}; q=${spec#*=}
  ( BN24_Q="$q" ./build-mac/cli weights/v24.bn24 1 $W /tmp/qs_$lab.f32 2>/dev/null && $PY -I scripts/cmp.py $R /tmp/qs_$lab.f32 "$lab [$q]"; rm -f /tmp/qs_$lab.f32 ) &
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 1; done
done
wait
