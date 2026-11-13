#!/bin/bash
# Bit-identity suite in the arm64 eval container (gcc 14 = the Pis'): build the current tree, then cmp
#   v3   cli v3nae-fp16.bnpk 3 v3dev32.f32: md5 0e0271c5... (the K2B-G1 gated BirdNET v3.0 NA-east neon-fp16 package)
#   v2.4 fp16 (mode 0) on /eng/data/devA64.f32 == data/bitref/a17.f32 (gated A017 build)
#   v2.4 fp32 (mode 2)                          == data/bitref/c2.f32  (gated A019 build)
# (references kept in the repo's data/bitref/ with SHA256SUMS, not in container /tmp)
# plus any extra "label weights mode windows reference" lines given as arguments (quoted).
set -e
OUT=/root/bS
C=${CONTAINER:-bnneon-eval-local}
cd "$(dirname "$0")/.."
for f in data/bitref/a17.f32 data/bitref/c2.f32 data/bitref/v3dev32.f32 data/devA64.f32; do
  [ -f "$f" ] || { echo "cmp_suite: $f is missing. This suite compares the build with the gated reference outputs (data/bitref/*, data/devA64.f32)"
    echo "and the gated weights in the container (/root/v24.bn24, /root/v3nae-fp16.bnpk); those files are not in the repository (guide, appendix A)."
    echo "Without them, check a model of your own: python -I compiler/bnc.py check <model.onnx> ... --docker $C"; exit 2; }
done
docker inspect "$C" >/dev/null 2>&1 || { echo "cmp_suite: no container named $C (start one with scripts/eval_container.sh start, then CONTAINER=<name>)"; exit 2; }
docker exec "$C" sh -c 'test -f /root/v24.bn24 && test -f /root/v3nae-fp16.bnpk' || { echo "cmp_suite: $C has no /root/v24.bn24 and /root/v3nae-fp16.bnpk (the gated weights; guide appendix A)"; exit 2; }
mkdir -p eval_tmp; tok=cmp-mount-$$; echo "$PWD" > "eval_tmp/$tok"
docker exec "$C" test -f "/eng/eval_tmp/$tok" || { rm -f "eval_tmp/$tok"; echo "cmp_suite: $C does not mount this checkout ($PWD) at /eng (scripts/eval_container.sh start <name>, then CONTAINER=<name>)"; exit 2; }
rm -f "eval_tmp/$tok"
[ -n "$NOBUILD" ] || docker exec "$C" sh -c "cd /eng && rm -rf $OUT && make -s OUT=$OUT -j4 2>&1 | grep -E 'error|warning: (implicit|unused var)' ; test -x $OUT/cli" || { echo BUILD FAILED; exit 1; }
docker exec "$C" sh -c "cd /root && $OUT/cli v3nae-fp16.bnpk 3 /eng/data/bitref/v3dev32.f32 /root/s3.f32 1 2>/dev/null && [ \"\$(md5sum < /root/s3.f32 | cut -c1-32)\" = 0e0271c5b636a912028bc33f465b9caa ] && echo 'v3 neon-fp16: bit-identical' || echo 'v3 neon-fp16: DIFFERS'
$OUT/cli v24.bn24 0 /eng/data/devA64.f32 /root/s0.f32 1 2>/dev/null && cmp /root/s0.f32 /eng/data/bitref/a17.f32 && echo 'v2.4 fp16: bit-identical' || echo 'v2.4 fp16: DIFFERS'
$OUT/cli v24.bn24 2 /eng/data/devA64.f32 /root/s2.f32 1 2>/dev/null && cmp /root/s2.f32 /eng/data/bitref/c2.f32 && echo 'v2.4 fp32: bit-identical' || echo 'v2.4 fp32: DIFFERS'"
for x in "$@"; do
  set -- $x
  docker exec "$C" sh -c "cd /root && $OUT/cli $2 $3 $4 /root/sx.f32 1 2>/root/sx.log; head -1 /root/sx.log >&2; cmp /root/sx.f32 $5 && echo '$1: bit-identical' || echo '$1: DIFFERS'"
done
