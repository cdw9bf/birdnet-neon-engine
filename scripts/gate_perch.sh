#!/bin/bash
# Score Perch v2 engine packages through the shared harness, one scoring process at a time (container CLI fed over
# stdin), then replay each against its fp32 reference cache. Two sets:
#   all: the global packages against the fp32 ORT reference (perch-runtime/data/ref_fp32)
#        N012 perch_v2-neon-fp16-s8   N013 perch_v2-neon-fp16   N014 perch_v2-neon-fp32   (mode 0)
#   nae: the north-america-east slice packages (criterion N016 in docs/GATES.md) against the sliced fp32 reference
#        (perch-runtime/data/ref_fp32_na_east)
#        N016 perch_v2_nae-neon-fp32 (+ per-window LSE for the calibration report)   N017 -neon-fp16-s8   N018 -neon-fp16
# usage: scripts/gate_perch.sh all|nae [ids...]   (container build /root/bG and the packages copied to /root first)
#   scripts/gate_perch_all.sh and scripts/gate_nae.sh are the two sets under their old names. The audio is decoded
#   once per run: modelgate extern keeps the decoded inputs for the next id (--keep-inputs) and they are deleted
#   after the last one (needs the external harness's modelgate with --keep-inputs; docs/ADDING_A_MODEL.md, appendix A).
set -e
E=$(cd "$(dirname "$0")/.." && pwd)
KL=$(cd "$E/.." && pwd)
PY=${PY:-$KL/v24-runtime/venv/bin/python}
GPY=${GPY:-$(ls ${BIRDLAB_DATA:-$KL/../lab-data}/venv/bin/python 2>/dev/null || echo $PY)}
set_=${1:-}
[ $# -gt 0 ] && shift
case $set_ in
  all) adapter=perch_v2; TMP=${TMP:-$E/data/perch/gtmp}; REF=${REF:-$KL/perch-runtime/data/ref_fp32}
       ids=${@:-N012 N013 N014} ;;
  nae) adapter=perch_v2_na_east; TMP=${TMP:-$E/data/perch/gtmp_nae}; REF=${REF:-$KL/perch-runtime/data/ref_fp32_na_east}
       ids=${@:-N016 N017 N018} ;;
  *) echo "usage: scripts/gate_perch.sh all|nae [ids...]" >&2; exit 2 ;;
esac
pkg() {
  case $1 in
    N012) echo perch_v2-neon-fp16-s8 ;; N013) echo perch_v2-neon-fp16 ;; N014) echo perch_v2-neon-fp32 ;;
    N016) echo perch_v2_nae-neon-fp32 ;; N017) echo perch_v2_nae-neon-fp16-s8 ;; N018) echo perch_v2_nae-neon-fp16 ;;
    *) echo "gate_perch.sh: unknown id $1 (N012-N014, N016-N018)" >&2; exit 2 ;;
  esac
}
for id in $ids; do pkg $id > /dev/null; done  # every id known before any scoring (pkg exits 2)
mkdir -p "$TMP"
for id in $ids; do
  out=$E/data/perch/cand_$id
  extra=""
  [ $id = N016 ] && extra="--lse $E/data/perch/lse_nae"
  echo "== $id $(pkg $id) $(date)"
  (cd $KL/tools && $GPY -I modelgate.py extern $adapter $out 1 --tmp $TMP --keep-inputs -- $PY -I $E/scripts/pipe_score.py \
     bnneon-eval /root/bG/cli /root/$(pkg $id).bnpk 0 {in} {off} {out} $extra) > $E/results/gate_${id}_infer.log 2>&1
  (cd $KL/tools && $GPY -I modelgate.py sim $adapter $REF $out $id) > $E/results/gate_${id}_result.txt 2>&1
  tail -30 $E/results/gate_${id}_result.txt
done
# the decoded inputs extern kept for the next candidate (--keep-inputs)
rm -f "$TMP"/*.f32 "$TMP"/*.off "$TMP"/*.key
[ $set_ = nae ] && rmdir "$TMP" 2>/dev/null || true
