# Sourced by the precision screens (perch_qsweep.sh, bat_qsweep.sh): a windows file in 8 chunks, 8 processes, one
# output again.
#   split8 SRC PREFIX BYTES    writes PREFIX0.f32 .. PREFIX7.f32, BYTES each (skipped when PREFIX7.f32 exists)
#   cat8 PREFIX OUT            concatenates PREFIX.0.f32 .. PREFIX.7.f32 into OUT
split8() {
  [ -f "${2}7.f32" ] && return 0
  for i in 0 1 2 3 4 5 6 7; do dd if="$1" of="$2$i.f32" bs="$3" skip=$i count=1 2>/dev/null; done
}
cat8() {
  cat "$1".{0,1,2,3,4,5,6,7}.f32 > "$2"
}
