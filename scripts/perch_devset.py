"""Perch v2 dev set (precision screen / parity; NOT the gate set).

Random 5 s windows at 32 kHz from the 8 held-out SSW calibration hours
(upstream-quant/data/calibration_ssw_hours.txt; disjoint from the 34.4 h
gate set). Audio path = the Perch gate's (BirdNET-Go's): ffmpeg PCM16 at the
native rate (48 kHz for SSW), then the go-audio-resampler CLI (resample32,
QualityMedium, 1 s chunks) to 32 kHz, via perch-runtime/scripts/perchgate.py.

usage: python -I perch_devset.py <calib_audio_dir> <out.f32> <n_windows>
           [seed]
Writes n_windows x 160000 float32 (row-major) and <out>.txt with
"file start_sample" per window.
"""
import os
import sys

import numpy as np

from bnneon import extern

HERE = os.path.dirname(os.path.abspath(__file__))
PG = os.path.join(HERE, "..", "..", "perch-runtime", "scripts")
LAB_DATA = os.environ.get("BIRDLAB_DATA",
                          os.path.join(HERE, "..", "..", "..", "lab-data"))
TUNING = os.path.join(LAB_DATA, "evalset")
HOURS = os.path.join(HERE, "..", "..", "..", "upstream-quant", "data",
                     "calibration_ssw_hours.txt")
W = 160000  # samples per window (5 s at 32 kHz)
HOP = 320  # window starts are multiples of 10 ms


def read_hours() -> list[str]:
    """The 8 calibration hours' file names."""
    with open(HOURS, encoding="utf-8") as f:
        hours = [line.split("\t")[0] for line in f if line.strip()]
    assert len(hours) == 8, hours
    return hours


def per_hour(n: int) -> list[int]:
    """n windows split over the 8 hours, the remainder to the first."""
    return [n // 8 + (1 if i < n % 8 else 0) for i in range(8)]


def main(argv: list[str]) -> int:
    """Writes the dev-set windows and their index file."""
    # infer.py reads sys.argv[1] when it is imported (here the audio dir;
    # only its load_stream is used)
    infer = extern.load_with_argv(os.path.join(TUNING, "scripts"), "infer",
                                  *argv[1:])
    perchgate = extern.load_module(PG, "perchgate")
    audio_dir, out, n = argv[1], argv[2], int(argv[3])
    seed = int(argv[4]) if len(argv) > 4 else 2411
    hours = read_hours()
    rng = np.random.default_rng(seed)
    x_all = np.zeros((n, W), np.float32)
    meta = []
    k = 0
    for h, m in zip(hours, per_hour(n)):
        stream = infer.load_stream([os.path.join(audio_dir, h)])
        x = perchgate.resample_to_32k(stream)
        starts = np.sort(rng.choice((len(x) - W) // HOP, size=m,
                                    replace=False)) * HOP
        for s in starts:
            x_all[k] = x[s:s + W]
            meta.append(f"{h} {s}")
            k += 1
        print(h, len(x) / 32000.0, "s", m, "windows", flush=True)
    x_all.tofile(out)
    with open(out + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(meta) + "\n")
    print("wrote", out, x_all.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
