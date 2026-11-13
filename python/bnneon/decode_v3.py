"""Audio decoding as the BirdNET v3.0 gate harness does it.

ffmpeg (imageio_ffmpeg's) to 16-bit PCM at the native rate, mono, / 32768,
then soxr VHQ to 32 kHz. The v3 gate scorers (scripts/gate_v3_*.py) use it;
it needs soundfile, soxr and imageio-ffmpeg, which bnneon does not require.
"""
import functools
import os
import subprocess

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr

SR = 32000


@functools.cache
def ffmpeg_exe() -> str:
    """The ffmpeg executable imageio_ffmpeg provides."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_stream(files: list[str], root: str = "") -> np.ndarray:
    """Decodes and concatenates a stream's files at 32 kHz.

    Args:
        files: the audio files, relative to root (or absolute).
        root: the directory relative paths start from ("" = the current
            one).

    Returns:
        float32 samples (PCM16 / 32768, soxr VHQ resampled if needed).
    """
    parts = []
    for f in files:
        f = os.path.join(root, f)
        sr = sf.info(f).samplerate
        raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", f, "-f",
                              "s16le", "-ac", "1", "-"],
                             check=True, capture_output=True).stdout
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if sr != SR:
            x = soxr.resample(x, sr, SR, quality="VHQ").astype(np.float32)
        parts.append(x)
    return np.concatenate(parts)
