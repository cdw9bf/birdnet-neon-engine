"""The audio windows bnc check runs: from a file, from audio, or synthetic."""
from collections.abc import Callable, Sequence
import types
import wave

import numpy as np

from . import errs

# a window cut from audio counts when its peak exceeds this (not silence)
SIGNAL_PEAK = 1e-4


def _soundfile() -> types.ModuleType | None:
    """The soundfile module, or None when it is not installed."""
    try:
        # optional dependency, only `check --audio` reads with it (16-bit PCM
        # wav files work without it)
        import soundfile  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return soundfile


def from_file(path: str, nsamp: int, n: int = 0) -> np.ndarray:
    """Windows from --windows-file: a .npy array, or raw float32 (.f32).

    Args:
        path: the file ([n][nsamp] float32 PCM in [-1, 1)).
        nsamp: samples per window.
        n: how many windows to keep (0: all).

    Returns:
        float32 [windows][nsamp].

    Raises:
        errs.Usage: not float data, or not a whole number of windows.
    """
    if path.endswith(".npy"):
        x = np.load(path)
    else:
        x = np.fromfile(path, np.float32)
    if x.dtype.kind != "f":
        raise errs.Usage(f"--windows-file {path}: dtype {x.dtype}; expected "
                         f"float32 PCM in [-1, 1) (scale integer PCM first)")
    x = np.asarray(x, np.float32).reshape(-1)
    if x.size % nsamp:
        raise errs.Usage(f"--windows-file {path}: {x.size} floats is not a "
                         f"multiple of {nsamp} samples per window")
    x = x.reshape(-1, nsamp)
    return x[:n] if n else x


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Reads an audio file as mono: soundfile if installed, else 16-bit wav.

    Returns:
        (samples in [-1, 1), sample rate).

    Raises:
        errs.Usage: a wav file that is not 16-bit PCM (without soundfile).
    """
    sf = _soundfile()
    if sf is not None:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(1), fs
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise errs.Usage(f"{path}: {8 * width}-bit PCM; install soundfile or "
                         f"convert to 16-bit PCM")
    pcm = np.frombuffer(raw, np.int16).reshape(-1, channels)
    return pcm.mean(1) / 32768.0, fs


def _enough(xs: list, n: int) -> bool:
    """True when n windows are wanted (n > 0) and xs holds them."""
    return bool(n) and len(xs) >= n


def from_audio(paths: Sequence[str], nsamp: int, n: int, sr: int | None,
               log: Callable[..., None],
               skip_silence: bool = True) -> np.ndarray:
    """Up to n consecutive windows cut from audio files, in order.

    Args:
        paths: the --audio files.
        nsamp: samples per window.
        n: how many windows (0: every full window).
        sr: the model's rate (a file at another rate is warned about; nothing
            is resampled); None or 0 when unknown.
        log: the print function for the warning.
        skip_silence: leave out windows whose peak is at most SIGNAL_PEAK.

    Returns:
        float32 [windows][nsamp].

    Raises:
        errs.Usage: no (non-silent) window in any file.
    """
    xs = []
    for path in paths:
        x, fs = read_audio(path)
        if sr and fs != sr:
            log(f"WARN: {path} is {fs} Hz but the model expects {sr} Hz: "
                f"windows are cut without resampling")
        for i in range(0, len(x) - nsamp + 1, nsamp):
            w = x[i:i + nsamp].astype(np.float32)
            if not skip_silence or np.abs(w).max() > SIGNAL_PEAK:
                xs.append(w)
            if _enough(xs, n):
                break
        if _enough(xs, n):
            break
    if not xs:
        what = " with signal" if skip_silence else ""
        raise errs.Usage(f"--audio: no window of {nsamp} samples{what} "
                         f"found in {paths}")
    return np.stack(xs[:n] if n else xs)


def synthetic(nsamp: int, n: int, sr: int | None) -> np.ndarray:
    """n synthetic windows: noise, two chirps and an impulse each.

    Enough to exercise every layer; not a substitute for real audio. The same
    arguments always give the same windows.
    """
    rng = np.random.default_rng(0)
    t = np.arange(nsamp) / float(sr or 48000)
    xs = []
    for _ in range(n):
        x = rng.standard_normal(nsamp) * 0.01
        for _ in range(2):
            f0, f1 = rng.uniform(500, 3000), rng.uniform(1000, 8000)
            amp = rng.uniform(0.05, 0.3)
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * t[-1]))
            x += amp * np.sin(phase)
        x[rng.integers(nsamp)] += 0.5
        xs.append(np.clip(x, -0.99, 0.99).astype(np.float32))
    return np.stack(xs)


def make_windows(nsamp: int, n: int, audio: Sequence[str], sr: int | None,
                 log: Callable[..., None],
                 wfile: str | None = None) -> np.ndarray:
    """The windows check runs: --windows-file, else --audio, else synthetic.

    Args:
        nsamp: samples per window.
        n: how many windows (with a windows file, 0 keeps them all).
        audio: the --audio files (may be empty).
        sr: the model's sample rate, None when not given.
        log: the print function.
        wfile: the --windows-file, if any.

    Returns:
        float32 [windows][nsamp].

    Raises:
        errs.Usage: the windows file or the audio cannot be used.
    """
    if wfile:
        return from_file(wfile, nsamp, n)
    if audio:
        return from_audio(audio, nsamp, n, sr, log)
    return synthetic(nsamp, n, sr)
