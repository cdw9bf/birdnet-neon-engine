"""The engine CLI's --pipe protocol and the stream pool of the gate scorers.

`cli <weights> <mode> --pipe[-emb]` reads int32 n, then n int32 window
offsets (samples into the stream), then the float32 stream until its stdin
closes, and writes n x D float32 outputs (D = classes, or the embedding size
for --pipe-emb) to stdout. feed() writes that input, start() runs a command
with a feeder thread, pipe_once() runs it to the end, read_exact() reads a
batch of outputs, and pooled() runs the streams of a gate on a few threads.
It needs numpy only.
"""
from collections.abc import Callable, Iterator
import concurrent.futures
import subprocess
import threading
from typing import Any, BinaryIO

import numpy as np

CHUNK = 1 << 22  # bytes per stdin write


def feed(stdin: BinaryIO, offs: np.ndarray, x: np.ndarray) -> None:
    """Writes int32 n, the int32 offsets and the float32 stream; closes stdin.

    The stream is written in CHUNK-byte slices of its own buffer (no copy of
    the stream). A reader that exits early ends the feeding quietly: the
    caller sees it as a short output and the command's exit code.

    Args:
        stdin: the command's stdin.
        offs: the window offsets in samples.
        x: the stream (float32 samples; another dtype is converted).
    """
    try:
        stdin.write(np.int32(len(offs)).tobytes())
        stdin.write(np.ascontiguousarray(offs, np.int32).tobytes())
        mv = memoryview(np.ascontiguousarray(x, np.float32)).cast("B")
        for k in range(0, len(mv), CHUNK):
            stdin.write(mv[k:k + CHUNK])
        stdin.close()
    except BrokenPipeError:
        pass


def start(cmd: list[str], offs: np.ndarray, x: np.ndarray,
          stdout: Any = subprocess.PIPE,
          stderr: Any = None) -> tuple[subprocess.Popen, threading.Thread]:
    """Starts cmd and a thread that feeds it (feed()).

    Args:
        cmd: the engine command (ending in --pipe or --pipe-emb).
        offs: the window offsets.
        x: the stream.
        stdout: passed to subprocess.Popen (default: a pipe to read).
        stderr: passed to subprocess.Popen (None: inherited).

    Returns:
        (the process, the started feeder thread; join it when done).
    """
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout,
                         stderr=stderr)
    th = threading.Thread(target=feed, args=(p.stdin, offs, x))
    th.start()
    return p, th


def pipe_once(cmd: list[str], offs: np.ndarray, x: np.ndarray,
              stderr: Any = None) -> tuple[int, bytes]:
    """Runs cmd once with feed() on its stdin; reads its whole stdout.

    Args:
        cmd: the engine command (ending in --pipe or --pipe-emb).
        offs: the window offsets.
        x: the stream.
        stderr: passed to subprocess.Popen (None: inherited).

    Returns:
        (exit code, the output bytes).
    """
    p, th = start(cmd, offs, x, stderr=stderr)
    out = p.stdout.read()
    th.join()
    p.wait()
    return p.returncode, out


def read_exact(stdout: BinaryIO, nbytes: int) -> bytes:
    """Up to nbytes from stdout; fewer only when the command's output ended.

    Args:
        stdout: the command's stdout (a buffered pipe: read(n) waits for n
            bytes or the end of the output).
        nbytes: the bytes wanted (4 x windows x outputs).

    Returns:
        The bytes read; the caller treats a short result as an incomplete
        output.
    """
    return stdout.read(nbytes)


def pooled(streams: dict[str, Any], jobs: int,
           load: Callable[[list[str]], np.ndarray],
           run: Callable[[str, np.ndarray], Any]) -> Iterator[Any]:
    """Runs run(name, load(files)) for every stream on `jobs` threads.

    A stream is decoded only when a slot is free (memory); the results are
    yielded in stream order once every stream has been submitted.
    """
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        futs = []
        running = set()
        for name, st in streams.items():
            while len(running) >= jobs:
                _, running = concurrent.futures.wait(
                    running, return_when=concurrent.futures.FIRST_COMPLETED)
            f = ex.submit(run, name, load(st["files"]))
            futs.append(f)
            running.add(f)
        for f in futs:
            yield f.result()
