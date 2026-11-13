"""Unit tests for bnneon.pipe (the --pipe protocol and the stream pool).

  python -P -s tests/test_pipe.py     (bnneon installed, or
                                       PYTHONPATH=<checkout>/python)

No engine is needed: a Python child process stands in for the CLI. Exit 0
when every check passes.
"""
import io
import sys
import threading
import time

import numpy as np

from bnneon import pipe
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check

# a child that echoes its stdin to its stdout
ECHO = [sys.executable, "-c",
        "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"]
# a child that exits without reading its stdin
QUIT = [sys.executable, "-c", "import sys; sys.exit(3)"]


class _Sink(io.BytesIO):
    """A BytesIO that keeps its bytes when closed."""

    def __init__(self):
        super().__init__()
        self.data = b""
        self.closed_by_feed = False

    def close(self):
        self.data = self.getvalue()
        self.closed_by_feed = True
        super().close()


def _expected(offs, x):
    """The --pipe input bytes of offsets offs and stream x."""
    return (np.int32(len(offs)).tobytes() + offs.astype(np.int32).tobytes()
            + x.astype(np.float32).tobytes())


def test_feed():
    offs = np.array([0, 5, 9], np.int32)
    x = np.arange(20, dtype=np.float32)
    sink = _Sink()
    pipe.feed(sink, offs, x)
    check("feed writes n, offsets, stream", sink.data == _expected(offs, x))
    check("feed closes stdin", sink.closed_by_feed)
    sink = _Sink()
    pipe.feed(sink, offs.astype(np.int64), x.astype(np.float64))
    check("feed converts to int32 / float32",
          sink.data == _expected(offs, x))
    big = np.arange(3 * pipe.CHUNK // 4 + 7, dtype=np.float32)
    sink = _Sink()
    pipe.feed(sink, offs, big)
    check("feed writes a stream longer than a chunk",
          sink.data == _expected(offs, big))


def test_pipe_once():
    offs = np.array([0, 3], np.int32)
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    rc, out = pipe.pipe_once(ECHO, offs, x)
    check("pipe_once: exit code and output", rc == 0
          and out == _expected(offs, x), (rc, len(out)))
    big = np.zeros(2 * pipe.CHUNK, np.float32)
    rc, out = pipe.pipe_once(QUIT, offs, big)
    check("pipe_once: a reader that quits early (no BrokenPipeError)",
          rc == 3 and out == b"", (rc, out))


def test_start_and_read():
    offs = np.array([0], np.int32)
    x = np.arange(4, dtype=np.float32)
    p, th = pipe.start(ECHO, offs, x)
    head = pipe.read_exact(p.stdout, 8)
    rest = pipe.read_exact(p.stdout, 100)
    th.join()
    p.wait()
    want = _expected(offs, x)
    check("read_exact reads exactly n bytes", head == want[:8], head)
    check("read_exact is short at the end of the output",
          rest == want[8:] and p.returncode == 0, (len(rest), p.returncode))


def test_pooled():
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def run(name, x):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return name, x

    streams = {f"s{i}": {"files": [i]} for i in range(6)}
    out = list(pipe.pooled(streams, 2, lambda files: files[0] * 10, run))
    check("pooled: results in stream order",
          out == [(f"s{i}", 10 * i) for i in range(6)], out)
    check("pooled: at most `jobs` streams at a time", state["peak"] == 2,
          state)


def main():
    test_feed()
    test_pipe_once()
    test_start_and_read()
    test_pooled()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
