"""Runs the engine CLI for bnc check.

The CLI is an existing binary (--cli), a native build of this checkout, a
container that mounts this checkout at /eng (--docker NAME), or a throwaway
container started from an image (--docker-image IMAGE, removed afterwards).
"""
from collections.abc import Callable, Sequence
import os
import pathlib
import subprocess

import numpy as np

from . import cliargs
from . import errs

# the engine checkout this package lives in (python/bnneon/compiler/): check
# builds the C runtime there. That holds for a checkout and for the editable
# install (`pip install -e .`); a copied, non-editable install has no engine
# sources next to it, so check then needs --cli, --docker or --docker-image.
ROOT = str(pathlib.Path(os.path.abspath(__file__)).parents[3])
# where a container build of the CLI goes (inside the container)
CONTAINER_DIR = "/root/bnc_check"


def _run(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run with text output captured; the caller checks the code."""
    return subprocess.run(cmd, capture_output=True, text=True, check=False,
                          **kwargs)


def run_cli(cli: str, pkg: str, mode: int, win: str,
            out: str) -> subprocess.CompletedProcess:
    """Runs `cli PKG MODE WIN OUT 1` natively; the caller checks the code."""
    return _run([cli, pkg, str(mode), win, out, "1"])


def read_output(out: str, n: int, nc: int) -> tuple[np.ndarray | None, str]:
    """The CLI's output file as [n][nc] float32.

    Returns:
        (the outputs, "") or (None, what is wrong with the file's size).
    """
    y = np.fromfile(out, np.float32)
    if y.size != n * nc:
        return None, f"the CLI wrote {y.size} floats, expected {n} x {nc}"
    return y.reshape(n, nc), ""


class Runner:
    """The engine CLI, built or found as the check options say.

    Attributes:
        cli: the CLI's path (inside the container when docker is set).
        docker: the container that runs the CLI, or None.
        own: the name of a container this Runner started (removed by close()).
    """

    def __init__(self, args: Sequence[str], log: Callable[..., None]):
        """Finds or builds the CLI.

        Args:
            args: the check command line (--cli, --docker, --docker-image).
            log: the print function.

        Raises:
            errs.Usage: --cli is not an executable.
            RuntimeError: a container or the build failed (the message has a
                hint).
        """
        self.log = log
        self.docker: str | None = cliargs.opt(args, "--docker")
        self.cli: str = cliargs.opt(args, "--cli")
        self.own: str | None = None
        if self.cli:
            if not os.access(self.cli, os.X_OK):
                raise errs.Usage(f"--cli {self.cli}: not an executable")
            return
        image = cliargs.opt(args, "--docker-image")
        if image:
            self.docker = self._start_container(image)
        if self.docker:
            self._check_mount()
            self.cli = self._build_in_container()
            return
        self.cli = self._build_native()

    def _start_container(self, image: str) -> str:
        """Starts a throwaway container from image; returns its name."""
        self.own = f"bnc-check-{os.getpid()}"
        r = _run(["docker", "run", "-d", "--name", self.own,
                  "--platform", "linux/arm64", "--memory", "2g",
                  "--memory-swap", "2g", "-v", f"{ROOT}:/eng", image,
                  "sleep", "infinity"])
        if r.returncode:
            raise RuntimeError(
                f"docker run {image} failed: {r.stderr.strip()[:300]}; hint: "
                f"build the image with scripts/eval_container.sh build "
                f"(integration/Dockerfile.eval)")
        return self.own

    def _check_mount(self) -> None:
        """Checks that the container mounts THIS checkout at /eng.

        A token written here must show up there (another tree would build
        other code).
        """
        os.makedirs(os.path.join(ROOT, "eval_tmp"), exist_ok=True)
        token = f"bnc-mount-{os.getpid()}"
        token_path = os.path.join(ROOT, "eval_tmp", token)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(ROOT)
        try:
            chk = _run(["docker", "exec", self.docker, "sh", "-c",
                        f"test -f /eng/eval_tmp/{token} && echo MOUNT_OK"])
        finally:
            os.remove(token_path)
        if "MOUNT_OK" in chk.stdout:
            return
        self.close()
        detail = chk.stderr.strip()
        detail = f" ({detail[:160]})" if detail else ""
        raise RuntimeError(
            f"container {self.docker} does not mount this checkout ({ROOT}) at "
            f"/eng{detail}; hint: start one with scripts/eval_container.sh "
            f"start (it mounts this checkout), or pass --docker-image <image>")

    def _build_in_container(self) -> str:
        """Builds the CLI inside the container; returns its path there."""
        out = _run(["docker", "exec", self.docker, "sh", "-c",
                    f"cd /eng && make -s OUT={CONTAINER_DIR} -j4 2>&1 | "
                    f"grep -E ' error' ; test -x {CONTAINER_DIR}/cli && "
                    f"echo BUILD_OK"])
        if "BUILD_OK" not in out.stdout:
            self.close()
            raise RuntimeError(
                f"container build failed in {self.docker}:/eng: "
                f"{out.stdout.strip()} {out.stderr.strip()}; hint: the "
                f"container needs gcc >= 10, make and a C library "
                f"(integration/Dockerfile.eval)")
        return f"{CONTAINER_DIR}/cli"

    def _build_native(self) -> str:
        """Builds the CLI of this checkout natively; returns its path."""
        out = _run(["make", "-s", f"OUT={ROOT}/build/check", "-j4"], cwd=ROOT)
        cli = f"{ROOT}/build/check/cli"
        if out.returncode or not os.access(cli, os.X_OK):
            lines = (out.stderr + out.stdout).splitlines()
            err = [line for line in lines if "error" in line][:5]
            detail = " | ".join(err) or out.stderr.strip()[:300]
            raise RuntimeError(
                f"native build failed ({detail}); hint: this needs an AArch64 "
                f"C compiler (clang or gcc >= 10); on another machine pass "
                f"--cli <built cli>, --docker <container> or --docker-image "
                f"<image>")
        return cli

    def close(self) -> None:
        """Removes the container this Runner started, if any."""
        if self.own:
            subprocess.run(["docker", "rm", "-f", self.own],
                           capture_output=True, check=False)
            self.own = None

    def _run_in_container(self, pkg: str, mode: int, win: str,
                          out: str) -> subprocess.CompletedProcess:
        """Copies package and windows in, runs the CLI, copies the output."""
        d = CONTAINER_DIR
        for f in (pkg, win):
            subprocess.run(["docker", "cp", f, f"{self.docker}:{d}/"],
                           check=True)
        r = _run(["docker", "exec", self.docker, self.cli,
                  f"{d}/{os.path.basename(pkg)}", str(mode),
                  f"{d}/{os.path.basename(win)}", f"{d}/out.f32", "1"])
        if r.returncode == 0:
            subprocess.run(["docker", "cp", f"{self.docker}:{d}/out.f32", out],
                           check=True)
        return r

    def run(self, pkg: str, mode: int, win: str, n: int, nc: int,
            workdir: str) -> tuple[np.ndarray | None, str, int]:
        """Runs one package on the windows file.

        Args:
            pkg: the package.
            mode: the CLI's mode (2 fp32, 3 fp16, 4 int8).
            win: the windows file (float32 [n][samples]).
            n: the number of windows.
            nc: the number of classes.
            workdir: where the output goes.

        Returns:
            (logits [n][nc] or None on failure, the CLI's messages, exit code).
        """
        out = os.path.join(workdir, f"out_{mode}.f32")
        if self.docker:
            r = self._run_in_container(pkg, mode, win, out)
        else:
            r = run_cli(self.cli, pkg, mode, win, out)
        if r.returncode:
            return None, (r.stderr + r.stdout).strip(), r.returncode
        y, msg = read_output(out, n, nc)
        if y is None:
            return None, msg, 1
        return y, r.stderr.strip(), 0
