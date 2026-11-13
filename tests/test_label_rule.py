"""The label-line rule: the compiler, the Python binding and the README's C.

  python -P -s tests/test_label_rule.py [--cc cc] [--labels FILE ...]
      (bnneon installed, or PYTHONPATH=<checkout>/python)

The compiler (bnneon.compiler.prep.class_labels), the binding
(bnneon.binding.class_labels, its self-contained twin) and the README's C
example (class_label / label_len, compiled from the README's first ```c
block) must number a package's label lines alike. Checks the rule on cases,
that all three agree on every case, and that the compiler refuses the files
other line rules would number differently. --labels FILE: also checks that
the compiler accepts that file and the three readers agree on it. Without a C
compiler the C part is a clean skip. Exit 0 when every check passes.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from bnneon import binding
from bnneon import testutil
from bnneon.compiler import prep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKS = testutil.Checks()
check = CHECKS.check
# the C side prints each label followed by this separator
SEP = b"\x1f"
# reads a label file on stdin, prints every class label (the README's rule)
C_MAIN = r"""
int main(void) {
  static char buf[1 << 20];
  size_t n = fread(buf, 1, sizeof buf - 1, stdin);
  buf[n] = 0;
  for (int i = 0;; i++) {
    const char *l = class_label(buf, i);
    if (!l) return 0;
    printf("%.*s\x1f", label_len(l), l);
  }
}
"""
# (name, file text, the class labels)
CASES = [
    ("plain", "A_a\nB_b\n", ["A_a", "B_b"]),
    ("no final newline", "A_a\nB_b", ["A_a", "B_b"]),
    ("CRLF", "A_a\r\nB_b\r\n", ["A_a", "B_b"]),
    ("blank lines are not labels", "\nA_a\n\n \t\nB_b\r\n \r\n\n",
     ["A_a", "B_b"]),
    ("spaces inside and around a label are kept", " A a_B b \n",
     [" A a_B b "]),
    ("Perch's header line is not a class",
     "inat2024_fsd50k\nAbavorana luctuosa\n", ["Abavorana luctuosa"]),
    ("the header after blank lines", "\n\ninat2024_fsd50k\r\nA\n", ["A"]),
    ("the header string later is a class", "A\ninat2024_fsd50k\n",
     ["A", "inat2024_fsd50k"]),
    ("a longer first line is a class", "inat2024_fsd50k2\nA\n",
     ["inat2024_fsd50k2", "A"]),
    ("a header with spaces is a class", "inat2024_fsd50k \nA\n",
     ["inat2024_fsd50k ", "A"]),
    ("the header alone", "inat2024_fsd50k\n", []),
    ("empty file", "", []),
    ("UTF-8 labels", "Corvus corone_Corneille noire é\n",
     ["Corvus corone_Corneille noire é"]),
]
# files the compiler refuses: another line rule would number them differently
REFUSED = [
    ("a lone CR", "A\rB\n"),
    ("CR-only line ends", "A\rB\rC\r"),
    ("two CRs before the newline", "A\r\r\nB\n"),
    ("a form feed", "A\x0cB\n"),
    ("a vertical tab", "A\x0bB\n"),
    ("a file separator", "A\x1cB\n"),
    ("a next-line character", "A\u0085B\n"),
    ("a Unicode line separator", "A B\n"),
    ("a Unicode paragraph separator", "A B\n"),
    ("a line of no-break spaces", "A\n \nB\n"),
    ("a NUL byte", "A\nB\x00C\n"),
]


def readme_label_functions():
    """label_len and class_label from the README's first ```c block."""
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        m = re.search(r"```c\n(.*?)```", f.read(), re.S)
    if m is None:
        return None
    code = m.group(1)
    parts = []
    for name in ("label_len", "class_label"):
        fn = re.search(r"^static [^\n]*\b" + name + r"\(.*?^}\n", code,
                       re.S | re.M)
        if fn is None:
            return None
        parts.append(fn.group(0))
    return "".join(parts)


def build_c_reader(cc, tmp):
    """Compiles the README's label functions; the executable or None."""
    functions = readme_label_functions()
    check("README has label_len and class_label", functions is not None)
    if functions is None:
        return None
    src = os.path.join(tmp, "labels.c")
    exe = os.path.join(tmp, "labels")
    with open(src, "w", encoding="utf-8") as f:
        f.write("#include <stdio.h>\n#include <string.h>\n\n")
        f.write(functions + C_MAIN)
    r = subprocess.run([cc, "-std=gnu11", "-O2", "-Wall", "-Wextra",
                        "-Werror", src, "-o", exe], capture_output=True,
                       text=True, check=False)
    check("README label functions compile with -Werror", r.returncode == 0,
          r.stdout + r.stderr)
    return None if r.returncode else exe


def c_labels(exe, raw):
    """The README reader's class labels of a file's bytes."""
    out = subprocess.run([exe], input=raw, capture_output=True, check=True)
    return [s.decode("utf-8", "replace") for s in out.stdout.split(SEP)[:-1]]


def test_cases(exe):
    for name, text, want in CASES:
        raw = text.encode()
        check(f"compiler: {name}", prep.class_labels(raw) == want,
              prep.class_labels(raw))
        check(f"binding: {name}", binding.class_labels(text) == want,
              binding.class_labels(text))
        check(f"compiler accepts: {name}",
              prep.label_rule_error("l.txt", raw) is None)
        if exe:
            check(f"README C: {name}", c_labels(exe, raw) == want,
                  c_labels(exe, raw))


def test_refused(exe):
    for name, text in REFUSED:
        raw = text.encode()
        err = prep.label_rule_error("x/l.txt", raw)
        check(f"compiler refuses: {name}", err is not None
              and err.startswith("label file x/l.txt: non-blank line ")
              and "hint:" in err, err)
        # the three readers still agree with each other on such a file
        check(f"binding = compiler: {name}",
              binding.class_labels(raw.decode()) == prep.class_labels(raw))
        if exe and "\0" not in text:
            check(f"README C = compiler: {name}",
                  c_labels(exe, raw) == prep.class_labels(raw),
                  c_labels(exe, raw))


def test_files(exe, paths):
    for path in paths:
        with open(path, "rb") as f:
            raw = f.read()
        name = os.path.basename(path)
        check(f"{name}: the compiler accepts it",
              prep.label_rule_error(path, raw) is None,
              prep.label_rule_error(path, raw))
        want = prep.class_labels(raw)
        check(f"{name}: binding = compiler ({len(want)} labels)",
              binding.class_labels(raw.decode("utf-8", "replace")) == want)
        if exe:
            check(f"{name}: README C = compiler", c_labels(exe, raw) == want)


def main():
    cc = testutil.opt("--cc", os.environ.get("CC", "cc"))
    with tempfile.TemporaryDirectory(prefix="bnn_labels_") as tmp:
        exe = build_c_reader(cc, tmp) if shutil.which(cc) else None
        if exe is None:
            print(f"skip README C part (no C compiler '{cc}', or it failed)")
        test_cases(exe)
        test_refused(exe)
        test_files(exe, testutil.opt_all("--labels"))
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
