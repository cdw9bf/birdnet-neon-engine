"""Unit tests for bnneon.labels and the package's binding import.

  python -P -s tests/test_package_units.py     (bnneon installed, or
                                                PYTHONPATH=<checkout>/python)

Exit 0 when every check passes.
"""
import contextlib
import io
import os
import sys
import tempfile

import bnneon
from bnneon import labels
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check


def test_classify():
    cases = {
        "Turdus merula_Common Blackbird": ("Turdus merula_Common Blackbird",
                                           "kept"),
        "Common_Blackbird_Turdus_merula": ("Turdus merula_Common Blackbird",
                                           "split"),
        "Eurasian_Golden_Oriole": ("Eurasian Golden Oriole_Eurasian Golden "
                                   "Oriole", "species?"),
        "Nothing": ("Nothing_Nothing", "doubled"),
        "Background_birds": ("Background birds_Background birds", "doubled"),
        "Myotis sp._Myotis": ("Myotis sp._Myotis", "kept"),
    }
    for line, want in cases.items():
        got = labels.classify(line, {})
        check(f"classify {line}", got == want, got)
    got = labels.classify("Eurasian_Golden_Oriole",
                          {"Eurasian_Golden_Oriole": "Oriolus oriolus"})
    check("classify with --sci", got == ("Oriolus oriolus_Eurasian Golden "
                                         "Oriole", "sci"), got)
    check("parse --sci", labels.parse_sci(["A_B=X y "]) == {"A_B": "X y"})
    with contextlib.redirect_stderr(io.StringIO()):
        check("parse --sci refuses", labels.parse_sci(["A_B"]) is None)


def _relabel(args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(io.StringIO()):
        rc = labels.main(args)
    return rc, out.getvalue()


def test_relabel_main():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.txt")
        dst = os.path.join(d, "out.txt")
        with open(src, "wb") as f:
            f.write(b"Common_Blackbird_Turdus_merula\r\nNothing\r\n")
        rc, out = _relabel([src, dst, "--check"])
        check("--check exit 1", rc == 1, out)
        check("--check writes nothing", not os.path.exists(dst))
        rc, out = _relabel([src, dst])
        with open(dst, "rb") as f:
            body = f.read()
        check("write keeps CRLF and the final newline", rc == 0 and body
              == b"Turdus merula_Common Blackbird\r\nNothing_Nothing\r\n",
              body)
        with open(src, "w", encoding="utf-8") as f:
            f.write("Eurasian_Golden_Oriole\n")
        check("species without --sci refused", _relabel([src, dst])[0] == 2)
        check("--allow-doubled writes", _relabel([src, dst,
                                                 "--allow-doubled"])[0] == 0)
        check("usage error", _relabel([])[0] == 3)


def test_binding_import():
    check("binding API", bnneon.API_VERSION == 2 and bnneon.Model
          is bnneon.binding.Model and callable(bnneon.load_library))


def main():
    test_classify()
    test_relabel_main()
    test_binding_import()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
