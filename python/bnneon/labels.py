"""Rewrite a label file into the form BirdNET-Go reads.

  python -I scripts/relabel_for_birdnetgo.py IN.txt OUT.txt
      [--sci LINE=Genus species ...] [--allow-doubled] [--check]

BirdNET-Go reads one `Scientific name_Common name` per line: it splits each
line on `_` into the scientific and the common name (an optional third field
is a code), so a file published as `Common_Name_Genus_species`
(BirdNET-Analyzer's default) must be rewritten. The rule: the last two
underscore-separated tokens form the scientific name when they look like
`Genus species` (a capitalized word followed by a lower-case word); everything
before them, with underscores turned into spaces, is the common name. A line
already in the target form (one or two `_`, a spaced capitalized left part) is
kept.

A line with no such pair falls into one of two groups:
  - a single word or a non-species class (`Anthropogenic`, `Nothing`,
    `Background_birds`, `Cricket`): written as `Name_Name`
    (`Anthropogenic_Anthropogenic`) and listed in the summary;
  - two or more capitalized words (`Eurasian_Golden_Oriole`): most likely a
    real species whose scientific name the file lacks. These are flagged
    loudly in --check and REFUSED when writing unless each gets
    `--sci LINE=Genus species` (e.g. --sci "Eurasian_Golden_Oriole=Oriolus
    oriolus") or --allow-doubled accepts the doubled form for all of them.
--check only reports (exit 1 when anything would change or needs --sci). The
line count and order never change. The rule is a heuristic: run --check first
and read the printed rewrites.
"""
import argparse
from collections.abc import Sequence
import re
import sys

SCI = re.compile(r"^[A-Z][a-z-]+ [a-z][a-z-]+$")
# `Corvus corone cornix`, `Myotis sp.`: a spaced, capitalized part
SCI_ANY = re.compile(r"^[A-Z][A-Za-z.-]* [a-z.][A-Za-z. -]*$")
CAP = re.compile(r"^[A-Z][a-z-]+$")
# the first paragraph of the module docstring as it was first written (the
# --help description; kept byte for byte)
DESCRIPTION = ("Rewrite a label file into the form BirdNET-Go reads: one "
               "`Scientific name_Common name` per line.")


def looks_birdnetgo(line: str) -> bool:
    """True when a line is already in BirdNET-Go's form."""
    parts = line.split("_")
    if len(parts) not in (2, 3):
        return False
    return SCI_ANY.match(parts[0]) is not None or parts[0] == parts[1]


def classify(line: str, sci_map: dict[str, str]) -> tuple[str, str]:
    """Rewrites one label line.

    Args:
        line: the line (stripped).
        sci_map: --sci: line -> scientific name.

    Returns:
        (new line, kind): kind is 'kept', 'sci' (from --sci), 'split' (Genus
        species found), 'doubled' (Name_Name) or 'species?' (two or more
        capitalized words, no scientific name: needs --sci or
        --allow-doubled).
    """
    if line in sci_map:
        return f"{sci_map[line]}_{line.replace('_', ' ')}", "sci"
    if looks_birdnetgo(line):
        return line, "kept"
    tok = line.split("_")
    if len(tok) >= 3 and " " not in tok[0]:
        pair = f"{tok[-2]} {tok[-1]}"
        if SCI.match(pair):
            return f"{pair}_{' '.join(tok[:-2])}", "split"
    name = line.replace("_", " ")
    capitalized = all(CAP.match(t) for t in tok)
    kind = "species?" if len(tok) >= 2 and capitalized else "doubled"
    return f"{name}_{name}", kind


def rewrite(line: str, sci_map: dict[str, str]) -> str:
    """The BirdNET-Go form of one label line (see classify)."""
    return classify(line, sci_map)[0]


def _parser() -> argparse.ArgumentParser:
    """The command line parser (its --help text is part of the interface)."""
    ap = argparse.ArgumentParser(
        prog="relabel_for_birdnetgo.py", description=DESCRIPTION,
        epilog="flags may come before or after the file arguments")
    ap.add_argument("infile",
                    help="the label file as published (one class per line)")
    ap.add_argument("outfile", help="where to write the BirdNET-Go form "
                                    "(ignored with --check)")
    ap.add_argument("--sci", action="append", default=[],
                    metavar="LINE=Genus species",
                    help="the scientific name of a line the file lacks it "
                         "for (repeatable)")
    ap.add_argument("--allow-doubled", action="store_true",
                    help="accept Name_Name for lines that look like species "
                         "names")
    ap.add_argument("--check", action="store_true",
                    help="only report what would change (exit 1 when "
                         "anything would)")
    return ap


def parse_sci(values: Sequence[str]) -> dict[str, str] | None:
    """--sci LINE=Genus species values as a map; None (error printed) if bad."""
    sci = {}
    for s in values:
        k, sep, v = s.partition("=")
        if not sep or not v.strip():
            print(f"error: --sci expects LINE=Genus species, got '{s}'",
                  file=sys.stderr)
            return None
        sci[k] = v.strip()
    return sci


def _rewrite_lines(lines: Sequence[str], sci: dict[str, str]):
    """Rewrites every line, printing each change.

    Returns:
        (new lines, changed count, doubled lines, species? lines).
    """
    out, changed, doubled, species = [], 0, [], []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        new, kind = classify(ln.strip(), sci)
        if new != ln.strip():
            changed += 1
            flag = ("   <- NO SCIENTIFIC NAME: a species? give --sci"
                    if kind == "species?" else "")
            print(f"{ln.strip():45s} -> {new}{flag}")
        if kind == "doubled":
            doubled.append(ln.strip())
        if kind == "species?":
            species.append(ln.strip())
        out.append(new)
    return out, changed, doubled, species


def _summary(lines: Sequence[str], sci: dict[str, str],
             doubled: Sequence[str], species: Sequence[str]) -> None:
    """Prints the unknown --sci lines, the doubled and the species? lines."""
    present = {line.strip() for line in lines}
    unknown = [k for k in sci if k not in present]
    if unknown:
        print(f"WARN: --sci names lines that are not in the file: {unknown}")
    if doubled:
        print(f"written as Name_Name (no scientific name, taken as non-species "
              f"classes): {', '.join(doubled)}")
    if species:
        print(f"NO SCIENTIFIC NAME FOUND for {len(species)} line(s) that look "
              f"like species names: {', '.join(species)}")
        print("  give each with --sci \"LINE=Genus species\" (look the species "
              "up), or --allow-doubled to write them as Name_Name")


def main(argv: Sequence[str]) -> int:
    """The relabel command; returns its exit code.

    Returns:
        0 written (or nothing to change with --check); 1 --check found
        changes; 2 refused (lines need --sci); 3 usage error.
    """
    try:
        a = _parser().parse_args(argv)
    except SystemExit as e:
        return 3 if e.code else 0
    sci = parse_sci(a.sci)
    if sci is None:
        return 3
    try:
        with open(a.infile, "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except OSError as e:
        print(f"error: cannot read {a.infile}: {e.strerror}", file=sys.stderr)
        return 3
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines()
    out, changed, doubled, species = _rewrite_lines(lines, sci)
    n = sum(1 for line in lines if line.strip())
    _summary(lines, sci, doubled, species)
    if a.check:
        need = f"; {len(species)} need --sci" if species else ""
        print(f"{changed} of {n} lines would change{need}")
        return 1 if changed or species else 0
    if species and not a.allow_doubled:
        print(f"error: not written: {len(species)} line(s) need --sci or "
              f"--allow-doubled (see above)", file=sys.stderr)
        return 2
    end = nl if raw.endswith(("\n", "\r\n")) else ""
    with open(a.outfile, "wb") as f:
        f.write((nl.join(out) + end).encode("utf-8"))
    doubled_note = (f" ({len(species)} doubled by --allow-doubled)"
                    if species else "")
    print(f"wrote {a.outfile}: {changed} lines rewritten, {n} labels"
          f"{doubled_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
