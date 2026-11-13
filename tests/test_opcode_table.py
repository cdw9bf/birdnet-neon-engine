"""The compiler's opcode and activation numbers against the runtime's.

  python -I tests/test_opcode_table.py     (bnneon installed, or
                                            PYTHONPATH=<checkout>/python -P -s)

The registry opcodes are written down twice: the RT_OP_* enum of src/rt.h (the
C runtime) and the OP tables of bnneon.compiler.lower and lower_dot (the
lowering that writes them into packages). Nothing generates one from the
other, so this test reads the enum from the header and compares names and
numbers, and does the same for the fused-activation codes. Exit 0 when every
check passes.
"""
import os
import re
import sys

from bnneon import testutil
from bnneon.compiler import lower
from bnneon.compiler import lower_dot

CHECKS = testutil.Checks()
check = CHECKS.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the compiler's name of each RT_ACT_* code
ACT_NAMES = {"NONE": None, "SILU": "swish", "RELU": "relu"}


def enum_values(header, prefix):
    """{name without prefix: number} of the `prefix NAME = N` enumerators."""
    found = re.findall(rf"\b{prefix}([A-Z0-9_]+)\s*=\s*(\d+)", header)
    return {name: int(number) for name, number in found}


def test_opcodes(header):
    runtime = enum_values(header, "RT_OP_")
    compiler = lower.OP | lower_dot.OP
    shared = set(lower.OP) & set(lower_dot.OP)
    check("the two lowering tables name different opcodes", not shared, shared)
    check("every runtime opcode has the compiler's number, and the reverse",
          runtime == compiler,
          sorted(set(runtime.items()) ^ set(compiler.items())))
    numbers = sorted(runtime.values())
    check("opcode numbers are 1..N without gaps or repeats (append only)",
          numbers == list(range(1, len(numbers) + 1)), numbers)
    check("the enum ends with the count RT_OP_N",
          re.search(r"RT_OP_[A-Z0-9_]+\s*=\s*\d+,[^\n]*\n\s*RT_OP_N\n",
                    header) is not None)


def test_activations(header):
    runtime = enum_values(header, "RT_ACT_")
    check("the runtime has the three activation codes",
          set(runtime) == set(ACT_NAMES), runtime)
    as_compiler = {ACT_NAMES.get(name): code for name, code in runtime.items()}
    check("every activation code has the compiler's number",
          as_compiler == lower.ACT, (as_compiler, lower.ACT))


def main():
    with open(os.path.join(ROOT, "src", "rt.h"), encoding="utf-8") as f:
        header = f.read()
    test_opcodes(header)
    test_activations(header)
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
