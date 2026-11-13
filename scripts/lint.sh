#!/bin/sh
# Lint gate for every tracked Python file: pylint with the [tool.pylint.*] settings of pyproject.toml (Google's
# pylintrc, a little stricter). Exit 0 when there are no findings.
#   scripts/lint.sh [PYLINT]     PYLINT: the pylint to run (default: pylint on PATH; pip install -e '.[lint]')
cd "$(dirname "$0")/.." || exit 3
PYLINT=${1:-pylint}
git ls-files '*.py' | xargs "$PYLINT" --rcfile=pyproject.toml --score=n
