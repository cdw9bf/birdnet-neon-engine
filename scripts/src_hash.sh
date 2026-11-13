#!/bin/sh
# engine source hash used to label gate runs: sha256 over src/*.c src/*.h in C-locale sorted order
cd "$(dirname "$0")/.."
# shasum (macOS, most Linux) or sha256sum (coreutils only), chosen before the pipeline runs
if command -v shasum > /dev/null 2>&1; then SHA="shasum -a 256"; else SHA=sha256sum; fi
LC_ALL=C ls src/*.c src/*.h | LC_ALL=C sort | xargs cat | $SHA | cut -c1-16
