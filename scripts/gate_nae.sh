#!/bin/bash
# The Perch v2 north-america-east packages N016-N018 through the shared harness: scripts/gate_perch.sh nae (see there).
exec "$(dirname "$0")/gate_perch.sh" nae "$@"
