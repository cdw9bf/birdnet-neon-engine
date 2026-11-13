#!/bin/bash
# The Perch v2 global packages N012-N014 through the shared harness: scripts/gate_perch.sh all (see there).
exec "$(dirname "$0")/gate_perch.sh" all "$@"
