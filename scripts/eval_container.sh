#!/bin/bash
# The engine's arm64 evaluation container, from your own checkout (see integration/Dockerfile.eval).
#   scripts/eval_container.sh build  [IMAGE]        build the image (default bnneon-eval:local)
#   scripts/eval_container.sh start  [NAME] [IMAGE] run it detached, --memory 2g, this checkout mounted at /eng
#   scripts/eval_container.sh stop   [NAME]         remove it
#   scripts/eval_container.sh status [NAME]         what it mounts and whether gcc is there
# NAME defaults to bnneon-eval-local. Then: bnc.py check ... --docker NAME, or scripts/cmp_suite.sh (CONTAINER=NAME).
set -e
cd "$(dirname "$0")/.."
cmd=${1:-status}
name=${2:-bnneon-eval-local}
case "$cmd" in
  build)
    docker build --platform linux/arm64 -f integration/Dockerfile.eval -t "${2:-bnneon-eval:local}" integration ;;
  start)
    image=${3:-bnneon-eval:local}
    docker image inspect "$image" >/dev/null 2>&1 || { echo "image $image not found: run '$0 build' first"; exit 1; }
    docker run -d --name "$name" --platform linux/arm64 --memory 2g --memory-swap 2g -v "$PWD:/eng" "$image" >/dev/null
    if ! docker exec "$name" test -f /eng/Makefile; then
      docker rm -f "$name" >/dev/null
      echo "$PWD is not visible inside the container (/eng is empty): this path is not shared with Docker (Docker Desktop: Settings > Resources > File sharing; Colima: a mounted directory); container removed"; exit 1
    fi
    echo "started $name: $image, $PWD mounted at /eng (2 GB); use: bnc.py check ... --docker $name" ;;
  stop)
    docker rm -f "$name" >/dev/null && echo "removed $name" ;;
  status)
    docker inspect "$name" --format 'container {{.Name}}: {{.State.Status}}, mounts {{range .Mounts}}{{.Source}} -> {{.Destination}} {{end}}' 2>/dev/null \
      || { echo "no container named $name (start one: $0 start)"; exit 1; }
    docker exec "$name" sh -c 'gcc --version | head -1; test -f /eng/Makefile && echo "/eng holds the engine checkout" || echo "/eng does not hold the engine checkout"' ;;
  *) sed -n 2,8p "$0"; exit 3 ;;
esac
