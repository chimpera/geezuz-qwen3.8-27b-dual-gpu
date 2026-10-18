#!/usr/bin/env bash
# Build the megakernel on a remote box, all inside a container (no GPU used):
#   scripts/box-build.sh <ssh-host> <remote-dir>
# Expects the repo synced to <remote-dir>/geezuz-qwen3.8-27b-dual-gpu (box-sync.sh).
# Clones llama.cpp (tag below) to <remote-dir>/llama.cpp if missing, builds a
# CPU-only shared libllama from it, then runs make in the sources. The whole
# <remote-dir> is bind-mounted into the container at its own absolute path.
set -euo pipefail

HOST=${1:?usage: box-build.sh <ssh-host> <remote-dir>}
REMOTE_DIR=${2:?usage: box-build.sh <ssh-host> <remote-dir>}

ssh "$HOST" bash -s -- "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
REMOTE_DIR=$1
# a positional param is not tilde-expanded, so expand it here (the remote user's home)
case "$REMOTE_DIR" in
	"~") REMOTE_DIR=$HOME ;;
	"~/"*) REMOTE_DIR=$HOME/"${REMOTE_DIR#~/}" ;;
esac
SRC=$REMOTE_DIR/geezuz-qwen3.8-27b-dual-gpu
LLAMA=$REMOTE_DIR/llama.cpp
LLAMA_TAG=b10246
LLAMA_REPO=https://github.com/ggml-org/llama.cpp
IMAGE=mk-build:local

[ -f "$SRC/Makefile" ] || { echo "no repo at $SRC; run box-sync.sh first" >&2; exit 1; }

# build image: the CUDA devel image plus cmake and git (missing from it)
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "== build image $IMAGE =="
    docker build -t "$IMAGE" -f "$SRC/docker/build.Dockerfile" "$SRC/docker"
fi

# llama.cpp at the pinned tag, next to the sources (LLAMA ?= ../llama.cpp)
if [ ! -e "$LLAMA/.git" ]; then
    echo "== clone llama.cpp $LLAMA_TAG =="
    git clone --depth 1 --branch "$LLAMA_TAG" "$LLAMA_REPO" "$LLAMA"
else
    # annotated tags list a "<sha> refs/tags/<tag>^{}" line for the commit
    WANT=$(git ls-remote "$LLAMA_REPO" "refs/tags/$LLAMA_TAG^{}" | awk '{print $1}')
    [ -n "$WANT" ] || WANT=$(git ls-remote "$LLAMA_REPO" "refs/tags/$LLAMA_TAG" | awk '{print $1}')
    if [ "$WANT" != "$(git -C "$LLAMA" rev-parse HEAD)" ]; then
        echo "== update llama.cpp to $LLAMA_TAG =="
        git -C "$LLAMA" fetch --depth 1 origin tag "$LLAMA_TAG"
        git -C "$LLAMA" checkout -q "$LLAMA_TAG"
    fi
fi

echo "== build (CPU-only libllama, then the engine) =="
# run as the remote user (not root) so build artifacts stay theirs; bind-mount the dir at its own
# absolute path so the rpath make bakes ($(abspath $(LLAMA)/build/bin)) is valid on the box too
docker run --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-v "$REMOTE_DIR:$REMOTE_DIR" -w "$REMOTE_DIR" \
	"$IMAGE" bash -c '
	set -euo pipefail
	cmake -S llama.cpp -B llama.cpp/build \
		-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
		-DGGML_BACKEND=CPU -DGGML_CUDA=OFF -DBUILD_EXAMPLES=OFF
	cmake --build llama.cpp/build -j"$(nproc)"
	make -C geezuz-qwen3.8-27b-dual-gpu -j"$(nproc)"
'
REMOTE
