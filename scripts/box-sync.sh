#!/usr/bin/env bash
# Sync this repo (without .git) to a remote box:
#   scripts/box-sync.sh <ssh-host> <remote-dir>
# The sources land in <remote-dir>/dual-gpu-megakernel. Build outputs generated on
# the box (see .gitignore) are excluded so a re-sync does not delete them.
set -euo pipefail

HOST=${1:?usage: box-sync.sh <ssh-host> <remote-dir>}
REMOTE_DIR=${2:?usage: box-sync.sh <ssh-host> <remote-dir>}
SRC="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" mkdir -p -- "$REMOTE_DIR/dual-gpu-megakernel"
rsync -avz --delete \
	--exclude .git \
	--exclude '*.o' \
	--exclude qwen38_mk \
	--exclude libqwen38mk.so \
	--exclude 'ptxas*.log' \
	--exclude tools/vocab_freq \
	--exclude tools/llama_ref \
	--exclude bench/ffn_bench \
	--exclude 'bench/ffn_bench_h*' \
	--exclude bench/gemv_bench \
	--exclude bench/mmq_bench \
	--exclude 'bench/mmq_e*' \
	--exclude bench/resid_bench \
	"$SRC/" "$HOST:$REMOTE_DIR/dual-gpu-megakernel/"
