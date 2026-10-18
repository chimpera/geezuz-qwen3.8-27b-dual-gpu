#!/usr/bin/env bash
# Sync this repo (without .git) to a remote box:
#   scripts/box-sync.sh <ssh-host> <remote-dir>
# The sources land in <remote-dir>/geezuz-qwen3.8-27b-dual-gpu. Build outputs generated on
# the box (see .gitignore) are excluded so a re-sync does not delete them.
set -euo pipefail

HOST=${1:?usage: box-sync.sh <ssh-host> <remote-dir>}
REMOTE_DIR=${2:?usage: box-sync.sh <ssh-host> <remote-dir>}
SRC="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" mkdir -p -- "$REMOTE_DIR/geezuz-qwen3.8-27b-dual-gpu"
rsync -avz --delete \
	--exclude .git \
	--exclude '*.o' \
	--exclude geezuz \
	--exclude libgeezuz.so \
	--exclude geezuz_tp2 \
	--exclude libgeezuz_tp2.so \
	--exclude 'ptxas*.log' \
	--exclude tools/vocab_freq \
	--exclude tools/llama_ref \
	--exclude bench/ffn_bench \
	--exclude 'bench/ffn_bench_h*' \
	--exclude bench/gemv_bench \
	--exclude bench/mmq_bench \
	--exclude 'bench/mmq_e*' \
	--exclude bench/resid_bench \
	"$SRC/" "$HOST:$REMOTE_DIR/geezuz-qwen3.8-27b-dual-gpu/"
