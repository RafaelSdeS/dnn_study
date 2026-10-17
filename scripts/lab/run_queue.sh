#!/usr/bin/env bash
# Runs a queue of training commands one at a time on a lab GPU box without Slurm (nyx; hefesto would do too). QUEUE
# holds one shell command per line (`.venv/bin/python -m scripts.train --experiment E --runtime pcad --model M`); its
# head is re-read before every run, so lines can be added or reordered while it runs (prepend to prioritise).
# success -> QUEUE.done, failure -> QUEUE.failed. The box is shared: before each run it waits while any other process
# computes on the GPU (a run is never killed, the next one just waits). Exits when QUEUE is empty. Start it detached,
# from the repo, with the dataset the PCAD runs read (docs/logs/PHASE11_LOG.md "Input 224 and VGG kernel blocks"):
#   PCAD_DATASET_ROOT=~/.cache/kagglehub/datasets/akash2sharma/tiny-imagenet/versions/1 \
#     setsid nohup bash scripts/lab/run_queue.sh ~/queue_nyx.txt >> ~/queue_nyx.log 2>&1 < /dev/null &
# Stop: pkill -f "run_[q]ueue" (the run in progress goes on; kill its python too -- the same line rerun resumes it).
set -uo pipefail
if [[ -z "${RUN_QUEUE_NICED:-}" ]]; then
  export RUN_QUEUE_NICED=1
  exec nice -n 10 ionice -c3 bash "$(realpath "$0")" "$@"
fi
QUEUE=$(realpath "$1"); WAIT=${WAIT:-600}
cd "$(dirname "$(realpath "$0")")/../.."

while [[ -s "$QUEUE" ]]; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then sleep "$WAIT"; continue; fi
  cmd=$(head -n1 "$QUEUE"); sed -i 1d "$QUEUE"
  echo "$(date -Is) START $cmd"
  # up to TRIES attempts: a run resumes from its per-epoch checkpoint, so a transient crash (an intermittent CUDA error
  # in the DataLoader's pin-memory thread was seen once on nyx, 2026-10-17) costs one epoch; a real bug fails each time
  ok=0
  for try in $(seq "${TRIES:-3}"); do
    if bash -c "$cmd"; then ok=1; break; fi
    echo "$(date -Is) RETRY $try $cmd"; sleep 60
  done
  if (( ok )); then
    echo "$cmd" >> "$QUEUE.done"; echo "$(date -Is) OK $cmd"
  else
    echo "$cmd" >> "$QUEUE.failed"; echo "$(date -Is) FAILED $cmd"
  fi
done
echo "$(date -Is) queue empty"
