#!/usr/bin/env bash
# Keeps the Slurm queue fed under PCAD's QOS per-user submit limit (MaxSubmitJobs 50, pending + running;
# found 2026-09-30 -- sacctmgr's association shows no limit, the QOS does). QUEUE holds one shell command per
# line, each submitting ONE job (`python -m scripts.cluster submit ... --model m`, or
# `scripts/pcad/rerun_qat_fused.sh <experiment>/<model>`). Every 10 min, while fewer than MAX jobs are queued,
# it pops the first line and runs it: success -> QUEUE.done; the QOS limit (raced by a manual submit) -> back
# to the head, retried later; any other failure -> QUEUE.failed. Lines may be added or reordered while it runs
# (prepend to prioritise). Exits when QUEUE is empty. Run on PCAD, repo up to date, clean code:
#   setsid nohup scripts/pcad/feed_queue.sh ~/queue_phase11.txt >> ~/queue_phase11.log 2>&1 < /dev/null &
set -uo pipefail
QUEUE=$(realpath "$1"); MAX=${MAX:-48}; SLEEP=${SLEEP:-600}  # 2 slots left for manual submits
cd "$(dirname "$(realpath "$0")")/../.." && source .venv/bin/activate

while [[ -s "$QUEUE" ]]; do
  if (( $(squeue -h -u "$USER" | wc -l) >= MAX )); then sleep "$SLEEP"; continue; fi
  cmd=$(head -n1 "$QUEUE"); sed -i 1d "$QUEUE"
  if out=$(bash -c "$cmd" 2>&1); then
    echo "$cmd" >> "$QUEUE.done"; echo "$(date -Is) OK $cmd :: $(tail -n1 <<<"$out")"
  elif grep -q QOSMaxSubmitJobPerUserLimit <<<"$out"; then
    { echo "$cmd"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; sleep "$SLEEP"
  else
    echo "$cmd" >> "$QUEUE.failed"; echo "$(date -Is) FAILED $cmd :: $(tail -n3 <<<"$out" | tr '\n' ' ')"
  fi
done
echo "$(date -Is) queue empty"
