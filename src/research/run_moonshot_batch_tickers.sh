#!/bin/bash
# Run the moonshot pipeline for a list of tickers on Claude/Sonnet, falling back to
# Nemotron automatically if a real Claude usage-limit error is hit (detected from
# the actual CLI error text, not just any Nemotron-side noise in the log).
#
# Usage: src/research/run_moonshot_batch_tickers.sh TICKER [TICKER ...]
# Example: src/research/run_moonshot_batch_tickers.sh IOT AHR ROKU KNSA MRP ANIP DAVE HNGE ZVRA LFST
#
# Run from the repo root. Logs go to logs/moonshot_batch/.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOGDIR="$REPO_ROOT/logs/moonshot_batch"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/_summary.log"
LIMIT_RE='claude CLI failed.*(limit|429|quota|overloaded_error|rate_limit)'

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 TICKER [TICKER ...]" >&2
  exit 1
fi

backend="nemotron"
needs_rescore=()

for t in "$@"; do
  attempt=1
  while true; do
    if [ "$backend" = "claude" ]; then
      label="${t} (claude/sonnet)"
      logfile="$LOGDIR/${t}_sonnet.log"
      args=(--backend claude --model sonnet --ledger-backend claude --ledger-model sonnet)
    else
      label="${t} (nemotron)"
      logfile="$LOGDIR/${t}.log"
      args=()
    fi
    echo "=== $label starting $(date) ===" | tee -a "$SUMMARY"
    .venv/bin/python src/research/run_moonshot_pipeline.py "$t" "${args[@]}" > "$logfile" 2>&1
    rc=$?
    echo "=== $label finished rc=$rc $(date) ===" | tee -a "$SUMMARY"

    if [ "$backend" = "claude" ] && [ $rc -ne 0 ] && grep -qiE "$LIMIT_RE" "$logfile"; then
      echo "=== usage limit detected on $t, switching remaining batch to nemotron $(date) ===" | tee -a "$SUMMARY"
      backend="nemotron"
      attempt=$((attempt+1))
      if [ $attempt -le 2 ]; then
        continue
      fi
    fi

    # Score-only gap: analysis finished but the always-Claude scoring step was
    # still rate-limited. Record it and move on; retried in the final pass below.
    if [ $rc -ne 0 ] && grep -q "score\[identified\] *INCOMPLETE" "$logfile" && grep -q "analyze\[identified\] *done" "$logfile"; then
      needs_rescore+=("$t")
    fi
    break
  done
done

if [ "${#needs_rescore[@]}" -gt 0 ]; then
  echo "=== final rescore pass for: ${needs_rescore[*]} $(date) ===" | tee -a "$SUMMARY"
  for t in "${needs_rescore[@]}"; do
    echo "=== $t rescoring starting $(date) ===" | tee -a "$SUMMARY"
    .venv/bin/python src/research/run_moonshot_pipeline.py "$t" > "$LOGDIR/${t}_rescore.log" 2>&1
    rc=$?
    echo "=== $t rescoring finished rc=$rc $(date) ===" | tee -a "$SUMMARY"
  done
fi

echo "=== batch ($*) complete $(date) ===" | tee -a "$SUMMARY"
