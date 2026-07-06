#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Collect classification accuracies from a folder of per-job .log files.
#
# Each JEPAVTS/baseline classification run prints its final test accuracy once
# as "accuracy: <fraction>" (0..1). The per-job log file name encodes the run
# as <dataset>_<RENDER>_<single|dual>.log, so we key on the file name.
#
# Usage:
#   bash scripts/collect_cls_acc.sh <log_dir> [<log_dir> ...]
#   TSV=1 bash scripts/collect_cls_acc.sh <log_dir> [...]   # spreadsheet paste mode
#
# Examples:
#   bash scripts/collect_cls_acc.sh logs/jepavts_crossformer_cls
#   TSV=1 bash scripts/collect_cls_acc.sh logs/jepavts_crossformer_cls
#
# Default output: one line per run "<name>  <acc%>", then per-folder AVG single/dual.
# TSV=1 output:  tab-separated "<dataset>\t<acc%>" blocks (one per single/dual
#                config), datasets alphabetical, ending with "Average Accuracy" —
#                ready to paste straight into a spreadsheet column.
# Runs with no "accuracy:" line (crashed / still training) are listed as MISSING.
# ═══════════════════════════════════════════════════════════════════════════

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/collect_cls_acc.sh <log_dir> [<log_dir> ...]" >&2
  exit 1
fi

for dir in "$@"; do
  if [ ! -d "$dir" ]; then
    echo "!! not a directory: $dir" >&2
    continue
  fi

  # Any logs present?
  shopt -s nullglob
  logs=("$dir"/*.log)
  shopt -u nullglob
  echo "══════════════════════════════════════════════════════════════"
  echo "  $dir"
  echo "══════════════════════════════════════════════════════════════"
  if [ "${#logs[@]}" -eq 0 ]; then
    echo "  (no .log files)"
    echo
    continue
  fi

  # Collect "<name> <acc|MISSING>" records once, reuse for either output mode.
  records=$(
    for f in "${logs[@]}"; do
      b=$(basename "$f" .log)
      acc=$(grep 'accuracy:' "$f" | tail -1 | sed 's/.*accuracy: *//')
      [ -n "$acc" ] && echo "$b $acc" || echo "$b MISSING"
    done | sort
  )

  if [ "$TSV" = "1" ]; then
    # Spreadsheet paste mode: one block per config (single/dual), tab-separated
    # "<dataset>\t<acc%>", datasets alphabetical, then "Average Accuracy".
    # Strip the trailing _<render>_<single|dual> to recover the bare dataset name
    # (UEA dataset names have no underscores).
    for cfg in single dual; do
      block=$(echo "$records" | awk -v cfg="_${cfg}$" '$1 ~ cfg')
      [ -z "$block" ] && continue
      echo "# $cfg"
      echo "$block" | awk '
        { name=$1; sub(/_[^_]*_[^_]*$/, "", name)
          if ($2=="MISSING") { printf "%s\t%s\n", name, "MISSING"; next }
          printf "%s\t%.2f\n", name, $2*100; s+=$2; n++ }
        END { if (n) printf "Average Accuracy\t%.3f\n", s/n*100 }'
      echo
    done
  else
    echo "$records" | awk '
      { if ($2=="MISSING") { printf "%-48s %8s\n", $1, "MISSING"; next }
        printf "%-48s %8.2f\n", $1, $2*100
        if ($1 ~ /_single$/) { ss+=$2; sn++ } else if ($1 ~ /_dual$/) { ds+=$2; dn++ }
        total+=$2; tn++ }
      END {
        print  "  ------------------------------------------------------------"
        if (sn) printf "%-48s %8.3f  (%d runs)\n", "AVG single", ss/sn*100, sn
        if (dn) printf "%-48s %8.3f  (%d runs)\n", "AVG dual",   ds/dn*100, dn
        if (tn) printf "%-48s %8.3f  (%d runs)\n", "AVG all",    total/tn*100, tn
      }'
    echo
  fi
done
