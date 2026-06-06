#!/usr/bin/env bash
# benchmark_combine.sh — Merge Bedrock + Ollama raw results into final report.
# Run on either machine once both raw files are present in results/.
#
# Inputs  (produced independently):
#   results/benchmark_raw_bedrock.jsonl  — from benchmark_bedrock.sh
#   results/benchmark_raw_ollama.jsonl   — from benchmark_ollama.sh
#
# Output:
#   results/benchmark_raw.jsonl                  — merged, deduplicated
#   results/bedrock_vs_ollama_vs_mistral.md      — final report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# Configuration — change here only, not below
# ============================================================
RESULTS_DIR="results"
BEDROCK_RAW="$RESULTS_DIR/benchmark_raw_bedrock.jsonl"
OLLAMA_RAW="$RESULTS_DIR/benchmark_raw_ollama.jsonl"
MERGED_RAW="$RESULTS_DIR/benchmark_raw.jsonl"
FINAL_REPORT="$RESULTS_DIR/bedrock_vs_ollama_vs_mistral.md"
# ============================================================

echo "============================================================"
echo " EIA Benchmark — Combining Results"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Check both input files exist
if [[ ! -f "$BEDROCK_RAW" ]]; then
    echo "ERROR: $BEDROCK_RAW not found."
    echo "Run scripts/benchmark_bedrock.sh and copy the output here."
    exit 1
fi

if [[ ! -f "$OLLAMA_RAW" ]]; then
    echo "ERROR: $OLLAMA_RAW not found."
    echo "Run scripts/benchmark_ollama.sh and copy the output here."
    exit 1
fi

# Merge with deduplication — last write per (backend, event_index) wins
# Protects against duplicate rows if either script was run more than once
echo "Merging and deduplicating results..."
python3 -c "
import json
from pathlib import Path

bedrock = Path('$BEDROCK_RAW').read_text().splitlines()
ollama  = Path('$OLLAMA_RAW').read_text().splitlines()

# Deduplicate by (backend, event_index) — last occurrence wins
seen = {}
for line in bedrock + ollama:
    if not line.strip():
        continue
    r = json.loads(line)
    key = (r['backend'], r['event_index'])
    seen[key] = r

# Write in display order: bedrock backends first, ollama second
display_order = ['bedrock', 'mistral-bedrock', 'mistral-local',
                 'ollama:llama3.2:3b', 'ollama:phi4-mini', 'ollama:qwen3:14b']

def sort_key(r):
    backend = r['backend']
    order = display_order.index(backend) if backend in display_order else 99
    return (order, r['event_index'])

rows = sorted(seen.values(), key=sort_key)

Path('$MERGED_RAW').write_text(
    '\n'.join(json.dumps(r) for r in rows) + '\n'
)
print(f'  Merged {len(rows)} unique rows into $MERGED_RAW')
"
echo ""

# Sanity check — verify event counts and catch zero-result backends
echo "Event counts per backend:"
python3 -c "
import json, sys
from collections import Counter
from pathlib import Path

lines = Path('$MERGED_RAW').read_text().splitlines()
counts = Counter(
    json.loads(l)['backend']
    for l in lines if l.strip()
)

# Use majority count as the expected value
from statistics import mode
try:
    expected = mode(counts.values())
except Exception:
    expected = list(counts.values())[0] if counts else 0

all_ok = True
for backend, count in sorted(counts.items()):
    if count == 0:
        status = '✗ ZERO RESULTS — re-run this backend'
        all_ok = False
    elif count != expected:
        status = f'⚠  MISMATCH (expected {expected})'
        all_ok = False
    else:
        status = '✓'
    print(f'  {backend:<35} {count:>4} events  {status}')

print()
if all_ok:
    print(f'  All backends: {expected} events each — consistent ✓')
else:
    print('  Fix mismatches before generating the final report.')
    print('  Clear the affected raw file and re-run that script.')
    sys.exit(1)
"

echo ""
echo "------------------------------------------------------------"
echo " Generating final report"
echo "------------------------------------------------------------"
python -m stream1_log_anomaly.benchmark_runner \
    --report \
    --raw-input "$MERGED_RAW" \
    --report-output "$FINAL_REPORT"

echo ""
echo "============================================================"
echo " Final report: $FINAL_REPORT"
echo "============================================================"