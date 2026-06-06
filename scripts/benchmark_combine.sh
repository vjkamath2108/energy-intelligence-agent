#!/usr/bin/env bash
# benchmark_combine.sh — Merge Bedrock + Ollama raw results and generate
# the final benchmark report.
#
# Run on either machine once both raw files are present in results/:
#   results/benchmark_raw_bedrock.jsonl  (from benchmark_bedrock.sh)
#   results/benchmark_raw_ollama.jsonl   (from benchmark_ollama.sh)
#
# Output: results/benchmark_raw.jsonl (merged)
#         results/bedrock_vs_ollama_vs_mistral.md (final report)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="results"
BEDROCK_RAW="$RESULTS_DIR/benchmark_raw_bedrock.jsonl"
OLLAMA_RAW="$RESULTS_DIR/benchmark_raw_ollama.jsonl"
MERGED_RAW="$RESULTS_DIR/benchmark_raw.jsonl"

echo "============================================================"
echo " EIA Benchmark — Combining Results"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Check both input files exist
if [[ ! -f "$BEDROCK_RAW" ]]; then
    echo "ERROR: $BEDROCK_RAW not found. Run scripts/benchmark_bedrock.sh first."
    exit 1
fi

if [[ ! -f "$OLLAMA_RAW" ]]; then
    echo "ERROR: $OLLAMA_RAW not found. Run scripts/benchmark_ollama.sh first."
    exit 1
fi

# Merge — Bedrock first (display order), Ollama second
echo "Merging results..."
cat "$BEDROCK_RAW" "$OLLAMA_RAW" > "$MERGED_RAW"
echo "Merged into $MERGED_RAW"
echo ""

# Verify event counts per backend
echo "Event counts per backend:"
python3 -c "
import json
from collections import Counter
from pathlib import Path

lines = Path('$MERGED_RAW').read_text().splitlines()
counts = Counter(
    json.loads(l)['backend']
    for l in lines if l.strip()
)
first_count = list(counts.values())[0] if counts else 0
for backend, count in sorted(counts.items()):
    status = '✓' if count == first_count else '⚠  MISMATCH'
    print(f'  {backend:<35} {count:>4} events  {status}')

all_counts = list(counts.values())
if len(set(all_counts)) == 1:
    print()
    print(f'  All backends: {all_counts[0]} events each — consistent ✓')
else:
    print()
    print('  WARNING: Event counts differ across backends.')
    print('  Clear raw files and re-run to fix:')
    print('    > results/benchmark_raw_bedrock.jsonl')
    print('    > results/benchmark_raw_ollama.jsonl')
    print('    scripts/benchmark_bedrock.sh')
    print('    scripts/benchmark_ollama.sh')
    print('    scripts/benchmark_combine.sh')
"

echo ""
echo "------------------------------------------------------------"
echo " Generating final report"
echo "------------------------------------------------------------"
python -m stream1_log_anomaly.benchmark_runner \
    --report \
    --raw-input "$MERGED_RAW" \
    --report-output "$RESULTS_DIR/bedrock_vs_ollama_vs_mistral.md"

echo ""
echo "============================================================"
echo " Final report: $RESULTS_DIR/bedrock_vs_ollama_vs_mistral.md"
echo "============================================================"