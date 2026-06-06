#!/usr/bin/env bash
# benchmark_bedrock.sh — Run AWS Bedrock backends (Claude Sonnet + Mistral 7B Bedrock)
# Run on any machine with AWS access and AWS_PROFILE=containers configured.
#
# Output: results/benchmark_raw_bedrock.jsonl
#         results/benchmark_report_bedrock.md (partial — Bedrock backends only)
#
# Then copy results/benchmark_raw_bedrock.jsonl to the Ollama machine and run
# benchmark_ollama.sh, followed by benchmark_combine.sh for the final report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="results"
RAW_FILE="$RESULTS_DIR/benchmark_raw_bedrock.jsonl"

echo "============================================================"
echo " EIA Benchmark — Bedrock Backends"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Verify AWS access before starting
echo "Checking AWS access..."
if ! aws sts get-caller-identity --profile containers > /dev/null 2>&1; then
    echo "ERROR: AWS access failed. Check AWS_PROFILE=containers and credentials."
    exit 1
fi
echo "AWS access OK."
echo ""

# Ensure results dir exists and clear any previous Bedrock run
mkdir -p "$RESULTS_DIR"
> "$RAW_FILE"
echo "Cleared $RAW_FILE"
echo ""

# --- Backend 1: Claude Sonnet (Bedrock) ---
echo "------------------------------------------------------------"
echo " Running: Claude Sonnet (Bedrock)"
echo "------------------------------------------------------------"
AWS_PROFILE=containers \
AWS_REGION=eu-west-2 \
    python -m stream1_log_anomaly.benchmark_runner \
    --explainer bedrock \
    --raw-output "$RAW_FILE"
echo ""

# --- Backend 2: Mistral 7B (Bedrock) ---
echo "------------------------------------------------------------"
echo " Running: Mistral 7B (Bedrock)"
echo "------------------------------------------------------------"
AWS_PROFILE=containers \
BEDROCK_REGION=eu-west-2 \
    python -m stream1_log_anomaly.benchmark_runner \
    --explainer mistral-bedrock \
    --raw-output "$RAW_FILE"
echo ""

# --- Partial report (Bedrock backends only) ---
echo "------------------------------------------------------------"
echo " Generating partial report (Bedrock backends)"
echo "------------------------------------------------------------"
python -m stream1_log_anomaly.benchmark_runner \
    --report \
    --raw-input "$RAW_FILE" \
    --report-output "$RESULTS_DIR/benchmark_report_bedrock.md"

echo ""
echo "============================================================"
echo " Bedrock run complete."
echo " Raw results:    $RAW_FILE"
echo " Partial report: $RESULTS_DIR/benchmark_report_bedrock.md"
echo ""
echo " Next step — copy to Ollama machine:"
echo "   scp $RAW_FILE <ollama-machine>:~/path/to/EIA/results/"
echo " Then run: scripts/benchmark_ollama.sh"
echo "============================================================"