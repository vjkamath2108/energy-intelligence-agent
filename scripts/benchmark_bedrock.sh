#!/usr/bin/env bash
# benchmark_bedrock.sh — Run AWS Bedrock backends (Claude Sonnet + Mistral 7B Bedrock)
# Runs independently — no dependency on benchmark_ollama.sh.
#
# Output: results/benchmark_raw_bedrock.jsonl
#         results/benchmark_report_bedrock.md (partial — Bedrock backends only)
#
# Workflow:
#   1. Run this script on the Bedrock machine
#   2. Run benchmark_ollama.sh on the Ollama machine (independently)
#   3. Bring both raw files to one machine
#   4. Run benchmark_combine.sh for the final report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# Configuration — change here only, not below
# ============================================================
AWS_PROFILE="${AWS_PROFILE:-bedrock}"
AWS_REGION="${AWS_REGION:-eu-west-2}"
RESULTS_DIR="results"
RAW_FILE="$RESULTS_DIR/benchmark_raw_bedrock.jsonl"
REPORT_FILE="$RESULTS_DIR/benchmark_report_bedrock.md"
# ============================================================

export AWS_PROFILE
export AWS_REGION

echo "============================================================"
echo " EIA Benchmark — Bedrock Backends"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " AWS_PROFILE : $AWS_PROFILE"
echo " AWS_REGION  : $AWS_REGION"
echo "============================================================"
echo ""

# Verify AWS access before starting
echo "Checking AWS access..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "ERROR: AWS access failed."
    echo "Try: aws sso login --profile $AWS_PROFILE"
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
python -m stream1_log_anomaly.benchmark_runner \
    --explainer bedrock \
    --raw-output "$RAW_FILE"
echo ""

# --- Backend 2: Mistral 7B (Bedrock) ---
echo "------------------------------------------------------------"
echo " Running: Mistral 7B (Bedrock)"
echo "------------------------------------------------------------"
BEDROCK_REGION="$AWS_REGION" \
    python -m stream1_log_anomaly.benchmark_runner \
    --explainer mistral-bedrock \
    --raw-output "$RAW_FILE"
echo ""

# --- Partial report (Bedrock backends only) ---
echo "------------------------------------------------------------"
echo " Generating partial report (Bedrock backends only)"
echo "------------------------------------------------------------"
python -m stream1_log_anomaly.benchmark_runner \
    --report \
    --raw-input "$RAW_FILE" \
    --report-output "$REPORT_FILE"

echo ""
echo "============================================================"
echo " Bedrock run complete."
echo " Raw results:    $RAW_FILE"
echo " Partial report: $REPORT_FILE"
echo ""
echo " Git add and push this file, then run benchmark_combine.sh"
echo " once benchmark_raw_ollama.jsonl is also present."
echo "============================================================"