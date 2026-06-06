#!/usr/bin/env bash
# benchmark_ollama.sh — Run Ollama backends (Mistral local + other models)
# Runs independently — no dependency on benchmark_bedrock.sh.
#
# Prerequisites:
#   - Ollama running: ollama serve
#   - Models pulled: ollama pull mistral && ollama pull llama3.2:3b && ollama pull phi4-mini
#
# Output: results/benchmark_raw_ollama.jsonl
#         results/benchmark_report_ollama.md (partial — Ollama backends only)
#
# Workflow:
#   1. Run benchmark_bedrock.sh on the Bedrock machine (independently)
#   2. Run this script on the Ollama machine (independently)
#   3. Bring both raw files to one machine
#   4. Run benchmark_combine.sh for the final report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# Configuration — change here only, not below
# ============================================================
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S:-120}"
RESULTS_DIR="results"
RAW_FILE="$RESULTS_DIR/benchmark_raw_ollama.jsonl"
REPORT_FILE="$RESULTS_DIR/benchmark_report_ollama.md"
REQUIRED_MODELS=("mistral" "llama3.2:3b" "phi4-mini")
# ============================================================

export OLLAMA_TIMEOUT_S

echo "============================================================"
echo " EIA Benchmark — Ollama Backends"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " OLLAMA_BASE_URL : $OLLAMA_BASE_URL"
echo " OLLAMA_TIMEOUT_S: $OLLAMA_TIMEOUT_S"
echo "============================================================"
echo ""

# Check Ollama is running
echo "Checking Ollama..."
if ! curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama not running at $OLLAMA_BASE_URL"
    echo "Start it with: ollama serve"
    exit 1
fi
echo "Ollama OK."
echo ""

# Check required models are pulled — pull if missing
echo "Checking models..."
for model in "${REQUIRED_MODELS[@]}"; do
    if ! ollama list 2>/dev/null | grep -q "^${model}"; then
        echo "  '$model' not found — pulling now..."
        ollama pull "$model"
    else
        echo "  $model OK"
    fi
done
echo ""

# Clear previous Ollama run
mkdir -p "$RESULTS_DIR"
> "$RAW_FILE"
echo "Cleared $RAW_FILE"
echo ""

# --- Backend 1: Mistral 7B (Sovereign/Ollama) ---
echo "------------------------------------------------------------"
echo " Running: Mistral 7B (Sovereign/Ollama)"
echo " Note: ~3-4s per event, ~3min total"
echo "------------------------------------------------------------"
OLLAMA_MODEL=mistral:latest \
    python -m stream1_log_anomaly.benchmark_runner \
    --explainer mistral-local \
    --raw-output "$RAW_FILE"
echo ""

# --- Backend 2: llama3.2:3b ---
echo "------------------------------------------------------------"
echo " Running: llama3.2:3b (Ollama)"
echo "------------------------------------------------------------"
OLLAMA_MODEL=llama3.2:3b \
    python -m stream1_log_anomaly.benchmark_runner \
    --explainer ollama \
    --raw-output "$RAW_FILE"
echo ""

# --- Backend 3: phi4-mini ---
echo "------------------------------------------------------------"
echo " Running: phi4-mini (Ollama)"
echo "------------------------------------------------------------"
OLLAMA_MODEL=phi4-mini \
    python -m stream1_log_anomaly.benchmark_runner \
    --explainer ollama \
    --raw-output "$RAW_FILE"
echo ""

# --- Partial report (Ollama backends only) ---
echo "------------------------------------------------------------"
echo " Generating partial report (Ollama backends only)"
echo "------------------------------------------------------------"
python -m stream1_log_anomaly.benchmark_runner \
    --report \
    --raw-input "$RAW_FILE" \
    --report-output "$REPORT_FILE"

echo ""
echo "============================================================"
echo " Ollama run complete."
echo " Raw results:    $RAW_FILE"
echo " Partial report: $REPORT_FILE"
echo ""
echo " Git add and push this file, then run benchmark_combine.sh"
echo " once benchmark_raw_bedrock.jsonl is also present."
echo "============================================================"