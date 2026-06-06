#!/usr/bin/env bash
# benchmark_ollama.sh — Run Ollama backends (Mistral local + other models)
# Run on any machine with Ollama installed, after copying
# benchmark_raw_bedrock.jsonl here from the Bedrock machine.
#
# Prerequisites:
#   - results/benchmark_raw_bedrock.jsonl present (copy from Bedrock machine)
#   - Ollama running: ollama serve
#   - Models pulled: ollama pull mistral && ollama pull llama3.2:3b && ollama pull phi4-mini
#
# Output: results/benchmark_raw_ollama.jsonl
#         results/benchmark_report_ollama.md (partial — Ollama backends only)
#
# Then run benchmark_combine.sh for the final combined report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="results"
RAW_FILE="$RESULTS_DIR/benchmark_raw_ollama.jsonl"
BEDROCK_RAW="$RESULTS_DIR/benchmark_raw_bedrock.jsonl"

echo "============================================================"
echo " EIA Benchmark — Ollama Backends"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Check Bedrock results are present
if [[ ! -f "$BEDROCK_RAW" ]]; then
    echo "ERROR: $BEDROCK_RAW not found."
    echo "Copy it from the Bedrock machine first:"
    echo "  scp <bedrock-machine>:~/path/to/EIA/results/benchmark_raw_bedrock.jsonl results/"
    exit 1
fi
echo "Bedrock raw results found: $BEDROCK_RAW"
echo ""

# Check Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama not running. Start it with: ollama serve"
    exit 1
fi
echo "Ollama OK."
echo ""

# Check required models are pulled
echo "Checking models..."
REQUIRED_MODELS=("mistral" "llama3.2:3b" "phi4-mini")
for model in "${REQUIRED_MODELS[@]}"; do
    if ! ollama list 2>/dev/null | grep -q "^${model}"; then
        echo "WARNING: Model '$model' not found. Pulling now..."
        ollama pull "$model"
    else
        echo "  $model OK"
    fi
done
echo ""

# Clear previous Ollama run
> "$RAW_FILE"
echo "Cleared $RAW_FILE"
echo ""

# --- Backend 1: Mistral 7B (Sovereign/Ollama) ---
echo "------------------------------------------------------------"
echo " Running: Mistral 7B (Sovereign/Ollama)"
echo " Note: ~3-4s per event, ~3min total"
echo "------------------------------------------------------------"
OLLAMA_MODEL=mistral:latest \
OLLAMA_TIMEOUT_S=120 \
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
echo " Generating partial report (Ollama backends)"
echo "------------------------------------------------------------"
python -m stream1_log_anomaly.benchmark_runner \
    --report \
    --raw-input "$RAW_FILE" \
    --report-output "$RESULTS_DIR/benchmark_report_ollama.md"

echo ""
echo "============================================================"
echo " Ollama run complete."
echo " Raw results:    $RAW_FILE"
echo " Partial report: $RESULTS_DIR/benchmark_report_ollama.md"
echo ""
echo " Next step — generate final combined report:"
echo "   scripts/benchmark_combine.sh"
echo "============================================================"