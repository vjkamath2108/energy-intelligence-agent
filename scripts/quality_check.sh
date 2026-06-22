#!/usr/bin/env bash
# W6.4 quality gate: mypy --strict, ruff, coverage ≥80%
# Run from repo root: bash scripts/quality_check.sh

set -euo pipefail

VENV=.venv/bin/python
RUFF=.venv/bin/ruff
PYTEST="PYTHONPATH=. $VENV -m pytest"

PASS=0
FAIL=0

_ok()   { echo "  ✓  $1"; PASS=$((PASS+1)); }
_fail() { echo "  ✗  $1"; FAIL=$((FAIL+1)); }

echo "═══════════════════════════════════════"
echo " EIA Quality Gate — W6.4"
echo "═══════════════════════════════════════"

# ── 1. mypy --strict ────────────────────────────────────────────────────────
echo
echo "1/3  mypy --strict src/"
if $VENV -m mypy --strict src/ --no-error-summary 2>&1; then
    _ok "mypy: 0 errors"
else
    _fail "mypy: errors found (see above)"
fi

# ── 2. ruff ─────────────────────────────────────────────────────────────────
echo
echo "2/3  ruff check + format check"
if $RUFF check src/ && $RUFF format --check src/ 2>&1; then
    _ok "ruff: clean"
else
    _fail "ruff: issues found (see above)"
fi

# ── 3. pytest + coverage ────────────────────────────────────────────────────
echo
echo "3/3  pytest with coverage ≥80%"
IGNORE="--ignore=tests/test_db.py --ignore=tests/test_bm25.py"
IGNORE="$IGNORE --ignore=tests/test_hybrid_retriever.py --ignore=tests/test_embeddings.py"

COV_OUTPUT=$(eval "$PYTEST tests/ $IGNORE --cov --cov-report=term-missing -q 2>&1")
echo "$COV_OUTPUT" | grep -E "^TOTAL|^[0-9]+ passed"

TOTAL_LINE=$(echo "$COV_OUTPUT" | grep "^TOTAL")
if [ -n "$TOTAL_LINE" ]; then
    COV_PCT=$(echo "$TOTAL_LINE" | awk '{print $NF}' | tr -d '%')
    if [ "$COV_PCT" -ge 80 ] 2>/dev/null; then
        _ok "coverage: ${COV_PCT}% (gate ≥80%)"
    else
        _fail "coverage: ${COV_PCT}% is below 80% gate"
    fi
else
    _fail "coverage: could not parse report"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════"
echo " PASSED: $PASS / $((PASS+FAIL))  |  FAILED: $FAIL"
echo "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
