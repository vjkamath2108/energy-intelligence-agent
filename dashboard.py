"""EIA metrics dashboard — RAG pipeline + Anomaly feed.

Two-tab Streamlit dashboard covering both EIA pipelines:

  Tab 1 — RAG Pipeline
    P50/P95 dispatch latency trend
    Confidence distribution
    Citation coverage %
    Failure rate (requires_human_review)
    Hallucination detection rate
    Live recommendation feed

  Tab 2 — Anomaly Feed
    Mirrors stream1_log_anomaly/dashboard.py:
    Severity breakdown, explanation latency trend, live anomaly feed.

Data sources:
  results/rag_log.jsonl      — one record per dispatch() call
  results/anomaly_log.jsonl  — one record per detected anomaly + explanation

Run with:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RAG_LOG = Path("results/rag_log.jsonl")
ANOMALY_LOG = Path("results/anomaly_log.jsonl")

st.set_page_config(
    page_title="EIA Metrics Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Intelligence Agent — Metrics Dashboard")
st.caption("RAG dispatch pipeline · Real-time anomaly detection · Bedrock + Ollama backends")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame. Returns empty DataFrame on missing/empty file."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    return df.sort_values("logged_at", ascending=False)


rag_df = _load_jsonl(RAG_LOG)
anomaly_df = _load_jsonl(ANOMALY_LOG)

tab_rag, tab_anomaly = st.tabs(["RAG Pipeline", "Anomaly Feed"])


# ===========================================================================
# Tab 1 — RAG Pipeline
# ===========================================================================

with tab_rag:
    if rag_df.empty:
        st.warning(
            "No RAG dispatch results logged yet.  "
            "Run `python -c \"import asyncio; from src.agent.energy_agent import "
            "EnergyIntelligenceAgent; asyncio.run(EnergyIntelligenceAgent().dispatch('test'))\"` "
            "or use the demo scripts."
        )
        st.stop()

    # ── Key metrics ──────────────────────────────────────────────────────────

    total = len(rag_df)
    p50_lat = rag_df["latency_ms"].quantile(0.50)
    p95_lat = rag_df["latency_ms"].quantile(0.95)
    failure_rate = rag_df["requires_human_review"].mean() * 100
    halluc_rate = rag_df["hallucination_flagged"].mean() * 100
    avg_coverage = rag_df["citation_coverage"].mean() * 100
    avg_confidence = rag_df["confidence"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Dispatches", total)
    c2.metric("P50 Latency", f"{p50_lat:.0f}ms")
    c3.metric("P95 Latency", f"{p95_lat:.0f}ms", delta=f"gate <500ms" if p95_lat < 500 else "⚠ >500ms gate")
    c4.metric("Avg Confidence", f"{avg_confidence:.2f}")
    c5.metric("Citation Coverage", f"{avg_coverage:.0f}%")
    c6.metric("Failure Rate", f"{failure_rate:.0f}%")

    st.divider()

    # ── Latency trend ────────────────────────────────────────────────────────

    st.subheader("Dispatch Latency Trend (ms)")

    lat_df = rag_df[["logged_at", "latency_ms", "backend"]].copy().sort_values("logged_at")
    lat_df = lat_df.set_index("logged_at")

    # Rolling P50 / P95 per backend
    col_lat, col_backend = st.columns([2, 1])
    with col_lat:
        st.line_chart(lat_df["latency_ms"])
        st.caption("P95 gate: 500ms  |  P50 target: <200ms")

    with col_backend:
        st.markdown("**Backend breakdown**")
        backend_stats = (
            rag_df.groupby("backend")["latency_ms"]
            .agg(["count", "median", lambda q: q.quantile(0.95)])
            .rename(columns={"count": "calls", "median": "P50 (ms)", "<lambda_0>": "P95 (ms)"})
            .round(0)
        )
        st.dataframe(backend_stats, use_container_width=True)

    st.divider()

    # ── Confidence + Citation coverage ───────────────────────────────────────

    st.subheader("Quality Metrics")

    qual_l, qual_r = st.columns(2)

    with qual_l:
        st.markdown("**Confidence distribution**")
        conf_bins = pd.cut(
            rag_df["confidence"],
            bins=[0, 0.40, 0.60, 0.80, 1.01],
            labels=["<0.40 (poor)", "0.40–0.60 (low)", "0.60–0.80 (ok)", "0.80–1.00 (good)"],
        )
        st.bar_chart(conf_bins.value_counts().sort_index())

    with qual_r:
        st.markdown("**Citation coverage % over time**")
        cov_df = rag_df[["logged_at", "citation_coverage"]].copy().sort_values("logged_at")
        cov_df["citation_coverage"] = (cov_df["citation_coverage"] * 100).round(1)
        cov_df = cov_df.set_index("logged_at")
        st.line_chart(cov_df["citation_coverage"])
        st.caption("Target: ≥90%  |  Citation coverage = cited docs / retrieved docs")

    st.divider()

    # ── Safety metrics ───────────────────────────────────────────────────────

    st.subheader("Safety Signals")

    safe_l, safe_r = st.columns(2)

    with safe_l:
        safety_counts = pd.Series(
            {
                "Clean": int((~rag_df["requires_human_review"]).sum()),
                "Human review": int(rag_df["requires_human_review"].sum()),
                "Hallucination": int(rag_df["hallucination_flagged"].sum()),
            }
        )
        st.bar_chart(safety_counts)
        st.caption("Hallucination: term overlap < 30%  |  Human review: confidence < 0.60 or safety flag")

    with safe_r:
        st.markdown("**Failure rate over time**")
        fail_df = (
            rag_df[["logged_at", "requires_human_review"]]
            .copy()
            .sort_values("logged_at")
            .set_index("logged_at")
        )
        fail_df["requires_human_review"] = fail_df["requires_human_review"].astype(int)
        st.line_chart(fail_df["requires_human_review"])
        st.caption("1 = flagged for human review, 0 = clean")

    st.divider()

    # ── Live recommendation feed ─────────────────────────────────────────────

    st.subheader("Live Recommendation Feed (last 20)")

    feed_cols = [
        "logged_at", "backend", "latency_ms", "confidence",
        "citation_coverage", "hallucination_flagged", "requires_human_review", "action",
    ]
    st.dataframe(
        rag_df[feed_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "logged_at": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "backend": "Backend",
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "citation_coverage": st.column_config.NumberColumn("Citation %", format="%.0f%%"),
            "hallucination_flagged": st.column_config.CheckboxColumn("Halluc."),
            "requires_human_review": st.column_config.CheckboxColumn("Human Review"),
            "action": st.column_config.TextColumn("Action", width="large"),
        },
    )

    st.caption("Refresh page to reload latest records from results/rag_log.jsonl")


# ===========================================================================
# Tab 2 — Anomaly Feed (extends stream1_log_anomaly/dashboard.py)
# ===========================================================================

with tab_anomaly:
    if anomaly_df.empty:
        st.warning(
            "No anomalies logged yet. Run the streaming pipeline to generate data:\n"
            "`python main.py --mode file --explainer bedrock`"
        )
        st.stop()

    anomaly_df["explanation_latency_ms"] = anomaly_df["explanation_latency_ms"].round(0)

    # ── Anomaly key metrics ──────────────────────────────────────────────────

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Anomalies", len(anomaly_df))
    a2.metric("HIGH Severity", int((anomaly_df["severity"] == "HIGH").sum()))
    a3.metric("Avg Explanation Latency", f"{anomaly_df['explanation_latency_ms'].mean():.0f}ms")
    a4.metric("Backends Used", anomaly_df["backend"].nunique())

    st.divider()

    # ── Backend breakdown ────────────────────────────────────────────────────

    st.subheader("Backend Breakdown")
    b_l, b_r = st.columns(2)

    backend_counts = anomaly_df["backend"].value_counts()
    b_l.dataframe(
        backend_counts.rename("calls").reset_index().rename(columns={"index": "backend"}),
        use_container_width=True,
        hide_index=True,
    )

    backend_latency = anomaly_df.groupby("backend")["explanation_latency_ms"].mean().round(0)
    b_r.dataframe(
        backend_latency.rename("avg_latency_ms").reset_index().rename(columns={"index": "backend"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Severity breakdown ───────────────────────────────────────────────────

    st.subheader("Severity Breakdown")
    st.bar_chart(anomaly_df["severity"].value_counts())

    st.divider()

    # ── Explanation latency trend ────────────────────────────────────────────

    st.subheader("Explanation Latency Trend (ms)")
    lat_trend = (
        anomaly_df[["logged_at", "explanation_latency_ms"]]
        .sort_values("logged_at")
        .set_index("logged_at")
    )
    st.line_chart(lat_trend["explanation_latency_ms"])

    st.divider()

    # ── Live anomaly feed ────────────────────────────────────────────────────

    st.subheader("Live Anomaly Feed (last 20)")
    anomaly_feed_cols = [
        "logged_at", "service", "level", "event_latency_ms",
        "severity", "backend", "explanation_latency_ms", "summary",
    ]
    st.dataframe(
        anomaly_df[anomaly_feed_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "logged_at": st.column_config.DatetimeColumn("Logged At", format="HH:mm:ss"),
            "service": "Service",
            "level": "Level",
            "event_latency_ms": st.column_config.NumberColumn("Event Latency (ms)"),
            "severity": "Severity",
            "backend": "Backend",
            "explanation_latency_ms": st.column_config.NumberColumn("Explanation Latency (ms)"),
            "summary": st.column_config.TextColumn("Summary", width="large"),
        },
    )

    st.divider()

    # ── Recommended actions ──────────────────────────────────────────────────

    st.subheader("Recommended Actions (most recent 5)")
    for _, row in anomaly_df.head(5).iterrows():
        with st.expander(
            f"{row['service']} — {row['severity']} — {row['logged_at'].strftime('%H:%M:%S')}"
        ):
            st.markdown(f"**Summary:** {row['summary']}")
            st.markdown(f"**Likely Cause:** {row['likely_cause']}")
            st.markdown(f"**Recommended Action:** {row['recommended_action']}")
            st.caption(f"Backend: {row['backend']} | Latency: {row['explanation_latency_ms']:.0f}ms")

    st.caption("Refresh page to reload latest anomalies from results/anomaly_log.jsonl")
