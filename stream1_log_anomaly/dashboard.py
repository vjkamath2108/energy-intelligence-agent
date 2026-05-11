"""
Streamlit dashboard — Live Anomaly Feed + Metrics.

Reads from results/anomaly_log.jsonl and displays:
- Key metrics: total anomalies, HIGH severity count, avg explanation latency
- Backend breakdown: Bedrock vs Ollama call counts
- Live anomaly feed: most recent 20 anomalies
- Latency trend chart

Run with:
    streamlit run stream1_log_anomaly/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path("stream1_log_anomaly/results/anomaly_log.jsonl")

st.set_page_config(
    page_title="Log Anomaly Monitor",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Log Anomaly Monitor")
st.caption("Stream 1 — Real-Time Log Anomaly Pipeline | Isolation Forest + LLM Explanation")
# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return pd.DataFrame()

    records = []
    with LOG_PATH.open() as f:
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
    df["explanation_latency_ms"] = df["explanation_latency_ms"].round(0)
    return df.sort_values("logged_at", ascending=False)


df = load_data()

if df.empty:
    st.warning("No anomalies logged yet. Run the pipeline to generate data.")
    st.stop()

# ── Key metrics ───────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Anomalies",
    len(df),
)
col2.metric(
    "HIGH Severity",
    len(df[df["severity"] == "HIGH"]),
)
col3.metric(
    "Avg Explanation Latency",
    f"{df['explanation_latency_ms'].mean():.0f}ms",
)
col4.metric(
    "Backends Used",
    df["backend"].nunique(),
)

st.divider()

# ── Backend breakdown ─────────────────────────────────────────────────────────

st.subheader("Backend Breakdown")

backend_col1, backend_col2 = st.columns(2)

backend_counts = df["backend"].value_counts()
backend_col1.dataframe(
    backend_counts.rename("calls").reset_index().rename(columns={"index": "backend"}),
    use_container_width=True,
    hide_index=True,
)

backend_latency = df.groupby("backend")["explanation_latency_ms"].mean().round(0)
backend_col2.dataframe(
    backend_latency.rename("avg_latency_ms").reset_index().rename(columns={"index": "backend"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Severity breakdown ────────────────────────────────────────────────────────

st.subheader("Severity Breakdown")

severity_counts = df["severity"].value_counts()
st.bar_chart(severity_counts)

st.divider()

# ── Latency trend ─────────────────────────────────────────────────────────────

st.subheader("Explanation Latency Trend (ms)")

latency_df = df[["logged_at", "explanation_latency_ms", "backend"]].copy()
latency_df = latency_df.sort_values("logged_at")
latency_df = latency_df.set_index("logged_at")

st.line_chart(latency_df["explanation_latency_ms"])

st.divider()

# ── Live anomaly feed ─────────────────────────────────────────────────────────

st.subheader("Live Anomaly Feed (last 20)")

feed_cols = [
    "logged_at",
    "service",
    "level",
    "event_latency_ms",
    "severity",
    "backend",
    "explanation_latency_ms",
    "summary",
]

st.dataframe(
    df[feed_cols].head(20),
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

# ── Recommended actions ───────────────────────────────────────────────────────

st.subheader("Recommended Actions (most recent)")

for _, row in df.head(5).iterrows():
    with st.expander(f"{row['service']} — {row['severity']} — {row['logged_at'].strftime('%H:%M:%S')}"):
        st.markdown(f"**Summary:** {row['summary']}")
        st.markdown(f"**Likely Cause:** {row['likely_cause']}")
        st.markdown(f"**Recommended Action:** {row['recommended_action']}")
        st.caption(f"Backend: {row['backend']} | Explanation latency: {row['explanation_latency_ms']:.0f}ms")

# ── Auto-refresh ──────────────────────────────────────────────────────────────

st.divider()
st.caption("💡 Refresh the page to load latest anomalies from the log file.")