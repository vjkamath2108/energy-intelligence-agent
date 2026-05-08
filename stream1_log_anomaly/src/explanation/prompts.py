"""
Shared prompt templates for anomaly explanation.

Both BedrockExplainer and OllamaExplainer import from here.
Prompt logic lives in one place — changes apply to both backends.
"""

import json
from stream1_log_anomaly.src.models import LogEvent


def build_prompt(
    anomalous_event: LogEvent,
    context_events: list[LogEvent],
) -> str:
    """
    Build the user prompt for anomaly explanation.

    Args:
        anomalous_event: The event flagged as anomalous by Isolation Forest.
        context_events: Surrounding ±5 events for causality context.
    """
    context_str = json.dumps([e.model_dump() for e in context_events], indent=2)
    event_str = json.dumps(anomalous_event.model_dump(), indent=2)

    return f"""You are an expert site reliability engineer analysing log anomalies.

Surrounding context events (±5 around the anomaly):
{context_str}

Anomalous event flagged by Isolation Forest:
{event_str}

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON. Do not use markdown code fences. Do not write ```json.
Start your response with {{ and end with }}.

{{
  "severity": "<HIGH|MEDIUM|LOW>",
  "summary": "<one sentence: what happened>",
  "likely_cause": "<most probable root cause>",
  "recommended_action": "<what an engineer should do next>"
}}"""