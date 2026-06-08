"""Internal JSON parsing utilities shared by claude_client and ollama_client.

Both backends expect the same LLM output schema — centralising parsing here
prevents drift between the two clients.
"""

import json
import logging
import re
from typing import TypedDict

from src.models import RetrievedDocument

logger = logging.getLogger(__name__)


class ParsedLLMOutput(TypedDict):
    """Typed result of parsing a raw LLM JSON response."""

    action: str
    confidence: float
    reasoning: str
    supporting_documents: list[RetrievedDocument]


def extract_json(text: str) -> str:
    """Strip markdown fences and extract the first JSON object from LLM text.

    Args:
        text: Raw LLM response string (may contain ``` fences).

    Returns:
        JSON string suitable for json.loads().

    Raises:
        ValueError: If no JSON object is found in text.
    """
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in LLM response: {text[:200]!r}")
    return cleaned[start : end + 1]


def parse_llm_output(text: str, context_docs: list[RetrievedDocument]) -> ParsedLLMOutput:
    """Parse and validate the LLM JSON response into a typed dict.

    Cross-references cited_doc_ids against the provided context docs.
    Falls back to all context docs when no valid citations are found.

    Args:
        text: Raw LLM text response.
        context_docs: Retrieved docs used to resolve cited_doc_ids.

    Returns:
        ParsedLLMOutput with action, confidence, reasoning, supporting_documents.

    Raises:
        ValueError: If action is missing or JSON is malformed.
        json.JSONDecodeError: If the extracted string is not valid JSON.
    """
    raw = json.loads(extract_json(text))

    doc_by_id = {d.doc_id: d for d in context_docs}
    cited_ids = [str(i) for i in raw.get("cited_doc_ids", [])]
    supporting = [doc_by_id[i] for i in cited_ids if i in doc_by_id]

    unknown = [i for i in cited_ids if i not in doc_by_id]
    if unknown:
        logger.warning("LLM cited unknown doc_ids (hallucination risk): %s", unknown)
    if not supporting:
        logger.warning("No valid citations found — falling back to all context docs")
        supporting = list(context_docs)

    action = str(raw.get("action", "")).strip()
    if not action:
        raise ValueError(f"LLM response missing 'action' field. Raw: {text[:200]!r}")

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        logger.warning("Invalid confidence value %r — defaulting to 0.5", raw.get("confidence"))
        confidence = 0.5

    return ParsedLLMOutput(
        action=action,
        confidence=confidence,
        reasoning=str(raw.get("reasoning", "")).strip(),
        supporting_documents=supporting,
    )
