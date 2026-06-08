"""Prompt construction for EIA dispatch recommendations.

Prompts are loaded from YAML configs in prompt_configs/ — version-controlled
so changes are tracked in Git. To update the system prompt, edit the relevant
YAML file and bump the version string in calling code.
"""

import logging
import pathlib
from typing import Any

import yaml

from src.models import RetrievedDocument

logger = logging.getLogger(__name__)

_CONFIGS_DIR = pathlib.Path(__file__).parent / "prompt_configs"
_MAX_CONTENT_CHARS = 600  # truncate long chunks to keep context window manageable


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the prompt_configs directory.

    Args:
        filename: Filename relative to prompt_configs/.

    Returns:
        Parsed YAML content as a dict.
    """
    path = _CONFIGS_DIR / filename
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def load_system_prompt(version: str = "v1") -> str:
    """Load the versioned system prompt from YAML config.

    Args:
        version: Config version string (e.g. 'v1'). Resolves to system_{version}.yaml.

    Returns:
        System prompt string for the Bedrock / Ollama Messages API.
    """
    cfg = _load_yaml(f"system_{version}.yaml")
    return str(cfg["system_prompt"])


def build_user_message(query: str, context_docs: list[RetrievedDocument]) -> str:
    """Build the user message combining the operator query with retrieved context.

    Formats each document with its doc_id and source so the LLM can cite
    specific doc_ids in its response.

    Args:
        query: The operator's natural-language dispatch query.
        context_docs: Retrieved document chunks from the hybrid retriever.

    Returns:
        Formatted user message string for the Messages API.
    """
    if not context_docs:
        logger.warning("build_user_message called with no context documents")

    context_parts = []
    for i, doc in enumerate(context_docs, start=1):
        snippet = doc.content[:_MAX_CONTENT_CHARS]
        if len(doc.content) > _MAX_CONTENT_CHARS:
            snippet += "…"
        context_parts.append(
            f"[{i}] doc_id: {doc.doc_id}\n"
            f"    source: {doc.source}\n"
            f"    content: {snippet}"
        )

    context_block = "\n\n".join(context_parts) if context_parts else "(no documents retrieved)"

    return (
        f"Operator query: {query}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Generate a dispatch recommendation as a JSON object."
    )
