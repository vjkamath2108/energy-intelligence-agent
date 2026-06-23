"""Unit tests for Stream 4 data generation scripts.

Tests cover DPO pair and SFT dataset generation logic using synthetic data.
No benchmark files, GPU, or model weights are needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from stream4_finetuning.training.generate_dpo_pairs import (
    DPOPair,
    _build_completion,
    _build_prompt,
    generate_dpo_pairs,
    load_benchmark_records,
    load_log_events,
    write_dpo_pairs,
)
from stream4_finetuning.training.generate_sft_dataset import (
    SFTExample,
    _build_input,
    _build_output,
    generate_sft_dataset,
    write_sft_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic data
# ---------------------------------------------------------------------------


def _make_log_events(n: int = 20) -> list[dict[str, object]]:
    """Generate n synthetic log events for testing."""
    return [
        {
            "ts": f"2026-05-01T08:00:{i:02d}",
            "service": "api-gateway",
            "level": "ERROR" if i % 5 == 0 else "INFO",
            "latency_ms": 150 + i * 10,
            "status_code": 500 if i % 5 == 0 else 200,
        }
        for i in range(n)
    ]


def _make_record(
    backend: str,
    event_index: int,
    anomaly_pattern: str = "cascade",
    ground_truth: str = "HIGH",
    predicted: str = "HIGH",
    severity_correct: bool | None = True,
) -> dict[str, object]:
    """Build a synthetic benchmark record."""
    return {
        "backend": backend,
        "event_index": event_index,
        "service": "api-gateway",
        "level": "ERROR",
        "latency_ms": 1800,
        "status_code": 500,
        "anomaly_pattern": anomaly_pattern,
        "ground_truth_severity": ground_truth,
        "predicted_severity": predicted,
        "severity_correct": severity_correct,
        "json_valid": True,
        "elapsed_ms": 1316,
        "summary": f"Service failure detected at event {event_index}.",
        "likely_cause": "Database connection pool exhaustion.",
        "recommended_action": "Check connection pool metrics and restart service.",
    }


@pytest.fixture()
def log_events() -> list[dict[str, object]]:
    """20 synthetic log events."""
    return _make_log_events(20)


@pytest.fixture()
def benchmark_records() -> dict[str, dict[int, dict[str, object]]]:
    """Benchmark records for two backends with 3 shared events."""
    chosen_backend = "mistral-bedrock"
    rejected_backend = "ollama:llama3.2:3b"

    # Events 5, 7, 10 are injected patterns
    # Event 12 is isolated (should be excluded from DPO unless include_isolated)
    records: dict[str, dict[int, dict[str, object]]] = {
        chosen_backend: {
            5: _make_record(chosen_backend, 5, "cascade", "HIGH", "HIGH", True),
            7: _make_record(chosen_backend, 7, "latency_spike", "HIGH", "HIGH", True),
            10: _make_record(chosen_backend, 10, "error_surge", "HIGH", "HIGH", True),
            12: _make_record(chosen_backend, 12, "isolated", None, "HIGH", None),  # type: ignore[arg-type]
        },
        rejected_backend: {
            5: _make_record(rejected_backend, 5, "cascade", "HIGH", "MEDIUM", False),
            7: _make_record(rejected_backend, 7, "latency_spike", "HIGH", "LOW", False),
            10: _make_record(rejected_backend, 10, "error_surge", "HIGH", "HIGH", True),  # both correct
            12: _make_record(rejected_backend, 12, "isolated", None, "MEDIUM", None),  # type: ignore[arg-type]
        },
    }
    return records


# ---------------------------------------------------------------------------
# _build_prompt / _build_input
# ---------------------------------------------------------------------------


def test_build_prompt_contains_context_and_event(
    log_events: list[dict[str, object]],
) -> None:
    """_build_prompt includes context events and the anomalous event JSON."""
    context = log_events[3:8]
    anomalous = log_events[5]
    prompt = _build_prompt(anomalous, context)

    assert "api-gateway" in prompt
    assert "Isolation Forest" in prompt
    assert "context events" in prompt.lower()


def test_build_prompt_is_string(log_events: list[dict[str, object]]) -> None:
    """_build_prompt returns a non-empty string."""
    result = _build_prompt(log_events[0], log_events[1:3])
    assert isinstance(result, str)
    assert len(result) > 50


def test_build_input_matches_prompt_content(
    log_events: list[dict[str, object]],
) -> None:
    """_build_input (SFT version) contains anomalous event and context."""
    context = log_events[0:3]
    anomalous = log_events[1]
    inp = _build_input(anomalous, context)

    assert "api-gateway" in inp
    assert "Isolation Forest" in inp


# ---------------------------------------------------------------------------
# _build_completion / _build_output
# ---------------------------------------------------------------------------


def test_build_completion_valid_json() -> None:
    """_build_completion returns valid JSON with all required fields."""
    record = _make_record("mistral-bedrock", 5)
    result = _build_completion(record)
    parsed = json.loads(result)

    assert "severity" in parsed
    assert "summary" in parsed
    assert "likely_cause" in parsed
    assert "recommended_action" in parsed


def test_build_output_matches_record_severity() -> None:
    """_build_output preserves the predicted_severity from the record."""
    record = _make_record("mistral-bedrock", 5, predicted="MEDIUM")
    result = _build_output(record)
    parsed = json.loads(result)
    assert parsed["severity"] == "MEDIUM"


# ---------------------------------------------------------------------------
# generate_dpo_pairs
# ---------------------------------------------------------------------------


def test_generate_dpo_pairs_count(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """generate_dpo_pairs returns only pairs where chosen=correct, rejected=wrong."""
    pairs = generate_dpo_pairs(log_events, benchmark_records)
    # Events 5 and 7 qualify; event 10 has both correct; event 12 is isolated
    assert len(pairs) == 2


def test_generate_dpo_pairs_schema(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """Each DPO pair has prompt, chosen, rejected, and metadata fields."""
    pairs = generate_dpo_pairs(log_events, benchmark_records)
    assert pairs
    for pair in pairs:
        assert isinstance(pair.prompt, str)
        assert isinstance(pair.chosen, str)
        assert isinstance(pair.rejected, str)
        assert "event_index" in pair.metadata
        assert "chosen_backend" in pair.metadata
        assert "rejected_backend" in pair.metadata


def test_generate_dpo_pairs_chosen_is_correct_severity(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """chosen completion reflects the ground-truth severity (HIGH)."""
    pairs = generate_dpo_pairs(log_events, benchmark_records)
    for pair in pairs:
        chosen_parsed = json.loads(pair.chosen)
        assert chosen_parsed["severity"] == "HIGH"


def test_generate_dpo_pairs_rejects_wrong_severity(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """rejected completion has a different (wrong) severity than ground truth."""
    pairs = generate_dpo_pairs(log_events, benchmark_records)
    for pair in pairs:
        rejected_parsed = json.loads(pair.rejected)
        assert rejected_parsed["severity"] != "HIGH"


def test_generate_dpo_pairs_excludes_isolated(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """Isolated events are excluded from DPO pairs regardless of severity."""
    pairs = generate_dpo_pairs(log_events, benchmark_records)
    for pair in pairs:
        assert pair.metadata.get("anomaly_pattern") != "isolated"


def test_generate_dpo_pairs_missing_backend_raises(
    log_events: list[dict[str, object]],
) -> None:
    """generate_dpo_pairs raises ValueError if chosen_backend has no records."""
    with pytest.raises(ValueError, match="No records found for chosen backend"):
        generate_dpo_pairs(log_events, {}, chosen_backend="nonexistent")


def test_dpo_pair_to_dict_round_trips() -> None:
    """DPOPair.to_dict() serialises all fields and is JSON-serialisable."""
    pair = DPOPair(
        prompt="Analyse this event.",
        chosen='{"severity": "HIGH", "summary": "x", "likely_cause": "y", "recommended_action": "z"}',
        rejected='{"severity": "LOW", "summary": "x", "likely_cause": "y", "recommended_action": "z"}',
        metadata={"event_index": 5, "source": "test"},
    )
    d = pair.to_dict()
    assert d["prompt"] == pair.prompt
    assert d["chosen"] == pair.chosen
    assert json.dumps(d)  # must be JSON-serialisable


# ---------------------------------------------------------------------------
# write_dpo_pairs
# ---------------------------------------------------------------------------


def test_write_dpo_pairs_creates_file(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
    tmp_path: Path,
) -> None:
    """write_dpo_pairs creates a JSONL file with one line per pair."""
    pairs = generate_dpo_pairs(log_events, benchmark_records)
    output = tmp_path / "dpo_pairs.jsonl"
    write_dpo_pairs(pairs, output)

    assert output.exists()
    lines = [l for l in output.read_text().splitlines() if l]
    assert len(lines) == len(pairs)
    for line in lines:
        row = json.loads(line)
        assert "prompt" in row
        assert "chosen" in row
        assert "rejected" in row


# ---------------------------------------------------------------------------
# generate_sft_dataset
# ---------------------------------------------------------------------------


def test_generate_sft_dataset_excludes_wrong_predictions(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """SFT dataset only includes mistral-bedrock events with severity_correct=True."""
    examples = generate_sft_dataset(log_events, benchmark_records)
    # Events 5, 7, 10 are correct for mistral-bedrock; 12 is isolated → excluded
    assert len(examples) == 3


def test_generate_sft_dataset_includes_isolated_when_flag_set(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """SFT dataset includes isolated events when include_isolated=True."""
    examples = generate_sft_dataset(
        log_events, benchmark_records, include_isolated=True
    )
    assert len(examples) == 4


def test_generate_sft_dataset_schema(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
) -> None:
    """Each SFT example has instruction, input, output, and metadata."""
    examples = generate_sft_dataset(log_events, benchmark_records)
    for ex in examples:
        assert isinstance(ex.instruction, str) and len(ex.instruction) > 20
        assert isinstance(ex.input, str) and len(ex.input) > 10
        assert isinstance(ex.output, str)
        output_parsed = json.loads(ex.output)
        assert "severity" in output_parsed
        assert "summary" in output_parsed


def test_generate_sft_dataset_missing_backend_raises(
    log_events: list[dict[str, object]],
) -> None:
    """generate_sft_dataset raises ValueError if source_backend has no records."""
    with pytest.raises(ValueError, match="No records found for backend"):
        generate_sft_dataset(log_events, {}, source_backend="nonexistent")


def test_sft_example_to_dict_round_trips() -> None:
    """SFTExample.to_dict() serialises all fields and is JSON-serialisable."""
    ex = SFTExample(
        instruction="You are an expert.",
        input="Analyse this event.",
        output='{"severity": "HIGH", "summary": "x", "likely_cause": "y", "recommended_action": "z"}',
        metadata={"event_index": 5},
    )
    d = ex.to_dict()
    assert d["instruction"] == ex.instruction
    assert json.dumps(d)


# ---------------------------------------------------------------------------
# write_sft_dataset
# ---------------------------------------------------------------------------


def test_write_sft_dataset_creates_file(
    log_events: list[dict[str, object]],
    benchmark_records: dict[str, dict[int, dict[str, object]]],
    tmp_path: Path,
) -> None:
    """write_sft_dataset creates a JSONL file with one line per example."""
    examples = generate_sft_dataset(log_events, benchmark_records)
    output = tmp_path / "sft_dataset.jsonl"
    write_sft_dataset(examples, output)

    assert output.exists()
    lines = [l for l in output.read_text().splitlines() if l]
    assert len(lines) == len(examples)
    for line in lines:
        row = json.loads(line)
        assert "instruction" in row
        assert "input" in row
        assert "output" in row


# ---------------------------------------------------------------------------
# load_log_events / load_benchmark_records
# ---------------------------------------------------------------------------


def test_load_log_events(tmp_path: Path) -> None:
    """load_log_events reads a JSONL file and returns a list of dicts."""
    events = _make_log_events(5)
    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text("\n".join(json.dumps(e) for e in events))

    loaded = load_log_events(jsonl)
    assert len(loaded) == 5
    assert loaded[0]["service"] == "api-gateway"


def test_load_benchmark_records(tmp_path: Path) -> None:
    """load_benchmark_records groups records by backend → event_index."""
    records = [
        _make_record("mistral-bedrock", 5),
        _make_record("mistral-bedrock", 7),
        _make_record("ollama:llama3.2:3b", 5),
    ]
    jsonl = tmp_path / "benchmark.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records))

    loaded = load_benchmark_records(jsonl)
    assert "mistral-bedrock" in loaded
    assert 5 in loaded["mistral-bedrock"]
    assert 7 in loaded["mistral-bedrock"]
    assert "ollama:llama3.2:3b" in loaded
    assert 5 in loaded["ollama:llama3.2:3b"]
