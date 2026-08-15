from __future__ import annotations

from pathlib import Path

import pytest

from nexus_errors.config_error import ConfigurationError
from nexus_telemetry.run_metrics import RunMetricsWriter, read_run_metrics


def test_written_records_round_trip(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"throughput_reward": 1.0, "latency_penalty": -0.5}, {"p95_latency_ms": 12.0})
        writer.write(1, {"throughput_reward": 1.2, "latency_penalty": -0.3}, {"p95_latency_ms": 10.0})

    records = read_run_metrics(tmp_path)

    assert [r["timestep"] for r in records] == [0, 1]
    assert records[0]["reward_components"] == {"throughput_reward": 1.0, "latency_penalty": -0.5}


def test_total_is_derived_so_it_cannot_disagree_with_the_components(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"a": 1.0, "b": -0.25}, {})

    assert read_run_metrics(tmp_path)[0]["reward_total"] == 0.75


def test_scalar_only_reward_is_refused(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer, pytest.raises(ConfigurationError, match="individually"):
        writer.write(0, {}, {"p95_latency_ms": 12.0})


def test_writer_appends_rather_than_truncating(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"a": 1.0}, {})
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(1, {"a": 2.0}, {})

    assert len(read_run_metrics(tmp_path)) == 2


def test_truncated_final_line_from_a_killed_run_still_reads(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"a": 1.0}, {})
    with (tmp_path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"timestep": 1, "reward_com')

    records = read_run_metrics(tmp_path)

    assert len(records) == 1
    assert records[0]["timestep"] == 0


def test_missing_metrics_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        read_run_metrics(tmp_path / "no-such-run")


def test_non_finite_values_still_produce_valid_json(tmp_path: Path) -> None:
    # Python emits bare NaN/Infinity, which is not valid JSON: jq and JS JSON.parse both reject it. A run
    # that went NaN produced a metrics file nothing could read, exactly when /diagnose-reward needs it.
    import json as _json

    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"throughput_reward": float("nan")}, {"p95_latency_ms": float("inf")})

    line = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip()

    def reject_constants(constant: str) -> object:
        raise AssertionError(f"non-JSON constant emitted: {constant}")

    parsed = _json.loads(line, parse_constant=reject_constants)  # strict RFC-8259
    assert parsed["reward_components"]["throughput_reward"] is None
    assert parsed["p95_latency_ms"] is None


def test_a_non_finite_record_is_flagged_so_the_anomaly_is_not_silent(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"a": float("nan")}, {})

    assert read_run_metrics(tmp_path)[0]["has_non_finite"] is True


def test_finite_records_carry_no_flag(tmp_path: Path) -> None:
    with RunMetricsWriter(tmp_path) as writer:
        writer.write(0, {"a": 1.0}, {"b": 2.0})

    assert "has_non_finite" not in read_run_metrics(tmp_path)[0]
