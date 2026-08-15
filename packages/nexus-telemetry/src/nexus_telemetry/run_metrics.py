"""Append-only run metrics.

One JSON object per timestep, written to ``experiments/runs/<run-id>/metrics.jsonl``. Append-only and
line-delimited so a crashed run still leaves readable data, and so ``/diagnose-reward`` can stream a long run
without loading it whole.

Reward components are written **individually**. A scalar-only reward is undebuggable: a flat total with a
rising throughput term and a falling fairness term tells you exactly what happened, where the scalar tells you
nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import TracebackType

from nexus_errors.config_error import ConfigurationError


def _json_safe(value: float) -> float | None:
    """Map NaN/inf to ``None`` so the record stays RFC-8259 valid.

    Losing the distinction between NaN and inf is an acceptable trade for a file that can actually be
    parsed; ``has_non_finite`` marks the record so the anomaly is not silent.
    """
    return value if math.isfinite(value) else None


class RunMetricsWriter:
    """Writes one metrics line per timestep for a single run.

    Use as a context manager so the file handle closes even when training raises — which, with RL, it will.
    """

    def __init__(self, run_dir: Path, filename: str = "metrics.jsonl") -> None:
        self._path = run_dir / filename
        run_dir.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(
        self,
        timestep: int,
        reward_components: dict[str, float],
        metrics: dict[str, float],
    ) -> None:
        """Append one record.

        Args:
            timestep: The step this record describes. The join key for everything downstream.
            reward_components: Named reward terms, e.g. ``{"throughput_reward": 0.8, ...}``. Required and
                non-empty — an unnamed reward defeats the point of the file.
            metrics: QoS metrics for the step (p95 latency, throughput, fairness, drops).
        """
        if not reward_components:
            raise ConfigurationError(
                "reward_components is empty. Log every named term individually — a scalar-only reward "
                "cannot be diagnosed when training plateaus.",
                {"timestep": timestep},
            )
        record: dict[str, object] = {
            "timestep": timestep,
            "reward_components": {k: _json_safe(v) for k, v in reward_components.items()},
            "reward_total": _json_safe(sum(reward_components.values())),
            **{k: _json_safe(v) for k, v in metrics.items()},
        }
        # allow_nan=False: Python happily emits bare NaN/Infinity, which is not valid JSON. jq, JS
        # JSON.parse and most tooling reject it — so a run that went NaN produced a metrics file nothing
        # could read, exactly when /diagnose-reward most needs to read it. Non-finite values become null
        # (plus a flag) so the anomaly stays visible and the line stays parseable.
        if any(value is None for value in record.values()):
            record["has_non_finite"] = True
        self._handle.write(json.dumps(record, default=str, allow_nan=False) + "\n")

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()

    def __enter__(self) -> RunMetricsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_run_metrics(run_dir: Path, filename: str = "metrics.jsonl") -> list[dict[str, object]]:
    """Read a run's metrics back.

    Skips blank lines so a run killed mid-write still parses — a truncated final line is normal after a
    crash, and refusing to read the whole file because of it would be actively unhelpful.
    """
    path = run_dir / filename
    if not path.exists():
        raise ConfigurationError(f"No metrics at {path}.", {"run_dir": str(run_dir)})

    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed: dict[str, object] = json.loads(stripped)
        except json.JSONDecodeError:
            continue  # truncated final line from an interrupted run
        records.append(parsed)
    return records
