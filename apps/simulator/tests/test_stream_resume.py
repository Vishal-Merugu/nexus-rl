"""WebSocket downsampling / resume tests.

Regression: the stream tracked a high-water timestep and never reset it, so after `/reset` the dashboard
went permanently silent — a frozen chart that still looked live.
"""

from __future__ import annotations

from nexus_simulator.main import should_send_frame as should_send


def _stream(timesteps: list[int]) -> list[int]:
    sent: list[int] = []
    last_sent: int | None = None
    for timestep in timesteps:
        if should_send(timestep, last_sent):
            sent.append(timestep)
            last_sent = timestep
    return sent


def test_the_first_frame_is_sent_immediately() -> None:
    # Starting from -1 meant the dashboard waited a whole downsample window before its first paint.
    assert _stream([0])[0] == 0


def test_frames_are_downsampled() -> None:
    sent = _stream(list(range(20)))

    assert sent == [0, 5, 10, 15]


def test_the_stream_resumes_after_a_reset() -> None:
    running = list(range(100))
    after_reset = list(range(0, 20))

    sent = _stream(running + after_reset)

    post_reset = list(sent[len(_stream(running)) :])
    assert post_reset, "dashboard received nothing after the simulator reset"
    assert post_reset[0] == 0


def test_a_reset_sends_the_new_first_frame_promptly() -> None:
    assert should_send(0, last_sent=95) is True
