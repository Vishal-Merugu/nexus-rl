---
name: block-bare-except
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: \.py$
  - field: new_text
    operator: regex_match
    pattern: (except\s*:|except\s+Exception\s*:(?!\s*\n\s*(raise|.*capture_exception)))
---

**Bare `except:` and unhandled `except Exception:` are forbidden.** Catch the specific error, or re-raise, or log
it with `telemetry.capture_exception(err, "...")` before handling.

Swallowed exceptions in a control loop produce a system that looks healthy while doing nothing — the worst
possible failure mode for an orchestrator, and one that will not show up until a demo.

Raised errors must subclass `NexusError` from `nexus-errors`. Never raise bare `Exception` or `ValueError`.
