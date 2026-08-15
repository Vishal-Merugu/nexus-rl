---
name: block-commit-run-artifacts
enabled: true
event: bash
action: block
conditions:
  - field: command
    operator: regex_match
    pattern: git\s+add\b.*(experiments/runs|\.zip|\.pt\b|\.ckpt\b|\.env(\s|$))
---

**Never commit run artifacts, checkpoints, or `.env`.**

`experiments/runs/` is gitignored except `.gitkeep`. Checkpoints are large, binary, and invalidated by the next
`MDP_VERSION` bump — committing one preserves a file that will be meaningless within weeks while bloating the
repo permanently.

If a specific result needs preserving, commit the **numbers** — `metrics.jsonl` summaries into
`docs/evaluation/results.md`, with the run-id, seed count, `MDP_VERSION` and git SHA — not the weights.
