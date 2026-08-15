---
name: block-training-in-tests
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: (tests?/.*\.py$|test_.*\.py$|.*_test\.py$)
  - field: new_text
    operator: regex_match
    pattern: (\.learn\(|nexus_train|train_policy\(|model\.fit\()
---

**Never train a model inside a test.** It is slow, non-deterministic in wall-clock, flaky in CI, and proves
nothing that the environment contract tests do not already prove.

Test the contract instead:

- observation shape, dtype, bounds
- action space bounds and decoding
- reward components present, named, and correctly signed
- determinism under a fixed seed

Training belongs in `make evaluate` and `/train-and-evaluate`, not in the test suite. See `CLAUDE.md` →
Testing & Documentation.
