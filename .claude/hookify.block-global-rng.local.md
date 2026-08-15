---
name: block-global-rng
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: \.py$
  - field: new_text
    operator: regex_match
    pattern: (^\s*import\s+random\b|^\s*from\s+random\s+import|np\.random\.(seed|rand|randn|randint|choice|uniform|normal|permutation|shuffle)\b)
---

**Global RNG is forbidden.** Every stochastic draw takes an explicitly injected `numpy.random.Generator`:

```python
def sample_demand(rng: np.random.Generator) -> NDArray[np.float32]: ...
```

Create generators with `np.random.default_rng(seed)` at the composition root and pass them down.

A simulator whose trajectory depends on hidden global state cannot be debugged. "Same seed, same config,
different reward" burns hours and looks like an RL problem when it is a plumbing problem. See `tasks/lessons.md`
→ Determinism.
