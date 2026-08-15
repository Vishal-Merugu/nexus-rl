---
name: block-any-type
enabled: true
event: file
action: block
conditions:
  - field: file_path
    operator: regex_match
    pattern: \.py$
  - field: new_text
    operator: regex_match
    pattern: (:\s*Any\b|->\s*Any\b|\[Any[\],]|\bcast\()
---

**`Any` and `cast()` are forbidden.** Use protocols, generics, `object` with narrowing, or a real type. If mypy
disagrees with you, narrow properly — a `cast()` that hides a wrong type surfaces as a shape error 40 minutes
into a training run. `# type: ignore` is allowed only with an inline reason comment. See `CLAUDE.md` typing rules.
