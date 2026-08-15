---
name: warn-star-import
enabled: true
event: file
action: warn
conditions:
  - field: file_path
    operator: regex_match
    pattern: \.py$
  - field: new_text
    operator: regex_match
    pattern: (from\s+[\w.]+\s+import\s+\*|^__all__\s*=)
---

**Star imports and re-export barrels make imports untraceable.** Import directly from the module that defines
the symbol:

```python
from nexus_sim.link import Link  # yes
from nexus_sim import *  # no
```

`__init__.py` files should be empty or hold only the package version. A reader should be able to find any
symbol's definition from its import line alone.
