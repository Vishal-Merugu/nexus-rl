---
description: Create a plan document and follow best practices. Usage: /create-plan Description of feature
---

Create a plan for the following task:

$ARGUMENTS

### 1. Plan Mode Default

- Write detailed specs upfront to reduce ambiguity
- At the end of each plan, conduct a rigorous interview with me asking for every unresolved detail.
- Encode in the plan that an **Agent Team** should be used. Explain the composition of the agent team for each phase. **It is important to use the term "Agent Team"**

### 2. Subagent Strategy

- Use agents liberally to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via agents
- If a component or module can be developed independently, use an agent from the agent team

Typical team composition for this repo:

| Phase              | Agents                                                            |
| ------------------ | ----------------------------------------------------------------- |
| Design             | `nexus-system-architect`, plus `simulator-architect` if sim-touching |
| MDP change         | `mdp-reviewer` (mandatory), `rl-training-debugger`                 |
| Agent-layer change | `mcp-tool-designer`                                                |
| Implementation     | `test-writer-fixer` (tests first), then the domain agent           |
| Results            | `evaluation-analyst`                                               |
| Gate               | `python-code-reviewer`, then `pre-commit-auditor`                  |

### 3. Answer These Before Implementing

Every plan in this repo must state:

- **Which package or app** each piece lands in. `packages/` is libraries, `apps/` is services.
- **Is this an MDP change?** If observation, action, or reward move at all — say so, plan the `MDP_VERSION` bump, and list the checkpoints it invalidates.
- **Does it touch determinism?** Any new stochastic draw needs an injected `Generator` in the plan, not retrofitted after.
- **Does it change agent authority?** If the agent layer gains any capability, state why it stays inside constraints. If it does not, the plan is an ADR instead.
- **What does it cost per `step()`?** Anything in the hot path needs a budget stated upfront.

### 4. Verification Before Done

- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- For anything performance-touching: benchmark before and after, report the number

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Before implementing: can I reuse existing code? Would extracting shared logic serve this and existing callers? If yes, do that.

## Task Management

1. **Plan First**: Write plan with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to plan

## Core Principles

- **TDD**: Write tests first that test observable behavior. Never test implementation details. Plan the tests explicitly.
- **Pseudo code only**: Plans contain pseudo code, never actual code.
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Be extremely concise. Sacrifice grammar for concision.**
