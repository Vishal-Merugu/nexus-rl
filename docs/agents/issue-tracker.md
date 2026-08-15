# Issue Tracker

Issues live in **GitHub Issues**, accessed via the `gh` CLI.

## Reading

```bash
gh issue list --limit 30
gh issue view <number>
gh issue list --label ready-for-agent
```

## Creating

```bash
gh issue create --title "<title>" --body "<body>" --label <label>
```

A good issue in this repo answers four questions:

1. **What is wrong or missing**, concretely — with a `file:line` reference if it is code, or a run-id if it
   is a result.
2. **How to reproduce**, including the seed. This project is deterministic; an issue without a seed is
   throwing that away.
3. **What was expected** instead.
4. **Whether it touches the MDP.** If observation, action, or reward are involved, say so in the title — it
   changes who reviews it and whether checkpoints die.

## Labels

| Label              | Meaning                                                        |
| ------------------ | -------------------------------------------------------------- |
| `needs-triage`     | Default for new issues                                         |
| `needs-info`       | Blocked on a reproduction, a seed, or a run-id                 |
| `ready-for-agent`  | Well-specified enough for an agent to implement unattended     |
| `ready-for-human`  | Needs a judgement call — usually an MDP or authority question  |
| `mdp-change`       | Touches observation / action / reward. Invalidates checkpoints |
| `determinism`      | Reproducibility broken. Treat as CRITICAL                      |
| `wontfix`          | Closed deliberately, with the reasoning recorded               |

## Priority

`determinism` outranks everything. A repo that cannot reproduce a run cannot diagnose anything else, so a
determinism issue blocks other work rather than queueing behind it.
