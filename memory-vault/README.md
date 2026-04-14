# Atlas Swarm Memory Vault

Persistent, self-evolving memory for the AI Factory agent swarm.
Each agent reads/writes markdown files here. Obsidian indexes them.

## Structure

- **agents/**    — Agent profiles, capabilities, performance history
- **decisions/** — Strategic decisions with rationale (CEO-Agent)
- **learnings/** — Self-improving prompt adjustments, Q-learning router state
- **products/**  — Product pipeline: ideas → designs → listings → revenue
- **metrics/**   — KPI snapshots, DMAIC cycle data (Lean Six Sigma)
- **feedback/**  — Customer feedback, marketplace signals, quality audits

## Convention

Every file: YAML frontmatter + markdown body.
Every file MUST have: created_at, updated_at, agent_id, confidence (0-1).
