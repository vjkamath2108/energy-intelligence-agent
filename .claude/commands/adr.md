# /adr — Architecture Decision Record Generator

Generate a structured ADR for an architectural decision.

## Input
A decision topic — e.g. "Use Bedrock AgentCore over custom orchestrator"

## Process
1. Ask for the decision topic if not provided
2. Generate a structured ADR
3. Determine the next ADR number from existing files in /docs/adr/
4. Save to /docs/adr/ADR-NNN-title.md

## Output format
# ADR-NNN: [Decision Title]

**Date:** YYYY-MM-DD
**Status:** Proposed

## Context
What is the situation that requires a decision?

## Decision
What was decided?

## Consequences
What are the results of this decision — positive and negative?

## Alternatives Considered
| Option | Reason rejected |
|---|---|
| Alternative 1 | Why not chosen |
| Alternative 2 | Why not chosen |

## Rules
- Always save to /docs/adr/ — never to root or other directories
- Number sequentially from existing ADRs in that folder
- Title must be kebab-case in the filename: ADR-001-use-bedrock-agentcore.md
- Apply root CLAUDE.md rules only — /docs has no sub-level CLAUDE.md