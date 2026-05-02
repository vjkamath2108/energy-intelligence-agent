# /refactor — Refactor with Plan Mode Gate

Refactor a file or function. Always plans before making any changes.

## Input
A file path and a refactor goal.
Example: /refactor scripts/deploy.sh — extract environment validation 
into a separate function

## Process
1. Read the file and understand current structure
2. Identify what needs to change to meet the refactor goal
3. Output a step-by-step plan
4. STOP — do not make any edits until the user confirms the plan

## Output format
### Refactor Plan: [filename]

**Goal:** [restate the refactor goal]

**Proposed changes:**
1. [specific change — what and where]
2. [specific change — what and where]
3. [specific change — what and where]

**Risk assessment:**
- Files affected: [list]
- Potential side effects: [list]
- Reversibility: [easy/medium/hard]

**Awaiting confirmation — reply "proceed" to apply these changes.**

## Rules
- NEVER make any edits before the user confirms the plan
- If the refactor touches more than one file, list all affected files
- If risk is high, say so explicitly — do not downplay it
- Plan mode is mandatory for this command — it cannot be bypassed