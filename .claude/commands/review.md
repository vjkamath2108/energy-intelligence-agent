# /review — Code Review

Review the file or diff provided against the conventions in CLAUDE.md.

## Input
A file path or staged diff to review.

## Process
1. Read the active CLAUDE.md rules for the file's directory
2. Check for violations across these categories:
   - Commit format compliance
   - Error handling — no silent failures
   - Secrets or hardcoded credentials
   - Code style and conventions

## Output format
### Review: [filename]

**p1 issues** (must fix before merge)
- [issue]: [specific line or pattern] — [why it violates the rule]

**p2 issues** (should fix)
- [issue]: [specific line or pattern] — [recommendation]

**p3 issues** (minor, optional)
- [issue]: [suggestion]

**Verdict: PASS / FAIL**
A passing review means zero p1 issues.

## Rules
- Be specific — reference the actual line or pattern, not a generic observation
- Do not suggest fixes that contradict the active CLAUDE.md rules
- If no issues found at a severity level, write "None"