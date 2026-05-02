# /commt-msg - Conventional commit behaviour

Generate a commit message from the current staged diff

## Input
The staged git diff - run 'git diff - staged' to get it

## Process
1. Read the diff
2. Identify what changed and why
3. Determine the correct type: feat,fix,chore,docs,refactor,test
4. Determine the scope from the directory or file attached

## Output format:
type(scope): short description (72 chars)
    - bullet: what changed
    - bullet: Why it changed
    - bullet: any side effects or dependencies affected

## Rules

    - Never exceed 72 characters on the first line
    - Use imperative mood - "add" not "added","fix" not "fixed"
    - Scope should be the directory or module name
    - If the changes span multiple scopes, use the primary one
    - Validate against root CLAUDE.md commit format before outputting