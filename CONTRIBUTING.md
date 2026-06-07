# Contributing

## Branch naming
- `feature/`
- `fix/`
- `docs/`

## Commit format
`[scope] short imperative description`
Examples: `[indexer] add FAISS batch insert`, `[rank] fix honeypot purge edge case`

## Atomic commit rules
- One logical change per commit
- Never bundle unrelated files together
- Each commit must leave the repo in a working state

## Changelog rules
- Update `docs/CHANGELOG.md` with every meaningful change
- New entries go at the top, most recent first
- Always explain WHY a change was made, not just what

## Pull request process
1. 1 approval required before merge
2. Ensure the PR checklist in the template is fully checked

## Key constraints to never break
- rank.py must complete in under 5 minutes with no internet or GPU access
- Never commit files from artifacts/ or data/ directories
