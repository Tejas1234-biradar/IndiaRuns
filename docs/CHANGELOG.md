# Changelog

All notable changes to IndiaRuns are documented here, most recent first.
Format: date, title, what changed, why, scope.

---

## [2026-06-07] — add gitignore

### What changed
- Created `.gitignore` file to specify untracked files and directories

### Why
Committed gitignore first before any other files so that artifact binaries and raw data files are never accidentally tracked, which would bloat the repo and risk leaking sensitive candidate data.

### Scope
Repo Infrastructure

## [2026-06-07] — initialize repository structure

### What changed
- Created `docs/CHANGELOG.md` to track all repo changes with context

### Why
Changelog initialized as the first file so every subsequent change in this sprint has a documented record. Future teammates and agents can read this file to understand not just what exists in the repo but why decisions were made.

### Scope
Repo Infrastructure
