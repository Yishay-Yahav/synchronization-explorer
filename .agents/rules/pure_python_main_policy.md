# Pure Python Main Branch Invariant

## Core Invariant
- **The `main` branch must strictly contain ONLY pure Python files**:
  - Allowed on `main`: `.py` files, `.gitignore`, `.md` documentation files.
  - FORBIDDEN on `main`: `.html`, `.css`, `.js`, `server.py` or any web server files.
  - All web-related files belong exclusively to the `v2-web-dashboard` branch.

## Commit Protocol
- Never stage or commit web frontend assets to `main`.
- Always inspect `git status` / `git ls-files` to ensure no non-Python artifacts are staged on `main`.
