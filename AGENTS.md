# Workspace Agent Rules

## Always Use GRAVITAS

On every software engineering, debugging, refactoring, review, or migration task in this workspace:

1. Read and follow the `gravitas` skill (`C:\Users\Lenovo\.agents\skills\gravitas\SKILL.md`) before planning or editing.
2. Also load the matching extension skill when applicable:
   - `gravitas-python` for Python code (`backend/`, `scripts/`)
   - `gravitas-typescript` for TypeScript/Next.js code (`frontend/`)
   - `gravitas-highstakes` for irreversible/high-stakes changes (deletions, schema/data migrations, auth/security, git history/push)
3. Enforce the core invariants on every turn:
   - Read target files, tests, and callers before mutating.
   - Keep changes strictly within declared scope.
   - Never exact-retry a failed approach without a material change.
   - Verify with fresh executable evidence after the final relevant edit (`VERDICT: PASS` or `VERDICT: FAIL -- <reason>`).
   - Report only what was directly observed; label unverified claims explicitly.

## Storage & Environment Constraints

- **Do not install, clone, or store artifacts on `C:`** (`C:` has low free space). Keep virtual environments, packages, caches, datasets, model weights, audio, previews, and scratch files on `D:`.
- Keep `data/`, `models/weights/`, `.env`, previews, audio, motion, and catalogue files gitignored and local.
- Do not commit, push, or rewrite git history without explicit user approval.
