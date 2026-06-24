# Note — prior audit reports intentionally removed (most recent: 2026-06-24)

Audit reports are **deliberately removed from the working tree before each new
independent-audit round**, so the incoming auditors form their own view from the
code, data, and artifacts rather than anchoring on prior auditors' conclusions.
This has now happened twice.

**Removed 2026-06-24 (before the FOURTH round):** the two third-round reports —
`INDEPENDENT_AUDIT_2026-06-22_{GPT5,OPUS}.md`. Recoverable from git history (they
were committed in `742a2a2` and merged to `main` in `d16af5a`) and copied outside
the repo to `C:\Users\COWLAR\projects\_lstm_audit_backup_2026-06-24\`.

**Removed 2026-06-22 (before the third round):** the original four reports (origin
project audit 2026-06-11, internal physics-loss audit 2026-06-14, and the two
external science audits 2026-06-14 / 2026-06-16) — in git history and backed up to
`C:\Users\COWLAR\projects\_lstm_audit_backup_2026-06-22\`.

**Why a fourth round:** a pre-registered **n=12 strengthen grid (PROTOCOL §8.8,
`results/p7_strengthen/`)** was run to stress-test the one positive the third round
had let stand (a narrow same-architecture 5 dB noise-robustness benefit, then based
on n=3). The new auditors are asked to verify by execution what that grid shows and
whether the study as a whole is publishable.

Notes:
- Recover any removed report with e.g. `git log --diff-filter=D -- audit_reports/`
  then `git checkout <commit>~1 -- audit_reports/<file>`.
- Other maintainer documents (`PROJECT_STATE.md`, `results/FINDINGS.md`,
  `results/phase5_bandenergy/findings_bandenergy.md`) may still reference removed
  reports, and **`results/FINDINGS.md` §0 has NOT yet been updated for the §8.8
  result** — treat all maintainer prose as **claims to verify**, not facts.
- The new reports will be written here under new names
  (`INDEPENDENT_AUDIT_2026-06-24_{GPT5,CLAUDE}.md`).
- `dashboard_alive_2026-06-11.html` is a Phase-1 dashboard-boot screenshot, not an
  audit report — retained.
