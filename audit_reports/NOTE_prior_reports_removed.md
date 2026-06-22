# Note — prior audit reports intentionally removed (2026-06-22)

The previous audit reports (the origin project audit, the internal physics-loss
audit, the two independent external science audits, and the audit prompt template)
were **deliberately removed from the working tree** before commissioning a new,
**fresh pair of independent audits** — so the new auditors form their own view
from the code, data, and artifacts rather than anchoring on prior auditors'
conclusions.

- They are **not lost**: recoverable from git history (e.g.
  `git log --diff-filter=D -- audit_reports/`, then
  `git checkout <commit>~1 -- audit_reports/`), and a copy was kept outside the
  repo by the owner.
- Other documents (`PROJECT_STATE.md`, `results/FINDINGS.md`, `README.md`) may
  still reference the removed reports by filename. Those references are stale by
  design; treat the maintainer documents as **claims to verify**, not facts.
- The new audit reports will be written here under new names.

Nothing else in `audit_reports/` is an audit report (`dashboard_alive_*.html` is a
Phase-1 dashboard-boot screenshot, retained).
