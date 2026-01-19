> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see the main documentation.
>
> *Archived on: 2026-01-20*
> *Reason: Superseded by consolidated documentation*
# 🔀 MERGE CONFLICTS RESOLUTION GUIDE

**Generated**: 2025-11-22
**Current Branch**: `main`
**Branches to Merge**: 5 branches with conflicts

---

## 📊 SUMMARY

| Branch | Team | Conflicts | Severity | Resolution Time |
|---|---|---|---|---|
| `add-email-digest-ui` | Team 9 | 2 files | 🟡 Medium | 15 min |
| `fix-auth-callbacks-batch-2` | Team 4 | 1 file | 🟢 Easy | 10 min |
| `fix-hardcoded-user-id` | Team 3 | 1 file | 🟢 Easy | 10 min |
| `fix-magic-numbers` | Team 6 | 1 file | 🟡 Medium | 15 min |
| `implement-2fa-sessions` | Team 8 | 1 file | 🟢 Easy | 5 min |

**Total Conflicted Files**: 6
**Estimated Total Resolution Time**: 55 minutes

---

## ✅ ALREADY MERGED (Confirmed Working)

These branches were successfully merged to `main`:
- ✅ `claude/fix-hardcoded-credentials` - Team 1: Security fixes
- ✅ `claude/fix-password-hashing` - Team 2: Password hashing
- ✅ `claude/add-database-indexes` - Team 5: Database performance
- ✅ `claude/fix-n-plus-one-queries` - Team 7: Query optimization
- ✅ `claude/cleanup-imports-validation` - Team 10: Code cleanup

**Status**: Main branch has improvements from 5 teams ✓

---

## 🔴 BRANCH 1: `add-email-digest-ui-01Y9BnxZPQsoiK6srWG77751`

**Team**: Team 9 (Email Digest Management UI)
**Conflicts**: 2 files
**Cause**: Both main and branch added new sections to same files

### Conflicted Files:

#### 1. `packages/dashboard/config.py`
**Conflict Type**: Both added new sections at end of file

**Main has**: Startup validation section (Team 1's work)
```python
# =============================================================================
# Startup Validation
# =============================================================================
def _validate_configuration():
    ...
```

**Branch has**: Email digest configuration
```python
# Email Digest Queue Configuration
EMAIL_DIGEST_ENABLED = os.getenv("EMAIL_DIGEST_ENABLED", "True").lower() == "true"
DIGEST_FREQUENCY_HOURS = int(os.getenv("DIGEST_FREQUENCY_HOURS", "24"))
...
```

**Resolution**: **KEEP BOTH**
- Accept branch's email digest config
- Keep main's validation section
- Place email digest config BEFORE validation section

**Steps**:
```bash
git merge --no-commit origin/claude/add-email-digest-ui-01Y9BnxZPQsoiK6srWG77751
# Edit config.py:
# 1. Keep email digest config from branch
# 2. Keep validation section from main
# 3. Ensure both sections are present
git add packages/dashboard/config.py
```

---

#### 2. `packages/dashboard/models/email_log.py`
**Conflict Type**: Different index strategies

**Main has**: Minimal indexes (Team 5's optimization for write performance)
```python
__table_args__ = (
    Index('idx_email_logs_sent_at', 'sent_at'),
    Index('ix_email_logs_created_at', 'created_at'),
    # Composite indexes removed - log tables should minimize indexes for write performance
)
```

**Branch has**: Comprehensive indexes (Team 9's UI query optimization)
```python
__table_args__ = (
    Index('idx_email_logs_sent_at', 'sent_at'),
    Index('idx_email_logs_time_status', 'created_at', 'status'),
    Index('idx_email_logs_user_time', 'user_id', 'created_at'),
    Index('idx_email_logs_recipient_time', 'recipient_email', 'created_at'),
)
```

**Resolution**: **KEEP BRANCH** (Team 9's comprehensive indexes)
- **Reasoning**: Team 9 added UI that queries this table - need indexes for performance
- Team 5 was correct for write-heavy tables, but email logs will be queried frequently by UI
- Trade-off: Slightly slower writes for much faster reads

**Steps**:
```bash
# Accept branch version (theirs)
git checkout --theirs packages/dashboard/models/email_log.py
git add packages/dashboard/models/email_log.py
```

---

## 🟡 BRANCH 2: `fix-auth-callbacks-batch-2-012DVCXu4SkYBP7jwLmB9EDT`

**Team**: Team 4 (Authentication Integration Batch 2)
**Conflicts**: 1 file
**Cause**: Team 8 modified security callbacks (2FA), Team 4 added auth helper

### Conflicted Files:

#### 1. `packages/dashboard/callbacks/security_callbacks.py`
**Conflict Type**: Both modified imports and same function

**Main has**: No changes (baseline)

**Branch has**:
- Added `from utils.auth_utils import get_current_user_id`
- Replaced `user_id = 1` with `get_current_user_id()`

**Resolution**: **KEEP BRANCH** if Team 3 already merged, otherwise **MANUAL MERGE**

**Check first**:
```bash
# Check if auth_utils.py exists (Team 3's work)
ls -la packages/dashboard/utils/auth_utils.py
```

**If file exists** (Team 3 merged):
```bash
# Accept branch version
git checkout --theirs packages/dashboard/callbacks/security_callbacks.py
git add packages/dashboard/callbacks/security_callbacks.py
```

**If file doesn't exist** (Team 3 not merged):
```bash
# This branch will fail at runtime - need Team 3 first
# ABORT and merge Team 3 first
git merge --abort
echo "ERROR: Team 3 must be merged before Team 4"
```

**Dependency**: Requires `packages/dashboard/utils/auth_utils.py` from Team 3's branch

---

## 🟢 BRANCH 3: `fix-hardcoded-user-id-01BckhunZFZEAxp6BA6N8GbY`

**Team**: Team 3 (Authentication Integration Batch 1)
**Conflicts**: 1 file
**Cause**: Overlapping changes with another branch

### Conflicted Files:

#### 1. `packages/dashboard/callbacks/webhook_callbacks.py`
**Conflict Type**: Same lines modified

**Main has**: Original code with `user_id = 1`

**Branch has**:
- Added `from utils.auth_utils import get_current_user_id`
- Replaced `user_id = 1` with `get_current_user_id()`

**Resolution**: **KEEP BRANCH**

**Steps**:
```bash
git merge --no-commit origin/claude/fix-hardcoded-user-id-01BckhunZFZEAxp6BA6N8GbY
git checkout --theirs packages/dashboard/callbacks/webhook_callbacks.py
git add packages/dashboard/callbacks/webhook_callbacks.py
```

**Note**: This branch creates `utils/auth_utils.py` - Team 4 depends on this!

---

## 🟡 BRANCH 4: `fix-magic-numbers-01T2bWdYWoCanxW33rK1wVDw`

**Team**: Team 6 (Magic Numbers to Constants)
**Conflicts**: 1 file
**Cause**: File was modified by another team

### Conflicted Files:

#### 1. `packages/dashboard/callbacks/experiment_wizard_callbacks.py`
**Conflict Type**: Different sections modified

**Main has**: Possible changes from another team

**Branch has**:
- Added `from utils.constants import MAX_SAMPLES_PER_DATASET, DEFAULT_EPOCHS`
- Replaced magic numbers like `20480`, `50` with constants

**Resolution**: **MANUAL MERGE** (but likely clean)

**Steps**:
```bash
git merge --no-commit origin/claude/fix-magic-numbers-01T2bWdYWoCanxW33rK1wVDw

# Check the conflict
git diff packages/dashboard/callbacks/experiment_wizard_callbacks.py

# If conflict is just imports or different sections:
# 1. Keep both sets of imports
# 2. Keep main's logic changes
# 3. Keep branch's constant replacements

# Edit file manually to combine:
# - Main's imports + Branch's constant imports
# - Main's logic changes + Branch's number replacements

git add packages/dashboard/callbacks/experiment_wizard_callbacks.py
```

**Pattern to follow**:
```python
# BEFORE (main):
import xxx
samples = min(n, 20480)  # Magic number

# AFTER (merged):
import xxx
from utils.constants import MAX_SAMPLES_PER_DATASET
samples = min(n, MAX_SAMPLES_PER_DATASET)  # Named constant
```

---

## 🟢 BRANCH 5: `implement-2fa-sessions-01Etw1s7XCyLZVs3MwzwUjdi`

**Team**: Team 8 (2FA and Session Tracking)
**Conflicts**: 1 file
**Cause**: Both branches created same documentation file

### Conflicted Files:

#### 1. `SECURITY_IMPLEMENTATION_ANALYSIS.md`
**Conflict Type**: Add/Add - both branches created same file with different content

**Main has**: Security analysis from Team 1/2
**Branch has**: 2FA implementation analysis from Team 8

**Resolution**: **KEEP BOTH** - Merge documents

**Steps**:
```bash
git merge --no-commit origin/claude/implement-2fa-sessions-01Etw1s7XCyLZVs3MwzwUjdi

# Manually merge the markdown files
# Keep all sections from both versions
# Organize into coherent structure

git add SECURITY_IMPLEMENTATION_ANALYSIS.md
```

**Suggested structure**:
```markdown
# SECURITY IMPLEMENTATION ANALYSIS

## Overview
[From main]

## Password Hashing Implementation
[From main - Team 2]

## 2FA Implementation
[From branch - Team 8]

## Session Tracking
[From branch - Team 8]

## Remaining Work
[Combine from both]
```

---

## 📋 RECOMMENDED MERGE ORDER

Merge in this order to minimize conflicts and dependencies:

### **Step 1**: Foundation - Authentication (Team 3)
```bash
git checkout main
git merge --no-ff origin/claude/fix-hardcoded-user-id-01BckhunZFZEAxp6BA6N8GbY
# Resolve webhook_callbacks.py conflict
# Creates utils/auth_utils.py for Team 4
git commit -m "Merge Team 3: Authentication integration batch 1"
```

### **Step 2**: Auth Completion (Team 4)
```bash
git merge --no-ff origin/claude/fix-auth-callbacks-batch-2-012DVCXu4SkYBP7jwLmB9EDT
# Resolve security_callbacks.py conflict
# Depends on Team 3's auth_utils.py
git commit -m "Merge Team 4: Authentication integration batch 2"
```

### **Step 3**: Documentation (Team 8)
```bash
git merge --no-ff origin/claude/implement-2fa-sessions-01Etw1s7XCyLZVs3MwzwUjdi
# Merge SECURITY_IMPLEMENTATION_ANALYSIS.md
git commit -m "Merge Team 8: 2FA and session tracking implementation"
```

### **Step 4**: Code Quality (Team 6)
```bash
git merge --no-ff origin/claude/fix-magic-numbers-01T2bWdYWoCanxW33rK1wVDw
# Resolve experiment_wizard_callbacks.py conflict
git commit -m "Merge Team 6: Replace magic numbers with constants"
```

### **Step 5**: UI Features (Team 9)
```bash
git merge --no-ff origin/claude/add-email-digest-ui-01Y9BnxZPQsoiK6srWG77751
# Resolve config.py and email_log.py conflicts
git commit -m "Merge Team 9: Email digest management UI"
```

---

## 🛠️ CONFLICT RESOLUTION SCRIPTS

### Quick Resolution Script
```bash
#!/bin/bash
# File: resolve_all_conflicts.sh

set -e  # Exit on error

echo "🔀 Starting conflict resolution..."

# Branch 1: Team 3 - Auth Batch 1
echo "📝 Merging Team 3..."
git merge --no-ff origin/claude/fix-hardcoded-user-id-01BckhunZFZEAxp6BA6N8GbY || true
git checkout --theirs packages/dashboard/callbacks/webhook_callbacks.py
git add packages/dashboard/callbacks/webhook_callbacks.py
git commit -m "Merge Team 3: Authentication integration batch 1"

# Branch 2: Team 4 - Auth Batch 2
echo "📝 Merging Team 4..."
git merge --no-ff origin/claude/fix-auth-callbacks-batch-2-012DVCXu4SkYBP7jwLmB9EDT || true
git checkout --theirs packages/dashboard/callbacks/security_callbacks.py
git add packages/dashboard/callbacks/security_callbacks.py
git commit -m "Merge Team 4: Authentication integration batch 2"

# Branch 3: Team 8 - 2FA
echo "📝 Merging Team 8..."
git merge --no-ff origin/claude/implement-2fa-sessions-01Etw1s7XCyLZVs3MwzwUjdi || true
# Manual merge required for SECURITY_IMPLEMENTATION_ANALYSIS.md
echo "⚠️  Manual merge needed for SECURITY_IMPLEMENTATION_ANALYSIS.md"
echo "Press Enter after resolving..."
read
git add SECURITY_IMPLEMENTATION_ANALYSIS.md
git commit -m "Merge Team 8: 2FA and session tracking implementation"

# Branch 4: Team 6 - Constants
echo "📝 Merging Team 6..."
git merge --no-ff origin/claude/fix-magic-numbers-01T2bWdYWoCanxW33rK1wVDw || true
echo "⚠️  Manual merge needed for experiment_wizard_callbacks.py"
echo "Press Enter after resolving..."
read
git add packages/dashboard/callbacks/experiment_wizard_callbacks.py
git commit -m "Merge Team 6: Replace magic numbers with constants"

# Branch 5: Team 9 - Email Digest UI
echo "📝 Merging Team 9..."
git merge --no-ff origin/claude/add-email-digest-ui-01Y9BnxZPQsoiK6srWG77751 || true
git checkout --theirs packages/dashboard/models/email_log.py
echo "⚠️  Manual merge needed for config.py"
echo "Press Enter after resolving..."
read
git add packages/dashboard/config.py packages/dashboard/models/email_log.py
git commit -m "Merge Team 9: Email digest management UI"

echo "✅ All merges complete!"
```

---

## ✅ POST-MERGE VERIFICATION

After all merges complete, run these checks:

### 1. **Code Verification**
```bash
# Check no conflict markers remain
grep -r "<<<<<<< HEAD" packages/dashboard/
grep -r ">>>>>>>" packages/dashboard/
# Should return nothing

# Check all imports resolve
python -c "import sys; sys.path.insert(0, 'dash_app'); import config"
python -c "import sys; sys.path.insert(0, 'dash_app'); from utils.auth_utils import get_current_user_id"
```

### 2. **Security Verification**
```bash
# No hardcoded user_id = 1
grep -rn "user_id = 1" packages/dashboard/callbacks/
# Should return 0 results

# No hardcoded credentials
grep -rn "lstm_password" packages/dashboard/
# Should return 0 results
```

### 3. **Functional Testing**
```bash
# Start app
cd dash_app
python app.py

# Should start without errors
# Test in browser:
# - Login page works
# - Settings page loads
# - Email digest management visible
```

### 4. **Database Migration**
```bash
# If using Alembic
alembic revision --autogenerate -m "Merge all team changes"
alembic upgrade head

# Check new tables
psql -d lstm_dashboard -c "\dt"
# Should see: session_logs, login_history
```

---

## 🚨 TROUBLESHOOTING

### **Issue**: Team 4 merge fails with "auth_utils not found"
**Cause**: Team 3 not merged yet
**Fix**: Merge Team 3 first (creates auth_utils.py)

### **Issue**: Imports fail after merge
**Cause**: Circular dependency
**Fix**: Check import order in `__init__.py` files

### **Issue**: Database errors after merge
**Cause**: New models not migrated
**Fix**: Run `alembic upgrade head`

### **Issue**: App crashes on startup
**Cause**: Missing environment variables
**Fix**: Copy `.env.example` to `.env` and configure

---

## 📊 FINAL STATUS

After successful merge:
- ✅ All 10 teams' work integrated
- ✅ No hardcoded credentials
- ✅ No hardcoded user IDs
- ✅ Password hashing implemented
- ✅ Database optimized
- ✅ Code quality improved
- ✅ 2FA infrastructure ready
- ✅ Email digest management UI
- ✅ Magic numbers replaced

**Estimated Time to Complete All Merges**: 1-2 hours (with breaks for testing)

---

**Last Updated**: 2025-11-22
**Status**: Ready for resolution
