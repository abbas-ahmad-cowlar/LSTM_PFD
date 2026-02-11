# IDB 2.2: Backend Services Best Practices

> **IDB ID:** 2.2  
> **Domain:** Dashboard Platform  
> **Source Directory:** `packages/dashboard/services/`  
> **Extraction Date:** 2026-01-23

---

## Overview

This document extracts **proven patterns and conventions** from the Backend Services sub-block that should be adopted by other teams for consistency across the LSTM_PFD project.

---

## 1. Service Method Conventions

### 1.1 Static Method Pattern

Most services use `@staticmethod` for stateless operations:

```python
# ✅ Preferred: Stateless service methods
class HPOService:
    @staticmethod
    def create_campaign(name: str, method: str, ...) -> Optional[Dict]:
        with get_db_session() as session:
            # Business logic
            pass
```

**When to Use**: Operations that don't require instance state.

### 1.2 Return Type Standardization

**Pattern A — Tuple Returns** (for operations with error states):

```python
# AuthenticationService pattern
def verify_totp(user_id: int, code: str) -> Tuple[bool, str]:
    """Returns: (success, error_message)"""
    if not is_valid:
        return False, "Invalid code"
    return True, ""
```

**Pattern B — Dict Returns** (for data retrieval):

```python
# HPOService pattern
def get_campaign(campaign_id: int) -> Optional[Dict]:
    """Returns: Campaign dict or None"""
```

**Pattern C — Result Dict** (for complex operations):

```python
# EmailProvider pattern
def send(...) -> Dict[str, Any]:
    """Returns: {'success': bool, 'message_id': str, 'error': str}"""
```

### 1.3 Comprehensive Docstrings

All public methods should include:

```python
def verify_backup_code(user_id: int, code: str, ip_address: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verify and consume a backup code.

    Args:
        user_id: ID of the user
        code: Backup code to verify
        ip_address: IP address (for audit trail)

    Returns:
        Tuple of (success, error_message)

    Example:
        >>> success, error = AuthenticationService.verify_backup_code(1, "ABCD-EFGH-IJKL")
        >>> if success:
        ...     print("Code verified and consumed")
    """
```

---

## 2. Error Handling Patterns

### 2.1 Graceful Degradation

Services should fail gracefully when optional dependencies are unavailable:

```python
# CacheService - cache miss doesn't break functionality
@staticmethod
def get(key: str) -> Optional[Any]:
    if redis_client is None:
        return None  # Graceful degradation
    try:
        value = redis_client.get(key)
        return json.loads(value) if value else None
    except Exception as e:
        logger.error(f"Cache get error for key '{key}': {e}")
        return None  # Don't crash on cache errors
```

### 2.2 Fail-Open for Non-Critical Paths

Rate limiters should fail open to avoid blocking core functionality:

```python
# EmailRateLimiter pattern
def can_send(self) -> bool:
    try:
        # Rate limiting logic
        pass
    except Exception as e:
        logger.error(f"Rate limiter error (failing open): {e}")
        return True  # Fail open to avoid blocking emails
```

### 2.3 Structured Error Returns

Never raise exceptions from service methods — return structured errors:

```python
# ✅ Good: Structured error return
def setup_2fa(user_id: int) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    try:
        # ... logic
        return True, qr_code, secret, None
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return False, None, None, str(e)

# ❌ Avoid: Raising exceptions
def setup_2fa(user_id: int) -> str:
    raise AuthenticationError("User not found")  # Forces try/catch on caller
```

### 2.4 Logging with Context

Always include relevant context in error logs:

```python
logger.error(f"Error verifying 2FA code for user {user_id}: {e}", exc_info=True)
logger.warning(f"Invalid backup code attempt for user {user_id} from IP {ip_address}")
```

---

## 3. Transaction Management

### 3.1 Context Manager Pattern

**Always** use the context manager for database sessions:

```python
# ✅ Standard pattern
with get_db_session() as session:
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        return False, "User not found"

    user.some_field = new_value
    session.add(new_object)
    session.commit()

    return True, ""
```

### 3.2 Early Returns Within Transaction

Validate and return early before committing:

```python
with get_db_session() as session:
    user = session.query(User).filter_by(id=user_id).first()

    # ✅ Early validation before any mutations
    if not user:
        return False, None, "User not found"
    if not user.totp_secret:
        return False, None, "2FA not set up"

    # Proceed with mutations only after validation passes
    session.commit()
```

### 3.3 Atomic Operations

Group related operations in a single transaction:

```python
with get_db_session() as session:
    # Delete old codes
    session.query(BackupCode).filter_by(user_id=user_id).delete()

    # Create new codes (all or nothing)
    for code in new_codes:
        session.add(BackupCode(user_id=user_id, code_hash=hash(code)))

    session.commit()  # Single commit for entire operation
```

---

## 4. Caching Strategies

### 4.1 Cache-Aside Pattern

```python
def get_dataset_stats(dataset_id: int) -> Dict:
    # 1. Check cache first
    cache_key = f"dataset:stats:{dataset_id}"
    cached = CacheService.get(cache_key)
    if cached:
        return cached

    # 2. Compute on cache miss
    with get_db_session() as session:
        stats = _compute_stats(dataset_id, session)

    # 3. Store in cache with TTL
    CacheService.set(cache_key, stats, ttl=3600)
    return stats
```

### 4.2 Cache Invalidation by Pattern

```python
# Invalidate all cached stats for a dataset
CacheService.invalidate_pattern(f"dataset:stats:{dataset_id}*")

# Invalidate all user-related cache
CacheService.invalidate_pattern(f"user:{user_id}:*")
```

### 4.3 TTL Configuration

Use centralized TTL constants:

```python
from dashboard_config import CACHE_TTL_SHORT, CACHE_TTL_MEDIUM, CACHE_TTL_LONG

# Short TTL (5 min) - frequently changing data
CacheService.set(key, value, ttl=CACHE_TTL_SHORT)

# Medium TTL (1 hour) - moderately stable data
CacheService.set(key, value, ttl=CACHE_TTL_MEDIUM)

# Long TTL (24 hours) - rarely changing data
CacheService.set(key, value, ttl=CACHE_TTL_LONG)
```

---

## 5. Logging Conventions

### 5.1 Logger Initialization

Every service file starts with:

```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

### 5.2 Log Levels

| Level     | Use Case                     | Example                                         |
| --------- | ---------------------------- | ----------------------------------------------- |
| `DEBUG`   | Development tracing          | `logger.debug(f"Processing item {i}")`          |
| `INFO`    | Normal operations            | `logger.info(f"Email sent to {email}")`         |
| `WARNING` | Recoverable issues           | `logger.warning(f"Rate limit exceeded")`        |
| `ERROR`   | Failures requiring attention | `logger.error(f"DB error: {e}", exc_info=True)` |

### 5.3 Structured Logging Context

Include actionable context:

```python
# ✅ Good: Actionable context
logger.info(f"Created session for user {user_id} from IP {ip_address}")
logger.warning(f"User {user_id} locked out due to {MAX_2FA_ATTEMPTS} failed attempts")

# ❌ Avoid: Vague messages
logger.info("Session created")
logger.warning("User locked out")
```

---

## 6. Dependency Injection Patterns

### 6.1 Factory Pattern for Providers

Abstract provider creation behind factories:

```python
class NotificationProviderFactory:
    _providers: Dict[str, NotificationProvider] = {}  # Instance cache

    @staticmethod
    def get_provider(provider_type: str, config: Dict = None) -> NotificationProvider:
        # 1. Check feature flags
        if provider_type == 'slack' and not NOTIFICATIONS_SLACK_ENABLED:
            raise ValueError("Slack notifications are disabled")

        # 2. Return cached or create new
        cache_key = f"{provider_type}_{id(config) if config else 'default'}"
        if cache_key not in cls._providers:
            if provider_type == 'slack':
                from providers.slack_notifier import SlackNotifier
                cls._providers[cache_key] = SlackNotifier(config or {})

        return cls._providers[cache_key]
```

### 6.2 Lazy Imports for Circular Dependencies

When ServiceA and ServiceB reference each other:

```python
# ✅ Lazy import inside method
class SearchService:
    @staticmethod
    def _get_suggestions(session, parsed_query: Dict) -> List[str]:
        from services.tag_service import TagService  # Lazy import
        tags = TagService.suggest_tags(session, parsed_query.get('keywords', [''])[0])
        return [f"tag:{t.name}" for t in tags]
```

### 6.3 Abstract Base Classes

Define interfaces for swappable implementations:

```python
from abc import ABC, abstractmethod

class EmailProvider(ABC):
    @abstractmethod
    def send(self, to_email: str, subject: str, html_body: str, text_body: str) -> Dict[str, Any]:
        pass

class SendGridProvider(EmailProvider):
    def send(self, ...) -> Dict[str, Any]:
        # SendGrid-specific implementation

class SMTPProvider(EmailProvider):
    def send(self, ...) -> Dict[str, Any]:
        # SMTP-specific implementation
```

### 6.4 Feature Flag Checks

Guard provider instantiation with feature flags:

```python
@staticmethod
def get_enabled_providers() -> List[str]:
    from dashboard_config import (
        EMAIL_ENABLED,
        NOTIFICATIONS_SLACK_ENABLED,
        NOTIFICATIONS_TEAMS_ENABLED
    )

    enabled = []
    if EMAIL_ENABLED:
        enabled.append('email')
    if NOTIFICATIONS_SLACK_ENABLED:
        enabled.append('slack')
    return enabled
```

---

## 7. Rate Limiting

### 7.1 Token Bucket Algorithm (Redis-Backed)

```python
class EmailRateLimiter:
    def __init__(self, redis_client, max_emails_per_minute: int = 100):
        self.redis = redis_client
        self.max_emails_per_minute = max_emails_per_minute
        self.key = "email_rate_limit"

    def can_send(self) -> bool:
        current = self.redis.get(self.key)
        if current is None:
            self.redis.setex(self.key, 60, 1)  # 60s TTL
            return True

        if int(current) < self.max_emails_per_minute:
            self.redis.incr(self.key)
            return True

        return False
```

### 7.2 In-Memory Rate Limiting (Development Only)

```python
# ⚠️ WARNING: Not suitable for production multi-process deployments
_attempts: Dict[int, List[datetime]] = defaultdict(list)

def _check_rate_limit(user_id: int) -> Tuple[bool, str]:
    recent = [a for a in _attempts[user_id] if datetime.utcnow() - a < timedelta(minutes=15)]
    if len(recent) >= MAX_ATTEMPTS:
        return True, f"Too many attempts. Try again later."
    return False, ""
```

---

## 8. Security Patterns

### 8.1 Secure Secret Generation

```python
import secrets
import pyotp

# TOTP secrets
secret = pyotp.random_base32()  # 32+chars base32

# Session tokens
token = secrets.token_urlsafe(32)  # URL-safe random token

# Backup codes
code = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(12))
```

### 8.2 Credential Hashing

```python
import bcrypt

# Hash with cost factor 12
code_hash = bcrypt.hashpw(code.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

# Verify
if bcrypt.checkpw(input_code.encode('utf-8'), stored_hash.encode('utf-8')):
    # Valid
```

### 8.3 Prefix-Based API Key Lookup

O(1) key lookup without full bcrypt comparison on every request:

```python
# Key format: sk_live_[prefix][secret]
# Store: prefix (indexed), bcrypt(secret)

def verify_key(api_key: str) -> Optional[APIKey]:
    prefix = api_key[8:28]  # Extract prefix portion
    candidates = session.query(APIKey).filter_by(prefix=prefix).all()  # Fast indexed lookup

    for candidate in candidates:
        if bcrypt.checkpw(api_key.encode(), candidate.key_hash.encode()):
            return candidate
    return None
```

---

## Quick Reference Card

| Category     | Pattern            | Example                             |
| ------------ | ------------------ | ----------------------------------- |
| DB Session   | Context manager    | `with get_db_session() as session:` |
| Error Return | Tuple or Dict      | `return False, "error message"`     |
| Logging      | `__name__` logger  | `logger = setup_logger(__name__)`   |
| Caching      | Cache-aside        | Check → Compute → Store             |
| Provider     | Factory + ABC      | `Factory.get_provider(type)`        |
| Rate Limit   | Redis token bucket | `redis.setex(key, 60, 1)`           |
| Secrets      | `secrets` module   | `secrets.token_urlsafe(32)`         |
| Hashing      | bcrypt rounds 12   | `bcrypt.gensalt(rounds=12)`         |

---

_Extracted from IDB 2.2 Backend Services for LSTM_PFD project standardization._
