> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see the main documentation.
>
> *Archived on: 2026-01-20*
> *Reason: Superseded by consolidated documentation*
# 🎯 TIER 1 IMPLEMENTATION PLAN: QUICK WINS (9 Features)

**Timeline:** 8 weeks  
**Team Size:** 2 developers (1 backend-focused, 1 full-stack)  
**Methodology:** Agile sprints (2-week iterations)

---

# FEATURE #1: API KEYS & RATE LIMITING

**Duration:** 1 week (5 days)  
**Priority:** P0 (Highest - Foundational)  
**Assigned To:** Backend Developer

---

## 1.1 OBJECTIVES

### Primary Objective
Enable programmatic access to the ML platform via secure API keys with rate limiting to prevent abuse.

### Success Criteria
- Users can generate/revoke API keys through dashboard
- API endpoints accept authentication via `X-API-Key` header
- Rate limiting enforces 1,000 requests/hour per key
- Exceeded rate limits return HTTP 429 with clear error message
- Admin can view all API keys and their usage statistics

### Business Value
- **External Integration:** Customers can integrate with CI/CD pipelines, notebooks, scripts
- **Developer Adoption:** Makes platform attractive to technical users
- **Scalability Foundation:** Required for future SaaS deployment
- **Revenue Enabler:** Paid tiers can offer higher rate limits

---

## 1.2 TECHNICAL SPECIFICATIONS

### Database Schema

```sql
-- Migration: 001_add_api_keys.sql

CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL UNIQUE,  -- bcrypt hash of the key
    name VARCHAR(100) NOT NULL,  -- User-provided name (e.g., "CI/CD Pipeline")
    prefix VARCHAR(20) NOT NULL,  -- First 8 chars for display (e.g., "sk_live_abc")
    scopes TEXT[] DEFAULT ARRAY['read', 'write'],  -- Permissions array
    rate_limit INTEGER DEFAULT 1000,  -- Requests per hour
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,  -- NULL = never expires
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_prefix ON api_keys(prefix);  -- For fast lookup
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;

-- Migration: 002_add_api_usage_tracking.sql

CREATE TABLE api_usage (
    id SERIAL PRIMARY KEY,
    api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,  -- GET, POST, etc.
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_usage_key_timestamp ON api_usage(api_key_id, timestamp DESC);
CREATE INDEX idx_api_usage_timestamp ON api_usage(timestamp) WHERE timestamp > NOW() - INTERVAL '30 days';  -- Partial index for recent data
```

### API Key Format

```
Format: sk_{env}_{random_32_bytes}

Examples:
- sk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6  (Production)
- sk_test_x9y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4  (Testing)

Components:
- sk: Prefix indicating "secret key"
- env: Environment (live/test)
- random: Cryptographically secure random 32 bytes (base64url encoded)

Security:
- Only shown ONCE at creation (like GitHub, Stripe)
- Stored as bcrypt hash in database (cost factor: 12)
- Prefix stored separately for UI display (e.g., "sk_live_a1b2...")
```

### Rate Limiting Strategy

```python
"""
Rate Limiting Architecture:

Storage: Redis (fast, atomic operations)
Algorithm: Sliding Window Counter
Window: 1 hour (3600 seconds)
Limit: 1000 requests per hour (configurable per key)

Redis Key Structure:
  rate_limit:{api_key_prefix}:{window_timestamp}
  
  Example: rate_limit:sk_live_abc:1718460000
  
Value: Integer counter
Expiry: 2 hours (auto-cleanup)

Sliding Window Implementation:
  1. Current timestamp: 1718461234
  2. Window start: 1718460000 (round down to hour)
  3. Increment: INCR rate_limit:sk_live_abc:1718460000
  4. Check: If count > 1000, reject with 429
  5. Set expiry: EXPIRE rate_limit:sk_live_abc:1718460000 7200
"""

# Pseudocode
def check_rate_limit(api_key_prefix):
    current_time = int(time.time())
    window_start = current_time - (current_time % 3600)  # Round to hour
    redis_key = f"rate_limit:{api_key_prefix}:{window_start}"
    
    # Increment counter
    count = redis.incr(redis_key)
    
    # Set expiry on first request in window
    if count == 1:
        redis.expire(redis_key, 7200)  # 2 hours
    
    # Check limit
    if count > rate_limit:
        return False, count, rate_limit
    
    return True, count, rate_limit
```

---

## 1.3 IMPLEMENTATION TASKS

### Day 1: Database & Models

**Task 1.1:** Create Database Migration
- **File:** `database/migrations/001_add_api_keys.sql`
- **Action:** Write migration SQL (see schema above)
- **Validation:** Run migration on dev database, verify tables created
- **Rollback:** Write corresponding down migration

**Task 1.2:** Implement SQLAlchemy Models
- **File:** `models/api_key.py`
- **Code:**
  ```python
  from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP, ARRAY
  from sqlalchemy.orm import relationship
  from models.base import Base
  
  class APIKey(Base):
      __tablename__ = 'api_keys'
      
      id = Column(Integer, primary_key=True)
      user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
      key_hash = Column(String(255), nullable=False, unique=True)
      name = Column(String(100), nullable=False)
      prefix = Column(String(20), nullable=False)
      scopes = Column(ARRAY(String), default=['read', 'write'])
      rate_limit = Column(Integer, default=1000)
      last_used_at = Column(TIMESTAMP)
      is_active = Column(Boolean, default=True)
      expires_at = Column(TIMESTAMP)
      created_at = Column(TIMESTAMP, default=func.now())
      updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
      
      user = relationship("User", back_populates="api_keys")
      
      def __repr__(self):
          return f"<APIKey(id={self.id}, name='{self.name}', prefix='{self.prefix}')>"
  ```

**Task 1.3:** Update User Model
- **File:** `models/user.py`
- **Action:** Add relationship: `api_keys = relationship("APIKey", back_populates="user")`

**Testing Criteria:**
- ✅ Migration runs without errors
- ✅ Can create APIKey record via SQLAlchemy session
- ✅ Foreign key constraint works (delete user → cascade deletes keys)
- ✅ `user.api_keys` returns list of keys

---

### Day 2: Service Layer

**Task 2.1:** Implement API Key Generation Service
- **File:** `services/api_key_service.py`
- **Code:**
  ```python
  import secrets
  import bcrypt
  from datetime import datetime, timedelta
  from models.api_key import APIKey
  from database.connection import get_db_session
  
  class APIKeyService:
      
      @staticmethod
      def generate_key(user_id: int, name: str, environment: str = 'live', 
                       rate_limit: int = 1000, expires_in_days: int = None) -> dict:
          """
          Generate a new API key for a user.
          
          Args:
              user_id: ID of the user
              name: Descriptive name (e.g., "CI/CD Pipeline")
              environment: 'live' or 'test'
              rate_limit: Requests per hour
              expires_in_days: Key expiry in days (None = never expires)
              
          Returns:
              dict with 'api_key' (plain text, show once) and 'record' (database object)
              
          Raises:
              ValueError: If user doesn't exist or name is empty
          """
          
          # Validate inputs
          if not name or len(name.strip()) == 0:
              raise ValueError("API key name cannot be empty")
          
          session = get_db_session()
          user = session.query(User).get(user_id)
          if not user:
              raise ValueError(f"User {user_id} not found")
          
          # Generate random key
          random_bytes = secrets.token_urlsafe(32)  # 32 bytes = 43 chars base64url
          api_key = f"sk_{environment}_{random_bytes}"
          
          # Hash for storage
          key_hash = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt(rounds=12))
          
          # Extract prefix for display
          prefix = api_key[:20]  # "sk_live_abc12345678"
          
          # Calculate expiry
          expires_at = None
          if expires_in_days:
              expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
          
          # Create database record
          api_key_record = APIKey(
              user_id=user_id,
              key_hash=key_hash.decode('utf-8'),
              name=name.strip(),
              prefix=prefix,
              rate_limit=rate_limit,
              expires_at=expires_at
          )
          
          session.add(api_key_record)
          session.commit()
          session.refresh(api_key_record)
          
          return {
              'api_key': api_key,  # Plain text, return to user ONCE
              'record': api_key_record
          }
      
      @staticmethod
      def verify_key(api_key: str) -> APIKey:
          """
          Verify API key and return corresponding database record.
          
          Args:
              api_key: Plain text API key from request header
              
          Returns:
              APIKey object if valid, None if invalid
          """
          
          # Extract prefix for fast lookup
          if len(api_key) < 20:
              return None
          
          prefix = api_key[:20]
          
          session = get_db_session()
          
          # Find keys with matching prefix (much faster than checking all hashes)
          candidates = session.query(APIKey).filter(
              APIKey.prefix == prefix,
              APIKey.is_active == True
          ).all()
          
          # Check hash for each candidate
          for candidate in candidates:
              if bcrypt.checkpw(api_key.encode('utf-8'), candidate.key_hash.encode('utf-8')):
                  # Update last used timestamp
                  candidate.last_used_at = datetime.utcnow()
                  session.commit()
                  return candidate
          
          return None
      
      @staticmethod
      def revoke_key(api_key_id: int, user_id: int) -> bool:
          """
          Revoke (deactivate) an API key.
          
          Args:
              api_key_id: ID of the key to revoke
              user_id: ID of the user (authorization check)
              
          Returns:
              True if revoked, False if not found or unauthorized
          """
          session = get_db_session()
          
          key = session.query(APIKey).filter(
              APIKey.id == api_key_id,
              APIKey.user_id == user_id
          ).first()
          
          if not key:
              return False
          
          key.is_active = False
          session.commit()
          return True
      
      @staticmethod
      def list_user_keys(user_id: int) -> list:
          """Get all API keys for a user."""
          session = get_db_session()
          return session.query(APIKey).filter(APIKey.user_id == user_id).all()
  ```

**Testing Criteria:**
- ✅ `generate_key()` returns valid key format (`sk_live_...`)
- ✅ Generated key is 52+ characters long
- ✅ Database stores hashed version (not plain text)
- ✅ `verify_key()` returns APIKey object for valid key
- ✅ `verify_key()` returns None for invalid/inactive key
- ✅ `revoke_key()` sets `is_active = False`
- ✅ Performance: `verify_key()` completes in <5ms (index on prefix)

---

### Day 3: Rate Limiting Middleware

**Task 3.1:** Implement Rate Limiter
- **File:** `api/rate_limiter.py`
- **Code:**
  ```python
  import time
  from functools import wraps
  from flask import request, jsonify
  from config import Config
  import redis
  
  # Initialize Redis connection
  redis_client = redis.Redis(
      host=Config.REDIS_HOST,
      port=Config.REDIS_PORT,
      db=Config.REDIS_DB,
      decode_responses=True
  )
  
  class RateLimiter:
      
      @staticmethod
      def check_rate_limit(api_key_id: int, rate_limit: int) -> tuple:
          """
          Check if request is within rate limit.
          
          Args:
              api_key_id: Database ID of the API key
              rate_limit: Requests per hour limit
              
          Returns:
              (allowed: bool, current_count: int, limit: int, reset_time: int)
          """
          
          current_time = int(time.time())
          window_start = current_time - (current_time % 3600)  # Round to hour
          redis_key = f"rate_limit:key_{api_key_id}:{window_start}"
          
          # Increment counter
          try:
              count = redis_client.incr(redis_key)
              
              # Set expiry on first request
              if count == 1:
                  redis_client.expire(redis_key, 7200)  # 2 hours
              
              # Calculate reset time (next hour boundary)
              reset_time = window_start + 3600
              
              # Check limit
              allowed = count <= rate_limit
              
              return allowed, count, rate_limit, reset_time
              
          except redis.RedisError as e:
              # Fail open (allow request if Redis is down)
              logger.error(f"Redis error in rate limiting: {e}")
              return True, 0, rate_limit, current_time + 3600
      
      @staticmethod
      def rate_limit_decorator(f):
          """
          Decorator to apply rate limiting to API endpoints.
          
          Usage:
              @app.route('/api/v1/predict')
              @RateLimiter.rate_limit_decorator
              def predict():
                  ...
          """
          @wraps(f)
          def decorated_function(*args, **kwargs):
              # Get API key from request (set by auth middleware)
              api_key_record = getattr(request, 'api_key', None)
              
              if not api_key_record:
                  # No API key (should be caught by auth middleware)
                  return jsonify({'error': 'Unauthorized'}), 401
              
              # Check rate limit
              allowed, count, limit, reset_time = RateLimiter.check_rate_limit(
                  api_key_id=api_key_record.id,
                  rate_limit=api_key_record.rate_limit
              )
              
              # Add rate limit headers to response
              response_headers = {
                  'X-RateLimit-Limit': str(limit),
                  'X-RateLimit-Remaining': str(max(0, limit - count)),
                  'X-RateLimit-Reset': str(reset_time)
              }
              
              if not allowed:
                  # Rate limit exceeded
                  return jsonify({
                      'error': 'rate_limit_exceeded',
                      'message': f'Rate limit of {limit} requests per hour exceeded. Resets at {reset_time}.',
                      'current_usage': count,
                      'limit': limit,
                      'reset_at': reset_time
                  }), 429, response_headers
              
              # Execute endpoint
              response = f(*args, **kwargs)
              
              # Add rate limit headers to successful response
              if isinstance(response, tuple):
                  # Response is (data, status_code, headers)
                  if len(response) == 3:
                      response[2].update(response_headers)
                  elif len(response) == 2:
                      response = (response[0], response[1], response_headers)
              else:
                  # Response is just data
                  response = (response, 200, response_headers)
              
              return response
          
          return decorated_function
  ```

**Testing Criteria:**
- ✅ First request: `X-RateLimit-Remaining` = 999
- ✅ After 1000 requests in 1 hour: Returns HTTP 429
- ✅ After hour resets: Counter resets to 0
- ✅ Redis down: Requests still allowed (fail-open)
- ✅ Different API keys have independent counters
- ✅ Response headers present on all requests

---

### Day 4: API Middleware & Endpoints

**Task 4.1:** Authentication Middleware
- **File:** `api/middleware.py`
- **Code:**
  ```python
  from flask import request, jsonify
  from functools import wraps
  from services.api_key_service import APIKeyService
  
  class AuthMiddleware:
      
      @staticmethod
      def require_api_key(f):
          """
          Decorator requiring valid API key for endpoint access.
          
          Checks for API key in:
            1. Header: X-API-Key
            2. Header: Authorization: Bearer <key>
            3. Query param: api_key (discouraged, but supported)
            
          Usage:
              @app.route('/api/v1/predict')
              @AuthMiddleware.require_api_key
              @RateLimiter.rate_limit_decorator
              def predict():
                  ...
          """
          @wraps(f)
          def decorated_function(*args, **kwargs):
              # Extract API key from request
              api_key = None
              
              # Method 1: X-API-Key header (preferred)
              api_key = request.headers.get('X-API-Key')
              
              # Method 2: Authorization Bearer header
              if not api_key:
                  auth_header = request.headers.get('Authorization', '')
                  if auth_header.startswith('Bearer '):
                      api_key = auth_header[7:]  # Remove "Bearer " prefix
              
              # Method 3: Query parameter (not recommended, but supported)
              if not api_key:
                  api_key = request.args.get('api_key')
              
              if not api_key:
                  return jsonify({
                      'error': 'missing_api_key',
                      'message': 'API key required. Provide via X-API-Key header or Authorization: Bearer header.'
                  }), 401
              
              # Verify API key
              api_key_record = APIKeyService.verify_key(api_key)
              
              if not api_key_record:
                  return jsonify({
                      'error': 'invalid_api_key',
                      'message': 'Invalid or inactive API key.'
                  }), 401
              
              # Check expiry
              if api_key_record.expires_at:
                  from datetime import datetime
                  if datetime.utcnow() > api_key_record.expires_at:
                      return jsonify({
                          'error': 'expired_api_key',
                          'message': 'API key has expired.'
                      }), 401
              
              # Attach API key record to request for downstream use
              request.api_key = api_key_record
              request.user_id = api_key_record.user_id
              
              # Execute endpoint
              return f(*args, **kwargs)
          
          return decorated_function
  ```

**Task 4.2:** API Key Management Endpoints
- **File:** `api/v1/api_keys.py`
- **Code:**
  ```python
  from flask import Blueprint, request, jsonify
  from auth.decorators import login_required
  from services.api_key_service import APIKeyService
  
  api_keys_bp = Blueprint('api_keys', __name__)
  
  @api_keys_bp.route('/api/v1/api-keys', methods=['GET'])
  @login_required
  def list_api_keys():
      """List all API keys for current user."""
      user_id = request.user_id
      keys = APIKeyService.list_user_keys(user_id)
      
      return jsonify({
          'api_keys': [{
              'id': key.id,
              'name': key.name,
              'prefix': key.prefix,
              'rate_limit': key.rate_limit,
              'scopes': key.scopes,
              'is_active': key.is_active,
              'last_used_at': key.last_used_at.isoformat() if key.last_used_at else None,
              'expires_at': key.expires_at.isoformat() if key.expires_at else None,
              'created_at': key.created_at.isoformat()
          } for key in keys]
      }), 200
  
  @api_keys_bp.route('/api/v1/api-keys', methods=['POST'])
  @login_required
  def create_api_key():
      """Generate a new API key."""
      user_id = request.user_id
      data = request.get_json()
      
      # Validate input
      name = data.get('name', '').strip()
      if not name:
          return jsonify({'error': 'name is required'}), 400
      
      rate_limit = data.get('rate_limit', 1000)
      environment = data.get('environment', 'live')
      expires_in_days = data.get('expires_in_days')
      
      # Generate key
      result = APIKeyService.generate_key(
          user_id=user_id,
          name=name,
          environment=environment,
          rate_limit=rate_limit,
          expires_in_days=expires_in_days
      )
      
      return jsonify({
          'api_key': result['api_key'],  # Plain text, shown ONCE
          'id': result['record'].id,
          'name': result['record'].name,
          'prefix': result['record'].prefix,
          'rate_limit': result['record'].rate_limit,
          'expires_at': result['record'].expires_at.isoformat() if result['record'].expires_at else None,
          'message': 'API key generated. Save it securely - you won\'t be able to see it again.'
      }), 201
  
  @api_keys_bp.route('/api/v1/api-keys/<int:key_id>', methods=['DELETE'])
  @login_required
  def revoke_api_key(key_id):
      """Revoke an API key."""
      user_id = request.user_id
      success = APIKeyService.revoke_key(key_id, user_id)
      
      if not success:
          return jsonify({'error': 'API key not found or unauthorized'}), 404
      
      return jsonify({'message': 'API key revoked successfully'}), 200
  ```

**Testing Criteria:**
- ✅ `GET /api/v1/api-keys` returns user's keys (hidden full key)
- ✅ `POST /api/v1/api-keys` returns plain text key once
- ✅ `DELETE /api/v1/api-keys/{id}` revokes key
- ✅ Revoked key cannot authenticate subsequent requests
- ✅ Unauthorized user cannot delete another user's key

---

### Day 5: UI Integration & Testing

**Task 5.1:** Settings Page UI
- **File:** `layouts/settings.py` (enhance existing)
- **Add API Keys Tab:**
  ```python
  # In settings.py, add to tabs
  
  dbc.Tab(label="API Keys", tab_id="api-keys", children=[
      html.Div([
          html.H4("API Keys", className="mt-3"),
          html.P("Use API keys to authenticate programmatic access to the platform."),
          
          # Existing keys table
          html.Div(id='api-keys-table'),
          
          # Generate new key button
          dbc.Button("+ Generate New API Key", id='generate-key-btn', color="primary", className="mt-3"),
          
          # Modal for key generation
          dbc.Modal([
              dbc.ModalHeader("Generate New API Key"),
              dbc.ModalBody([
                  dbc.Label("Name (e.g., 'CI/CD Pipeline')"),
                  dbc.Input(id='key-name-input', placeholder="Enter descriptive name"),
                  dbc.Label("Rate Limit (requests/hour)", className="mt-2"),
                  dbc.Input(id='key-rate-limit-input', type="number", value=1000),
                  html.Div(id='generated-key-display', className="mt-3")
              ]),
              dbc.ModalFooter([
                  dbc.Button("Cancel", id='cancel-key-btn', className="mr-2"),
                  dbc.Button("Generate", id='confirm-generate-btn', color="primary")
              ])
          ], id='generate-key-modal', is_open=False)
      ])
  ])
  ```

**Task 5.2:** Callbacks
- **File:** `callbacks/api_key_callbacks.py`
- **Implement:**
  - Load keys table
  - Generate key modal
  - Display generated key (with copy button)
  - Revoke key action

**Task 5.3:** End-to-End Testing
- **Manual Test Plan:**
  ```
  TEST CASE 1: Generate API Key
  1. Navigate to Settings → API Keys
  2. Click "Generate New API Key"
  3. Enter name: "Test Key"
  4. Click "Generate"
  5. ✅ Success modal shows full key (starts with "sk_live_")
  6. ✅ Key appears in table with masked format (sk_***abc)
  7. Copy key to clipboard
  
  TEST CASE 2: Authenticate with API Key
  1. Open terminal
  2. Run: curl -H "X-API-Key: sk_live_..." http://localhost:8050/api/v1/experiments
  3. ✅ Returns 200 OK with experiments list
  4. ✅ Response headers include X-RateLimit-Remaining: 999
  
  TEST CASE 3: Rate Limiting
  1. Write script to make 1001 requests
  2. Run script
  3. ✅ First 1000 requests: 200 OK
  4. ✅ Request 1001: 429 Rate Limit Exceeded
  5. ✅ Error message includes reset time
  6. Wait 1 hour
  7. ✅ Rate limit resets, requests succeed again
  
  TEST CASE 4: Revoke Key
  1. In UI, click "Revoke" on test key
  2. Confirm deletion
  3. ✅ Key disappears from table
  4. Use revoked key in API request
  5. ✅ Returns 401 Unauthorized
  
  TEST CASE 5: Invalid Key
  1. curl -H "X-API-Key: sk_live_invalid" http://localhost:8050/api/v1/experiments
  2. ✅ Returns 401 Unauthorized with clear error message
  ```

---

## 1.4 DO'S AND DON'TS

### ✅ DO's

1. **DO use bcrypt for hashing** (not SHA256 or MD5)
   - Reason: Bcrypt is designed for password/key hashing (slow by design)

2. **DO show the full API key ONLY ONCE**
   - After generation, only display prefix (e.g., `sk_***abc`)
   - Cannot retrieve full key later (security best practice)

3. **DO use atomic Redis operations**
   - `INCR` is atomic (thread-safe)
   - Prevents race conditions in rate limiting

4. **DO fail open if Redis is down**
   - Allow requests if rate limiter fails
   - Reason: Availability > strict rate limiting

5. **DO index the prefix column**
   - Makes `verify_key()` fast (O(1) instead of O(n))

6. **DO set Redis key expiry**
   - Prevents memory leak if cleanup fails
   - 2-hour expiry ensures old windows are deleted

7. **DO return clear error messages**
   - Example: `"Rate limit of 1000 req/hr exceeded. Resets at 1634567890."`
   - Helps developers debug issues

8. **DO log API key usage**
   - Insert into `api_usage` table (asynchronously)
   - Enables analytics and abuse detection

9. **DO validate input in service layer**
   - Don't trust controller/UI to validate
   - Service layer is source of truth

10. **DO use environment-specific prefixes**
    - `sk_live_` for production
    - `sk_test_` for development
    - Makes it obvious which keys are which

### ❌ DON'Ts

1. **DON'T store plain text keys in database**
   - Security violation
   - Use bcrypt hash

2. **DON'T use MD5 or SHA1 for hashing**
   - Too fast (vulnerable to brute force)
   - Use bcrypt with cost factor 12

3. **DON'T allow unlimited rate limits**
   - Even admins should have limits (e.g., 10,000/hr)
   - Prevents accidental DOS

4. **DON'T return full key after creation**
   - List endpoint should return prefix only
   - Prevents key leakage in logs

5. **DON'T hard-code rate limits**
   - Store in database (per-key configuration)
   - Allows flexible pricing tiers later

6. **DON'T forget to update `last_used_at`**
   - Used for analytics and inactive key cleanup
   - Update on every successful auth

7. **DON'T allow API keys in URL query params in production**
   - Query params logged by proxies/load balancers
   - Support it, but warn users to use headers

8. **DON'T forget timezone handling**
   - Store all timestamps in UTC
   - Convert to user timezone in UI only

9. **DON'T block on Redis writes**
   - Update `last_used_at` asynchronously
   - Don't delay response for analytics

10. **DON'T skip migration rollback**
    - Always write down migration
    - Test rollback before deploying

---

## 1.5 TESTING CHECKLIST

### Unit Tests (`tests/test_api_key_service.py`)

```python
def test_generate_key_returns_valid_format():
    """Generated key should match format sk_{env}_{32_bytes}"""
    result = APIKeyService.generate_key(user_id=1, name="Test")
    assert result['api_key'].startswith('sk_live_')
    assert len(result['api_key']) > 50

def test_generate_key_stores_hash_not_plaintext():
    """Database should store bcrypt hash, not plain key"""
    result = APIKeyService.generate_key(user_id=1, name="Test")
    assert result['record'].key_hash != result['api_key']
    assert result['record'].key_hash.startswith('$2b$')  # bcrypt prefix

def test_verify_key_accepts_valid_key():
    """Valid key should authenticate successfully"""
    result = APIKeyService.generate_key(user_id=1, name="Test")
    verified = APIKeyService.verify_key(result['api_key'])
    assert verified is not None
    assert verified.user_id == 1

def test_verify_key_rejects_invalid_key():
    """Invalid key should return None"""
    verified = APIKeyService.verify_key("sk_live_invalid123")
    assert verified is None

def test_revoke_key_deactivates():
    """Revoked key should not authenticate"""
    result = APIKeyService.generate_key(user_id=1, name="Test")
    APIKeyService.revoke_key(result['record'].id, user_id=1)
    verified = APIKeyService.verify_key(result['api_key'])
    assert verified is None

def test_rate_limiter_enforces_limit():
    """Rate limiter should reject after limit"""
    from api.rate_limiter import RateLimiter
    
    # Mock Redis to return counts
    for i in range(1, 1002):
        allowed, count, limit, _ = RateLimiter.check_rate_limit(1, 1000)
        if i <= 1000:
            assert allowed == True
        else:
            assert allowed == False
```

### Integration Tests (`tests/integration/test_api_endpoints.py`)

```python
def test_api_endpoint_requires_key(test_client):
    """Endpoint should return 401 without API key"""
    response = test_client.get('/api/v1/experiments')
    assert response.status_code == 401

def test_api_endpoint_accepts_valid_key(test_client, api_key):
    """Endpoint should return 200 with valid API key"""
    response = test_client.get('/api/v1/experiments', headers={'X-API-Key': api_key})
    assert response.status_code == 200

def test_rate_limit_headers_present(test_client, api_key):
    """Response should include rate limit headers"""
    response = test_client.get('/api/v1/experiments', headers={'X-API-Key': api_key})
    assert 'X-RateLimit-Limit' in response.headers
    assert 'X-RateLimit-Remaining' in response.headers
    assert 'X-RateLimit-Reset' in response.headers

def test_rate_limit_enforced(test_client, api_key):
    """Should return 429 after exceeding limit"""
    # Make 1001 requests
    for i in range(1001):
        response = test_client.get('/api/v1/experiments', headers={'X-API-Key': api_key})
        if i < 1000:
            assert response.status_code == 200
        else:
            assert response.status_code == 429
```

### Manual QA Checklist

- [ ] Generate key through UI → Key displayed once
- [ ] Copy key, refresh page → Key not visible (prefix only)
- [ ] Use key in curl request → Authenticates successfully
- [ ] Check response headers → Rate limit headers present
- [ ] Make 1001 requests → 1001st returns 429
- [ ] Wait 1 hour → Rate limit resets
- [ ] Revoke key in UI → Key disappears
- [ ] Use revoked key → Returns 401
- [ ] Try invalid key → Returns 401 with clear message
- [ ] Admin views all users' keys → Only own keys visible

---

## 1.6 SUCCESS METRICS

### Quantitative
- 100% of API endpoints protected by auth middleware
- Rate limiter responds in <5ms (Redis latency)
- Zero plain text keys in database (audit with: `SELECT * FROM api_keys WHERE key_hash NOT LIKE '$2b$%'`)
- 95%+ test coverage for `api_key_service.py` and `rate_limiter.py`

### Qualitative
- Developers can generate keys without contacting support
- Error messages are clear and actionable
- UI is intuitive (no documentation needed for basic use)

---

## 1.7 ROLLOUT PLAN

### Phase 1: Internal Testing (Day 1-2 of Week 2)
- Deploy to dev environment
- Test with 3 internal users
- Collect feedback on UI and error messages

### Phase 2: Beta (Day 3-4 of Week 2)
- Deploy to staging environment
- Invite 10 power users to generate keys
- Monitor for errors in Sentry

### Phase 3: Production (Day 5 of Week 2)
- Deploy to production (Friday evening, low traffic)
- Announce feature in Monday team meeting
- Create documentation in wiki
- Monitor usage for first week

### Rollback Plan
If critical issues found:
1. Disable rate limiting (set all limits to 1,000,000)
2. Revert middleware changes (allow non-key access temporarily)
3. Fix issues in dev
4. Redeploy next week

---

**END OF FEATURE #1 PLAN**

---
