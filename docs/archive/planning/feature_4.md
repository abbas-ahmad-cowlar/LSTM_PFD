# FEATURE #4: SLACK/TEAMS WEBHOOK INTEGRATION

**Duration:** 1 week (5 days)  
**Priority:** P0 (High - Team collaboration, viral adoption)  
**Assigned To:** Backend Developer

---

## 4.1 OBJECTIVES

### Primary Objective
Enable users to receive ML experiment notifications directly in their team's Slack or Microsoft Teams channels via webhook integration, facilitating team collaboration and increasing platform visibility within organizations.

### Success Criteria
- Users can configure Slack/Teams webhooks in settings
- Notifications post to channels within 30 seconds of event occurrence
- Rich message formatting (buttons, colors, structured data)
- Webhook failures don't break core notification system (graceful degradation)
- Feature can be enabled/disabled globally via config flag
- Individual users can enable/disable per event type
- Webhook logs stored for debugging
- System handles webhook rate limits (Slack: 1 msg/sec per webhook)

### Business Value
- **Viral Growth:** Notifications in team channels expose product to colleagues → organic adoption
- **Team Collaboration:** Entire team sees training results → faster decision-making
- **Professional Image:** Rich Slack/Teams messages = enterprise-ready
- **Reduced Context Switching:** Stay in Slack, no need to open dashboard for updates
- **Async Communication:** Night training finishes → Team sees results next morning

---

## 4.2 ARCHITECTURAL PRINCIPLES (MODULARITY)

### Feature Toggle System

```
Design Philosophy: All integrations should be PLUGGABLE

Key Requirements:
1. Global Kill Switch: Disable entire integration without code changes
2. Provider Abstraction: Easy to add new providers (Discord, Mattermost)
3. Graceful Degradation: If Slack fails, other channels (email) still work
4. User-Level Control: Each user opts in/out independently
5. Event-Level Control: User can enable Slack for "training.complete" but not "training.started"

Implementation Strategy:
- Feature flags in config file (environment variables)
- Provider factory pattern (easy to add new providers)
- Separate service class for each provider (SlackNotifier, TeamsNotifier)
- Notification routing logic decoupled from providers
```

### Configuration Structure

```yaml
# config/notifications.yaml (or environment variables)

# GLOBAL FEATURE FLAGS
NOTIFICATIONS_EMAIL_ENABLED: true
NOTIFICATIONS_SLACK_ENABLED: true           # ← Global Slack toggle
NOTIFICATIONS_TEAMS_ENABLED: true           # ← Global Teams toggle
NOTIFICATIONS_WEBHOOK_ENABLED: true         # ← Custom webhooks toggle
NOTIFICATIONS_SMS_ENABLED: false            # ← Future: SMS (disabled for now)

# PROVIDER-SPECIFIC SETTINGS
SLACK_RATE_LIMIT_PER_WEBHOOK: 1             # 1 message per second
SLACK_RETRY_ATTEMPTS: 3
SLACK_TIMEOUT_SECONDS: 10

TEAMS_RATE_LIMIT_PER_WEBHOOK: 2             # 2 messages per second
TEAMS_RETRY_ATTEMPTS: 3
TEAMS_TIMEOUT_SECONDS: 10

# WEBHOOK SETTINGS
WEBHOOK_CUSTOM_TIMEOUT_SECONDS: 5
WEBHOOK_CUSTOM_RETRY_ATTEMPTS: 2

# FEATURE-SPECIFIC TOGGLES
NOTIFICATIONS_ENABLE_RICH_FORMATTING: true  # Rich cards vs plain text
NOTIFICATIONS_ENABLE_MENTIONS: true         # Allow @channel, @user mentions
NOTIFICATIONS_ENABLE_DIGEST_SLACK: false    # Weekly digest to Slack (future)
```

### Modular Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  EVENT SOURCE (Training Task, HPO Task, etc.)          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  NOTIFICATION SERVICE (Central Router)                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 1. Check Global Feature Flags                     │ │
│  │    if not Config.NOTIFICATIONS_SLACK_ENABLED:     │ │
│  │        skip Slack                                  │ │
│  │                                                    │ │
│  │ 2. Load User Preferences (database)               │ │
│  │    if user.slack_enabled_for_event(event_type):   │ │
│  │        route to Slack                              │ │
│  │                                                    │ │
│  │ 3. Route to Enabled Channels                      │ │
│  └───────────────────────────────────────────────────┘ │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
   │Email │  │Slack │  │Teams │  │Custom│
   │      │  │      │  │      │  │Webhook│
   └──────┘  └──────┘  └──────┘  └──────┘
      │          │          │          │
      ▼          ▼          ▼          ▼
   [User's   [Team    [Team    [Custom
    Inbox]    Slack]   Teams]   System]

PLUGGABLE PROVIDERS:
- Each provider is independent class
- Implements common interface: NotificationProvider
- Easy to add new providers (Discord, Mattermost, etc.)
- Failures in one provider don't affect others
```

---

## 4.3 PROVIDER ABSTRACTION LAYER

### Interface Design

```python
# services/notification_providers/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class NotificationMessage:
    """
    Standardized message format across all providers.
    Providers translate this to their specific format.
    """
    title: str
    body: str
    event_type: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any]  # Event-specific data
    actions: Optional[list] = None  # Buttons/links
    color: Optional[str] = None  # Hex color for sidebar/accent
    
    # Provider-specific overrides (optional)
    slack_override: Optional[Dict] = None
    teams_override: Optional[Dict] = None


class NotificationProvider(ABC):
    """
    Abstract base class for all notification providers.
    
    New providers (Discord, Mattermost) simply implement this interface.
    """
    
    @abstractmethod
    def send(self, webhook_url: str, message: NotificationMessage) -> bool:
        """
        Send notification via provider.
        
        Args:
            webhook_url: Provider-specific webhook URL
            message: Standardized message object
            
        Returns:
            True if sent successfully, False if failed
            
        Raises:
            ProviderError: If provider-specific error occurs
        """
        pass
    
    @abstractmethod
    def validate_webhook_url(self, webhook_url: str) -> bool:
        """
        Validate that webhook URL is correctly formatted for this provider.
        
        Args:
            webhook_url: URL to validate
            
        Returns:
            True if valid format, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'slack', 'teams')"""
        pass
    
    @abstractmethod
    def supports_rich_formatting(self) -> bool:
        """Return True if provider supports rich cards/buttons"""
        pass
```

### Provider Factory Pattern

```python
# services/notification_providers/factory.py

from config import Config

class NotificationProviderFactory:
    """
    Factory for creating notification provider instances.
    
    Centralizes provider instantiation logic.
    Makes it easy to add new providers.
    """
    
    _providers = {}  # Cache provider instances
    
    @staticmethod
    def get_provider(provider_type: str) -> NotificationProvider:
        """
        Get provider instance by type.
        
        Args:
            provider_type: 'slack', 'teams', 'webhook', etc.
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider type unknown or disabled
        """
        
        # Check if provider globally enabled
        if provider_type == 'slack':
            if not Config.NOTIFICATIONS_SLACK_ENABLED:
                raise ValueError("Slack notifications are disabled globally")
        elif provider_type == 'teams':
            if not Config.NOTIFICATIONS_TEAMS_ENABLED:
                raise ValueError("Teams notifications are disabled globally")
        elif provider_type == 'webhook':
            if not Config.NOTIFICATIONS_WEBHOOK_ENABLED:
                raise ValueError("Custom webhooks are disabled globally")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Return cached instance or create new
        if provider_type not in _providers:
            if provider_type == 'slack':
                from .slack_notifier import SlackNotifier
                _providers[provider_type] = SlackNotifier()
            elif provider_type == 'teams':
                from .teams_notifier import TeamsNotifier
                _providers[provider_type] = TeamsNotifier()
            elif provider_type == 'webhook':
                from .custom_webhook_notifier import CustomWebhookNotifier
                _providers[provider_type] = CustomWebhookNotifier()
        
        return _providers[provider_type]
    
    @staticmethod
    def get_enabled_providers() -> list:
        """
        Get list of globally enabled provider types.
        
        Returns:
            List of enabled provider strings (e.g., ['email', 'slack', 'teams'])
        """
        enabled = []
        
        if Config.NOTIFICATIONS_EMAIL_ENABLED:
            enabled.append('email')
        if Config.NOTIFICATIONS_SLACK_ENABLED:
            enabled.append('slack')
        if Config.NOTIFICATIONS_TEAMS_ENABLED:
            enabled.append('teams')
        if Config.NOTIFICATIONS_WEBHOOK_ENABLED:
            enabled.append('webhook')
        
        return enabled
```

---

## 4.4 DATABASE SCHEMA (MODULAR)

### Extensible Webhook Configuration

```sql
-- Table: webhook_configurations
-- Stores webhook URLs for any provider (Slack, Teams, custom)

CREATE TABLE webhook_configurations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Provider info
    provider_type VARCHAR(50) NOT NULL,  -- 'slack', 'teams', 'webhook'
    webhook_url TEXT NOT NULL,  -- Provider-specific webhook URL
    
    -- User-provided metadata
    name VARCHAR(200),  -- User-friendly name (e.g., "#ml-experiments channel")
    description TEXT,   -- Optional description
    
    -- Configuration
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Event routing (JSON array of enabled events)
    -- Example: ["training.complete", "training.failed", "hpo.campaign_complete"]
    enabled_events JSONB DEFAULT '[]'::jsonb,
    
    -- Provider-specific settings (JSON, flexible for different providers)
    -- Example for Slack: {"mention_on_failure": true, "mention_user": "@abbas"}
    settings JSONB DEFAULT '{}'::jsonb,
    
    -- Status tracking
    last_used_at TIMESTAMP,
    last_error TEXT,  -- Last error message (for debugging)
    consecutive_failures INTEGER DEFAULT 0,  -- Auto-disable after N failures
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Ensure user can't add duplicate webhooks
    UNIQUE(user_id, webhook_url)
);

CREATE INDEX idx_webhook_configs_user ON webhook_configurations(user_id);
CREATE INDEX idx_webhook_configs_provider ON webhook_configurations(provider_type);
CREATE INDEX idx_webhook_configs_active ON webhook_configurations(is_active) 
    WHERE is_active = TRUE;

-- Table: webhook_logs (separate from email_logs for modularity)
CREATE TABLE webhook_logs (
    id SERIAL PRIMARY KEY,
    webhook_config_id INTEGER NOT NULL REFERENCES webhook_configurations(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE SET NULL,
    
    event_type VARCHAR(50) NOT NULL,
    provider_type VARCHAR(50) NOT NULL,
    
    -- Request details
    webhook_url TEXT NOT NULL,  -- Logged for debugging (can't get from config if config deleted)
    payload JSONB,  -- Full JSON payload sent
    
    -- Response details
    status VARCHAR(20) NOT NULL,  -- 'sent', 'failed', 'rate_limited'
    http_status_code INTEGER,  -- 200, 429, 500, etc.
    response_body TEXT,  -- Provider response (if error)
    error_message TEXT,
    
    retry_count INTEGER DEFAULT 0,
    sent_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_webhook_logs_config ON webhook_logs(webhook_config_id);
CREATE INDEX idx_webhook_logs_user ON webhook_logs(user_id);
CREATE INDEX idx_webhook_logs_status ON webhook_logs(status);
CREATE INDEX idx_webhook_logs_created ON webhook_logs(created_at DESC);
```

### Example Data

```sql
-- User configures Slack webhook
INSERT INTO webhook_configurations (
    user_id, 
    provider_type, 
    webhook_url, 
    name, 
    enabled_events,
    settings
) VALUES (
    42,
    'slack',
    'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX',
    '#ml-experiments channel',
    '["training.complete", "training.failed", "hpo.campaign_complete"]'::jsonb,
    '{
        "mention_on_failure": true,
        "mention_channel": "@channel",
        "use_rich_formatting": true
    }'::jsonb
);

-- User configures Microsoft Teams webhook
INSERT INTO webhook_configurations (
    user_id,
    provider_type,
    webhook_url,
    name,
    enabled_events,
    settings
) VALUES (
    42,
    'teams',
    'https://outlook.office.com/webhook/abc-def-ghi/IncomingWebhook/jkl-mno-pqr',
    'ML Team - General Channel',
    '["training.complete", "training.failed"]'::jsonb,
    '{
        "theme_color": "00ff00",
        "include_action_buttons": true
    }'::jsonb
);
```

---

## 4.5 SLACK INTEGRATION SPECIFICATION

### Slack Webhook URL Format

```
Format: https://hooks.slack.com/services/T{TEAM_ID}/B{CHANNEL_ID}/{SECRET_TOKEN}

Example: https://hooks.slack.com/services/T1234567890/B0987654321/abcdefghijklmnopqrstuvwx

Validation Regex:
^https://hooks\.slack\.com/services/T[A-Z0-9]{8,10}/B[A-Z0-9]{8,10}/[a-zA-Z0-9]{24}$

How to get webhook URL:
1. Go to Slack workspace
2. Navigate to Apps → Incoming Webhooks
3. Click "Add to Slack"
4. Select channel (e.g., #ml-experiments)
5. Copy webhook URL
```

### Slack Message Format (Block Kit)

**Rich Message Structure:**
```json
{
  "text": "Training Complete",  // Fallback text (for notifications)
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "🎉 Training Complete",
        "emoji": true
      }
    },
    {
      "type": "section",
      "fields": [
        {
          "type": "mrkdwn",
          "text": "*Experiment:*\nResNet34_Standard"
        },
        {
          "type": "mrkdwn",
          "text": "*Accuracy:*\n96.8%"
        },
        {
          "type": "mrkdwn",
          "text": "*Duration:*\n14m 32s"
        },
        {
          "type": "mrkdwn",
          "text": "*Model Type:*\nResNet"
        }
      ]
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Metrics:*\nPrecision: 96.5% | Recall: 96.7% | F1: 96.6%"
      }
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "View Results"
          },
          "url": "https://dashboard.com/experiment/1234/results",
          "style": "primary"
        },
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "Compare Models"
          },
          "url": "https://dashboard.com/compare?ids=1234"
        }
      ]
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "Experiment #1234 | Started by Abbas | 2025-06-15 14:32"
        }
      ]
    }
  ]
}
```

**Visual Appearance in Slack:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Training Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment:              Accuracy:
ResNet34_Standard        96.8%

Duration:                Model Type:
14m 32s                  ResNet

Metrics: Precision: 96.5% | Recall: 96.7% | F1: 96.6%

[View Results]  [Compare Models]

Experiment #1234 | Started by Abbas | 2025-06-15 14:32
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Slack-Specific Features

**1. Mentions (@channel, @user)**
```json
// Mention entire channel (use sparingly!)
{
  "text": "<!channel> Training failed! Immediate attention required.",
  "blocks": [...]
}

// Mention specific user
{
  "text": "<@U12345678> Your experiment failed.",
  "blocks": [...]
}
```

**2. Color Coding (Attachments - Legacy but useful)**
```json
// Green for success
{
  "attachments": [
    {
      "color": "#36a64f",  // Green
      "text": "Training completed successfully"
    }
  ]
}

// Red for failure
{
  "attachments": [
    {
      "color": "#ff0000",  // Red
      "text": "Training failed with error"
    }
  ]
}
```

**3. Threading (Reply to previous message)**
```
Use Case: Multiple updates to same experiment

Message 1 (new thread):
  "Training started for Experiment #1234"
  
Message 2 (reply in thread):
  POST with thread_ts parameter
  "Training completed for Experiment #1234"
  
Benefit: Keeps channel clean, related updates grouped
```

### Slack Rate Limiting

```
Slack Rate Limits (per webhook):
- 1 message per second (sustained)
- Burst: 10 messages in 1 second allowed
- Exceeding limit: HTTP 429 response

Rate Limit Headers:
- X-Rate-Limit-Limit: 1
- X-Rate-Limit-Remaining: 0
- X-Rate-Limit-Reset: 1718461234 (Unix timestamp)

Handling Strategy:
1. Implement token bucket locally (prevent hitting limit)
2. If 429 received, read X-Rate-Limit-Reset header
3. Queue message for retry after reset time
4. Don't retry immediately (wastes API calls)
```

---

## 4.6 MICROSOFT TEAMS INTEGRATION SPECIFICATION

### Teams Webhook URL Format

```
Format: https://outlook.office.com/webhook/{TENANT_ID}@{REGION}/IncomingWebhook/{CHANNEL_ID}/{SECRET_TOKEN}

Example: https://outlook.office.com/webhook/abc-123-def@00000000-0000-0000-0000-000000000000/IncomingWebhook/ghi-456-jkl/mno-789-pqr

Validation Regex:
^https://[a-z0-9]+\.office\.com/webhook/[a-zA-Z0-9-]+@[a-zA-Z0-9-]+/IncomingWebhook/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+$

How to get webhook URL:
1. Open Microsoft Teams
2. Navigate to channel (e.g., "ML Experiments")
3. Click "..." → "Connectors"
4. Search "Incoming Webhook"
5. Configure → Name it "ML Dashboard"
6. Copy webhook URL
```

### Teams Message Format (Adaptive Cards)

**Rich Message Structure:**
```json
{
  "@type": "MessageCard",
  "@context": "https://schema.org/extensions",
  "summary": "Training Complete",
  "themeColor": "00ff00",  // Green for success
  "title": "🎉 Training Complete",
  "sections": [
    {
      "activityTitle": "Experiment: **ResNet34_Standard**",
      "activitySubtitle": "Completed successfully",
      "facts": [
        {
          "name": "Accuracy:",
          "value": "96.8%"
        },
        {
          "name": "Precision:",
          "value": "96.5%"
        },
        {
          "name": "Recall:",
          "value": "96.7%"
        },
        {
          "name": "F1-Score:",
          "value": "96.6%"
        },
        {
          "name": "Duration:",
          "value": "14m 32s"
        }
      ],
      "markdown": true
    }
  ],
  "potentialAction": [
    {
      "@type": "OpenUri",
      "name": "View Results",
      "targets": [
        {
          "os": "default",
          "uri": "https://dashboard.com/experiment/1234/results"
        }
      ]
    },
    {
      "@type": "OpenUri",
      "name": "Compare Models",
      "targets": [
        {
          "os": "default",
          "uri": "https://dashboard.com/compare?ids=1234"
        }
      ]
    }
  ]
}
```

**Visual Appearance in Teams:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Training Complete                [Green bar on left]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment: ResNet34_Standard
Completed successfully

Accuracy:     96.8%
Precision:    96.5%
Recall:       96.7%
F1-Score:     96.6%
Duration:     14m 32s

[View Results]  [Compare Models]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Teams-Specific Features

**1. Color Themes (themeColor)**
```
Success: #00ff00 (green)
Warning: #ffcc00 (yellow)
Error:   #ff0000 (red)
Info:    #0078d4 (blue)
```

**2. Mentions (Not supported in Incoming Webhooks)**
```
Limitation: Teams Incoming Webhooks don't support @mentions
Workaround: Use bold text to highlight: "**@ML-Team**: Training failed"
```

**3. Adaptive Cards v2 (Future)**
```
Current: MessageCard (legacy, but well-supported)
Future: Adaptive Cards v2 (richer formatting, inputs)

Migration path:
1. Start with MessageCard (easier, more compatible)
2. Later migrate to Adaptive Cards v2 (when needed)
```

### Teams Rate Limiting

```
Teams Rate Limits (per webhook):
- 4 messages per second (more generous than Slack)
- Burst: 20 messages in 10 seconds
- Exceeding limit: HTTP 429 response

Handling Strategy:
1. Similar to Slack (token bucket)
2. Less restrictive, so lower priority concern
3. Still implement rate limiter for safety
```

---

## 4.7 IMPLEMENTATION PLAN (DAY-BY-DAY)

### Day 1: Database Schema & Provider Abstraction

**Morning: Database Setup**
1. Write migration for `webhook_configurations` table
2. Write migration for `webhook_logs` table
3. Run migrations on dev database
4. Create SQLAlchemy models (`models/webhook_configuration.py`)
5. Write seed data (test webhooks for dev environment)

**Afternoon: Provider Abstraction**
1. Create `services/notification_providers/` directory structure
2. Implement `base.py` (NotificationProvider interface, NotificationMessage dataclass)
3. Implement `factory.py` (NotificationProviderFactory with feature flags)
4. Write unit tests for factory (test feature flag enforcement)
5. Document provider interface (docstrings)

**Testing Criteria:**
- ✅ Migrations run without errors
- ✅ Can insert/query webhook_configurations
- ✅ Factory returns correct provider based on type
- ✅ Factory raises error when provider globally disabled
- ✅ Provider interface well-documented

**Deliverable:** Database schema ready, provider architecture implemented.

---

### Day 2: Slack Provider Implementation

**Morning: Slack Notifier Core**
1. Create `services/notification_providers/slack_notifier.py`
2. Implement `SlackNotifier` class (inherits from `NotificationProvider`)
3. Implement `send()` method (HTTP POST to webhook URL)
4. Implement `validate_webhook_url()` (regex check)
5. Implement `_build_slack_payload()` (convert NotificationMessage → Slack Block Kit JSON)

**Afternoon: Slack Features**
1. Implement rich formatting (blocks, sections, actions)
2. Implement color coding (attachments with colors)
3. Implement mentions (@channel, @user) - respect settings
4. Implement rate limiting (token bucket, 1 msg/sec)
5. Implement retry logic (3 attempts with exponential backoff)

**Key Implementation Details:**

```python
# Pseudocode structure

class SlackNotifier(NotificationProvider):
    
    def __init__(self):
        self.rate_limiter = TokenBucket(
            capacity=1,  # 1 message per second
            refill_rate=1  # 1 token per second
        )
    
    def send(self, webhook_url: str, message: NotificationMessage) -> bool:
        """Send notification to Slack."""
        
        # 1. Validate webhook URL format
        if not self.validate_webhook_url(webhook_url):
            raise ValueError("Invalid Slack webhook URL")
        
        # 2. Rate limit (wait if needed)
        self.rate_limiter.consume(1)
        
        # 3. Build payload
        payload = self._build_slack_payload(message)
        
        # 4. Send HTTP POST
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=Config.SLACK_TIMEOUT_SECONDS
            )
            
            # 5. Handle response
            if response.status_code == 200:
                return True
            elif response.status_code == 429:
                # Rate limited by Slack
                reset_time = response.headers.get('X-Rate-Limit-Reset')
                # Queue for retry after reset
                raise RateLimitError(reset_time)
            else:
                # Other error
                raise ProviderError(f"Slack API error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            raise ProviderError("Slack webhook timeout")
        except requests.exceptions.ConnectionError:
            raise ProviderError("Cannot connect to Slack")
    
    def _build_slack_payload(self, message: NotificationMessage) -> dict:
        """
        Convert NotificationMessage to Slack Block Kit format.
        
        Handles:
        - Rich formatting (blocks)
        - Color coding (based on priority)
        - Action buttons (from message.actions)
        - Mentions (from settings)
        """
        
        # Check if user provided Slack-specific override
        if message.slack_override:
            return message.slack_override
        
        # Build standard payload
        blocks = []
        
        # Header block
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": message.title,
                "emoji": True
            }
        })
        
        # Body section (with fields from message.data)
        fields = []
        for key, value in message.data.items():
            fields.append({
                "type": "mrkdwn",
                "text": f"*{key}:*\n{value}"
            })
        
        blocks.append({
            "type": "section",
            "fields": fields
        })
        
        # Actions (buttons)
        if message.actions:
            elements = []
            for action in message.actions:
                elements.append({
                    "type": "button",
                    "text": {"type": "plain_text", "text": action['label']},
                    "url": action['url'],
                    "style": action.get('style', 'default')  # 'primary', 'danger'
                })
            
            blocks.append({
                "type": "actions",
                "elements": elements
            })
        
        # Build final payload
        payload = {
            "text": message.title,  # Fallback
            "blocks": blocks
        }
        
        # Add color via attachments (if priority is high/critical)
        if message.priority in ['high', 'critical']:
            payload["attachments"] = [{
                "color": "#ff0000" if message.priority == 'critical' else "#ffcc00"
            }]
        
        return payload
    
    def validate_webhook_url(self, webhook_url: str) -> bool:
        """Validate Slack webhook URL format."""
        import re
        pattern = r'^https://hooks\.slack\.com/services/T[A-Z0-9]{8,10}/B[A-Z0-9]{8,10}/[a-zA-Z0-9]{24}$'
        return bool(re.match(pattern, webhook_url))
```

**Testing Criteria:**
- ✅ Send test message to real Slack webhook → Appears in channel
- ✅ Message has rich formatting (blocks, buttons)
- ✅ Invalid webhook URL → Raises validation error
- ✅ Rate limiter enforces 1 msg/sec limit
- ✅ HTTP 429 from Slack → Retries after cooldown
- ✅ Timeout after 10 seconds → Raises ProviderError

**Deliverable:** Fully functional Slack integration.

---

### Day 3: Microsoft Teams Provider Implementation

**Morning: Teams Notifier Core**
1. Create `services/notification_providers/teams_notifier.py`
2. Implement `TeamsNotifier` class (same interface as SlackNotifier)
3. Implement `send()` method (HTTP POST to Teams webhook)
4. Implement `validate_webhook_url()` (Teams URL regex)
5. Implement `_build_teams_payload()` (convert to MessageCard format)

**Afternoon: Teams Features**
1. Implement rich formatting (MessageCard with sections, facts)
2. Implement color themes (themeColor based on priority)
3. Implement action buttons (potentialAction)
4. Implement rate limiting (2 msg/sec for Teams)
5. Test with real Teams webhook

**Key Differences from Slack:**
- Different JSON structure (MessageCard vs Block Kit)
- More generous rate limits (2 msg/sec vs 1 msg/sec)
- No mention support in Incoming Webhooks
- Different URL validation regex

**Testing Criteria:**
- ✅ Send test message to real Teams webhook → Appears in channel
- ✅ Message has correct color theme
- ✅ Action buttons work (open URLs)
- ✅ Rate limiter enforces 2 msg/sec limit
- ✅ Invalid Teams URL → Raises validation error

**Deliverable:** Fully functional Teams integration.

---

### Day 4: Integration with Notification Service

**Morning: Routing Logic**
1. Modify `services/notification_service.py`
2. Add webhook routing in `emit_event()` method
3. Implement `_send_webhook_notification()` method
4. Query `webhook_configurations` table for user's webhooks
5. Filter by `enabled_events` (only send if event enabled)

**Routing Logic Pseudocode:**

```python
# services/notification_service.py

class NotificationService:
    
    @staticmethod
    def emit_event(event_type: str, user_id: int, data: dict):
        """
        Main entry point for all notifications.
        Routes to all enabled channels (email, Slack, Teams, etc.)
        """
        
        # 1. Check global feature flags
        enabled_providers = NotificationProviderFactory.get_enabled_providers()
        
        # 2. Load user preferences (email, in-app)
        preferences = get_user_preferences(user_id, event_type)
        
        # 3. Send email (if enabled)
        if 'email' in enabled_providers and preferences.email_enabled:
            NotificationService._send_email(user_id, event_type, data)
        
        # 4. Send webhook notifications (Slack, Teams, custom)
        if any(p in enabled_providers for p in ['slack', 'teams', 'webhook']):
            NotificationService._send_webhook_notifications(user_id, event_type, data)
        
        # 5. Send in-app notification (toast)
        if preferences.in_app_enabled:
            NotificationService._send_in_app(user_id, event_type, data)
    
    @staticmethod
    def _send_webhook_notifications(user_id: int, event_type: str, data: dict):
        """
        Send notification to all configured webhooks for this user/event.
        """
        
        # Query user's webhook configurations
        webhooks = db.session.query(WebhookConfiguration).filter(
            WebhookConfiguration.user_id == user_id,
            WebhookConfiguration.is_active == True,
            WebhookConfiguration.enabled_events.contains([event_type])  # JSONB contains
        ).all()
        
        if not webhooks:
            return  # No webhooks configured for this event
        
        # Build standardized message
        message = NotificationService._build_notification_message(event_type, data)
        
        # Send to each webhook (in parallel, non-blocking)
        for webhook in webhooks:
            # Use Celery task for async sending (don't block main thread)
            send_webhook_notification_task.delay(
                webhook_id=webhook.id,
                message=message.to_dict()
            )
    
    @staticmethod
    def _build_notification_message(event_type: str, data: dict) -> NotificationMessage:
        """
        Build standardized message from event data.
        
        Maps event types to message templates.
        """
        
        if event_type == 'training.complete':
            return NotificationMessage(
                title="🎉 Training Complete",
                body=f"Experiment {data['experiment_name']} finished training.",
                event_type=event_type,
                priority='medium',
                data={
                    'Experiment': data['experiment_name'],
                    'Accuracy': f"{data['accuracy']:.1%}",
                    'Duration': f"{data['duration_minutes']}m {data['duration_seconds']}s",
                    'Model Type': data['model_type']
                },
                actions=[
                    {
                        'label': 'View Results',
                        'url': data['results_url'],
                        'style': 'primary'
                    },
                    {
                        'label': 'Compare Models',
                        'url': f"{Config.DASHBOARD_URL}/compare?ids={data['experiment_id']}"
                    }
                ],
                color='#00ff00'  # Green
            )
        
        elif event_type == 'training.failed':
            return NotificationMessage(
                title="⚠️ Training Failed",
                body=f"Experiment {data['experiment_name']} encountered an error.",
                event_type=event_type,
                priority='high',
                data={
                    'Experiment': data['experiment_name'],
                    'Error': data['error_message'],
                    'Suggestion': data.get('error_suggestion', 'Check logs for details')
                },
                actions=[
                    {
                        'label': 'View Error Details',
                        'url': data['error_details_url'],
                        'style': 'danger'
                    },
                    {
                        'label': 'Start New Training',
                        'url': f"{Config.DASHBOARD_URL}/experiment/new"
                    }
                ],
                color='#ff0000'  # Red
            )
        
        # ... other event types ...
```

**Celery Task for Async Sending:**

```python
# tasks/webhook_tasks.py

@celery_app.task(bind=True, max_retries=3)
def send_webhook_notification_task(self, webhook_id: int, message_dict: dict):
    """
    Async task to send webhook notification.
    
    Runs in background (non-blocking).
    Handles retries automatically.
    Logs to webhook_logs table.
    """
    
    # Load webhook config
    webhook = db.session.query(WebhookConfiguration).get(webhook_id)
    if not webhook or not webhook.is_active:
        return  # Webhook deleted or disabled
    
    # Reconstruct NotificationMessage
    message = NotificationMessage(**message_dict)
    
    # Get provider
    try:
        provider = NotificationProviderFactory.get_provider(webhook.provider_type)
    except ValueError as e:
        # Provider disabled globally
        log_webhook_error(webhook_id, str(e))
        return
    
    # Send via provider
    try:
        success = provider.send(webhook.webhook_url, message)
        
        if success:
            # Log success
            log_webhook_send(webhook_id, message, status='sent', http_status=200)
            
            # Update webhook metadata
            webhook.last_used_at = datetime.utcnow()
            webhook.consecutive_failures = 0
            db.session.commit()
        
        else:
            raise ProviderError("Send failed")
    
    except RateLimitError as e:
        # Rate limited, retry after cooldown
        retry_after = e.reset_time - time.time()
        raise self.retry(countdown=retry_after)
    
    except ProviderError as e:
        # Provider error, log and retry
        log_webhook_error(webhook_id, str(e))
        
        # Increment failure count
        webhook.consecutive_failures += 1
        
        # Auto-disable after 10 consecutive failures
        if webhook.consecutive_failures >= 10:
            webhook.is_active = False
            webhook.last_error = "Auto-disabled after 10 consecutive failures"
            # TODO: Notify user via email that webhook was disabled
        
        db.session.commit()
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
```

**Testing Criteria:**
- ✅ Train model → Slack webhook receives notification
- ✅ Train model → Teams webhook receives notification
- ✅ User has Slack + Teams configured → Both receive notification
- ✅ User disables event in webhook config → No notification sent
- ✅ Webhook fails 10 times → Auto-disabled, user notified via email
- ✅ Webhook rate limited → Task retries after cooldown

**Deliverable:** Webhook routing integrated with training tasks.

---

### Day 5: Settings UI & Testing

**Morning: Settings UI**
1. Enhance `layouts/settings.py` (add Webhooks tab)
2. Create form for adding webhook (provider dropdown, URL input, name)
3. Display list of configured webhooks (table with edit/delete)
4. Add event toggles (checkboxes for each event type)
5. Implement "Test Webhook" button (sends sample notification)

**Settings UI Design:**

```
Settings → Webhooks Tab

┌────────────────────────────────────────────────────────┐
│  WEBHOOK INTEGRATIONS                                  │
├────────────────────────────────────────────────────────┤
│  Notify external services when events occur.           │
│                                                        │
│  CONFIGURED WEBHOOKS (2)                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Name: #ml-experiments (Slack)                   │  │
│  │ URL: https://hooks.slack.com/services/T.../B... │  │
│  │ Status: ✅ Active (Last used: 2 hours ago)      │  │
│  │ Events: Training Complete, Training Failed     │  │
│  │ [Edit] [Test] [Delete]                         │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Name: ML Team - General (Microsoft Teams)      │  │
│  │ URL: https://outlook.office.com/webhook/...    │  │
│  │ Status: ✅ Active (Last used: 1 day ago)        │  │
│  │ Events: Training Complete                      │  │
│  │ [Edit] [Test] [Delete]                         │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  [+ Add Webhook]                                       │
│                                                        │
├────────────────────────────────────────────────────────┤
│  ADD WEBHOOK MODAL (when clicking "+ Add Webhook")   │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Provider: [Slack ▼] (Slack, Teams, Custom)     │  │
│  │                                                 │  │
│  │ Name: [#ml-experiments________________]        │  │
│  │                                                 │  │
│  │ Webhook URL:                                    │  │
│  │ [https://hooks.slack.com/services/...________] │  │
│  │ [How to get webhook URL?]                      │  │
│  │                                                 │  │
│  │ Enable for events:                              │  │
│  │ [✓] Training Complete                          │  │
│  │ [✓] Training Failed                            │  │
│  │ [ ] Training Started                           │  │
│  │ [✓] HPO Campaign Complete                      │  │
│  │ [ ] HPO Campaign Failed                        │  │
│  │                                                 │  │
│  │ Advanced Settings (Slack):                      │  │
│  │ [✓] Use rich formatting (Block Kit)            │  │
│  │ [ ] Mention @channel on failures               │  │
│  │                                                 │  │
│  │ [Cancel]  [Save Webhook]                       │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Callback Implementation (High-Level):**

```python
# callbacks/webhook_callbacks.py

@callback(
    Output('webhook-save-confirmation', 'children'),
    Input('save-webhook-btn', 'n_clicks'),
    State('webhook-provider-dropdown', 'value'),
    State('webhook-url-input', 'value'),
    State('webhook-name-input', 'value'),
    State({'type': 'event-toggle', 'event': ALL}, 'checked')
)
def save_webhook_configuration(n_clicks, provider, url, name, event_toggles):
    """
    Save webhook configuration to database.
    """
    
    if not n_clicks:
        return no_update
    
    # Validate inputs
    if not provider or not url or not name:
        return dbc.Alert("All fields required", color="danger")
    
    # Validate webhook URL format
    provider_class = NotificationProviderFactory.get_provider(provider)
    if not provider_class.validate_webhook_url(url):
        return dbc.Alert(f"Invalid {provider} webhook URL format", color="danger")
    
    # Build enabled_events list
    enabled_events = [
        event_type for event_type, checked in zip(event_types, event_toggles) if checked
    ]
    
    if not enabled_events:
        return dbc.Alert("Enable at least one event", color="warning")
    
    # Save to database
    user_id = get_current_user_id()
    webhook = WebhookConfiguration(
        user_id=user_id,
        provider_type=provider,
        webhook_url=url,
        name=name,
        enabled_events=enabled_events
    )
    
    db.session.add(webhook)
    db.session.commit()
    
    return dbc.Alert(f"✓ Webhook '{name}' saved successfully", color="success", duration=3000)


@callback(
    Output('test-webhook-result', 'children'),
    Input('test-webhook-btn', 'n_clicks'),
    State('webhook-id-hidden', 'data')  # Webhook ID passed from table
)
def test_webhook(n_clicks, webhook_id):
    """
    Send test notification to webhook.
    """
    
    if not n_clicks:
        return no_update
    
    # Load webhook config
    webhook = db.session.query(WebhookConfiguration).get(webhook_id)
    if not webhook:
        return dbc.Alert("Webhook not found", color="danger")
    
    # Build test message
    test_message = NotificationMessage(
        title="🧪 Test Notification",
        body="This is a test notification from ML Dashboard.",
        event_type='test',
        priority='low',
        data={
            'Test': 'This is a test',
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        actions=[
            {
                'label': 'Go to Dashboard',
                'url': Config.DASHBOARD_URL,
                'style': 'primary'
            }
        ]
    )
    
    # Send via Celery task (async)
    send_webhook_notification_task.delay(webhook.id, test_message.to_dict())
    
    return dbc.Alert(
        f"✓ Test notification sent to {webhook.name}. Check your channel in a few seconds.",
        color="success",
        duration=5000
    )
```

**Afternoon: End-to-End Testing**
1. Create test Slack workspace + webhook
2. Create test Teams channel + webhook
3. Run full end-to-end tests (see test scenarios below)
4. Document setup instructions for users
5. Create video tutorial (5 minutes: "Setting up Slack Notifications")

**Testing Criteria:**
- ✅ Add Slack webhook via UI → Saves to database
- ✅ Click "Test" → Notification appears in Slack channel
- ✅ Add Teams webhook → Saves correctly
- ✅ Train model → Both Slack and Teams receive notification
- ✅ Edit webhook (disable event) → That event no longer triggers webhook
- ✅ Delete webhook → No longer receives notifications
- ✅ Invalid URL → Shows validation error

**Deliverable:** Fully functional webhook system with UI.

---

## 4.8 MODULARITY TESTING

### Feature Toggle Tests

```python
# tests/test_feature_toggles.py

def test_slack_disabled_globally():
    """When Slack globally disabled, no Slack notifications sent."""
    
    # Set config
    Config.NOTIFICATIONS_SLACK_ENABLED = False
    
    # User has Slack webhook configured
    webhook = create_test_webhook(provider='slack', user_id=1)
    
    # Emit event
    NotificationService.emit_event('training.complete', user_id=1, data={...})
    
    # Assert: No Slack notification sent
    logs = db.session.query(WebhookLog).filter_by(provider_type='slack').all()
    assert len(logs) == 0
    
    # Assert: Email still sent (other channels unaffected)
    email_logs = db.session.query(EmailLog).all()
    assert len(email_logs) == 1


def test_teams_disabled_globally():
    """When Teams globally disabled, factory raises error."""
    
    Config.NOTIFICATIONS_TEAMS_ENABLED = False
    
    with pytest.raises(ValueError, match="Teams notifications are disabled"):
        provider = NotificationProviderFactory.get_provider('teams')


def test_enable_provider_at_runtime():
    """Provider can be enabled/disabled without restart (hot reload)."""
    
    # Start with Slack disabled
    Config.NOTIFICATIONS_SLACK_ENABLED = False
    
    # Re-enable
    Config.NOTIFICATIONS_SLACK_ENABLED = True
    
    # Should work now
    provider = NotificationProviderFactory.get_provider('slack')
    assert provider is not None
```

### Graceful Degradation Tests

```python
def test_slack_fails_email_still_works():
    """If Slack fails, email notification still sent."""
    
    # Mock Slack to always fail
    with mock.patch('services.notification_providers.slack_notifier.SlackNotifier.send', side_effect=ProviderError):
        
        # Emit event
        NotificationService.emit_event('training.complete', user_id=1, data={...})
        
        # Assert: Slack failed
        slack_logs = db.session.query(WebhookLog).filter_by(status='failed').all()
        assert len(slack_logs) == 1
        
        # Assert: Email still sent
        email_logs = db.session.query(EmailLog).filter_by(status='sent').all()
        assert len(email_logs) == 1


def test_invalid_webhook_doesnt_break_system():
    """Invalid webhook URL logged as error, doesn't crash."""
    
    # Create webhook with invalid URL
    webhook = WebhookConfiguration(
        user_id=1,
        provider_type='slack',
        webhook_url='https://invalid.com/webhook',  # Wrong format
        enabled_events=['training.complete']
    )
    db.session.add(webhook)
    db.session.commit()
    
    # Emit event
    NotificationService.emit_event('training.complete', user_id=1, data={...})
    
    # Assert: System didn't crash
    # Assert: Error logged
    webhook_logs = db.session.query(WebhookLog).filter_by(status='failed').first()
    assert webhook_logs is not None
    assert 'Invalid' in webhook_logs.error_message
```

---

## 4.9 DO'S AND DON'TS

### ✅ DO's

1. **DO implement feature toggles at global level**
   - Reason: Disable entire integration if provider has outage
   - Example: Slack API down → Disable globally, no failed requests

2. **DO use provider abstraction (interface pattern)**
   - Reason: Easy to add new providers (Discord, Mattermost)
   - Each provider is independent module

3. **DO send webhooks asynchronously (Celery tasks)**
   - Reason: Don't block training task (webhook can take 500ms-2s)
   - User gets immediate response, webhook sent in background

4. **DO implement rate limiting locally**
   - Reason: Prevent hitting provider's rate limits
   - Cheaper to throttle ourselves than get 429 errors

5. **DO log all webhook sends (audit trail)**
   - Reason: Debug failures, measure reliability
   - Can answer "Was webhook sent for this event?"

6. **DO auto-disable after repeated failures**
   - Reason: Broken webhook shouldn't spam error logs forever
   - After 10 failures, disable + notify user via email

7. **DO validate webhook URLs before saving**
   - Reason: Catch typos early (wrong format)
   - Better UX than finding out at send time

8. **DO provide test button**
   - Reason: Users want to verify webhook works
   - Instant feedback, confidence

9. **DO support per-event configuration**
   - Reason: Users want granular control
   - Slack for failures, but not for every training start

10. **DO store provider-specific settings in JSONB**
    - Reason: Flexibility (Slack has mentions, Teams doesn't)
    - Schema doesn't need changes when adding provider features

### ❌ DON'Ts

1. **DON'T send webhooks synchronously**
   - Reason: Adds 500ms-2s latency to training task
   - Use Celery (async)

2. **DON'T hardcode provider logic in NotificationService**
   - Reason: Violates modularity, hard to maintain
   - Use factory pattern, provider classes

3. **DON'T retry infinitely**
   - Reason: Wastes resources, clogs queue
   - Max 3 retries, then give up

4. **DON'T ignore rate limits**
   - Reason: Provider blocks your account
   - Implement local rate limiter

5. **DON'T expose webhook URLs in logs/UI**
   - Reason: Security risk (webhook URL is like password)
   - Show only first/last few characters

6. **DON'T couple webhook logic to training tasks**
   - Reason: Training tasks shouldn't know about Slack
   - Emit generic events, NotificationService routes

7. **DON'T forget error handling**
   - Reason: Network failures, timeouts, invalid URLs
   - Graceful degradation, fallback to email

8. **DON'T send webhooks for every event by default**
   - Reason: Spam (training.started every 2 minutes)
   - Default: Only high-value events enabled

9. **DON'T assume webhook always works**
   - Reason: Webhooks can expire, be deleted, rate limited
   - Log failures, auto-disable after threshold

10. **DON'T skip documentation**
    - Reason: Users need to know how to get webhook URL
    - Provide step-by-step guide with screenshots

---

## 4.10 TESTING CHECKLIST

### Unit Tests

- [ ] SlackNotifier.send() sends correct payload
- [ ] SlackNotifier.validate_webhook_url() rejects invalid URLs
- [ ] TeamsNotifier.send() sends correct MessageCard
- [ ] NotificationProviderFactory enforces feature flags
- [ ] Rate limiter enforces 1 msg/sec (Slack)
- [ ] Rate limiter enforces 2 msg/sec (Teams)
- [ ] Retry logic retries 3 times with backoff
- [ ] Auto-disable after 10 consecutive failures

### Integration Tests

- [ ] Train model → Slack webhook receives notification
- [ ] Train model → Teams webhook receives notification
- [ ] User has 2 webhooks → Both receive notification
- [ ] Webhook disabled for event → No notification sent
- [ ] Invalid webhook URL → Error logged, doesn't crash
- [ ] Slack rate limited (429) → Retries after cooldown
- [ ] Feature flag disabled → No webhook sent

### Manual QA

- [ ] Add Slack webhook via UI → Saves successfully
- [ ] Click "Test" → Notification appears in Slack channel within 10 seconds
- [ ] Notification has rich formatting (blocks, buttons)
- [ ] Click "View Results" button → Opens correct page
- [ ] Add Teams webhook → Notification appears in Teams
- [ ] Train model → Both Slack and Teams receive notification
- [ ] Edit webhook (disable event) → That event doesn't trigger webhook
- [ ] Delete webhook → No longer receives notifications
- [ ] 10 consecutive failures → Webhook auto-disabled, user notified via email
- [ ] Feature flag disabled in config → Webhooks don't send

---

## 4.11 SUCCESS METRICS

### Quantitative
- Webhook delivery rate: >95% (lower than email due to external dependencies)
- Webhook send latency: <30 seconds from event to channel
- Auto-disable rate: <5% of webhooks (most should stay healthy)
- User configuration: 30%+ of users configure at least one webhook
- Zero system crashes due to webhook failures

### Qualitative
- Teams see notifications → Increased awareness → More users sign up
- Users report staying in Slack instead of switching to dashboard
- Positive feedback on message formatting (professional, actionable)
- No spam complaints (users find notifications valuable, not noisy)

---

## 4.12 DOCUMENTATION OUTLINE

### User Guide: "Slack & Teams Integration"

```markdown
# Slack & Teams Integration

## Overview
Receive ML experiment notifications directly in your team's Slack or Microsoft Teams channels.

## Setting Up Slack

1. **Get Webhook URL**
   - Go to your Slack workspace
   - Navigate to: Apps → Incoming Webhooks
   - Click "Add to Slack"
   - Select channel (e.g., #ml-experiments)
   - Copy webhook URL

2. **Configure in Dashboard**
   - Open Settings → Webhooks
   - Click "+ Add Webhook"
   - Provider: Slack
   - Paste webhook URL
   - Name: "#ml-experiments"
   - Enable events: Training Complete, Training Failed
   - Click "Save"

3. **Test**
   - Click "Test" button
   - Check Slack channel for test notification

## Setting Up Microsoft Teams

1. **Get Webhook URL**
   - Open Teams channel
   - Click "..." → Connectors
   - Search "Incoming Webhook"
   - Click "Configure"
   - Name: "ML Dashboard"
   - Copy webhook URL

2. **Configure in Dashboard**
   - (Same as Slack, select "Microsoft Teams" provider)

## Event Types
- **Training Complete**: Model finishes training successfully
- **Training Failed**: Model encounters error during training
- **HPO Campaign Complete**: Hyperparameter search finishes

## Troubleshooting
- **Notification not appearing**: Check webhook URL, verify channel exists
- **Webhook disabled**: Exceeded failure threshold, re-enable in settings
- **Too many notifications**: Disable noisy events (e.g., Training Started)
```

### Admin Guide: "Webhook System Architecture"

```markdown
# Webhook System Architecture

## Feature Flags
All webhook integrations can be disabled globally:

```bash
# .env file
NOTIFICATIONS_SLACK_ENABLED=true
NOTIFICATIONS_TEAMS_ENABLED=true
NOTIFICATIONS_WEBHOOK_ENABLED=true
```

## Adding New Provider

1. Create `services/notification_providers/new_provider_notifier.py`
2. Implement `NotificationProvider` interface
3. Add to factory in `factory.py`
4. Add feature flag in config
5. Update UI dropdown (settings.py)

## Monitoring
- Webhook logs: `/admin/webhook-logs`
- Delivery rate: 95%+ is healthy
- Auto-disabled webhooks: Investigate if >10% of webhooks disabled

## Troubleshooting
- High failure rate: Check provider status (Slack API down?)
- Rate limit errors: Increase local rate limiter capacity
- Timeouts: Increase `SLACK_TIMEOUT_SECONDS` in config
```

---

**END OF FEATURE #4 PLAN**

---

This completes the comprehensive planning document for **Feature #4: Slack/Teams Webhook Integration** with emphasis on modularity.

**Key Modularity Features:**
✅ Global feature toggles (disable provider with one config change)  
✅ Provider abstraction (easy to add Discord, Mattermost, etc.)  
✅ Graceful degradation (if Slack fails, email still works)  
✅ Per-user, per-event control (granular configuration)  
✅ Independent service classes (SlackNotifier, TeamsNotifier)  
✅ Async sending (doesn't block main thread)  
✅ Auto-disable on repeated failures (self-healing)  

**Ready for your team to implement as a pluggable module.**
