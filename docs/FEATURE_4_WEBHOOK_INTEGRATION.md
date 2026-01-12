# Feature #4: Slack/Teams Webhook Integration

## Overview

Feature #4 adds Slack, Microsoft Teams, and custom webhook integrations to the LSTM Dashboard, enabling users to receive ML experiment notifications directly in their team channels. This facilitates team collaboration and increases platform visibility within organizations.

## Features

- **Multi-Provider Support**: Slack, Microsoft Teams, and custom webhooks
- **Rich Formatting**: Beautiful messages with colors, buttons, and structured data
- **Rate Limiting**: Built-in rate limiting (1 msg/sec for Slack, 2 msg/sec for Teams)
- **Retry Logic**: Automatic retries with exponential backoff
- **Graceful Degradation**: Webhook failures don't break core notification system
- **Event-Level Control**: Users can enable/disable webhooks per event type
- **Auto-Disable**: Webhooks auto-disable after 10 consecutive failures
- **Logging**: Full audit trail of webhook deliveries for debugging

## Architecture

### Provider Abstraction Layer

```
┌─────────────────────────────────────────────────────────┐
│  EVENT SOURCE (Training Task, HPO Task, etc.)          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  NOTIFICATION SERVICE (Central Router)                  │
│  - Checks feature flags                                │
│  - Loads user preferences                              │
│  - Routes to enabled channels                          │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
   │Email │  │Slack │  │Teams │  │Custom│
   │      │  │      │  │      │  │Webhook│
   └──────┘  └──────┘  └──────┘  └──────┘
```

### Components

1. **Database Models**:
   - `WebhookConfiguration`: Stores webhook URLs and settings
   - `WebhookLog`: Tracks delivery attempts and status

2. **Provider Classes**:
   - `SlackNotifier`: Slack Block Kit formatting
   - `TeamsNotifier`: Microsoft Teams MessageCard formatting
   - `CustomWebhookNotifier`: Generic JSON webhooks

3. **Factory Pattern**:
   - `NotificationProviderFactory`: Creates provider instances based on type

4. **Notification Service**:
   - Integrated webhook routing in `NotificationService.emit_event()`

## Configuration

### Environment Variables

```bash
# Global Feature Flags
NOTIFICATIONS_SLACK_ENABLED=true
NOTIFICATIONS_TEAMS_ENABLED=true
NOTIFICATIONS_WEBHOOK_ENABLED=true

# Slack Configuration
SLACK_RATE_LIMIT_PER_WEBHOOK=1        # 1 message per second
SLACK_RETRY_ATTEMPTS=3
SLACK_TIMEOUT_SECONDS=10

# Teams Configuration
TEAMS_RATE_LIMIT_PER_WEBHOOK=2        # 2 messages per second
TEAMS_RETRY_ATTEMPTS=3
TEAMS_TIMEOUT_SECONDS=10

# Custom Webhook Configuration
WEBHOOK_CUSTOM_TIMEOUT_SECONDS=5
WEBHOOK_CUSTOM_RETRY_ATTEMPTS=2

# Feature Toggles
NOTIFICATIONS_ENABLE_RICH_FORMATTING=true
NOTIFICATIONS_ENABLE_MENTIONS=true
```

## Usage

### Setting Up Slack Webhooks

1. **Create Incoming Webhook in Slack**:
   - Go to your Slack workspace
   - Navigate to Apps → "Incoming Webhooks"
   - Click "Add to Slack"
   - Select channel (e.g., `#ml-experiments`)
   - Copy webhook URL

2. **Add Webhook to Dashboard** (via API or database):
   ```python
   webhook_config = WebhookConfiguration(
       user_id=42,
       provider_type='slack',
       webhook_url='https://hooks.slack.com/services/T.../B.../...',
       name='#ml-experiments channel',
       enabled_events=[
           'training.complete',
           'training.failed',
           'hpo.campaign_complete'
       ],
       settings={
           'mention_on_failure': True,
           'use_rich_formatting': True
       },
       is_active=True
   )
   ```

3. **Test Notification**:
   ```python
   from services.notification_service import NotificationService

   NotificationService.emit_event(
       event_type='training.complete',
       user_id=42,
       data={
           'experiment_name': 'ResNet34_Test',
           'accuracy': 0.968,
           'duration': '14m 32s',
           'experiment_id': 1234,
           'dashboard_url': 'http://localhost:8050/experiments/1234'
       }
   )
   ```

### Setting Up Microsoft Teams Webhooks

1. **Create Incoming Webhook in Teams**:
   - Open Microsoft Teams
   - Navigate to your channel (e.g., "ML Experiments")
   - Click "..." → "Connectors"
   - Search "Incoming Webhook"
   - Configure → Name it "ML Dashboard"
   - Copy webhook URL

2. **Add Webhook to Dashboard**:
   ```python
   webhook_config = WebhookConfiguration(
       user_id=42,
       provider_type='teams',
       webhook_url='https://outlook.office.com/webhook/...',
       name='ML Team - General Channel',
       enabled_events=['training.complete', 'training.failed'],
       settings={
           'theme_color': '00ff00',
           'include_action_buttons': True
       },
       is_active=True
   )
   ```

### Custom Webhooks

For integrating with custom systems:

```python
webhook_config = WebhookConfiguration(
    user_id=42,
    provider_type='webhook',
    webhook_url='https://your-system.com/api/webhooks/ml-notifications',
    name='Custom Monitoring System',
    enabled_events=['training.complete', 'training.failed'],
    settings={},
    is_active=True
)
```

**Payload Format** (simple JSON):
```json
{
  "event_type": "training.complete",
  "title": "Training Complete - ResNet34_Test",
  "body": "",
  "priority": "medium",
  "data": {
    "experiment_name": "ResNet34_Test",
    "accuracy": 0.968,
    "duration": "14m 32s"
  },
  "actions": [
    {
      "label": "View Results",
      "url": "http://localhost:8050/experiments/1234",
      "style": "primary"
    }
  ]
}
```

## Database Schema

### `webhook_configurations`

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `user_id` | INTEGER | User who owns this webhook |
| `provider_type` | VARCHAR(50) | 'slack', 'teams', or 'webhook' |
| `webhook_url` | TEXT | Provider-specific webhook URL |
| `name` | VARCHAR(200) | User-friendly name |
| `description` | TEXT | Optional description |
| `is_active` | BOOLEAN | Whether webhook is active |
| `enabled_events` | JSONB | Array of enabled event types |
| `settings` | JSONB | Provider-specific settings |
| `last_used_at` | TIMESTAMP | Last successful delivery |
| `last_error` | TEXT | Last error message |
| `consecutive_failures` | INTEGER | Count of consecutive failures |
| `created_at` | TIMESTAMP | Creation timestamp |
| `updated_at` | TIMESTAMP | Update timestamp |

### `webhook_logs`

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `webhook_config_id` | INTEGER | Reference to webhook config |
| `user_id` | INTEGER | User ID |
| `event_type` | VARCHAR(50) | Event type |
| `provider_type` | VARCHAR(50) | Provider type |
| `webhook_url` | TEXT | Webhook URL (logged for audit) |
| `payload` | JSONB | Full payload sent |
| `status` | VARCHAR(20) | 'sent', 'failed', 'rate_limited' |
| `http_status_code` | INTEGER | HTTP response code |
| `response_body` | TEXT | Provider response |
| `error_message` | TEXT | Error message if failed |
| `retry_count` | INTEGER | Number of retries |
| `sent_at` | TIMESTAMP | Delivery timestamp |
| `created_at` | TIMESTAMP | Log creation timestamp |

## Migrations

Run migrations to create database tables:

```bash
# Migration 005: Create webhook_configurations table
psql -U lstm_user -d lstm_dashboard -f packages/dashboard/database/migrations/005_add_webhook_configurations.sql

# Migration 006: Create webhook_logs table
psql -U lstm_user -d lstm_dashboard -f packages/dashboard/database/migrations/006_add_webhook_logs.sql
```

## Supported Events

- `training.started` - Training job started
- `training.complete` - Training completed successfully
- `training.failed` - Training failed with error
- `hpo.campaign_started` - HPO campaign started
- `hpo.trial_complete` - HPO trial completed
- `hpo.campaign_complete` - HPO campaign completed
- `hpo.campaign_failed` - HPO campaign failed

## Slack Message Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Training Complete - ResNet34_Standard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment Name:         Accuracy:
ResNet34_Standard        96.8%

Duration:                Model Type:
14m 32s                  ResNet

[View Results]  [Compare Models]

Experiment #1234 | Started by Abbas | 2025-01-21
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Teams Message Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Training Complete - ResNet34_Standard  [Green bar]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accuracy:     96.8%
Precision:    96.5%
Recall:       96.7%
F1-Score:     96.6%
Duration:     14m 32s

[View Results]  [Compare Models]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Error Handling

### Rate Limiting

- Slack: 1 message/second (enforced locally + respects HTTP 429)
- Teams: 2 messages/second (enforced locally)
- Custom: No local rate limiting (relies on server response)

### Retries

- Automatic retry with exponential backoff (2s, 4s, 8s)
- Configurable retry attempts (default: 3)
- Logs all retry attempts for debugging

### Auto-Disable

Webhooks are automatically disabled after 10 consecutive failures to prevent spam and resource waste. Users can re-enable manually after fixing issues.

### Graceful Degradation

Webhook failures are isolated and don't affect:
- Email notifications
- In-app notifications
- Other webhooks
- Core application functionality

## Monitoring

### Check Webhook Status

```python
from models.webhook_configuration import WebhookConfiguration
from database.connection import get_db_session

with get_db_session() as session:
    webhooks = session.query(WebhookConfiguration).filter_by(
        user_id=42,
        is_active=True
    ).all()

    for webhook in webhooks:
        print(f"{webhook.provider_type}: {webhook.consecutive_failures} failures")
        print(f"Last used: {webhook.last_used_at}")
        print(f"Last error: {webhook.last_error}")
```

### View Webhook Logs

```python
from models.webhook_log import WebhookLog
from database.connection import get_db_session

with get_db_session() as session:
    logs = session.query(WebhookLog).filter_by(
        user_id=42
    ).order_by(WebhookLog.created_at.desc()).limit(10).all()

    for log in logs:
        print(f"{log.event_type}: {log.status} ({log.provider_type})")
```

## Security

- **HTTPS Only**: Custom webhooks must use HTTPS
- **URL Validation**: Strict regex validation for Slack/Teams URLs
- **Masked URLs**: Webhook URLs masked in logs (only last 10 chars shown)
- **User Isolation**: Users can only access their own webhooks

## Future Enhancements

- **Discord Support**: Add Discord webhook provider
- **Mattermost Support**: Add Mattermost webhook provider
- **Webhook Testing**: UI for testing webhooks before enabling
- **Digest Mode**: Weekly digest of all events to single message
- **Threading**: Slack thread support for related notifications
- **Mentions**: User/channel mentions in Slack messages

## Troubleshooting

### Webhook Not Sending

1. Check global feature flag: `NOTIFICATIONS_SLACK_ENABLED=true`
2. Check webhook is active: `webhook_config.is_active == True`
3. Check event is enabled: `event_type in webhook_config.enabled_events`
4. Check consecutive failures: `webhook_config.consecutive_failures < 10`
5. View logs: `SELECT * FROM webhook_logs WHERE webhook_config_id = X ORDER BY created_at DESC LIMIT 10`

### Invalid Webhook URL

- **Slack**: Must match `https://hooks.slack.com/services/T.../B.../...`
- **Teams**: Must match `https://*.office.com/webhook/.../IncomingWebhook/...`
- **Custom**: Must be valid HTTPS URL

### Rate Limiting Issues

If experiencing rate limit errors:
- Reduce event frequency (use digest mode)
- Increase `SLACK_RATE_LIMIT_PER_WEBHOOK` (not recommended)
- Use separate webhooks for different event types

## API Reference

See API endpoints documentation (to be added in next phase).

## License

Part of the LSTM Dashboard project.
