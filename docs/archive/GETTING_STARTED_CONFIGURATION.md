# Configuration

Environment configuration for LSTM PFD.

## Environment Variables

| Variable               | Required | Default                    | Description                  |
| ---------------------- | -------- | -------------------------- | ---------------------------- |
| `DATABASE_URL`         | Yes      | -                          | Database connection string   |
| `SECRET_KEY`           | Yes      | -                          | Flask session encryption key |
| `JWT_SECRET_KEY`       | Yes      | -                          | JWT token signing key        |
| `REDIS_URL`            | No       | `redis://localhost:6379/0` | Redis connection             |
| `CUDA_VISIBLE_DEVICES` | No       | `0`                        | GPU device selection         |
| `LOG_LEVEL`            | No       | `INFO`                     | Logging verbosity            |

## Configuration Files

### `.env` (Development)

```bash
# Development configuration
DATABASE_URL=sqlite:///./lstm_dashboard.db
SECRET_KEY=your-32-char-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
DEBUG=True
```

### `config/` Directory

| File                 | Purpose                     |
| -------------------- | --------------------------- |
| `data_config.py`     | Dataset generation settings |
| `model_config.py`    | Model architecture defaults |
| `training_config.py` | Training hyperparameters    |

## See Also

- [Installation](installation.md)
- [First Experiment](first-experiment.md)
