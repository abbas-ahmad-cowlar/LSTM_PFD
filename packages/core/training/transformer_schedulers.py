"""
Transformer Schedulers — backward-compat re-export shim.

All implementations have been moved to ``training.schedulers``.
Import from ``training.schedulers`` instead.
"""

from .schedulers import (
    create_warmup_cosine_schedule,
    create_warmup_linear_schedule,
    create_noam_schedule,
    create_scheduler,
    WarmupCosineScheduler,
)

# Legacy alias
get_scheduler = create_scheduler

__all__ = [
    "create_warmup_cosine_schedule",
    "create_warmup_linear_schedule",
    "create_noam_schedule",
    "get_scheduler",
    "WarmupCosineScheduler",
]
