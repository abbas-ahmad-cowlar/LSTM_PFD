"""
CNN DataLoaders — backward-compatible re-export shim.

This module re-exports all public symbols from the unified ``data.dataloader``
module so that existing ``from data.cnn_dataloader import …`` statements
continue to work unchanged.

The canonical implementation now lives in ``data/dataloader.py``.

Author: Syed Abbas Ahmad
Date: 2025-11-20  (consolidated 2026-03-15)
"""

from data.dataloader import (
    collate_signals as collate_fn,
    create_dataloader as create_cnn_dataloader,
    create_cnn_dataloaders,
    DataLoaderConfig,
)

__all__ = [
    "collate_fn",
    "create_cnn_dataloader",
    "create_cnn_dataloaders",
    "DataLoaderConfig",
]
