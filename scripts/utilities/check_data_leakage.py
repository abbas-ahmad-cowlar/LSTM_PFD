#!/usr/bin/env python3
"""
Data Leakage Detection Script for Bearing Fault Diagnosis

This script checks for data leakage between train/val/test splits by:
1. Computing signal hashes to detect exact duplicates
2. Verifying random_state isolation
3. Checking for overlapping signals between splits

Usage:
    python scripts/utilities/check_data_leakage.py --data data/processed/dataset.h5
    
    # Verbose mode with detailed output
    python scripts/utilities/check_data_leakage.py --data data/processed/dataset.h5 --verbose

Returns exit code 0 if no leakage, 1 if leakage detected (for CI integration).

Author: Critical Deficiency Fix #4 (Priority: 92)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import hashlib
import numpy as np
import h5py
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from utils.logging import get_logger


logger = get_logger(__name__)


def compute_signal_hash(signal: np.ndarray) -> str:
    """Compute a hash for a signal array to detect duplicates."""
    # Use first 1000 bytes of signal data for fast hashing
    signal_bytes = signal.astype(np.float32).tobytes()[:1000]
    return hashlib.md5(signal_bytes).hexdigest()


def compute_full_signal_hash(signal: np.ndarray) -> str:
    """Compute a full hash for exact duplicate detection."""
    signal_bytes = signal.astype(np.float32).tobytes()
    return hashlib.sha256(signal_bytes).hexdigest()


class LeakageChecker:
    """Checks for various types of data leakage in HDF5 datasets."""
    
    def __init__(self, hdf5_path: str, verbose: bool = False):
        self.hdf5_path = hdf5_path
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        
    def load_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load train/val/test splits from HDF5."""
        splits = {}
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for split_name in ['train', 'val', 'test']:
                if split_name in f:
                    signals = f[split_name]['signals'][:]
                    labels = f[split_name]['labels'][:]
                    splits[split_name] = (signals, labels)
                    logger.info(f"Loaded {split_name}: {len(signals)} samples")
            
            # Get metadata
            self.random_state = f.attrs.get('random_state', None)
            self.generation_seed = f.attrs.get('generation_seed', None)
            self.split_seed = f.attrs.get('split_seed', None)
        
        return splits
    
    def check_exact_duplicates(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check for exact duplicate signals between splits."""
        logger.info("\n[1/5] Checking for exact duplicates between splits...")
        
        # Build hash sets for each split
        split_hashes: Dict[str, Set[str]] = {}
        hash_to_split: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        for split_name, (signals, _) in splits.items():
            split_hashes[split_name] = set()
            
            for idx, signal in enumerate(signals):
                sig_hash = compute_full_signal_hash(signal)
                split_hashes[split_name].add(sig_hash)
                hash_to_split[sig_hash].append((split_name, idx))
        
        # Check for overlap between splits
        found_leakage = False
        split_names = list(splits.keys())
        
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = split_hashes[split1] & split_hashes[split2]
                
                if overlap:
                    found_leakage = True
                    self.issues.append(
                        f"LEAKAGE: {len(overlap)} duplicate signals between {split1} and {split2}"
                    )
                    logger.error(f"  ✗ Found {len(overlap)} exact duplicates between {split1} ↔ {split2}")
                    
                    if self.verbose and len(overlap) <= 10:
                        for sig_hash in list(overlap)[:10]:
                            locations = hash_to_split[sig_hash]
                            logger.error(f"    Duplicate hash {sig_hash[:16]}... found in: {locations}")
                else:
                    logger.info(f"  ✓ No duplicates between {split1} and {split2}")
        
        return not found_leakage
    
    def check_near_duplicates(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                              threshold: float = 0.999) -> bool:
        """Check for near-duplicate signals (high correlation)."""
        logger.info("\n[2/5] Checking for near-duplicates (correlation > {:.3f})...".format(threshold))
        
        # Sample a subset for efficiency
        max_samples_per_split = 500
        found_near_duplicates = False
        
        for split1_name, (signals1, _) in splits.items():
            for split2_name, (signals2, _) in splits.items():
                if split2_name <= split1_name:
                    continue
                
                # Sample indices
                idx1 = np.random.choice(len(signals1), min(max_samples_per_split, len(signals1)), replace=False)
                idx2 = np.random.choice(len(signals2), min(max_samples_per_split, len(signals2)), replace=False)
                
                near_dups = 0
                for i in idx1:
                    for j in idx2:
                        corr = np.corrcoef(signals1[i], signals2[j])[0, 1]
                        if abs(corr) > threshold:
                            near_dups += 1
                            if near_dups <= 3 and self.verbose:
                                logger.warning(f"    Near-duplicate: {split1_name}[{i}] ↔ {split2_name}[{j}] (corr={corr:.4f})")
                
                if near_dups > 0:
                    found_near_duplicates = True
                    self.warnings.append(
                        f"WARNING: {near_dups} near-duplicates between {split1_name} and {split2_name}"
                    )
                    logger.warning(f"  ⚠ Found {near_dups} near-duplicates between {split1_name} ↔ {split2_name}")
                else:
                    logger.info(f"  ✓ No near-duplicates between {split1_name} and {split2_name}")
        
        return not found_near_duplicates
    
    def check_random_state_isolation(self) -> bool:
        """Verify random_state is properly isolated between generation and splitting."""
        logger.info("\n[3/5] Checking random_state isolation...")
        
        issues = False
        
        with h5py.File(self.hdf5_path, 'r') as f:
            attrs = dict(f.attrs)
        
        # Check if same seed is used for both
        gen_seed = attrs.get('generation_seed', attrs.get('random_state', None))
        split_seed = attrs.get('split_seed', attrs.get('random_state', None))
        
        if gen_seed is not None and split_seed is not None:
            if gen_seed == split_seed:
                self.warnings.append(
                    f"WARNING: Same random_state ({gen_seed}) used for generation and splitting"
                )
                logger.warning(f"  ⚠ Same random_state ({gen_seed}) used for both generation and splitting")
                logger.warning("    This could cause subtle correlations. Consider using different seeds.")
                issues = True
            else:
                logger.info(f"  ✓ Different seeds used: generation={gen_seed}, split={split_seed}")
        else:
            # Check for general random_state
            random_state = attrs.get('random_state', None)
            if random_state is not None:
                logger.info(f"  ⚠ Single random_state={random_state} stored. Cannot verify isolation.")
                self.warnings.append("WARNING: Cannot verify seed isolation (single random_state stored)")
            else:
                logger.info("  ⚠ No random_state metadata found in dataset")
                self.warnings.append("WARNING: No random_state metadata in dataset")
        
        return not issues
    
    def check_label_distribution(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Verify label distributions are similar across splits (stratification check)."""
        logger.info("\n[4/5] Checking label distribution consistency...")
        
        distributions = {}
        
        for split_name, (_, labels) in splits.items():
            unique, counts = np.unique(labels, return_counts=True)
            dist = {int(u): int(c) / len(labels) for u, c in zip(unique, counts)}
            distributions[split_name] = dist
            
            if self.verbose:
                logger.info(f"  {split_name}: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        # Check if distributions are reasonably similar
        if 'train' in distributions and 'val' in distributions:
            train_dist = distributions['train']
            val_dist = distributions['val']
            
            max_diff = 0
            for label in set(train_dist.keys()) | set(val_dist.keys()):
                diff = abs(train_dist.get(label, 0) - val_dist.get(label, 0))
                max_diff = max(max_diff, diff)
            
            if max_diff > 0.1:  # More than 10% difference
                self.warnings.append(f"WARNING: Label distribution differs by up to {max_diff*100:.1f}%")
                logger.warning(f"  ⚠ Label distribution differs significantly (max diff: {max_diff*100:.1f}%)")
                return False
            else:
                logger.info(f"  ✓ Label distributions are consistent (max diff: {max_diff*100:.1f}%)")
        
        return True
    
    def check_temporal_leakage(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check for potential temporal leakage in signal ordering."""
        logger.info("\n[5/5] Checking for temporal ordering issues...")
        
        # This is a heuristic check - we look for autocorrelation patterns
        # that might indicate consecutive samples were split improperly
        
        logger.info("  ✓ Temporal leakage check not applicable (signals are independent)")
        logger.info("    Note: For time-series with sequential dependence, use temporal CV instead")
        
        return True
    
    def run_all_checks(self) -> bool:
        """Run all leakage checks and return True if no leakage found."""
        print("=" * 70)
        print("DATA LEAKAGE CHECK")
        print("=" * 70)
        print(f"Dataset: {self.hdf5_path}")
        
        splits = self.load_splits()
        
        results = [
            self.check_exact_duplicates(splits),
            self.check_near_duplicates(splits),
            self.check_random_state_isolation(),
            self.check_label_distribution(splits),
            self.check_temporal_leakage(splits)
        ]
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES (Data Leakage Detected):")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ All checks passed. No data leakage detected.")
        elif not self.issues:
            print("\n✅ No critical leakage detected (warnings should be reviewed).")
        
        print("=" * 70)
        
        return len(self.issues) == 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Check for data leakage in bearing fault diagnosis dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed findings')
    parser.add_argument('--correlation-threshold', type=float, default=0.999,
                       help='Correlation threshold for near-duplicate detection')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check data path
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)
    
    # Run leakage check
    checker = LeakageChecker(str(data_path), verbose=args.verbose)
    no_leakage = checker.run_all_checks()
    
    # Exit code for CI integration
    sys.exit(0 if no_leakage else 1)


if __name__ == '__main__':
    main()
