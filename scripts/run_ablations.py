"""
Run ablation studies for PINN model evaluation.

This wrapper script provides a simple entry point for running ablation experiments
as specified in MASTER_ROADMAP_FINAL.md Chapter 3.1.

Usage:
    python scripts/run_ablations.py              # Run full ablation study
    python scripts/run_ablations.py --quick      # Run quick mode (key configs only)
    python scripts/run_ablations.py --seeds 42,123,456  # Run with multiple seeds
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Run PINN ablation studies for research validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablations.py                    # Full ablation study
  python scripts/run_ablations.py --quick            # Quick mode (3 key configs)
  python scripts/run_ablations.py --seeds 42,123,456 # Multiple seeds
  python scripts/run_ablations.py --config my_config.yaml  # Custom config
        """
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: only run key comparison configs'
    )
    
    parser.add_argument(
        '--seeds',
        type=str,
        default='42',
        help='Comma-separated list of random seeds (default: 42)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional custom configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/ablation',
        help='Output directory for results (default: results/ablation)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'efficientnet_b0'],
        help='Base model architecture (default: resnet18)'
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    print("=" * 60)
    print("PINN ABLATION STUDY")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Seeds: {seeds}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Import and run the ablation study
    try:
        from scripts.research.pinn_ablation import run_ablation_study, ABLATION_CONFIGS, QUICK_CONFIGS
        
        # Select configs based on mode
        if args.quick:
            configs = [c for c in ABLATION_CONFIGS if c['name'] in QUICK_CONFIGS]
        else:
            configs = ABLATION_CONFIGS
        
        print(f"\nRunning {len(configs)} ablation configurations...")
        
        # Run for each seed
        all_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            results = run_ablation_study(configs, quick=args.quick)
            for r in results:
                r['seed'] = seed
            all_results.extend(results)
        
        # Summary
        print("\n" + "=" * 60)
        print("ABLATION STUDY COMPLETE")
        print("=" * 60)
        print(f"Total runs: {len(all_results)}")
        print(f"Results saved to: {args.output_dir}")
        
    except ImportError as e:
        print(f"Error: Could not import ablation study module: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during ablation study: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
