#!/usr/bin/env python3
"""
Compare training results across dataset versions.

Usage:
    python scripts/compare_results.py --v1 results/ --v2 results_v2/
    python scripts/compare_results.py --v1 results/ --v2 results_v2/ --output comparison.json
    python scripts/compare_results.py --leaderboard results/
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.core.evaluation.dataset_comparison import DatasetComparisonEngine


def main():
    parser = argparse.ArgumentParser(description="Compare training results")
    parser.add_argument(
        "--v1", type=str, help="Path to V1 results directory"
    )
    parser.add_argument(
        "--v2", type=str, help="Path to V2 results directory"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for comparison JSON report"
    )
    parser.add_argument(
        "--leaderboard", type=str, default=None,
        help="Print leaderboard for a single results directory"
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top models for leaderboard"
    )
    args = parser.parse_args()

    engine = DatasetComparisonEngine()

    # Leaderboard mode
    if args.leaderboard:
        results = engine.load_results(Path(args.leaderboard))
        leaderboard = engine.generate_leaderboard(results, top_n=args.top)

        print("=" * 70)
        print(f"LEADERBOARD: {results.version_tag}")
        print(f"Models: {len(results.models)} | "
              f"Physics: {results.advanced_physics or 'basic'}")
        print("=" * 70)
        print(f"{'Rank':>4} {'Model':<30} {'Test Acc':>10} "
              f"{'Params':>12} {'Time':>8}")
        print("-" * 70)
        for entry in leaderboard:
            print(f"  {entry['rank']:>2}. {entry['model']:<28} "
                  f"{entry['test_accuracy']:>10.4f} "
                  f"{entry['params']:>12,} "
                  f"{entry['time_s']:>7.0f}s")
        return

    # Comparison mode
    if not args.v1 or not args.v2:
        parser.error("--v1 and --v2 are required for comparison mode")

    v1 = engine.load_results(Path(args.v1))
    v2 = engine.load_results(Path(args.v2))
    report = engine.compare(v1, v2)

    # Print comparison
    output_text = engine.print_comparison(report)
    print(output_text)

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("results") / "comparison_report.json"
    engine.save_report(report, output_path)
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
