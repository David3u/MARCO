#!/usr/bin/env python3
"""
Create evaluation subset with only tasks that have max grid size <= 15x15
This filters for "easy" and "medium" difficulty tasks based on grid size.

Usage:
    python create_eval_subset.py --input /path/to/evaluation --output eval_subset
"""

import json
import shutil
from pathlib import Path
import argparse


def get_max_grid_size(task_data):
    """Get maximum grid size (width or height) in a task"""
    max_size = 0

    # Check training examples
    for example in task_data.get('train', []):
        for grid in [example.get('input', []), example.get('output', [])]:
            if grid:
                height = len(grid)
                width = len(grid[0]) if grid else 0
                max_size = max(max_size, height, width)

    # Check test examples
    for example in task_data.get('test', []):
        for grid in [example.get('input', []), example.get('output', [])]:
            if grid:
                height = len(grid)
                width = len(grid[0]) if grid else 0
                max_size = max(max_size, height, width)

    return max_size


def filter_evaluation_tasks(input_dir, output_dir, max_grid_size=15):
    """Filter evaluation tasks by maximum grid size"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    task_files = list(input_path.glob("*.json"))

    if not task_files:
        print(f"Warning: no JSON files found in {input_dir}")
        return

    print(f"Found {len(task_files)} tasks in {input_dir}")
    print(f"Filtering for max grid size <= {max_grid_size}x{max_grid_size}")
    print("=" * 80)

    copied_count = 0
    skipped_count = 0

    for task_file in sorted(task_files):
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)

            max_size = get_max_grid_size(task_data)

            if max_size <= max_grid_size:
                shutil.copy2(task_file, output_path / task_file.name)
                copied_count += 1
                print(f"Copied {task_file.name}: max_size={max_size}x{max_size}")
            else:
                skipped_count += 1
                print(f"Skipped {task_file.name}: max_size={max_size}x{max_size}")

        except Exception as e:
            print(f"Error processing {task_file.name}: {e}")
            skipped_count += 1

    print("=" * 80)
    print(f"\nCreated evaluation subset in: {output_path}")
    print(f"  Copied: {copied_count} tasks (max grid size <= {max_grid_size})")
    print(f"  Skipped: {skipped_count} tasks (too large)")
    print(f"  Subset size: {copied_count}/{len(task_files)} ({100*copied_count/len(task_files):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter ARC evaluation tasks by grid size"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_subset",
        help="Path to output directory (default: eval_subset)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=15,
        help="Maximum grid size (width or height) to include (default: 15)"
    )

    args = parser.parse_args()

    filter_evaluation_tasks(args.input, args.output, args.max_size)
