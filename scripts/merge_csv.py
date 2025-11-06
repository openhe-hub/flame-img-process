import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger


def collect_csv_files(input_dir: Path, pattern: str) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    return csv_files


def merge_csv_files(csv_files: List[Path], output_path: Path) -> None:
    if not csv_files:
        logger.warning("No CSV files found to merge. Exiting.")
        return

    logger.info(f"Merging {len(csv_files)} CSV files into '{output_path}'")
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            logger.debug(f"Loaded '{csv_file}' with shape {df.shape}")
        except Exception as exc:
            logger.error(f"Failed to read '{csv_file}': {exc}")

    if not dataframes:
        logger.warning("No dataframes were loaded successfully. Exiting.")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    logger.success(f"Saved merged dataset to '{output_path}' ({merged_df.shape[0]} rows).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-experiment CSV files into a single dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Directory containing per-experiment CSV files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match CSV files within the input directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset/total_dataset.csv"),
        help="Path for the merged output CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        logger.error(
            f"Output file '{args.output}' already exists. Use --overwrite to replace it."
        )
        return

    try:
        csv_files = collect_csv_files(args.input_dir, args.pattern)
    except FileNotFoundError as exc:
        logger.error(exc)
        return

    merge_csv_files(csv_files, args.output)


if __name__ == "__main__":
    main()
