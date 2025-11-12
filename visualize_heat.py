#!/usr/bin/env python3

"""Visualize heat simulation output as a 2D heatmap."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_grid(path: Path) -> np.ndarray:
    """Load the saved simulation grid from disk."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            header = fh.readline()
            if not header:
                raise ValueError("File is empty.")
            grid_size = int(header.strip())
            values = []
            for row_idx in range(grid_size):
                line = fh.readline()
                if not line:
                    raise ValueError(f"Missing data for row {row_idx}.")
                row = [float(value) for value in line.strip().split() if value]
                if len(row) != grid_size:
                    raise ValueError(
                        f"Row {row_idx} has {len(row)} columns, expected {grid_size}."
                    )
                values.append(row)
    except OSError as exc:
        raise RuntimeError(f"Could not read '{path}': {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid grid file '{path}': {exc}") from exc

    return np.asarray(values, dtype=np.float32)


def plot_grid(grid: np.ndarray, title: str, output: Path | None) -> None:
    """Render the grid using matplotlib."""
    plt.figure(figsize=(6, 5))
    image = plt.imshow(grid, cmap="inferno", origin="lower")
    plt.colorbar(image, label="Temperature")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=300)
        print(f"Saved heatmap to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="simulation_output.txt",
        help="Path to the simulation output file (default: simulation_output.txt)",
    )
    parser.add_argument(
        "--save",
        dest="output",
        help="Optional path to save the rendered heatmap instead of displaying it",
    )
    parser.add_argument(
        "--title",
        default="Heat Diffusion",
        help="Custom title for the plot (default: Heat Diffusion)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    grid = load_grid(input_path)
    plot_grid(grid, args.title, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

