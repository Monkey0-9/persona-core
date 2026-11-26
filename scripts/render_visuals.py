"""CLI wrapper for generating diagnostic static PNG and interactive HTML visualizations."""

from __future__ import annotations

import argparse

from mall_seg.visualization import render_all_visuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Render diagnostic visualizations.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()
    paths = render_all_visuals(config_path=args.config)
    for path in paths:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
