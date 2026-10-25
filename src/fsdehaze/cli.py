"""Command-line interface for single-image dehazing."""

from __future__ import annotations

import argparse
from pathlib import Path

from fsdehaze.config import load_config
from fsdehaze.io_utils import load_image, save_image
from fsdehaze.pipelines import (
    dehaze_dcp,
    dehaze_fuzzy,
    dehaze_hard_fcm,
    dehaze_hard_threshold,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m fsdehaze.cli`."""
    parser = argparse.ArgumentParser(description="Fuzzy Sky-Aware Single-Image Dehazing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dehaze_p = subparsers.add_parser("dehaze", help="Dehaze a single image")
    dehaze_p.add_argument("--input", required=True, type=Path, help="Input hazy image path")
    dehaze_p.add_argument(
        "--method",
        default="dcp",
        choices=["dcp", "B0", "hard_threshold", "B1", "hard_fcm", "B2", "fuzzy", "F0", "F1"],
        help="Dehazing method",
    )
    dehaze_p.add_argument(
        "--config",
        default=Path("configs/default.yaml"),
        type=Path,
        help="Path to base configuration YAML",
    )
    dehaze_p.add_argument(
        "--params",
        default=None,
        type=Path,
        help="Optional tuned parameters YAML (for fuzzy/F1)",
    )
    dehaze_p.add_argument("--out", required=True, type=Path, help="Output dehazed image path")
    dehaze_p.add_argument(
        "--save-debug",
        action="store_true",
        help="Save intermediate maps (dark, t_dcp, t_final, mu/mask) next to output",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    img = load_image(args.input)

    if args.method in ("dcp", "B0"):
        res = dehaze_dcp(img, cfg)
    elif args.method in ("hard_threshold", "B1"):
        res = dehaze_hard_threshold(img, cfg)
    elif args.method in ("hard_fcm", "B2"):
        res = dehaze_hard_fcm(img, cfg)
    else:
        params = load_config(args.params) if args.params is not None else None
        res = dehaze_fuzzy(img, cfg, params=params)

    save_image(args.out, res.J)

    if args.save_debug:
        stem = args.out.with_suffix("")
        if "dark" in res.info:
            save_image(f"{stem}_dark.png", res.info["dark"])
        if "t_dcp" in res.info:
            save_image(f"{stem}_t_dcp.png", res.info["t_dcp"])
        save_image(f"{stem}_t_final.png", res.t)
        if res.mu is not None:
            save_image(f"{stem}_mu.png", res.mu)
        if res.mask is not None:
            save_image(f"{stem}_mask.png", res.mask.astype("float32"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
