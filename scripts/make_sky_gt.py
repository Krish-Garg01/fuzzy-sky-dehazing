"""Generate pseudo ground-truth sky masks from clear images for evaluation only.

Design notes:
- Ground-truth sky masks are used strictly for evaluation metrics (never read by dehazing methods).
- Uses a pretrained ADE20K semantic segmentation model (SegFormer) via HuggingFace `transformers`
  on the clear (haze-free) images, extracting the 'sky' class.
- Writes binary PNG masks to `data/sky_masks/<dataset>/<image_id>.png`, generates `sky_fraction.csv`,
  and optionally writes side-by-side preview overlays (`--preview`) for manual inspection.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fsdehaze.config import load_config
from fsdehaze.io_utils import load_image, pair_dataset, save_image


def segment_sky_segformer(
    clear_rgb: np.ndarray,
    processor: object,
    model: object,
    device: str,
) -> np.ndarray:
    """Predict a binary sky mask from a clear RGB float32 [0, 1] image using SegFormer."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    H, W = clear_rgb.shape[:2]
    pil_img = Image.fromarray(np.round(np.clip(clear_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, num_labels, h, w)
        upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

    # Find label index corresponding to 'sky' in model.config.id2label (default 2 in ADE20K)
    sky_ids = [
        int(k)
        for k, v in model.config.id2label.items()
        if "sky" in str(v).lower()
    ]
    if not sky_ids:
        sky_ids = [2]

    mask = np.isin(pred, sky_ids)
    return mask.astype(bool)


def make_preview_overlay(clear_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a red-tinted overlay on sky pixels for human spot-checking."""
    overlay = clear_rgb.copy()
    tint = np.array([1.0, 0.2, 0.2], dtype=np.float32)
    overlay[mask] = 0.65 * overlay[mask] + 0.35 * tint
    return np.clip(overlay, 0.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating sky GT masks."""
    parser = argparse.ArgumentParser(
        description="Generate pseudo-GT sky masks from clear images using SegFormer (ADE20K)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sots_outdoor",
        help="Dataset key in configs/datasets.yaml (default: sots_outdoor)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys in configs/datasets.yaml (overrides --dataset)",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to datasets YAML config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace ADE20K segmentation model identifier",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save preview overlays for human inspection (at least 30 recommended)",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=40,
        help="Maximum number of preview overlays to save (default: 40)",
    )

    args = parser.parse_args(argv)
    datasets_cfg = load_config(args.datasets_config)
    dataset_list = args.datasets if args.datasets is not None else [args.dataset]

    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model).to(device)
    model.eval()

    for ds_name in dataset_list:
        if ds_name not in datasets_cfg:
            raise KeyError(f"Dataset {ds_name!r} not found in {args.datasets_config}")

        ds_info = datasets_cfg[ds_name]
        pairs = pair_dataset(
            hazy_dir=ds_info["hazy_dir"],
            clear_dir=ds_info["clear_dir"],
            id_rule=ds_info.get("id_rule", "prefix_before_underscore"),
            check_sizes=False,
        )

        out_mask_dir = Path(ds_info.get("sky_mask_dir") or f"data/sky_masks/{ds_name}")
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = out_mask_dir / "_previews"
        if args.preview:
            preview_dir.mkdir(parents=True, exist_ok=True)

        mask_cache: dict[Path, np.ndarray] = {}
        records: list[dict[str, object]] = []
        for idx, pair in enumerate(tqdm(pairs, desc=f"Sky GT [{ds_name}]", leave=False)):
            clear_img = load_image(pair.clear_path)
            if pair.clear_path in mask_cache:
                mask = mask_cache[pair.clear_path]
            else:
                mask = segment_sky_segformer(clear_img, processor, model, device)
                mask_cache[pair.clear_path] = mask

            sky_frac = float(np.mean(mask))

            mask_path = out_mask_dir / f"{pair.image_id}.png"
            cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))

            if args.preview and idx < args.preview_limit:
                overlay = make_preview_overlay(clear_img, mask)
                save_image(preview_dir / f"{pair.image_id}_overlay.png", overlay)

            records.append(
                {
                    "image_id": pair.image_id,
                    "clear_file": pair.clear_path.name,
                    "hazy_file": pair.hazy_path.name,
                    "sky_fraction": sky_frac,
                    "sky_heavy": bool(sky_frac > 0.20),
                }
            )

        df = pd.DataFrame(records)
        csv_path = out_mask_dir / "sky_fraction.csv"
        df.to_csv(csv_path, index=False)
        n_heavy = int(df["sky_heavy"].sum())
        print(
            f"[{ds_name}] Saved {len(df)} sky masks to {out_mask_dir} | "
            f"Sky-heavy (>20%): {n_heavy}/{len(df)} ({100.0 * n_heavy / len(df):.1f}%) | CSV: {csv_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
