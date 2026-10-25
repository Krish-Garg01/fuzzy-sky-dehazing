"""Image I/O and dataset pairing utilities.

Conventions:
- All images in memory are numpy.float32, shape (H, W, 3), RGB order, range [0, 1].
- OpenCV loads/saves BGR; conversion is performed at the I/O boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Pair:
    """A paired hazy and clear ground-truth image entry."""

    image_id: str
    hazy_path: Path
    clear_path: Path
    sky_mask_path: Path | None = None


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as RGB float32 in [0, 1].

    Grayscale images are expanded to 3 channels; alpha channels are dropped.

    Args:
        path: File path to read.

    Returns:
        Image array of shape (H, W, 3), dtype float32, RGB order, values in [0, 1].
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Failed to decode image: {file_path}")

    if raw.dtype == np.uint8:
        img = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        img = raw.astype(np.float32) / 65535.0
    elif np.issubdtype(raw.dtype, np.floating):
        img = np.clip(raw.astype(np.float32), 0.0, 1.0)
    else:
        raise TypeError(f"Unsupported image dtype {raw.dtype} in {file_path}")

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count {img.shape[2]} in {file_path}")
    else:
        raise ValueError(f"Unsupported image dimensions {img.shape} in {file_path}")

    return np.ascontiguousarray(img, dtype=np.float32)


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Save an image to disk as 8-bit PNG/JPEG.

    Clips float inputs to [0, 1], converts RGB to BGR for 3-channel images.

    Args:
        path: Output file path.
        img: Array of shape (H, W, 3) in RGB [0, 1] or (H, W) grayscale in [0, 1].
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(img, dtype=np.float32)
    clipped = np.clip(arr, 0.0, 1.0)
    uint8_img = np.round(clipped * 255.0).astype(np.uint8)

    if uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
        to_write = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)
    elif uint8_img.ndim == 2:
        to_write = uint8_img
    else:
        raise ValueError(f"Expected (H, W, 3) or (H, W) array to save, got {uint8_img.shape}")

    ok = cv2.imwrite(str(out_path), to_write)
    if not ok:
        raise IOError(f"cv2.imwrite failed for path: {out_path}")


def load_mask(path: str | Path) -> np.ndarray:
    """Load a binary mask image as a 2D boolean array.

    Args:
        path: Mask file path.

    Returns:
        Boolean array of shape (H, W) where True indicates positive/sky pixels.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {file_path}")

    raw = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Failed to decode mask image: {file_path}")

    return np.asarray(raw > 127, dtype=bool)


def _extract_id(path: Path, rule: str | Callable[[Path], str]) -> str:
    """Extract canonical image identifier from a filename."""
    if callable(rule):
        return str(rule(path))

    stem = path.stem
    if rule == "prefix_before_underscore":
        return stem.split("_")[0]
    if rule == "exact_stem":
        return stem
    if rule == "ohaze":
        # O-HAZE filenames are typically e.g. '01_outdoor_hazy.jpg' and '01_outdoor_GT.jpg'
        for suffix in ("_hazy", "_GT", "_clear", "_gt"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem.split("_")[0]
    raise ValueError(
        f"Unknown id_rule {rule!r}. Use 'prefix_before_underscore', 'exact_stem', 'ohaze', or a callable."
    )


def _list_images(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    files = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: p.name)
    return files


def pair_dataset(
    hazy_dir: str | Path,
    clear_dir: str | Path,
    id_rule: str | Callable[[Path], str] = "prefix_before_underscore",
    sky_mask_dir: str | Path | None = None,
    check_sizes: bool = True,
) -> list[Pair]:
    """Pair hazy images with clear ground-truth images and validate integrity.

    Pairing rules:
    - For 'prefix_before_underscore' (RESIDE SOTS-outdoor), the hazy stem prefix before '_'
      matches the clear stem (or clear prefix before '_').
    - We assert strict pairing between hazy images and clear partners (every hazy image maps
      to an existing clear image and no clear image is left unmatched).
    - Asserts spatial dimensions (H, W) match for every pair when check_sizes is True.

    Args:
        hazy_dir: Directory containing hazy images.
        clear_dir: Directory containing clear ground-truth images.
        id_rule: Rule name ('prefix_before_underscore', 'exact_stem', 'ohaze') or callable.
        sky_mask_dir: Optional directory containing binary sky masks named '<image_id>.png' or '<hazy_stem>.png'.
        check_sizes: Whether to load and assert matching (H, W) dimensions for every pair.

    Returns:
        Sorted list of validated Pair objects.
    """
    hazy_path_dir = Path(hazy_dir)
    clear_path_dir = Path(clear_dir)
    mask_path_dir = Path(sky_mask_dir) if sky_mask_dir is not None else None

    hazy_files = _list_images(hazy_path_dir)
    clear_files = _list_images(clear_path_dir)

    assert len(hazy_files) > 0, f"No hazy images found in {hazy_path_dir}"
    assert len(clear_files) > 0, f"No clear images found in {clear_path_dir}"

    clear_map: dict[str, Path] = {}
    for c_path in clear_files:
        cid = _extract_id(c_path, id_rule)
        assert cid not in clear_map, (
            f"Duplicate clear image ID {cid!r}: {clear_map[cid].name} and {c_path.name}"
        )
        clear_map[cid] = c_path

    # Count how many hazy files map to each clear ID so image_id is always unique per Pair
    hid_counts: dict[str, int] = {}
    for h_path in hazy_files:
        hid = _extract_id(h_path, id_rule)
        assert hid in clear_map, (
            f"Hazy image {h_path.name} (id={hid!r}) has no matching clear image in {clear_path_dir}"
        )
        hid_counts[hid] = hid_counts.get(hid, 0) + 1

    matched_clear_ids: set[str] = set()
    pairs: list[Pair] = []

    for h_path in hazy_files:
        hid = _extract_id(h_path, id_rule)
        matched_clear_ids.add(hid)
        c_path = clear_map[hid]
        unique_pair_id = hid if hid_counts[hid] == 1 else h_path.stem

        mask_file: Path | None = None
        if mask_path_dir is not None and mask_path_dir.is_dir():
            cand_id = mask_path_dir / f"{hid}.png"
            cand_stem = mask_path_dir / f"{h_path.stem}.png"
            if cand_stem.is_file():
                mask_file = cand_stem
            elif cand_id.is_file():
                mask_file = cand_id

        if check_sizes:
            h_img = cv2.imread(str(h_path), cv2.IMREAD_UNCHANGED)
            c_img = cv2.imread(str(c_path), cv2.IMREAD_UNCHANGED)
            assert h_img is not None, f"Failed to read hazy image {h_path}"
            assert c_img is not None, f"Failed to read clear image {c_path}"
            assert h_img.shape[:2] == c_img.shape[:2], (
                f"Size mismatch for ID {hid!r}: hazy {h_img.shape[:2]} vs clear {c_img.shape[:2]}"
            )

        pairs.append(
            Pair(
                image_id=unique_pair_id,
                hazy_path=h_path,
                clear_path=c_path,
                sky_mask_path=mask_file,
            )
        )

    unmatched_clear = set(clear_map.keys()) - matched_clear_ids
    assert not unmatched_clear, (
        f"Unmatched clear images in {clear_path_dir}: {sorted(unmatched_clear)[:10]}"
    )

    pairs.sort(key=lambda p: p.image_id)
    return pairs
