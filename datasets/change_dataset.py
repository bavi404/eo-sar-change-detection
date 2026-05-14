from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Mandatory mapping for 4-class masks:
      0,1 -> 0 (no change)
      2,3 -> 1 (change)

    Some splits may already be binary (0/1). In that case, keep them as-is.
    """
    mask = mask.copy()
    uniq = np.unique(mask)
    if np.isin(uniq, [0, 1]).all():
        return mask.astype(np.uint8)

    mask[(mask == 0) | (mask == 1)] = 0
    mask[(mask == 2) | (mask == 3)] = 1
    return mask.astype(np.uint8)


def _normalize_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _extract_eo(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        eo = arr[..., :3].astype(np.float32) / 255.0
        return eo
    # EO missing in this file, return zeros to keep fixed 8-channel input.
    h, w = arr.shape[:2]
    return np.zeros((h, w, 3), dtype=np.float32)


def input_channel_count(*, concat_diff: bool) -> int:
    """8 = EO_pre(3)+EO_post(3)+SAR_pre(1)+SAR_post(1); +4 diff if concat_diff."""
    return 12 if concat_diff else 8


def _extract_sar(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        sar = _normalize_channel(arr)
        return sar[..., None]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        sar = _normalize_channel(arr[..., 0])
        return sar[..., None]
    if arr.ndim == 3 and arr.shape[-1] > 3:
        sar = _normalize_channel(arr[..., 3])
        return sar[..., None]
    # SAR missing in this file, return zeros to keep fixed 8-channel input.
    h, w = arr.shape[:2]
    return np.zeros((h, w, 1), dtype=np.float32)


class ChangeDataset(Dataset):
    def __init__(self, split_root: str | Path, transforms=None, concat_diff: bool = True):
        self.split_root = Path(split_root)
        self.transforms = transforms
        self.concat_diff = concat_diff

        self.pre_files = sorted((self.split_root / "pre-event").glob("*.tif"))
        self.post_files = sorted((self.split_root / "post-event").glob("*.tif"))
        self.mask_files = sorted((self.split_root / "target").glob("*.tif"))

        if not self.pre_files or not self.post_files or not self.mask_files:
            raise FileNotFoundError(
                f"Expected .tif files in {self.split_root / 'pre-event'}, "
                f"{self.split_root / 'post-event'}, and {self.split_root / 'target'}"
            )
        if not (len(self.pre_files) == len(self.post_files) == len(self.mask_files)):
            raise ValueError("Mismatch in number of pre/post/target files.")

    def __len__(self) -> int:
        return len(self.pre_files)

    def __getitem__(self, idx: int):
        pre = tifffile.imread(self.pre_files[idx])
        post = tifffile.imread(self.post_files[idx])
        mask = tifffile.imread(self.mask_files[idx])
        mask = remap_mask(mask)

        eo_pre = _extract_eo(pre)    # (H, W, 3)
        eo_post = _extract_eo(post)  # (H, W, 3)
        sar_pre = _extract_sar(pre)  # (H, W, 1)
        sar_post = _extract_sar(post)  # (H, W, 1)

        parts = [eo_pre, eo_post, sar_pre, sar_post]
        if self.concat_diff:
            eo_diff = np.abs(eo_post.astype(np.float32) - eo_pre.astype(np.float32))
            sar_diff = np.abs(sar_post.astype(np.float32) - sar_pre.astype(np.float32))
            parts.extend([eo_diff, sar_diff])
        image = np.concatenate(parts, axis=-1).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return image, mask
