import os
import random

import numpy as np
import py360convert
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import DEFAULT_VP_UV_LIST


def is_image_file(file_name):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return file_name.lower().endswith(valid_ext)


class ERPViewportDataset(Dataset):
    """Dataset that reads ERP images and extracts viewports via py360convert.e2p."""

    def __init__(
        self,
        root,
        split="train",
        fov=(90.0, 90.0),
        uv_list=None,
        num_viewports=None,
        random_rotate=False,
        random_viewport_subset=False,
    ):
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                f"Dataset split directory not found: {split_root}. "
                f"Expected structure: <dataset>/{split}/..."
            )

        self.samples = []
        for current_root, _, files in os.walk(split_root):
            for file_name in files:
                if is_image_file(file_name):
                    self.samples.append(os.path.join(current_root, file_name))
        self.samples.sort()

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found in {split_root}")

        self.fov = (float(fov[0]), float(fov[1]))
        self.uv_list = list(uv_list if uv_list is not None else DEFAULT_VP_UV_LIST)
        self.random_rotate = random_rotate
        self.random_viewport_subset = random_viewport_subset

        if num_viewports is None:
            self.num_viewports = len(self.uv_list)
        else:
            self.num_viewports = int(num_viewports)

        if self.num_viewports <= 0 or self.num_viewports > len(self.uv_list):
            raise ValueError(
                f"num_viewports must be in [1, {len(self.uv_list)}], got {self.num_viewports}"
            )

    def _extract_viewports(self, erp):
        yaw_offset = random.uniform(-180.0, 180.0) if self.random_rotate else 0.0
        erp_h, erp_w = erp.shape[:2]
        fov_h, fov_w = self.fov
        out_h = max(1, int(round(erp_h * (fov_h / 180.0))))
        out_w = max(1, int(round(erp_w * (fov_w / 360.0))))
        candidate_uv = self.uv_list

        if self.num_viewports < len(self.uv_list):
            if self.random_viewport_subset:
                indices = sorted(random.sample(range(len(self.uv_list)), self.num_viewports))
                candidate_uv = [self.uv_list[idx] for idx in indices]
            else:
                candidate_uv = self.uv_list[: self.num_viewports]

        vp_list = []
        for u_deg, v_deg in candidate_uv:
            vp = py360convert.e2p(
                e_img=erp,
                fov_deg=self.fov,
                u_deg=u_deg + yaw_offset,
                v_deg=v_deg,
                out_hw=(out_h, out_w),
                mode="bilinear",
            )
            vp_list.append(vp)
        return vp_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        erp = np.asarray(Image.open(self.samples[index]).convert("RGB"))
        viewports = self._extract_viewports(erp)
        stacked_viewports = np.stack(viewports, axis=0)
        return torch.from_numpy(stacked_viewports).permute(0, 3, 1, 2).float().div(255.0)
