"""Batch-level detection augmentations that run on the device, built on kornia.

``AugmentedDetectionDataset`` augments one sample at a time in the dataloader
workers. Mosaic needs several images at once, so it cannot live there; this
module picks up the collated batch after it reaches the GPU and applies mosaic
plus the photometric jitter there. That also takes ColorJitter off the CPU
workers, which is the expensive part of the per-sample pipeline.

The dataset must be built with ``model.image_transform_unnormed`` when this
pipeline is used: images arrive as float ``[0, 1]`` square tensors and are
normalised here, after the augmentations that assume that range.
"""

from __future__ import annotations

import kornia.augmentation as K
import torch
from torch import nn

from OWLv2torch.torch_version.owlv2 import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def _cxcywh_norm_to_xyxy(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    centre = boxes[:, 0:2] * torch.tensor([width, height], device=boxes.device, dtype=boxes.dtype)
    size = boxes[:, 2:4] * torch.tensor([width, height], device=boxes.device, dtype=boxes.dtype)
    return torch.cat([centre - size * 0.5, centre + size * 0.5], dim=1)


def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    size = boxes[:, 2:4] - boxes[:, 0:2]
    centre = boxes[:, 0:2] + size * 0.5
    scale = torch.tensor([width, height], device=boxes.device, dtype=boxes.dtype)
    return torch.cat([centre / scale, size / scale], dim=1)


class BatchAugmentor(nn.Module):
    """Mosaic + photometric jitter + normalisation for a collated detection batch.

    Call with the batch images (float ``[0, 1]``, ``B x 3 x S x S``) and the list
    of per-image targets holding normalised ``cxcywh`` boxes; returns normalised
    images and rebuilt targets. Mosaic keeps the output resolution, so it is a
    drop-in for the un-augmented batch.
    """

    def __init__(
        self,
        image_size: int,
        *,
        mosaic_prob: float = 0.5,
        mosaic_grid: tuple[int, int] = (2, 2),
        mosaic_start_ratio_range: tuple[float, float] = (0.3, 0.7),
        min_box_size: float = 2.0,
        min_box_visibility: float = 0.2,
        color_jitter_brightness: float = 0.3,
        color_jitter_contrast: float = 0.3,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.02,
    ):
        super().__init__()
        if not 0.0 <= mosaic_prob <= 1.0:
            raise ValueError(f"mosaic_prob must be in [0, 1], got {mosaic_prob}")
        if not 0.0 <= min_box_visibility <= 1.0:
            raise ValueError(f"min_box_visibility must be in [0, 1], got {min_box_visibility}")
        if min(mosaic_grid) < 1:
            raise ValueError(f"mosaic_grid entries must be positive, got {mosaic_grid}")
        self.min_box_size = min_box_size
        self.min_box_visibility = min_box_visibility
        self.mosaic = (
            K.RandomMosaic(
                output_size=(image_size, image_size),
                mosaic_grid=tuple(mosaic_grid),
                start_ratio_range=mosaic_start_ratio_range,
                data_keys=["input", "bbox_xyxy"],
                p=mosaic_prob,
                # ``slice`` crops by indexing instead of warping, so the mosaic
                # costs no interpolation and the boxes stay pixel-exact.
                cropping_mode="slice",
            )
            if mosaic_prob > 0
            else None
        )
        jitter = (
            color_jitter_brightness,
            color_jitter_contrast,
            color_jitter_saturation,
            color_jitter_hue,
        )
        # ColorJiggle reproduces torchvision's ColorJitter; kornia's ColorJitter
        # deliberately does not.
        self.photometric = K.ColorJiggle(*jitter, p=1.0) if any(jitter) else None
        self.normalize = K.Normalize(
            mean=torch.tensor(OPENAI_CLIP_MEAN), std=torch.tensor(OPENAI_CLIP_STD)
        )

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if len(targets) != images.shape[0]:
            raise ValueError(
                f"Received {len(targets)} targets for a batch of {images.shape[0]}"
            )
        # A mosaic of one image with itself is a plain crop, so skip it.
        if self.mosaic is not None and images.shape[0] > 1:
            images, targets = self._apply_mosaic(images, targets)
        if self.photometric is not None:
            images = self.photometric(images)
        return self.normalize(images), targets

    def _apply_mosaic(
        self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        device = images.device
        batch_size = images.shape[0]
        height, width = float(images.shape[-2]), float(images.shape[-1])

        # kornia wants one dense box tensor, so pad to the largest count in the
        # batch and carry a validity mask alongside. Images without boxes still
        # need a padded row, hence the ``max(..., 1)``.
        counts = [int(target["boxes"].shape[0]) for target in targets]
        max_boxes = max(max(counts), 1)
        padded = images.new_zeros((batch_size, max_boxes, 4))
        labels = torch.zeros((batch_size, max_boxes), dtype=torch.int64, device=device)
        valid = torch.zeros((batch_size, max_boxes), dtype=torch.bool, device=device)
        for index, target in enumerate(targets):
            count = counts[index]
            if count == 0:
                continue
            boxes = target["boxes"].to(device=device, dtype=images.dtype)
            padded[index, :count] = _cxcywh_norm_to_xyxy(boxes, width, height)
            labels[index, :count] = target["labels"].to(device=device, dtype=torch.int64)
            valid[index, :count] = True

        mosaic_images, mosaic_boxes = self.mosaic(images, padded)
        # RandomMosaic transforms boxes but not labels, so the sampled tiling has
        # to be read back off the module to reattach them. Fail loudly if a
        # kornia upgrade renames these, because the alternative is silently
        # scrambled labels.
        params = getattr(self.mosaic, "_params", None) or {}
        if "permutation" not in params or "batch_prob" not in params:
            raise RuntimeError(
                "kornia's RandomMosaic no longer exposes 'permutation'/'batch_prob' "
                f"parameters (got {sorted(params)}); mosaic labels cannot be recovered"
            )
        permutation = params["permutation"].to(device)
        applied = params["batch_prob"].to(device) > 0.5
        cells = permutation.shape[1]

        # RandomMosaic returns B x (cells * max_boxes) x 4: block ``k`` holds the
        # boxes of source image ``permutation[:, k]``, translated into the mosaic
        # and clamped to the crop. Gathering the labels the same way is what
        # keeps them attached to their boxes. For samples the per-sample
        # probability skipped, kornia keeps block 0 and zeroes the rest.
        out_labels = torch.cat([labels[permutation[:, k]] for k in range(cells)], dim=1)
        keep = torch.cat(
            [
                valid[permutation[:, k]] & (applied[:, None] if k else True)
                for k in range(cells)
            ],
            dim=1,
        )

        sizes = (mosaic_boxes[..., 2:4] - mosaic_boxes[..., 0:2]).clamp_min(0)
        keep &= (sizes[..., 0] >= self.min_box_size) & (sizes[..., 1] >= self.min_box_size)
        if self.min_box_visibility > 0:
            source_sizes = (padded[..., 2:4] - padded[..., 0:2]).clamp_min(0)
            source_areas = torch.cat(
                [(source_sizes[..., 0] * source_sizes[..., 1])[permutation[:, k]] for k in range(cells)],
                dim=1,
            )
            visibility = sizes[..., 0] * sizes[..., 1] / source_areas.clamp_min(1e-6)
            keep &= visibility >= self.min_box_visibility

        out_height, out_width = float(mosaic_images.shape[-2]), float(mosaic_images.shape[-1])
        mosaic_targets = [
            {
                "boxes": _xyxy_to_cxcywh_norm(mosaic_boxes[index][keep[index]], out_width, out_height),
                "labels": out_labels[index][keep[index]],
            }
            for index in range(batch_size)
        ]
        return mosaic_images, mosaic_targets
