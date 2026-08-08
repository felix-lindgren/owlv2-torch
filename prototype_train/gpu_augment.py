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
from torch.nn import functional as F

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
        mosaic_mode: str = "crop",
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
        if mosaic_mode not in ("crop", "downscale"):
            raise ValueError(
                f"mosaic_mode must be 'crop' or 'downscale', got {mosaic_mode!r}"
            )
        self.min_box_size = min_box_size
        self.min_box_visibility = min_box_visibility
        self.mosaic_prob = mosaic_prob
        self.mosaic_mode = mosaic_mode
        self.mosaic_grid = tuple(mosaic_grid)
        self.mosaic_enabled = mosaic_prob > 0
        self.mosaic = (
            K.RandomMosaic(
                output_size=(image_size, image_size),
                mosaic_grid=tuple(mosaic_grid),
                start_ratio_range=mosaic_start_ratio_range,
                data_keys=["input", "bbox_xyxy"],
                # Driven at p=1 and masked afterwards: on a partial batch kornia
                # builds the images from the applied samples' rank but transforms
                # the boxes by original batch position, so the boxes land at the
                # wrong offsets. Measured at 706 of 1,932 retained boxes wrong at
                # p=0.5, and 0 of 2,286 at p=1. The per-sample draw therefore
                # happens here instead, and kornia only ever sees a full batch.
                p=1.0,
                # ``slice`` crops by indexing instead of warping, so the mosaic
                # costs no interpolation and the boxes stay pixel-exact.
                cropping_mode="slice",
            )
            if mosaic_prob > 0 and mosaic_mode == "crop"
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
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        *,
        apply_mosaic: bool = True,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if len(targets) != images.shape[0]:
            raise ValueError(
                f"Received {len(targets)} targets for a batch of {images.shape[0]}"
            )
        # A mosaic of one image with itself is a plain crop, so skip it.
        if apply_mosaic and self.mosaic_enabled and images.shape[0] > 1:
            if self.mosaic_mode == "downscale":
                images, targets = self._apply_downscale_mosaic(images, targets)
            else:
                images, targets = self._apply_mosaic(images, targets)
        if self.photometric is not None:
            images = self.photometric(images)
        return self.normalize(images), targets

    def _apply_downscale_mosaic(
        self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """DEIM-style mosaic: each source is downscaled into its own grid cell.

        The crop-style mosaic concatenates images at their original scale and
        cuts an input-sized window out, so it discards content and never makes
        an object smaller. This one resizes a whole source image into each cell,
        so nothing is cropped, every box survives, and a ``rows x cols`` grid
        shrinks every object by that factor -- which is the small-object
        supervision the crop variant cannot produce.

        Built from plain tensor ops rather than kornia, which keeps it clear of
        the partial-batch box bug that made the crop path unusable below p=1.
        """
        device = images.device
        batch_size, _, height, width = images.shape
        rows, cols = self.mosaic_grid

        selected = torch.rand(batch_size, device=device) < self.mosaic_prob
        if not bool(selected.any()):
            return images, targets

        # Integer cell edges so the tiles tile exactly, even for an odd size.
        row_edges = [round(r * height / rows) for r in range(rows + 1)]
        col_edges = [round(c * width / cols) for c in range(cols + 1)]

        sources = torch.randint(batch_size, (batch_size, rows * cols), device=device)
        mosaic_images = images.new_empty(images.shape)
        boxes_per_sample: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        labels_per_sample: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for cell in range(rows * cols):
            row, col = divmod(cell, cols)
            top, bottom = row_edges[row], row_edges[row + 1]
            left, right = col_edges[col], col_edges[col + 1]
            cell_height, cell_width = bottom - top, right - left
            tiles = F.interpolate(
                images,
                size=(cell_height, cell_width),
                mode="bilinear",
                align_corners=False,
            )
            picks = sources[:, cell]
            mosaic_images[:, :, top:bottom, left:right] = tiles[picks]

            for index in range(batch_size):
                if not bool(selected[index]):
                    continue
                source = int(picks[index])
                boxes = targets[source]["boxes"]
                if boxes.shape[0] == 0:
                    continue
                boxes = boxes.to(device=device, dtype=images.dtype)
                # Normalised cxcywh maps onto the cell by an affine rescale; no
                # clipping is possible, so no visibility filter is needed.
                boxes_per_sample[index].append(
                    torch.stack(
                        [
                            (left + boxes[:, 0] * cell_width) / width,
                            (top + boxes[:, 1] * cell_height) / height,
                            boxes[:, 2] * cell_width / width,
                            boxes[:, 3] * cell_height / height,
                        ],
                        dim=1,
                    )
                )
                labels_per_sample[index].append(
                    targets[source]["labels"].to(device=device, dtype=torch.int64)
                )

        out_images = torch.where(selected[:, None, None, None], mosaic_images, images)
        out_targets = []
        for index in range(batch_size):
            if not bool(selected[index]):
                out_targets.append(targets[index])
                continue
            if boxes_per_sample[index]:
                boxes = torch.cat(boxes_per_sample[index], dim=0)
                labels = torch.cat(labels_per_sample[index], dim=0)
                keep = (boxes[:, 2] * width >= self.min_box_size) & (
                    boxes[:, 3] * height >= self.min_box_size
                )
                boxes, labels = boxes[keep], labels[keep]
            else:
                boxes = images.new_zeros((0, 4))
                labels = torch.zeros((0,), dtype=torch.int64, device=device)
            out_targets.append({"boxes": boxes, "labels": labels})
        return out_images, out_targets

    def _apply_mosaic(
        self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        device = images.device
        batch_size = images.shape[0]
        height, width = float(images.shape[-2]), float(images.shape[-1])

        # The per-sample draw lives here rather than inside kornia; see the p=1.0
        # note in __init__. Nothing selected means nothing to build.
        selected = torch.rand(batch_size, device=device) < self.mosaic_prob
        if not bool(selected.any()):
            return images, targets

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

        # Driving kornia at p=1 means every sample must come back mosaicked, with
        # one permutation row each. Anything else is the partial-batch path this
        # wrapper exists to avoid, and its boxes cannot be trusted.
        if not bool(applied.all()) or permutation.shape[0] != batch_size:
            raise RuntimeError(
                f"kornia mosaicked {int(applied.sum())} of {batch_size} samples "
                f"with {permutation.shape[0]} permutations; expected the full "
                "batch, so the box layout cannot be trusted"
            )

        # RandomMosaic returns B x (cells * max_boxes) x 4: block ``k`` holds the
        # boxes of source image ``permutation[:, k]``, translated into the mosaic
        # and clamped to the crop. Gathering the labels the same way is what
        # keeps them attached to their boxes.
        if mosaic_boxes.shape[1] != cells * max_boxes:
            raise RuntimeError(
                f"Expected {cells * max_boxes} mosaic box slots, got "
                f"{mosaic_boxes.shape[1]}; the block layout is not as assumed"
            )
        out_labels = torch.cat([labels[permutation[:, k]] for k in range(cells)], dim=1)
        keep = torch.cat([valid[permutation[:, k]] for k in range(cells)], dim=1)

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

        if mosaic_images.shape != images.shape:
            raise RuntimeError(
                f"Mosaic changed the batch shape from {tuple(images.shape)} to "
                f"{tuple(mosaic_images.shape)}; it cannot be masked back in"
            )

        # Every sample was mosaicked, so the ones the probability did not select
        # are dropped here and keep their untouched image and targets.
        out_height, out_width = float(mosaic_images.shape[-2]), float(mosaic_images.shape[-1])
        out_images = torch.where(
            selected[:, None, None, None], mosaic_images, images
        )
        out_targets = [
            {
                "boxes": _xyxy_to_cxcywh_norm(
                    mosaic_boxes[index][keep[index]], out_width, out_height
                ),
                "labels": out_labels[index][keep[index]],
            }
            if bool(selected[index])
            else targets[index]
            for index in range(batch_size)
        ]
        return out_images, out_targets
