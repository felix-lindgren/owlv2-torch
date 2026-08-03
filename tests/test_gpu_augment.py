import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype_train.gpu_augment import BatchAugmentor


IMAGE_SIZE = 64


def constant_colour_batch(batch_size, image_size=IMAGE_SIZE):
    """Image ``i`` is a uniform field of value ``i / batch_size``.

    Uniform images make the mosaic verifiable pixel by pixel: whatever ends up
    inside a box must be the colour of the image that box came from.
    """
    colours = torch.arange(batch_size, dtype=torch.float32) / batch_size
    return colours.view(batch_size, 1, 1, 1).expand(batch_size, 3, image_size, image_size).contiguous()


def colour_of_label(label, batch_size):
    return float(label) / batch_size


def box_interior_colours(image, box):
    """Every colour inside a normalised cxcywh box, inset by a pixel.

    The inset absorbs kornia's inclusive box convention; what is left must come
    entirely from one source image, so a mistranslated or mis-clamped box shows
    up as a second colour here.
    """
    height, width = image.shape[-2:]
    centre_x, centre_y, box_width, box_height = box.tolist()
    left = int(round((centre_x - box_width / 2) * width)) + 1
    right = int(round((centre_x + box_width / 2) * width)) - 1
    top = int(round((centre_y - box_height / 2) * height)) + 1
    bottom = int(round((centre_y + box_height / 2) * height)) - 1
    region = image[0, max(top, 0) : bottom + 1, max(left, 0) : right + 1]
    if region.numel() == 0:
        region = image[0, min(int(centre_y * height), height - 1), min(int(centre_x * width), width - 1)]
    return torch.unique(region)


def full_image_targets(batch_size):
    return [
        {
            "boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
            "labels": torch.tensor([index], dtype=torch.int64),
        }
        for index in range(batch_size)
    ]


def test_mosaic_keeps_each_label_attached_to_its_box():
    torch.manual_seed(0)
    batch_size = 6
    images = constant_colour_batch(batch_size)
    augmentor = BatchAugmentor(
        IMAGE_SIZE,
        mosaic_prob=1.0,
        min_box_visibility=0.0,
        color_jitter_brightness=0.0,
        color_jitter_contrast=0.0,
        color_jitter_saturation=0.0,
        color_jitter_hue=0.0,
    )

    mosaic_images, mosaic_targets = augmentor._apply_mosaic(images, full_image_targets(batch_size))

    assert mosaic_images.shape == images.shape
    total_boxes = 0
    for image, target in zip(mosaic_images, mosaic_targets):
        assert target["boxes"].shape[0] == target["labels"].shape[0]
        total_boxes += target["boxes"].shape[0]
        for box, label in zip(target["boxes"], target["labels"]):
            assert box_interior_colours(image, box).tolist() == pytest.approx(
                [colour_of_label(label, batch_size)]
            )
    # Every output should draw from more than one source image.
    assert total_boxes > batch_size


def test_mosaic_carries_labels_through_ragged_and_empty_targets():
    torch.manual_seed(1)
    batch_size = 4
    images = constant_colour_batch(batch_size)
    counts = [0, 1, 2, 3]
    targets = []
    for index, count in enumerate(counts):
        centres = torch.linspace(0.25, 0.75, max(count, 1))[:count]
        targets.append(
            {
                "boxes": torch.stack(
                    [centres, centres, torch.full((count,), 0.4), torch.full((count,), 0.4)],
                    dim=1,
                ),
                "labels": torch.full((count,), index, dtype=torch.int64),
            }
        )
    augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=1.0, min_box_visibility=0.0)

    mosaic_images, mosaic_targets = augmentor._apply_mosaic(images, targets)

    # Image 0 has no boxes but still gets a padded row; none of that padding may
    # survive as a label-0 detection.
    for image, target in zip(mosaic_images, mosaic_targets):
        sizes = target["boxes"][:, 2:4] * IMAGE_SIZE
        assert (sizes >= augmentor.min_box_size).all()
        for box, label in zip(target["boxes"], target["labels"]):
            assert box_interior_colours(image, box).tolist() == pytest.approx(
                [colour_of_label(label, batch_size)]
            )
            assert counts[int(label)] > 0


def test_mosaic_visibility_threshold_drops_heavily_clipped_boxes():
    torch.manual_seed(2)
    batch_size = 8
    images = constant_colour_batch(batch_size)
    targets = full_image_targets(batch_size)
    kept = {}
    for visibility in (0.0, 0.45):
        torch.manual_seed(2)
        augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=1.0, min_box_visibility=visibility)
        _, mosaic_targets = augmentor._apply_mosaic(images, targets)
        kept[visibility] = sum(target["boxes"].shape[0] for target in mosaic_targets)
    # A full-image box keeps at most 0.49 of its area after the mosaic crop, so
    # a 0.45 threshold has to remove some of them.
    assert kept[0.45] < kept[0.0]


def test_forward_normalises_and_keeps_the_model_input_shape():
    torch.manual_seed(3)
    batch_size = 4
    images = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=1.0)

    out_images, out_targets = augmentor(images, full_image_targets(batch_size))

    assert out_images.shape == images.shape
    assert len(out_targets) == batch_size
    # CLIP normalisation pushes [0, 1] well outside itself.
    assert out_images.min() < 0.0


def test_disabled_mosaic_leaves_targets_untouched():
    torch.manual_seed(4)
    batch_size = 3
    images = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    targets = full_image_targets(batch_size)
    augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=0.0)

    _, out_targets = augmentor(images, targets)

    assert out_targets is targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mosaic_runs_on_cuda():
    torch.manual_seed(5)
    batch_size = 4
    images = constant_colour_batch(batch_size).cuda()
    augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=1.0, min_box_visibility=0.0).cuda()

    out_images, out_targets = augmentor(images, full_image_targets(batch_size))

    assert out_images.is_cuda
    for target in out_targets:
        assert target["boxes"].is_cuda
        assert target["labels"].is_cuda
