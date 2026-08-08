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


def marked_colour_of_label(label, batch_size):
    # Offset by one so image 0 is distinguishable from the black background.
    return (float(label) + 1.0) / (batch_size + 1.0)


def marked_box_batch(batch_size, generator, image_size=IMAGE_SIZE):
    """Image ``i`` is black except for one filled rectangle: its own box.

    ``constant_colour_batch`` cannot see a box that lands in the correct tile at
    the wrong offset, because every pixel of that tile holds the same colour.
    Here a mistranslated box covers background instead of its own colour, which
    is exactly what kornia's partial-batch box path used to produce.
    """
    images = torch.zeros((batch_size, 3, image_size, image_size))
    targets = []
    for index in range(batch_size):
        centre = 0.3 + 0.4 * torch.rand(2, generator=generator)
        size = 0.15 + 0.15 * torch.rand(2, generator=generator)
        left = int(round(float(centre[0] - size[0] / 2) * image_size))
        right = int(round(float(centre[0] + size[0] / 2) * image_size))
        top = int(round(float(centre[1] - size[1] / 2) * image_size))
        bottom = int(round(float(centre[1] + size[1] / 2) * image_size))
        images[index, :, top:bottom, left:right] = marked_colour_of_label(
            index, batch_size
        )
        targets.append(
            {
                "boxes": torch.cat([centre, size]).unsqueeze(0),
                "labels": torch.tensor([index], dtype=torch.int64),
            }
        )
    return images, targets


def mosaicked_mask(out_targets, targets):
    """Which samples the augmentor actually mosaicked.

    Selection is the wrapper's own draw rather than anything kornia exposes. An
    unselected sample is passed through by identity, which is an exact signal --
    comparing images is not, since a mosaic whose cells all come from one source
    can be pixel-identical to that source.
    """
    return [out is not original for out, original in zip(out_targets, targets)]


@pytest.mark.parametrize("mosaic_prob", (0.25, 0.5, 0.75))
def test_mosaic_boxes_cover_their_own_content_on_a_partial_batch(mosaic_prob):
    """Regression: boxes must land on their own pixels, not just the right tile.

    kornia builds a partial batch's images by rank among the applied samples but
    translates the boxes by original batch position, so below ``p=1.0`` the boxes
    came back at the wrong offsets. The wrapper now drives kornia at ``p=1`` and
    masks afterwards. A full-image box cannot detect this -- it covers the whole
    tile either way -- which is why this uses marked sub-image boxes.
    """
    batch_size = 8
    saw_partial = False
    for seed in range(25):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 10_000)
        images, targets = marked_box_batch(batch_size, generator)
        augmentor = BatchAugmentor(
            IMAGE_SIZE,
            mosaic_prob=mosaic_prob,
            min_box_visibility=0.2,
            color_jitter_brightness=0.0,
            color_jitter_contrast=0.0,
            color_jitter_saturation=0.0,
            color_jitter_hue=0.0,
        )

        out_images, out_targets = augmentor._apply_mosaic(images, targets)

        applied = mosaicked_mask(out_targets, targets)
        saw_partial |= 0 < sum(applied) < batch_size
        for image, target in zip(out_images, out_targets):
            for box, label in zip(target["boxes"], target["labels"]):
                assert box_interior_colours(image, box).tolist() == pytest.approx(
                    [marked_colour_of_label(label, batch_size)]
                ), "a retained box does not cover its own source content"
    assert saw_partial, "no seed produced a partially mosaicked batch"


@pytest.mark.parametrize("mosaic_prob", (0.25, 0.5, 0.75))
def test_mosaic_selects_roughly_the_requested_fraction(mosaic_prob):
    """The probability moved out of kornia, so it has to still mean something."""
    batch_size = 8
    mosaicked = 0
    samples = 0
    for seed in range(60):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 10_000)
        images, targets = marked_box_batch(batch_size, generator)
        augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=mosaic_prob)

        out_images, out_targets = augmentor._apply_mosaic(images, targets)

        applied = mosaicked_mask(out_targets, targets)
        mosaicked += sum(applied)
        samples += batch_size
        for index, was_mosaicked in enumerate(applied):
            if not was_mosaicked:
                # An unselected sample keeps its own boxes, untouched.
                assert torch.equal(out_targets[index]["boxes"], targets[index]["boxes"])
                assert torch.equal(
                    out_targets[index]["labels"], targets[index]["labels"]
                )
    assert abs(mosaicked / samples - mosaic_prob) < 0.05


@pytest.mark.parametrize("mosaic_prob", (0.25, 0.5, 0.75))
def test_mosaic_keeps_labels_attached_when_only_part_of_the_batch_is_mosaicked(mosaic_prob):
    """Labels must stay on their boxes once the probability skips someone.

    Selection is now the wrapper's own draw, so appliedness is read back off the
    images rather than out of kornia's parameters.
    """
    batch_size = 6
    images = constant_colour_batch(batch_size)
    saw_partial = False
    for seed in range(12):
        torch.manual_seed(seed)
        augmentor = BatchAugmentor(
            IMAGE_SIZE,
            mosaic_prob=mosaic_prob,
            min_box_visibility=0.0,
            color_jitter_brightness=0.0,
            color_jitter_contrast=0.0,
            color_jitter_saturation=0.0,
            color_jitter_hue=0.0,
        )
        targets = full_image_targets(batch_size)
        mosaic_images, mosaic_targets = augmentor._apply_mosaic(images, targets)

        applied = mosaicked_mask(mosaic_targets, targets)
        saw_partial |= 0 < sum(applied) < batch_size
        for index, (image, target) in enumerate(zip(mosaic_images, mosaic_targets)):
            for box, label in zip(target["boxes"], target["labels"]):
                assert box_interior_colours(image, box).tolist() == pytest.approx(
                    [colour_of_label(label, batch_size)]
                )
            if not applied[index]:
                # A skipped sample must come back exactly as it went in.
                assert target["labels"].tolist() == [index]
                assert target["boxes"].flatten().tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert saw_partial, "no seed produced a partially mosaicked batch"


def test_mosaic_passes_the_batch_through_when_no_sample_is_selected():
    """Selecting nobody must short-circuit before kornia is ever called."""
    batch_size = 4
    images = constant_colour_batch(batch_size)
    for seed in range(200):
        torch.manual_seed(seed)
        augmentor = BatchAugmentor(IMAGE_SIZE, mosaic_prob=0.05, min_box_visibility=0.0)
        targets = full_image_targets(batch_size)
        out_images, out_targets = augmentor._apply_mosaic(images, targets)
        if out_targets is targets:
            assert torch.equal(out_images, images)
            return
    pytest.fail("no seed left the whole batch unmosaicked")


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


# --- DEIM-style downscaling mosaic -----------------------------------------


@pytest.mark.parametrize("mosaic_prob", (0.25, 0.5, 1.0))
def test_downscale_mosaic_boxes_cover_their_own_content(mosaic_prob):
    """The same coordinate check that caught the crop-mode bug.

    ``box_interior_colours`` is not usable here: downscaling resamples, so a
    small box's interior is largely bilinear edge blur rather than one exact
    value. The centre pixel is unambiguous, and a bigger canvas keeps the
    downscaled rectangles several pixels wide.
    """
    image_size = 128
    batch_size = 8
    for seed in range(15):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 10_000)
        images, targets = marked_box_batch(batch_size, generator, image_size)
        augmentor = BatchAugmentor(
            image_size,
            mosaic_prob=mosaic_prob,
            mosaic_mode="downscale",
            min_box_size=0.0,
            color_jitter_brightness=0.0,
            color_jitter_contrast=0.0,
            color_jitter_saturation=0.0,
            color_jitter_hue=0.0,
        )

        out_images, out_targets = augmentor._apply_downscale_mosaic(images, targets)

        for image, target in zip(out_images, out_targets):
            for box, label in zip(target["boxes"], target["labels"]):
                want = marked_colour_of_label(label, batch_size)
                x = min(int(float(box[0]) * image_size), image_size - 1)
                y = min(int(float(box[1]) * image_size), image_size - 1)
                assert float(image[0, y, x]) == pytest.approx(want, abs=1e-4), (
                    "a downscaled box does not cover its own source content"
                )


def test_downscale_mosaic_keeps_every_box_and_shrinks_it_by_the_grid():
    """Nothing is cropped, so all boxes survive at exactly 1/grid scale.

    This is the property the crop mosaic lacks and the reason to try this
    variant at all: it manufactures small objects instead of removing context.
    """
    torch.manual_seed(7)
    batch_size = 6
    rows, cols = 2, 2
    generator = torch.Generator().manual_seed(99)
    images, targets = marked_box_batch(batch_size, generator)
    augmentor = BatchAugmentor(
        IMAGE_SIZE,
        mosaic_prob=1.0,
        mosaic_mode="downscale",
        mosaic_grid=(rows, cols),
        min_box_size=0.0,
    )

    _, out_targets = augmentor._apply_downscale_mosaic(images, targets)

    source_area = targets[0]["boxes"][0, 2] * targets[0]["boxes"][0, 3]
    for target in out_targets:
        # One box per source image, rows*cols sources, none dropped.
        assert target["boxes"].shape[0] == rows * cols
        widths = target["boxes"][:, 2]
        heights = target["boxes"][:, 3]
        assert (widths < 1.0 / cols).all() and (heights < 1.0 / rows).all()
        # Every box stays inside the frame.
        assert (target["boxes"][:, 0] - widths / 2 >= -1e-5).all()
        assert (target["boxes"][:, 1] - heights / 2 >= -1e-5).all()
        assert (target["boxes"][:, 0] + widths / 2 <= 1 + 1e-5).all()
        assert (target["boxes"][:, 1] + heights / 2 <= 1 + 1e-5).all()
    # Areas shrink by the grid factor, which is what makes small objects.
    all_areas = torch.cat([t["boxes"][:, 2] * t["boxes"][:, 3] for t in out_targets])
    assert float(all_areas.max()) < float(source_area)


def test_downscale_mosaic_passes_unselected_samples_through():
    batch_size = 8
    for seed in range(50):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 10_000)
        images, targets = marked_box_batch(batch_size, generator)
        augmentor = BatchAugmentor(
            IMAGE_SIZE, mosaic_prob=0.5, mosaic_mode="downscale"
        )

        out_images, out_targets = augmentor._apply_downscale_mosaic(images, targets)

        for index, was_mosaicked in enumerate(mosaicked_mask(out_targets, targets)):
            if not was_mosaicked:
                assert torch.equal(out_images[index], images[index])


def test_downscale_mosaic_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mosaic_mode"):
        BatchAugmentor(IMAGE_SIZE, mosaic_prob=0.5, mosaic_mode="nonsense")
