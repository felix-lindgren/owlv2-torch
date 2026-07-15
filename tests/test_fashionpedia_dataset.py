"""Tests for the Hugging Face Fashionpedia evaluation adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "ovd_eval.py"
_spec = importlib.util.spec_from_file_location("ovd_eval_tool", _TOOL_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["ovd_eval_tool"] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]

PHOTO_QUERY_TEMPLATE = _module.PHOTO_QUERY_TEMPLATE
FashionpediaDetectionDataset = _module.FashionpediaDetectionDataset
class_name_to_query = _module.class_name_to_query


def _fashionpedia_fixture() -> Dataset:
    class_names = ["shirt, blouse", "shoe"]
    features = Features({
        "image_id": Value("int64"),
        "image": Image(),
        "width": Value("int64"),
        "height": Value("int64"),
        "objects": {
            "bbox_id": Sequence(Value("int64")),
            "category": Sequence(ClassLabel(names=class_names)),
            "bbox": Sequence(Sequence(Value("float64"), length=4)),
            "area": Sequence(Value("int64")),
        },
    })
    return Dataset.from_dict(
        {
            "image_id": [7, 8],
            "image": [
                PILImage.new("RGB", (20, 10), color="white"),
                PILImage.new("RGB", (12, 16), color="black"),
            ],
            "width": [20, 12],
            "height": [10, 16],
            "objects": [
                {
                    "bbox_id": [99],
                    "category": [1],
                    "bbox": [[2.0, 1.0, 12.0, 8.0]],
                    "area": [70],
                },
                {
                    "bbox_id": [100],
                    "category": [0],
                    "bbox": [[1.0, 2.0, 5.0, 10.0]],
                    "area": [32],
                },
            ],
        },
        features=features,
    )


def test_fashionpedia_adapter_converts_pascal_voc_boxes_to_coco():
    dataset = FashionpediaDetectionDataset(
        split="val",
        dataset=_fashionpedia_fixture(),
    )

    coco = dataset.build_coco_gt()

    assert dataset.class_names == ["shirt, blouse", "shoe"]
    assert coco["annotations"][0] == {
        "id": 1,
        "image_id": 7,
        "category_id": 1,
        "bbox": [2.0, 1.0, 10.0, 7.0],
        "area": 70.0,
        "iscrowd": 0,
    }
    assert coco["categories"] == [
        {"id": 0, "name": "shirt, blouse"},
        {"id": 1, "name": "shoe"},
    ]


def test_fashionpedia_adapter_decodes_images_and_applies_limit():
    dataset = FashionpediaDetectionDataset(
        split="val",
        limit=1,
        dataset=_fashionpedia_fixture(),
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].mode == "RGB"
    assert sample["image_id"] == 7
    assert sample["target_size"] == (10, 20)
    assert sample["offset"] == (0.0, 0.0)


def test_photo_query_template_is_used_for_fashion_categories():
    assert (
        class_name_to_query("top, t-shirt, sweatshirt", PHOTO_QUERY_TEMPLATE)
        == "a photo of top, t shirt, sweatshirt"
    )
