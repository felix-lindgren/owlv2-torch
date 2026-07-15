"""Tests for the Fashionpedia-to-COCO conversion tool."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "convert_fashionpedia_to_coco.py"
)
_spec = importlib.util.spec_from_file_location("convert_fashionpedia_tool", _TOOL_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["convert_fashionpedia_tool"] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]

convert_split = _module.convert_split
xyxy_to_xywh = _module.xyxy_to_xywh


def _fixture() -> Dataset:
    features = Features({
        "image_id": Value("int64"),
        "image": Image(),
        "width": Value("int64"),
        "height": Value("int64"),
        "objects": {
            "bbox_id": Sequence(Value("int64")),
            "category": Sequence(ClassLabel(names=["shirt", "shoe"])),
            "bbox": Sequence(Sequence(Value("float64"), length=4)),
            "area": Sequence(Value("int64")),
        },
    })
    return Dataset.from_dict(
        {
            "image_id": [7],
            "image": [PILImage.new("RGB", (20, 10), color="white")],
            "width": [20],
            "height": [10],
            "objects": [{
                "bbox_id": [99, 100],
                "category": [1, 0],
                "bbox": [[2.0, 1.0, 12.0, 8.0], [4.0, 3.0, 4.0, 9.0]],
                "area": [70, 0],
            }],
        },
        features=features,
    )


def test_xyxy_to_xywh():
    assert xyxy_to_xywh([2, 1, 12, 8]) == [2.0, 1.0, 10.0, 7.0]


def test_convert_split_writes_images_and_coco_annotations(tmp_path):
    annotations_path = convert_split(_fixture(), "train", tmp_path)

    coco = json.loads(annotations_path.read_text())
    image = coco["images"][0]
    assert image == {
        "id": 7,
        "width": 20,
        "height": 10,
        "file_name": "7.png",
    }
    assert (tmp_path / "train" / "images" / image["file_name"]).exists()
    assert coco["annotations"] == [{
        "id": 1,
        "image_id": 7,
        "category_id": 1,
        "bbox": [2.0, 1.0, 10.0, 7.0],
        "area": 70.0,
        "iscrowd": 0,
    }]
    assert coco["categories"] == [
        {"id": 0, "name": "shirt"},
        {"id": 1, "name": "shoe"},
    ]


def test_convert_split_refuses_to_replace_annotations_by_default(tmp_path):
    convert_split(_fixture(), "val", tmp_path)

    try:
        convert_split(_fixture(), "val", tmp_path)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected conversion to protect existing annotations")
