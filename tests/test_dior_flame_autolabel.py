"""Unit tests for the FLAME few-shot DIOR auto-labeling helpers in tools/."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load the tool module by path — tools/ is not a package.
_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "test_dior_flame.py"
_spec = importlib.util.spec_from_file_location("test_dior_flame_tool", _TOOL_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["test_dior_flame_tool"] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]

iou_xyxy_vs_xywh = _module.iou_xyxy_vs_xywh
auto_label_candidates = _module.auto_label_candidates


class _StubProposal:
    def __init__(self, box, image_idx):
        self.box = np.asarray(box, dtype=np.float32)
        self.image_idx = image_idx


class _StubFlame:
    def __init__(self, proposals):
        self.proposals = proposals


def test_iou_xyxy_vs_xywh_basic():
    # Identical boxes -> IoU 1.0
    assert iou_xyxy_vs_xywh(np.array([10, 10, 30, 30]), (10, 10, 20, 20)) == 1.0
    # Disjoint boxes -> IoU 0.0
    assert iou_xyxy_vs_xywh(np.array([0, 0, 10, 10]), (50, 50, 10, 10)) == 0.0
    # Half-overlap -> IoU 1/3
    iou = iou_xyxy_vs_xywh(np.array([0, 0, 20, 10]), (10, 0, 20, 10))
    assert abs(iou - (100 / 300)) < 1e-6


def test_auto_label_candidates_matches_iou_threshold():
    # Two support images with one GT each. Image 0: GT at (0,0)-(20,20).
    support_samples = [
        {
            "objects": {
                # COCO xywh
                "bbox": [[0, 0, 20, 20]],
                "category": [3],
            }
        },
        {
            "objects": {
                "bbox": [[100, 100, 10, 10]],
                "category": [3],
            }
        },
    ]
    proposals = [
        _StubProposal([0, 0, 20, 20], image_idx=0),       # exact match -> True
        _StubProposal([0, 0, 5, 5], image_idx=0),         # IoU 0.0625 -> False
        _StubProposal([100, 100, 110, 110], image_idx=1), # exact match -> True
        _StubProposal([200, 200, 210, 210], image_idx=1), # disjoint   -> False
    ]
    flame = _StubFlame(proposals)
    labels = auto_label_candidates(
        flame, [0, 1, 2, 3], support_samples, class_idx=3, iou_threshold=0.5
    )
    assert labels == [True, False, True, False]


def test_auto_label_skips_other_classes():
    # GT exists but for a different class -> all labels must be False.
    support_samples = [
        {
            "objects": {
                "bbox": [[0, 0, 20, 20]],
                "category": [7],  # different from class_idx
            }
        }
    ]
    proposals = [_StubProposal([0, 0, 20, 20], image_idx=0)]
    flame = _StubFlame(proposals)
    labels = auto_label_candidates(
        flame, [0], support_samples, class_idx=3, iou_threshold=0.5
    )
    assert labels == [False]
