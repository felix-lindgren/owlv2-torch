"""Two-phase oriented bounding-box (OBB) training on top of OWLv2.

Phase 1 – Freeze the entire pretrained network; only train the newly
          initialised angle head (and the prototype bank / logit_scale).
Phase 2 – Unfreeze the box head and jointly fine-tune it together with
          the angle head and prototype bank.

Dataset assumptions
-------------------
Uses torchvision ``CocoDetection`` with an extended COCO-style annotation
format where each object dict MAY contain an ``"angle"`` key (radians,
measured counter-clockwise from the positive x-axis, range arbitrary –
the loss uses a (sin 2θ, cos 2θ) representation that is π-periodic).

If ``"angle"`` is absent from every annotation in a batch the angle loss
is simply zero, so this script is backwards-compatible with standard COCO
datasets (the box/classification heads still train normally).
"""

from OWLv2torch.torch_version.prototypes import VisualPrototypeBank
from OWLv2torch.torch_version.owlv2 import OwlV2, OBBDetector
from OWLv2torch.torch_version.loss import compute_losses
import torch
from torchvision.datasets import CocoDetection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
import mlflow
import mlflow.pytorch
import os
import math
from datetime import datetime


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def obb_coco_collate_fn(batch, id2size=None, letterbox=False):
    """COCO collate that also extracts per-object ``angle`` when present."""
    images, targets = [], []
    for image, anns in batch:
        images.append(image)
        H1, W1 = image.shape[-2], image.shape[-1]

        if len(anns) > 0:
            b = torch.as_tensor(
                [a["bbox"] for a in anns], dtype=torch.float32, device=image.device
            )  # xywh (orig space)
            labels = torch.as_tensor(
                [a["category_id"] for a in anns], dtype=torch.int64, device=image.device
            )
            angles = torch.as_tensor(
                [a.get("angle", 0.0) for a in anns], dtype=torch.float32, device=image.device
            )

            if id2size is None:
                raise ValueError("id2size mapping required")

            img_id = anns[0]["image_id"]
            meta = id2size[img_id]
            W0, H0 = float(meta["width"]), float(meta["height"])

            sx, sy = W1 / W0, H1 / H0
            if letterbox and abs(sx - sy) > 1e-6:
                s = min(sx, sy)
                px = (W1 - W0 * s) * 0.5
                py = (H1 - H0 * s) * 0.5
            else:
                s, px, py = sx, 0.0, 0.0

            b[:, 0] = b[:, 0] * s + px
            b[:, 1] = b[:, 1] * s + py
            b[:, 2] = b[:, 2] * s
            b[:, 3] = b[:, 3] * s

            b[:, 0].clamp_(0, W1)
            b[:, 1].clamp_(0, H1)
            b[:, 2] = torch.minimum(b[:, 2], W1 - b[:, 0])
            b[:, 3] = torch.minimum(b[:, 3], H1 - b[:, 1])

            cx = b[:, 0] + 0.5 * b[:, 2]
            cy = b[:, 1] + 0.5 * b[:, 3]
            boxes = torch.stack((cx / W1, cy / H1, b[:, 2] / W1, b[:, 3] / H1), dim=1)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=image.device)
            labels = torch.zeros((0,), dtype=torch.int64, device=image.device)
            angles = torch.zeros((0,), dtype=torch.float32, device=image.device)

        targets.append({"boxes": boxes, "labels": labels, "angles": angles})

    return {"images": torch.stack(images, 0), "targets": targets}


# ---------------------------------------------------------------------------
# Evaluation (reuses standard AABB eval – angle is ignored for now)
# ---------------------------------------------------------------------------

OPENAI_CLIP_MEAN: list[float] = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD: list[float] = [0.26862954, 0.26130258, 0.27577711]


def obb_eval(model, val_loader, device):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        max_detection_thresholds=[1, 10, 100],
        class_metrics=False,
        backend="faster_coco_eval",
    )
    metric.warn_on_many_detections = False
    confidence_threshold = 0.05

    predictions, gts = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["images"].to(device)
            with torch.autocast(device):
                outputs = model(pixel_values)
            logits, pred_boxes = outputs[1], outputs[3]

            B = pixel_values.shape[0]
            H, W = pixel_values.shape[-2], pixel_values.shape[-1]
            target_sizes = torch.tensor(
                [(H, W)] * B, device=device, dtype=torch.float32
            )
            boxes = model.owl.postprocess_boxes(pred_boxes, target_sizes)
            probs = torch.max(logits, dim=-1)
            scores = torch.sigmoid(probs.values)
            labels = probs.indices

            for i in range(B):
                mask = scores[i] > confidence_threshold
                predictions.append(
                    {
                        "boxes": boxes[i][mask].cpu(),
                        "scores": scores[i][mask].cpu(),
                        "labels": labels[i][mask].cpu().to(torch.int64),
                    }
                )
                gt = batch["targets"][i]
                gt_boxes = gt["boxes"]
                if len(gt_boxes) > 0:
                    img_h, img_w = H, W
                    cx, cy = gt_boxes[:, 0] * img_w, gt_boxes[:, 1] * img_h
                    w, h = gt_boxes[:, 2] * img_w, gt_boxes[:, 3] * img_h
                    gt_xyxy = torch.stack(
                        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
                    )
                else:
                    gt_xyxy = torch.zeros((0, 4), dtype=torch.float32)
                gts.append(
                    {
                        "boxes": gt_xyxy.cpu(),
                        "labels": gt["labels"].cpu().to(torch.int64),
                    }
                )

    metric.update(predictions, gts)
    metrics = metric.compute()
    print(f"\n=== COCO Evaluation (AABB proxy) ===")
    print(f"mAP @.50:.95 = {metrics['map']:.4f}   mAP @.50 = {metrics['map_50']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    mlflow.set_experiment("OwlV2-OBB-Training")

    config = {
        "device": "cuda",
        "C": 1,
        "K": 4,
        "model_type": "large",
        # Phase 1
        "phase1_lr": 1e-4,
        "phase1_epochs": 3,
        # Phase 2
        "phase2_lr": 3e-5,
        "phase2_epochs": 5,
        # Shared
        "batch_size": 16,
        "lambda_cls": 1.0,
        "lambda_l1": 5.0,
        "lambda_giou": 2.0,
        "lambda_angle": 1.0,
        # Paths (user must override)
        "train_ann": "path/to/train_annotations.json",
        "train_img": "path/to/train_images",
        "val_ann": "path/to/val_annotations.json",
        "val_img": "path/to/val_images",
    }

    with mlflow.start_run(
        run_name=f"obb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_params(config)
        device = config["device"]
        C, K = config["C"], config["K"]

        # Build model
        model = OwlV2(config["model_type"])
        bank = VisualPrototypeBank(C, K, dim=model.text_dim)
        det = OBBDetector(model, bank, phase=1)
        det = det.to(device)

        # Datasets
        train_dataset = CocoDetection(
            annFile=config["train_ann"],
            root=config["train_img"],
            transform=model.image_transform_fast,
        )
        val_dataset = CocoDetection(
            annFile=config["val_ann"],
            root=config["val_img"],
            transform=model.image_transform_fast,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=partial(
                obb_coco_collate_fn,
                id2size=train_dataset.coco.imgs,
                letterbox=False,
            ),
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            collate_fn=partial(
                obb_coco_collate_fn,
                id2size=val_dataset.coco.imgs,
                letterbox=False,
            ),
            shuffle=False,
        )

        loss_kwargs = dict(
            lambda_cls=config["lambda_cls"],
            lambda_l1=config["lambda_l1"],
            lambda_giou=config["lambda_giou"],
            lambda_angle=config["lambda_angle"],
        )

        # ----------------------------------------------------------------
        # Phase 1 — angle head only
        # ----------------------------------------------------------------
        print("\n========== PHASE 1: Angle-head warm-up ==========")
        det.set_phase(1)
        trainable = [p for p in det.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=config["phase1_lr"])
        global_step = 0

        for epoch in range(config["phase1_epochs"]):
            det.train()
            pbar = tqdm(train_loader, desc=f"P1 Epoch {epoch+1}")
            for batch in pbar:
                pixel_values = batch["images"].to(device)
                with torch.autocast(device):
                    outputs = det(pixel_values)
                losses = compute_losses(outputs, batch["targets"], bank, **loss_kwargs)

                opt.zero_grad()
                losses["loss"].backward()
                opt.step()

                global_step += 1
                if global_step % 2 == 0:
                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            mlflow.log_metric(f"p1_{k}", v.item(), step=global_step)
                pbar.set_description(
                    f"P1 E{epoch+1} loss={losses['loss'].item():.4f} "
                    f"ang={losses['L_angle'].item():.4f}"
                )

            metrics = obb_eval(det, val_loader, device)
            mlflow.log_metric("p1_mAP", metrics["map"].item(), step=epoch)

        # ----------------------------------------------------------------
        # Phase 2 — joint box + angle fine-tuning
        # ----------------------------------------------------------------
        print("\n========== PHASE 2: Joint box + angle fine-tuning ==========")
        det.set_phase(2)
        trainable = [p for p in det.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=config["phase2_lr"])

        best_map = 0.0
        for epoch in range(config["phase2_epochs"]):
            det.train()
            pbar = tqdm(train_loader, desc=f"P2 Epoch {epoch+1}")
            for batch in pbar:
                pixel_values = batch["images"].to(device)
                with torch.autocast(device):
                    outputs = det(pixel_values)
                losses = compute_losses(outputs, batch["targets"], bank, **loss_kwargs)

                opt.zero_grad()
                losses["loss"].backward()
                opt.step()

                global_step += 1
                if global_step % 2 == 0:
                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            mlflow.log_metric(f"p2_{k}", v.item(), step=global_step)
                pbar.set_description(
                    f"P2 E{epoch+1} loss={losses['loss'].item():.4f} "
                    f"ang={losses['L_angle'].item():.4f}"
                )

            metrics = obb_eval(det, val_loader, device)
            current_map = metrics["map"].item()
            mlflow.log_metric("p2_mAP", current_map, step=epoch)
            if current_map > best_map:
                best_map = current_map
                mlflow.log_metric("best_mAP", best_map)

        # Save final checkpoint
        torch.save(
            {
                "model_state_dict": det.state_dict(),
                "config": config,
                "best_map": best_map,
            },
            "obb_final.pth",
        )
        mlflow.log_artifact("obb_final.pth")
        print(f"\nDone. Best mAP = {best_map:.4f}")


if __name__ == "__main__":
    main()
