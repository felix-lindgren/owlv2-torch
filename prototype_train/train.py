from OWLv2torch.torch_version.prototypes import VisualPrototypeBank
from OWLv2torch.torch_version.owlv2 import OwlV2, PrototypeDetector
from OWLv2torch.torch_version.loss import compute_losses
from OWLv2torch.utils.tokenizer import tokenize
import torch
from torchvision.datasets import CocoDetection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from functools import partial
import mlflow
import mlflow.pytorch
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.ovd_eval import (
    DiorDetectionDataset,
    DotaDetectionDataset,
    coco_eval_with_custom_sizes,
    filter_coco_gt_to_class_names,
    run_prototype_inference,
)


def coco_collate_fn(batch, id2size=None, letterbox=False, square_pad=True, category_id_to_label=None):
    images, targets = [], []
    for image, anns in batch:
        images.append(image)
        H1, W1 = image.shape[-2], image.shape[-1]

        if category_id_to_label is not None:
            anns = [a for a in anns if a["category_id"] in category_id_to_label]

        if anns:
            b = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32, device=image.device)  # xywh (orig space)
            labels = torch.as_tensor(
                [category_id_to_label.get(a["category_id"], a["category_id"]) if category_id_to_label else a["category_id"] for a in anns],
                dtype=torch.int64,
                device=image.device,
            )

            if id2size is None:
                raise ValueError("id2size mapping required to scale COCO boxes to transformed image size")

            # original size via image_id
            img_id = anns[0]["image_id"]
            meta = id2size[img_id]          # {'width': W0, 'height': H0, ...}
            W0, H0 = float(meta["width"]), float(meta["height"])

            if square_pad:
                padded_size = max(W0, H0)
                sx, sy = W1 / padded_size, H1 / padded_size
                px, py = 0.0, 0.0
            elif letterbox:
                sx, sy = W1 / W0, H1 / H0
                s = min(sx, sy)
                px = (W1 - W0 * s) * 0.5
                py = (H1 - H0 * s) * 0.5
                sx = sy = s
            else:
                sx, sy = W1 / W0, H1 / H0
                px, py = 0.0, 0.0

            # scale + offset into transformed space
            b[:, 0] = b[:, 0] * sx + px
            b[:, 1] = b[:, 1] * sy + py
            b[:, 2] = b[:, 2] * sx
            b[:, 3] = b[:, 3] * sy

            # clamp in transformed space
            b[:, 0].clamp_(0, W1)
            b[:, 1].clamp_(0, H1)
            b[:, 2] = torch.minimum(b[:, 2], W1 - b[:, 0])
            b[:, 3] = torch.minimum(b[:, 3], H1 - b[:, 1])

            # xywh -> normalized cxcywh (transformed space)
            cx = b[:, 0] + 0.5 * b[:, 2]
            cy = b[:, 1] + 0.5 * b[:, 3]
            boxes = torch.stack((cx / W1, cy / H1, b[:, 2] / W1, b[:, 3] / H1), dim=1)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32, device=image.device)
            labels = torch.zeros((0,),   dtype=torch.int64,   device=image.device)

        targets.append({"boxes": boxes, "labels": labels})

    return {"images": torch.stack(images, 0), "targets": targets}


OPENAI_CLIP_MEAN: list[float] = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD:  list[float] = [0.26862954, 0.26130258, 0.27577711]


def class_name_to_query(name: str, template: str) -> str:
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    return template.format(name=cleaned)


def coco_class_names(coco, category_ids):
    return [coco.cats[category_id]["name"] for category_id in category_ids]


def select_shot_indices(dataset: CocoDetection, category_ids, shots_per_class, seed: int):
    if shots_per_class is None:
        return list(range(len(dataset)))
    if shots_per_class <= 0:
        raise ValueError(f"shots_per_class must be positive or None, got {shots_per_class}")

    generator = torch.Generator().manual_seed(seed)
    image_id_to_index = {image_id: idx for idx, image_id in enumerate(dataset.ids)}
    selected = set()
    for category_id in category_ids:
        image_ids = [
            image_id
            for image_id in dataset.ids
            if any(ann["category_id"] == category_id for ann in dataset.coco.imgToAnns.get(image_id, []))
        ]
        if not image_ids:
            raise ValueError(f"No training images contain category_id={category_id}")
        order = torch.randperm(len(image_ids), generator=generator).tolist()
        for idx in order[:min(shots_per_class, len(image_ids))]:
            selected.add(image_id_to_index[image_ids[idx]])

    return sorted(selected)


class ZeroShotDetector(torch.nn.Module):
    def __init__(self, owl: OwlV2, class_names, prompt_template: str, device):
        super().__init__()
        self.owl = owl
        queries = [class_name_to_query(name, prompt_template) for name in class_names]
        token_ids = tokenize(queries, context_length=16, truncate=True).to(device)
        attention_mask = token_ids == 0
        self.register_buffer("token_ids", token_ids, persistent=False)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        token_ids = self.token_ids.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = self.attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return self.owl.forward_object_detection(pixel_values, token_ids, attention_mask)


def save_eval_sample(
    image,
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    output_path,
    score_threshold=0.01,
    topk=25,
):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = image.detach().to("cpu", dtype=torch.float32)
    mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD, dtype=torch.float32).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    image_np = image.permute(1, 2, 0).numpy()

    pred_boxes = pred_boxes.detach().cpu()
    pred_scores = pred_scores.detach().cpu()
    pred_labels = pred_labels.detach().cpu()
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    if pred_scores.numel() > topk:
        order = torch.argsort(pred_scores, descending=True)[:topk]
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    ax.axis("off")

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=1.5, edgecolor="lime"))
        ax.text(
            x1,
            max(0, y1 - 3),
            f"p{int(label)}:{float(score):.2f}",
            fontsize=7,
            color="black",
            bbox=dict(facecolor="lime", alpha=0.75, edgecolor="none", pad=1),
        )

    for box, label in zip(gt_boxes.detach().cpu(), gt_labels.detach().cpu()):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=2.0, edgecolor="red"))
        ax.text(
            x1,
            max(0, y1 - 3),
            f"gt{int(label)}",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="red", alpha=0.75, edgecolor="none", pad=1),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def coco_eval(
    model,
    val_loader,
    device,
    num_classes=None,
    debug: bool = False,
    debug_max_images: int = 12,
    confidence_threshold: float = 0.05,
    score_with_objectness: bool = True,
    top_k: int | None = 300,
    sample_output_dir=None,
    sample_prefix="eval",
    sample_score_threshold: float = 0.01,
    sample_topk: int = 25,
    log_to_mlflow=True,
    output_format: str = "prototype",
    title: str = "COCO Evaluation Results",
):
    model.eval()
    
    # Initialize torchmetrics mAP
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=None,
        rec_thresholds=None,
        max_detection_thresholds=[1, 10, 100],
        class_metrics=False,
        backend='faster_coco_eval'
    )
    metric.warn_on_many_detections = False

    sample_output_dir = Path(sample_output_dir) if sample_output_dir is not None else None
    drawn = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch in pbar:
            pixel_values = batch["images"].to(device, non_blocking=True)  # [B, C, H, W], normalized with CLIP stats
            batch_targets = batch["targets"]
            with torch.autocast(device):
                outputs = model(pixel_values)
            if output_format == "prototype":
                logits, objectness_logits, pred_boxes = outputs[1], outputs[2], outputs[3]
            elif output_format == "standard":
                logits, objectness_logits, pred_boxes = outputs[0], outputs[1], outputs[2]
            else:
                raise ValueError(f"Unknown output_format: {output_format}")

            # Assuming square 1024x1024 after your preprocessing
            batch_size = pixel_values.shape[0]
            H, W = pixel_values.shape[-2], pixel_values.shape[-1]
            target_sizes = torch.tensor([(H, W)] * batch_size, device=device, dtype=torch.float32)

            # Postprocess model boxes -> xyxy absolute coords
            postprocess_model = model.owl if hasattr(model, "owl") else model
            boxes = postprocess_model.postprocess_boxes(pred_boxes, target_sizes)  # [B, N, 4] xyxy

            # Scores/labels
            class_probs = torch.sigmoid(logits.float())
            if score_with_objectness:
                objectness_probs = torch.sigmoid(objectness_logits.float())
                scores_per_class = class_probs * objectness_probs.unsqueeze(-1)
            else:
                scores_per_class = class_probs
            scores, labels = scores_per_class.max(dim=-1)

            batch_predictions = []
            batch_metric_targets = []
            for i in range(batch_size):
                img_boxes = boxes[i]
                img_scores = scores[i]
                img_labels = labels[i]

                # Filter low confidence
                valid_mask = img_scores > confidence_threshold
                img_boxes = img_boxes[valid_mask]
                img_scores = img_scores[valid_mask]
                img_labels = img_labels[valid_mask]
                if top_k is not None and img_scores.numel() > top_k:
                    top_idx = torch.topk(img_scores, top_k).indices
                    img_boxes = img_boxes[top_idx]
                    img_scores = img_scores[top_idx]
                    img_labels = img_labels[top_idx]

                pred_dict = {
                    'boxes': img_boxes.detach().cpu(),
                    'scores': img_scores.detach().cpu(),
                    'labels': img_labels.detach().cpu().to(torch.int64),
                }
                batch_predictions.append(pred_dict)

                # Ground truth
                gt_target = batch_targets[i]
                gt_boxes = gt_target["boxes"]  # normalized cxcywh
                gt_labels = gt_target["labels"].to(torch.int64)

                if len(gt_boxes) > 0:
                    img_h, img_w = target_sizes[i][0].item(), target_sizes[i][1].item()
                    cx = gt_boxes[:, 0] * img_w
                    cy = gt_boxes[:, 1] * img_h
                    w = gt_boxes[:, 2] * img_w
                    h = gt_boxes[:, 3] * img_h
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    gt_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                else:
                    gt_boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)

                target_dict = {
                    'boxes': gt_boxes_xyxy.detach().cpu(),
                    'labels': gt_labels.detach().cpu(),
                }
                batch_metric_targets.append(target_dict)

                if debug and sample_output_dir is not None and drawn < debug_max_images:
                    save_eval_sample(
                        pixel_values[i],
                        boxes[i],
                        scores[i],
                        labels[i],
                        gt_boxes_xyxy,
                        gt_labels,
                        sample_output_dir / f"{sample_prefix}_{drawn:03d}.jpg",
                        score_threshold=sample_score_threshold,
                        topk=sample_topk,
                    )
                    drawn += 1

            metric.update(batch_predictions, batch_metric_targets)

    metrics = metric.compute()

    # Print results
    print(f"\n=== {title} ===")
    print(f"mAP (IoU=0.50:0.95): {metrics['map']:.4f}")
    print(f"mAP (IoU=0.50): {metrics['map_50']:.4f}")
    print(f"mAP (IoU=0.75): {metrics['map_75']:.4f}")
    print(f"mAP (small): {metrics['map_small']:.4f}")
    print(f"mAP (medium): {metrics['map_medium']:.4f}")
    print(f"mAP (large): {metrics['map_large']:.4f}")
    if debug and sample_output_dir is not None:
        print(f"Saved {drawn} evaluation samples to {sample_output_dir}")
        if log_to_mlflow and drawn:
            mlflow.log_artifacts(str(sample_output_dir), artifact_path=str(sample_output_dir))

    return metrics

def main():
    # MLFlow setup
    mlflow.set_experiment("OwlV2-PrototypeDetector-Training")
    
    # Configuration
    config = {
        "device": "cuda",
        "C": 1,
        "K": 4,
        "model_type": "base",
        "prototype_learning_rate": 1e-4,
        "head_learning_rate": 3e-5,
        "weight_decay": 0.0,
        "box_tuning_mode": "last_layer",  # none, bias, last_layer, full_head
        "finetune_objectness_head": True,
        "batch_size": 16,
        "num_workers": 8,
        "num_epochs": 100,
        "eval_every_epochs": 5,
        "eval_final_epoch": True,
        "shots_per_class": 30,
        "shot_seed": 0,
        "include_geometry_loss": True,
        "include_objectness_loss": True,
        "negative_ratio": 5,
        "max_negatives_per_image": 512,
        "negative_loss_weight": 1.0,
        "run_zero_shot_baseline": False,
        "zero_shot_prompt_template": "{name}",
        "confidence_threshold": 0.001,
        "score_with_objectness": True,
        "eval_top_k": 300,
        "eval_sample_images": 0,
        "eval_sample_score_threshold": 0.01,
        "eval_sample_topk": 25,
        "eval_sample_dir": "eval_samples",
        "external_eval_enabled": True,
        "external_eval_datasets": ["dior", "dota"],
        "external_eval_limit": 100,
        "external_eval_batch_size": 16,
        "external_eval_num_workers": 4,
        "external_eval_fast_preprocess": True,
        "external_eval_autocast": True,
        "dior_split": "test",
        "dota_data_root": "data/dotav1_5",
        "dota_tile_size": 0,
        "dota_tile_overlap": 200,
        "dota_include_difficult": False,
        "category_ids": [0],
        "train_dataset_path": "/home/fellin/repos/rnd/dataset-utils/out_ships_1024_train_val/annotations.json",
        "train_images_path": "/home/fellin/repos/rnd/dataset-utils/out_ships_1024_train_val",
        "val_dataset_path": "/home/fellin/repos/rnd/dataset-utils/out_ships_1024_test/coco_merged_tiles_1024.json",
        "val_images_path": "/home/fellin/repos/rnd/dataset-utils/out_ships_1024_test/images"
    }
    
    with mlflow.start_run(run_name=f"owlv2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_params(config)
        
        # Initialize model
        device = config["device"]
        C, K = config["C"], config["K"]
        model = OwlV2(config["model_type"])
        
        bank = VisualPrototypeBank(C, K, dim=model.text_dim)
        det = PrototypeDetector(
            model,
            bank,
            box_tuning_mode=config["box_tuning_mode"],
            train_objectness_head=config["finetune_objectness_head"],
        )
        det = det.to(device)
        param_groups = [
            {
                "params": list(det.bank.parameters()),
                "lr": config["prototype_learning_rate"],
                "name": "prototypes",
            }
        ]
        if config["box_tuning_mode"] == "bias":
            param_groups.append({
                "params": [det.box_delta],
                "lr": config["head_learning_rate"],
                "name": "box_bias_delta",
            })
        elif config["box_tuning_mode"] == "last_layer":
            param_groups.append({
                "params": list(det.owl.box_head.dense2.parameters()),
                "lr": config["head_learning_rate"],
                "name": "box_head_last_layer",
            })
        elif config["box_tuning_mode"] == "full_head":
            param_groups.append({
                "params": list(det.owl.box_head.parameters()),
                "lr": config["head_learning_rate"],
                "name": "box_head",
            })
        if config["finetune_objectness_head"]:
            param_groups.append({
                "params": list(det.owl.objectness_head.parameters()),
                "lr": config["head_learning_rate"],
                "name": "objectness_head",
            })
        opt = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
        
        # Log model architecture info
        mlflow.log_param("text_dim", model.text_dim)
        mlflow.log_param("prototype_params", sum(p.numel() for p in bank.parameters()))
        mlflow.log_param("trainable_params", sum(p.numel() for p in det.parameters() if p.requires_grad))
        
        # Setup datasets
        train_dataset = CocoDetection(
            annFile=config["train_dataset_path"],
            root=config["train_images_path"],
            transform=model.image_transform_fast
        )
        val_dataset = CocoDetection(
            annFile=config["val_dataset_path"],
            root=config["val_images_path"],
            transform=model.image_transform
        )
        category_ids = config["category_ids"] or sorted(train_dataset.coco.getCatIds())[:C]
        if len(category_ids) != C:
            raise ValueError(f"Expected {C} training categories, found {len(category_ids)}")
        category_id_to_label = {
            category_id: label
            for label, category_id in enumerate(category_ids)
        }
        train_indices = select_shot_indices(
            train_dataset,
            category_ids,
            config["shots_per_class"],
            config["shot_seed"],
        )
        train_loader_dataset = Subset(train_dataset, train_indices)
        
        # Log dataset info
        mlflow.log_param("train_dataset_size", len(train_dataset))
        mlflow.log_param("train_support_images", len(train_loader_dataset))
        mlflow.log_param("val_dataset_size", len(val_dataset))
        mlflow.log_param("category_id_to_label", str(category_id_to_label))
        class_names = coco_class_names(val_dataset.coco, category_ids)
        mlflow.log_param("class_names", str(class_names))
        
        train_loader = DataLoader(
            train_loader_dataset,
            batch_size=config["batch_size"],
            collate_fn=partial(
                coco_collate_fn,
                id2size=train_dataset.coco.imgs,
                square_pad=True,
                category_id_to_label=category_id_to_label,
            ),
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
            prefetch_factor=4 if config["num_workers"] > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            collate_fn=partial(
                coco_collate_fn,
                id2size=val_dataset.coco.imgs,
                square_pad=True,
                category_id_to_label=category_id_to_label,
            ),
            shuffle=False,
            num_workers=config["num_workers"] // 2,
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
            prefetch_factor=4 if config["num_workers"] > 0 else None,
        )

        if config["run_zero_shot_baseline"]:
            print("\n=== Zero-shot baseline before prototype training ===")
            zero_shot = ZeroShotDetector(
                model,
                class_names,
                config["zero_shot_prompt_template"],
                device,
            ).to(device)
            zero_shot.eval()
            zero_shot_metrics = coco_eval(
                zero_shot,
                val_loader,
                device,
                debug=config["eval_sample_images"] > 0,
                debug_max_images=config["eval_sample_images"],
                confidence_threshold=config["confidence_threshold"],
                score_with_objectness=config["score_with_objectness"],
                top_k=config["eval_top_k"],
                sample_output_dir=Path(config["eval_sample_dir"]) / "zero_shot",
                sample_prefix="zero_shot",
                sample_score_threshold=config["eval_sample_score_threshold"],
                sample_topk=config["eval_sample_topk"],
                output_format="standard",
                title="Zero-shot COCO Evaluation Results",
            )
            for metric_key, metric_value in zero_shot_metrics.items():
                metric_value = metric_value.item() if hasattr(metric_value, "item") else metric_value
                mlflow.log_metric(f"zero_shot_{metric_key}", metric_value)

        external_eval_sets = {}
        if config["external_eval_enabled"]:
            external_batch_size = config["external_eval_batch_size"]
            if external_batch_size is None:
                external_batch_size = 32 if config["model_type"] == "base" else 8
            for dataset_name in config["external_eval_datasets"]:
                try:
                    if dataset_name == "dior":
                        dataset = DiorDetectionDataset(
                            split=config["dior_split"],
                            limit=config["external_eval_limit"],
                        )
                    elif dataset_name == "dota":
                        dataset = DotaDetectionDataset(
                            data_root=Path(config["dota_data_root"]),
                            limit=config["external_eval_limit"],
                            tile_size=config["dota_tile_size"],
                            tile_overlap=config["dota_tile_overlap"],
                            include_difficult=config["dota_include_difficult"],
                        )
                    else:
                        print(f"[WARN] Unknown external eval dataset '{dataset_name}', skipping.")
                        continue
                    gt = filter_coco_gt_to_class_names(dataset.build_coco_gt(), class_names)
                    external_eval_sets[dataset_name] = {
                        "dataset": dataset,
                        "gt": gt,
                        "batch_size": external_batch_size,
                    }
                    print(
                        f"Prepared external eval '{dataset_name}': "
                        f"{len(dataset)} samples/crops, {len(gt['annotations'])} matching GT boxes"
                    )
                except Exception as exc:
                    print(f"[WARN] Failed to prepare external eval '{dataset_name}': {exc}")
        
        # Training loop
        best_map = 0.0
        global_step = 0
        current_map = None
        avg_epoch_loss = None
        
        for epoch in range(config["num_epochs"]):
            print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")
            
            # Training phase
            det.train()
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch["images"].to(device, non_blocking=True)
                with torch.autocast(device):
                    outputs = det(pixel_values)
                losses = compute_losses(
                    outputs,
                    batch["targets"],
                    bank,
                    include_geometry_loss=config["include_geometry_loss"],
                    include_objectness_loss=config["include_objectness_loss"],
                    negative_ratio=config["negative_ratio"],
                    max_negatives_per_image=config["max_negatives_per_image"],
                    negative_loss_weight=config["negative_loss_weight"],
                )
                
                opt.zero_grad()
                losses["loss"].backward()
                opt.step()
                
                # Log training metrics
                current_loss = losses["loss"].item()
                epoch_loss += current_loss
                num_batches += 1
                global_step += 1
                
                if global_step % 2 == 0:
                    mlflow.log_metric("train_loss_step", current_loss, step=global_step)
                    # Log individual loss components if available
                    for loss_name, loss_value in losses.items():
                        if loss_name != "loss" and isinstance(loss_value, torch.Tensor):
                            mlflow.log_metric(f"train_{loss_name}_step", loss_value.item(), step=global_step)
                
                pbar.set_description(f"Epoch {epoch + 1} - Loss: {current_loss:.4f}")
            
            # Log epoch training metrics
            avg_epoch_loss = epoch_loss / num_batches
            mlflow.log_metric("train_loss_epoch", avg_epoch_loss, step=epoch)
            for group in opt.param_groups:
                mlflow.log_metric(f"learning_rate_{group['name']}", group["lr"], step=epoch)
            
            print(f"Average training loss: {avg_epoch_loss:.4f}")
            
            # Evaluation phase
            is_eval_epoch = (
                config["eval_every_epochs"] > 0
                and (epoch + 1) % config["eval_every_epochs"] == 0
            )
            is_final_epoch = epoch + 1 == config["num_epochs"]
            if is_eval_epoch or (config["eval_final_epoch"] and is_final_epoch):
                print("Running evaluation...")
                metrics = coco_eval(
                    det,
                    val_loader,
                    device,
                    debug=config["eval_sample_images"] > 0,
                    debug_max_images=config["eval_sample_images"],
                    confidence_threshold=config["confidence_threshold"],
                    score_with_objectness=config["score_with_objectness"],
                    top_k=config["eval_top_k"],
                    sample_output_dir=Path(config["eval_sample_dir"]) / f"epoch_{epoch + 1:03d}",
                    sample_prefix=f"epoch_{epoch + 1:03d}",
                    sample_score_threshold=config["eval_sample_score_threshold"],
                    sample_topk=config["eval_sample_topk"],
                    output_format="prototype",
                    title="Prototype COCO Evaluation Results",
                )
                
                # Log validation metrics
                metric_mapping = {
                    'map': 'val_mAP',
                    'map_50': 'val_mAP_50',
                    'map_75': 'val_mAP_75',
                    'map_small': 'val_mAP_small',
                    'map_medium': 'val_mAP_medium',
                    'map_large': 'val_mAP_large'
                }
                
                for metric_key, metric_name in metric_mapping.items():
                    if metric_key in metrics:
                        metric_value = metrics[metric_key].item() if hasattr(metrics[metric_key], 'item') else metrics[metric_key]
                        mlflow.log_metric(metric_name, metric_value, step=epoch)
                
                # Check if this is the best model
                current_map = metrics['map'].item() if hasattr(metrics['map'], 'item') else metrics['map']
                if current_map > best_map:
                    best_map = current_map
                    mlflow.log_metric("best_mAP", best_map)
                    print(f"New best mAP: {best_map:.4f}")

                for dataset_name, spec in external_eval_sets.items():
                    detections = run_prototype_inference(
                        det,
                        spec["dataset"],
                        device=device,
                        score_threshold=config["confidence_threshold"],
                        top_k=config["eval_top_k"],
                        batch_size=spec["batch_size"],
                        fast_preprocess=config["external_eval_fast_preprocess"],
                        use_autocast=config["external_eval_autocast"],
                        num_workers=config["external_eval_num_workers"],
                        desc=f"{dataset_name.upper()} prototype inference epoch {epoch + 1}",
                    )
                    ext_metrics = coco_eval_with_custom_sizes(
                        spec["gt"],
                        detections,
                        class_names,
                        title=f"{dataset_name.upper()} Prototype Evaluation epoch {epoch + 1}",
                    )
                    for metric_key, metric_value in ext_metrics.items():
                        mlflow.log_metric(f"{dataset_name}_{metric_key}", metric_value, step=epoch)
            else:
                print(
                    f"Skipping evaluation at epoch {epoch + 1}; "
                    f"eval_every_epochs={config['eval_every_epochs']}"
                )
        
        # Log final metrics
        if current_map is not None:
            mlflow.log_metric("final_mAP", current_map)
        mlflow.log_metric("final_train_loss", avg_epoch_loss)
        
        print(f"\nTraining completed!")
        print(f"Best mAP achieved: {best_map:.4f}")
        if current_map is not None:
            print(f"Final mAP: {current_map:.4f}")
        else:
            print("Final mAP: not evaluated")

if __name__ == "__main__":
    main()
