from OWLv2torch.torch_version.prototypes import VisualPrototypeBank
from OWLv2torch.torch_version.owlv2 import OwlV2, PrototypeDetector
from OWLv2torch.torch_version.loss import compute_losses
from OWLv2torch.utils.tokenizer import tokenize
import torch
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from torchvision.transforms import v2 as Tv2
from torchvision.transforms.v2 import functional as TvF
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from functools import partial
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from datetime import datetime
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
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


class AugmentedDetectionDataset(Dataset):
    """Wraps a CocoDetection (PIL + raw anns) and applies bbox-aware augmentations,
    then the model's image transform. Emits (image_tensor, {"boxes": cxcywh_norm, "labels"}).

    Augmentations: discrete 0/90/180/270 rotation, horizontal/vertical flip,
    RandomZoomOut (smaller objects), RandomIoUCrop (larger / cropped objects),
    SanitizeBoundingBoxes, then ColorJitter on the image only.
    """

    def __init__(
        self,
        base,
        image_transform,
        category_id_to_label,
        augment=True,
        color_jitter_brightness=0.3,
        color_jitter_contrast=0.3,
        color_jitter_saturation=0.2,
        color_jitter_hue=0.02,
        zoom_out_prob=0.5,
        zoom_out_max=1.5,
        iou_crop_min_scale=0.5,
        iou_crop_max_scale=1.0,
        sanitize_min_size=2.0,
    ):
        self.base = base
        self.image_transform = image_transform
        self.category_id_to_label = category_id_to_label
        self.augment = augment
        if augment:
            self.geom = Tv2.Compose([
                Tv2.RandomZoomOut(fill=0, side_range=(1.0, zoom_out_max), p=zoom_out_prob),
                Tv2.RandomIoUCrop(
                    min_scale=iou_crop_min_scale,
                    max_scale=iou_crop_max_scale,
                    min_aspect_ratio=0.7,
                    max_aspect_ratio=1.3,
                    sampler_options=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
                    trials=20,
                ),
                Tv2.SanitizeBoundingBoxes(min_size=sanitize_min_size),
            ])
            self.photo = Tv2.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            )
        else:
            self.geom = None
            self.photo = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pil, anns = self.base[idx]
        anns = [a for a in anns if a["category_id"] in self.category_id_to_label]
        W, H = pil.size

        if anns:
            xywh = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32)
            xyxy = torch.stack(
                [
                    xywh[:, 0],
                    xywh[:, 1],
                    xywh[:, 0] + xywh[:, 2],
                    xywh[:, 1] + xywh[:, 3],
                ],
                dim=1,
            )
            labels = torch.as_tensor(
                [self.category_id_to_label[a["category_id"]] for a in anns],
                dtype=torch.int64,
            )
        else:
            xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        image = tv_tensors.Image(TvF.pil_to_tensor(pil))
        boxes = tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(H, W))

        if self.augment:
            # Discrete 0/90/180/270 rotation
            k = int(torch.randint(0, 4, ()).item())
            if k > 0:
                angle = float(k * 90)
                image = TvF.rotate(image, angle=angle, expand=True)
                boxes = TvF.rotate(boxes, angle=angle, expand=True)
            # H/V flips (aerial imagery is symmetric)
            if torch.rand(()).item() < 0.5:
                image = TvF.horizontal_flip(image)
                boxes = TvF.horizontal_flip(boxes)
            if torch.rand(()).item() < 0.5:
                image = TvF.vertical_flip(image)
                boxes = TvF.vertical_flip(boxes)
            # Joint scale aug (zoom out + iou crop) + sanitize
            sample = {"image": image, "boxes": boxes, "labels": labels}
            sample = self.geom(sample)
            image, boxes, labels = sample["image"], sample["boxes"], sample["labels"]
            # Photometric (image only)
            image = self.photo(image)

        # Apply OWL's image transform: SquarePad -> Resize -> Normalize
        image_tensor = self.image_transform(image)

        # Map current canvas-space boxes to model output space (square pad, then resize)
        canvas_h, canvas_w = (
            boxes.canvas_size if hasattr(boxes, "canvas_size") else image.shape[-2:]
        )
        H_in, W_in = float(canvas_h), float(canvas_w)
        H_out, W_out = float(image_tensor.shape[-2]), float(image_tensor.shape[-1])
        side = max(H_in, W_in)
        s = H_out / side  # H_out == W_out for OWL's square inputs

        if boxes.numel() > 0:
            b = boxes.as_subclass(torch.Tensor).clone() * s
            b[:, 0::2].clamp_(0, W_out)
            b[:, 1::2].clamp_(0, H_out)
            wh = b[:, 2:4] - b[:, 0:2]
            cxcy = b[:, 0:2] + wh * 0.5
            boxes_norm = torch.stack(
                [cxcy[:, 0] / W_out, cxcy[:, 1] / H_out, wh[:, 0] / W_out, wh[:, 1] / H_out],
                dim=1,
            )
            keep = (boxes_norm[:, 2] > 0) & (boxes_norm[:, 3] > 0)
            boxes_norm = boxes_norm[keep]
            labels = labels[keep]
        else:
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)

        return image_tensor, {"boxes": boxes_norm, "labels": labels.to(torch.int64)}


def aug_collate_fn(batch):
    images, targets = zip(*batch)
    return {"images": torch.stack(images, 0), "targets": list(targets)}


OPENAI_CLIP_MEAN: list[float] = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD:  list[float] = [0.26862954, 0.26130258, 0.27577711]


def class_name_to_query(name: str, template: str) -> str:
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    return template.format(name=cleaned)


def coco_class_names(coco, category_ids):
    return [coco.cats[category_id]["name"] for category_id in category_ids]


def mlflow_metric_value(value):
    return value.item() if hasattr(value, "item") else value


def configure_mlflow_tracking():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def validate_prototype_init_texts(texts_config, class_names, K):
    if texts_config is None:
        raise ValueError("prototype_init_mode='text' requires prototype_init_texts")
    if isinstance(texts_config, str):
        raise ValueError("prototype_init_texts must be a list of K prompt strings, not one string")

    if isinstance(texts_config, dict):
        out = []
        for class_idx, class_name in enumerate(class_names):
            if class_name in texts_config:
                texts = texts_config[class_name]
            elif str(class_idx) in texts_config:
                texts = texts_config[str(class_idx)]
            else:
                raise ValueError(
                    f"prototype_init_texts is missing prompts for class '{class_name}' "
                    f"(or index '{class_idx}')"
                )
            if isinstance(texts, str):
                raise ValueError(f"prototype_init_texts for '{class_name}' must be a list, not one string")
            out.append(list(texts))
    elif class_names and all(isinstance(item, str) for item in texts_config):
        if len(class_names) != 1:
            raise ValueError(
                "A flat prototype_init_texts list is only valid for single-class training; "
                "use a dict or list-of-lists for multiple classes."
            )
        out = [list(texts_config)]
    else:
        texts_config = list(texts_config)
        if len(texts_config) != len(class_names):
            raise ValueError(
                f"prototype_init_texts must contain one prompt list per class; "
                f"got {len(texts_config)} for {len(class_names)} classes"
            )
        out = []
        for class_name, texts in zip(class_names, texts_config):
            if isinstance(texts, str):
                raise ValueError(f"prototype_init_texts for '{class_name}' must be a list, not one string")
            out.append(list(texts))

    for class_name, texts in zip(class_names, out):
        if len(texts) != K:
            raise ValueError(
                f"prototype_init_texts for '{class_name}' must contain exactly K={K} prompts; "
                f"got {len(texts)}"
            )
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError(f"prototype_init_texts for '{class_name}' must be non-empty strings")
    return out


@torch.no_grad()
def init_prototypes_from_texts(bank, model, class_names, texts_config, device):
    texts_by_class = validate_prototype_init_texts(texts_config, class_names, bank.K)
    flat_texts = [text for texts in texts_by_class for text in texts]
    token_ids = tokenize(flat_texts, context_length=16, truncate=True).to(device)
    attention_mask = token_ids == 0
    text_features = model.get_text_features(token_ids, attention_mask)
    text_features = F.normalize(text_features.float(), dim=-1)
    bank.prototypes.copy_(text_features.view(bank.num_classes, bank.K, bank.dim))
    return texts_by_class


def _box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def _spherical_kmeans(features, K, steps=12):
    features = F.normalize(features.float(), dim=-1)
    n = features.shape[0]
    if n == 0:
        raise ValueError("Cannot initialize prototypes from an empty feature set")

    mean = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
    first = torch.argmax(features @ mean.squeeze(0))
    centers = [features[first]]
    min_dist = 1.0 - (features @ centers[0]).clamp(-1, 1)
    while len(centers) < K:
        if len(centers) >= n:
            centers.append(centers[len(centers) % n])
            continue
        idx = torch.argmax(min_dist)
        new_center = features[idx]
        centers.append(new_center)
        dist = 1.0 - (features @ new_center).clamp(-1, 1)
        min_dist = torch.minimum(min_dist, dist)

    centers = torch.stack(centers, dim=0)
    for _ in range(steps):
        assignments = torch.argmax(features @ centers.t(), dim=1)
        next_centers = []
        for k in range(K):
            mask = assignments == k
            if mask.any():
                next_centers.append(F.normalize(features[mask].mean(dim=0), dim=0))
            else:
                nearest = (features @ centers.t()).max(dim=1).values
                next_centers.append(features[torch.argmin(nearest)])
        centers = torch.stack(next_centers, dim=0)
    return F.normalize(centers, dim=-1)


@torch.no_grad()
def init_prototypes_from_visual_support(
    bank,
    det,
    train_loader,
    device,
    max_embeddings_per_class=None,
):
    det.eval()
    model = det.owl
    P = model.sqrt_num_patches
    grid_xy = model.normalize_grid_corner_coordinates(P).to(device)
    per_class_features = [[] for _ in range(bank.num_classes)]

    for batch in tqdm(train_loader, desc="Collecting visual prototype init features"):
        pixel_values = batch["images"].to(device, non_blocking=True)
        image_feats, _ = model._image_grid_features(pixel_values)
        class_embeds = model.class_head.dense0(image_feats)
        class_embeds = F.normalize(class_embeds.float(), dim=-1)

        for b, target in enumerate(batch["targets"]):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue

            boxes_xyxy = _box_cxcywh_to_xyxy(boxes)
            for box, label in zip(boxes_xyxy, labels):
                label_idx = int(label.item())
                if label_idx < 0 or label_idx >= bank.num_classes:
                    continue
                inside = (
                    (grid_xy[:, 0] >= box[0])
                    & (grid_xy[:, 0] <= box[2])
                    & (grid_xy[:, 1] >= box[1])
                    & (grid_xy[:, 1] <= box[3])
                )
                if inside.any():
                    patch_indices = torch.nonzero(inside, as_tuple=False).squeeze(1)
                else:
                    center = torch.stack(((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5))
                    patch_indices = torch.argmin(((grid_xy - center) ** 2).sum(dim=1)).view(1)
                per_class_features[label_idx].append(class_embeds[b, patch_indices].detach().cpu())

    centers_by_class = []
    feature_counts = []
    for class_idx, chunks in enumerate(per_class_features):
        if not chunks:
            raise ValueError(f"No support embeddings found for prototype class index {class_idx}")
        features = torch.cat(chunks, dim=0).to(device)
        if max_embeddings_per_class is not None and features.shape[0] > max_embeddings_per_class:
            generator = torch.Generator(device=device).manual_seed(0)
            keep = torch.randperm(features.shape[0], generator=generator, device=device)[:max_embeddings_per_class]
            features = features[keep]
        feature_counts.append(features.shape[0])
        centers_by_class.append(_spherical_kmeans(features, bank.K))

    bank.prototypes.copy_(torch.stack(centers_by_class, dim=0).to(bank.prototypes.dtype))
    return feature_counts


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
    configure_mlflow_tracking()
    mlflow.set_experiment("OwlV2-PrototypeDetector-Training")
    
    # Configuration
    config = {
        "device": "cuda",
        "C": 1,
        "K": 8,
        "model_type": "base",
        "prototype_init_mode": "visual",  # random, text, visual
        "prototype_init_texts": None,
        "prototype_visual_init_max_embeddings_per_class": 4096,
        "prototype_learning_rate": 1e-4,
        "head_learning_rate": 3e-5,
        "weight_decay": 0.0,
        "box_tuning_mode": "full_head",  # none, bias, last_layer, full_head
        "finetune_objectness_head": True,
        "batch_size": 16,
        "num_workers": 8,
        "num_epochs": 100,
        "eval_every_epochs": 5,
        "eval_final_epoch": True,
        "shots_per_class": 30,
        "shot_seed": 0,
        "augment": True,
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
        "external_eval_limit": 250,
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
        
        # Setup datasets. The training side returns PIL+raw anns and runs through the
        # augmentation wrapper, which also handles the model's image transform.
        train_dataset = CocoDetection(
            annFile=config["train_dataset_path"],
            root=config["train_images_path"],
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
        train_aug_dataset = AugmentedDetectionDataset(
            train_dataset,
            model.image_transform_fast,
            category_id_to_label,
            augment=config["augment"],
        )
        train_clean_dataset = AugmentedDetectionDataset(
            train_dataset,
            model.image_transform_fast,
            category_id_to_label,
            augment=False,
        )
        train_loader_dataset = Subset(train_aug_dataset, train_indices)
        train_clean_loader_dataset = Subset(train_clean_dataset, train_indices)

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
            collate_fn=aug_collate_fn,
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
            prefetch_factor=4 if config["num_workers"] > 0 else None,
        )
        train_init_loader = DataLoader(
            train_clean_loader_dataset,
            batch_size=config["batch_size"],
            collate_fn=aug_collate_fn,
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
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

        prototype_init_mode = config["prototype_init_mode"]
        if prototype_init_mode == "text":
            texts_by_class = init_prototypes_from_texts(
                bank,
                model,
                class_names,
                config["prototype_init_texts"],
                device,
            )
            mlflow.log_param("prototype_init_texts_resolved", str(texts_by_class))
            print(f"Initialized prototypes from text prompts for {len(class_names)} class(es).")
        elif prototype_init_mode == "visual":
            visual_feature_counts = init_prototypes_from_visual_support(
                bank,
                det,
                train_init_loader,
                device,
                max_embeddings_per_class=config["prototype_visual_init_max_embeddings_per_class"],
            )
            mlflow.log_param("prototype_visual_init_feature_counts", str(visual_feature_counts))
            print(
                "Initialized prototypes from visual support embeddings: "
                f"{visual_feature_counts}"
            )
        elif prototype_init_mode == "random":
            print("Using random prototype initialization.")
        else:
            raise ValueError(
                "prototype_init_mode must be one of: random, text, visual; "
                f"got {prototype_init_mode!r}"
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
                mlflow.log_metric(f"zero_shot/{metric_key}", mlflow_metric_value(metric_value))

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
                    mlflow.log_metric("train/loss_step", current_loss, step=global_step)
                    # Log individual loss components if available
                    for loss_name, loss_value in losses.items():
                        if loss_name != "loss" and isinstance(loss_value, torch.Tensor):
                            mlflow.log_metric(f"train/{loss_name}_step", loss_value.item(), step=global_step)
                
                pbar.set_description(f"Epoch {epoch + 1} - Loss: {current_loss:.4f}")
            
            # Log epoch training metrics
            avg_epoch_loss = epoch_loss / num_batches
            mlflow.log_metric("train/loss_epoch", avg_epoch_loss, step=epoch)
            for group in opt.param_groups:
                mlflow.log_metric(f"train/learning_rate/{group['name']}", group["lr"], step=epoch)
            
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
                    'map': 'val/mAP',
                    'map_50': 'val/mAP_50',
                    'map_75': 'val/mAP_75',
                    'map_small': 'val/mAP_small',
                    'map_medium': 'val/mAP_medium',
                    'map_large': 'val/mAP_large'
                }
                
                for metric_key, metric_name in metric_mapping.items():
                    if metric_key in metrics:
                        mlflow.log_metric(metric_name, mlflow_metric_value(metrics[metric_key]), step=epoch)
                
                # Check if this is the best model
                current_map = metrics['map'].item() if hasattr(metrics['map'], 'item') else metrics['map']
                if current_map > best_map:
                    best_map = current_map
                    mlflow.log_metric("best/mAP", best_map)
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
                        mlflow.log_metric(f"external/{dataset_name}/{metric_key}", metric_value, step=epoch)
            else:
                print(
                    f"Skipping evaluation at epoch {epoch + 1}; "
                    f"eval_every_epochs={config['eval_every_epochs']}"
                )
        
        # Log final metrics
        if current_map is not None:
            mlflow.log_metric("final/mAP", current_map)
        mlflow.log_metric("final/train_loss", avg_epoch_loss)
        
        print(f"\nTraining completed!")
        print(f"Best mAP achieved: {best_map:.4f}")
        if current_map is not None:
            print(f"Final mAP: {current_map:.4f}")
        else:
            print("Final mAP: not evaluated")

if __name__ == "__main__":
    main()
