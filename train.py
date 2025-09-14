from OWLv2torch.torch_version.prototypes import VisualPrototypeBank
from OWLv2torch.torch_version.owlv2 import OwlV2, PrototypeDetector
from OWLv2torch.torch_version.loss import compute_losses
import torch
from torchvision.datasets import CocoDetection
from torchmetrics.detection import MeanAveragePrecision
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch.nn.functional as F

def coco_collate_fn(batch):
    images: list[Tensor] = []
    targets = []
    
    for image, coco_targets in batch:
        images.append(image)
        
        if len(coco_targets) > 0:
            # Extract bboxes in [x, y, w, h] format (COCO format)
            batch_bboxes = torch.tensor([target['bbox'] for target in coco_targets], dtype=torch.float32)
            batch_labels = torch.tensor([target['category_id'] for target in coco_targets], dtype=torch.long)
            
            # Convert from [x, y, w, h] to [cx, cy, w, h] (center format)
            # x, y are top-left corner in COCO, we need center coordinates
            cx = batch_bboxes[:, 0] + batch_bboxes[:, 2] / 2  # x + w/2
            cy = batch_bboxes[:, 1] + batch_bboxes[:, 3] / 2  # y + h/2
            w = batch_bboxes[:, 2]  # width
            h = batch_bboxes[:, 3]  # height
            
            # Stack into [cx, cy, w, h] format
            cxcywh_boxes = torch.stack([cx, cy, w, h], dim=1)
            
            # Normalize to [0, 1] - assuming image dimensions
            # Note: You might need to adjust these based on your actual image size
            # If your images are already transformed, you may need the original size
            img_height, img_width = image.shape[-2], image.shape[-1]  # Assuming CHW format
            
            cxcywh_boxes[:, 0] /= img_width   # normalize cx
            cxcywh_boxes[:, 1] /= img_height  # normalize cy  
            cxcywh_boxes[:, 2] /= img_width   # normalize w
            cxcywh_boxes[:, 3] /= img_height  # normalize h
            
        else:
            # Handle images with no annotations
            cxcywh_boxes = torch.zeros((0, 4), dtype=torch.float32)
            batch_labels = torch.zeros((0,), dtype=torch.long)
        
        # Create target dict for this image
        target_dict = {
            "boxes": cxcywh_boxes,
            "labels": batch_labels
        }
        targets.append(target_dict)
    
    # Stack images into batch tensor
    images = torch.stack(images, dim=0)
    
    return {
        'images': images,
        'targets': targets
    }

def coco_eval(model, val_loader, device, num_classes=None):
    """
    Evaluate the model on validation set using COCO metrics.
    
    Args:
        model: The detection model (should have .owl.postprocess_boxes method)
        val_loader: DataLoader for validation data
        device: torch device
        num_classes: Number of classes (optional, will be inferred if not provided)
    
    Returns:
        dict: Dictionary containing mAP metrics
    """
    model.eval()
    
    # Initialize torchmetrics mAP
    # Note: torchmetrics uses 0-indexing for classes, background is not included
    metric = MeanAveragePrecision(
        box_format='xyxy',  # We'll convert to xyxy format
        iou_type='bbox',
        iou_thresholds=None,  # Use default COCO thresholds (0.5:0.05:0.95)
        rec_thresholds=None,  # Use default recall thresholds
        max_detection_thresholds=[1, 10, 100],  # COCO standard
        class_metrics=True,  # Compute per-class metrics
    )
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch in pbar:
            pixel_values = batch["images"].to(device)
            batch_targets = batch["targets"]
            
            # Get model outputs
            outputs = model(pixel_values)
            
            # Extract logits and predicted boxes
            logits, pred_boxes = outputs[1], outputs[3]  # Based on your example
            
            # Get target sizes for postprocessing (original image dimensions)
            # Assuming your images are square and of size 1024 based on your path
            batch_size = pixel_values.shape[0]
            target_sizes = torch.tensor([[1024, 1024]] * batch_size).to(device)
            
            # Postprocess boxes using the model's method
            boxes = model.owl.postprocess_boxes(pred_boxes, target_sizes)
            
            # Get probabilities and convert to scores and labels
            probs = torch.max(logits, dim=-1)
            scores = torch.sigmoid(probs.values)
            labels = probs.indices
            
            # Process each image in the batch
            for i in range(batch_size):
                # Get predictions for this image
                img_boxes = boxes[i]  # Shape: [N, 4] in xyxy format (after postprocess)
                img_scores = scores[i]  # Shape: [N]
                img_labels = labels[i]  # Shape: [N]
                
                # Filter out low confidence predictions (optional threshold)
                confidence_threshold = 0.05  # Adjust as needed
                valid_mask = img_scores > confidence_threshold
                
                img_boxes = img_boxes[valid_mask]
                img_scores = img_scores[valid_mask]
                img_labels = img_labels[valid_mask]
                
                # Prepare prediction dict for torchmetrics
                pred_dict = {
                    'boxes': img_boxes.cpu(),
                    'scores': img_scores.cpu(),
                    'labels': img_labels.cpu()
                }
                predictions.append(pred_dict)
                
                # Get ground truth for this image
                gt_target = batch_targets[i]
                gt_boxes = gt_target["boxes"]  # [M, 4] in cxcywh format, normalized
                gt_labels = gt_target["labels"]  # [M]
                
                # Convert ground truth boxes from cxcywh normalized to xyxy absolute
                if len(gt_boxes) > 0:
                    # Convert from normalized cxcywh to absolute xyxy
                    img_h, img_w = target_sizes[i][0].item(), target_sizes[i][1].item()
                    
                    # Denormalize
                    cx = gt_boxes[:, 0] * img_w
                    cy = gt_boxes[:, 1] * img_h
                    w = gt_boxes[:, 2] * img_w
                    h = gt_boxes[:, 3] * img_h
                    
                    # Convert to xyxy format
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    gt_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                else:
                    gt_boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
                
                # Prepare target dict for torchmetrics
                target_dict = {
                    'boxes': gt_boxes_xyxy,
                    'labels': gt_labels
                }
                targets.append(target_dict)
    
    # Compute metrics
    metric.update(predictions, targets)
    metrics = metric.compute()
    
    # Print results
    print(f"\n=== COCO Evaluation Results ===")
    print(f"mAP (IoU=0.50:0.95): {metrics['map']:.4f}")
    print(f"mAP (IoU=0.50): {metrics['map_50']:.4f}")
    print(f"mAP (IoU=0.75): {metrics['map_75']:.4f}")
    print(f"mAP (small): {metrics['map_small']:.4f}")
    print(f"mAP (medium): {metrics['map_medium']:.4f}")
    print(f"mAP (large): {metrics['map_large']:.4f}")
    
    if 'map_per_class' in metrics:
        print(f"\nPer-class mAP:")
        for i, class_map in enumerate(metrics['map_per_class']):
            if not torch.isnan(class_map):
                print(f"  Class {i}: {class_map:.4f}")
    
    return metrics

device = "cuda"
C, K = 1, 4      # K≈4 is a strong default
model = OwlV2("base")

bank = VisualPrototypeBank(C, K, dim=model.text_dim)
det = PrototypeDetector(model, bank)  # model is your OwlV2
det = det.to(device)
opt = torch.optim.AdamW(bank.parameters(), lr=1e-3, weight_decay=0.0)




train_dataset = CocoDetection(annFile="/home/fellin/repos/rnd/dataset-utils/out_ships_1024_val/coco_merged_tiles_1024.json", 
                root="/home/fellin/repos/rnd/dataset-utils/out_ships_1024_val/images",
                transform=model.image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=coco_collate_fn)

val_dataset = CocoDetection(annFile="/home/fellin/repos/rnd/dataset-utils/out_ships_1024/coco_merged_tiles_1024.json", 
                root="/home/fellin/repos/rnd/dataset-utils/out_ships_1024/images",
                transform=model.image_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, collate_fn=coco_collate_fn)



for epoch in range(1):
    metrics = coco_eval(det, val_loader, device, 1)
    break
    pbar = tqdm(train_loader)
    det.train()
    for batch in pbar:
        pixel_values = batch["images"].to(device)
        outputs = det(pixel_values)

        det.owl.postprocess_boxes()
        breakpoint()
        losses = compute_losses(outputs, batch["targets"], bank)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        pbar.set_description(f"{losses['loss'].item()}")


