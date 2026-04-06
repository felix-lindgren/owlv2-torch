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
from torch import Tensor
from functools import partial
import mlflow
import mlflow.pytorch
import os
from datetime import datetime


def coco_collate_fn(batch, id2size=None, letterbox=False):
    images, targets = [], []
    for image, anns in batch:
        images.append(image)
        H1, W1 = image.shape[-2], image.shape[-1]

        if len(anns) > 0:
            b = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32, device=image.device)  # xywh (orig space)
            labels = torch.as_tensor([a["category_id"] for a in anns], dtype=torch.int64, device=image.device)

            if id2size is None:
                raise ValueError("id2size mapping required to scale COCO boxes to transformed image size")

            # original size via image_id
            img_id = anns[0]["image_id"]
            meta = id2size[img_id]          # {'width': W0, 'height': H0, ...}
            W0, H0 = float(meta["width"]), float(meta["height"])

            sx, sy = W1 / W0, H1 / H0
            if letterbox and abs(sx - sy) > 1e-6:
                s = min(sx, sy)
                px = (W1 - W0 * s) * 0.5
                py = (H1 - H0 * s) * 0.5
            else:
                s, px, py = sx, 0.0, 0.0     # pure warp/resize (your 1024→960 case)

            # scale + offset into transformed space
            b[:, 0] = b[:, 0] * s + px
            b[:, 1] = b[:, 1] * s + py
            b[:, 2] = b[:, 2] * s
            b[:, 3] = b[:, 3] * s

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

def coco_eval(model, val_loader, device, num_classes=None, debug: bool = False, debug_max_images: int = 12, log_to_mlflow=True):
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

    # If debugging, set Qt5Agg before importing pyplot
    if debug:
        import matplotlib
        try:
            matplotlib.use("Qt5Agg", force=True)
        except Exception as e:
            print(f"[debug] Failed to set Qt5Agg backend ({e}); falling back to Agg.")
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.ion()

    predictions = []
    targets = []

    drawn = 0
    confidence_threshold = 0.05  # used both for metrics filtering and debug display

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch in pbar:
            pixel_values = batch["images"].to(device)  # [B, C, H, W], normalized with CLIP stats
            batch_targets = batch["targets"]
            with torch.autocast(device):
                outputs = model(pixel_values)
            logits, pred_boxes = outputs[1], outputs[3]  # class_logits, pred_boxes

            # Assuming square 1024x1024 after your preprocessing
            batch_size = pixel_values.shape[0]
            H, W = pixel_values.shape[-2], pixel_values.shape[-1]
            target_sizes = torch.tensor([(H, W)] * batch_size, device=device, dtype=torch.float32)

            # Postprocess model boxes -> xyxy absolute coords
            boxes = model.owl.postprocess_boxes(pred_boxes, target_sizes)  # [B, N, 4] xyxy

            # Scores/labels
            probs = torch.max(logits, dim=-1)   # values: logits, indices: labels
            scores = torch.sigmoid(probs.values)
            labels = probs.indices

            for i in range(batch_size):
                img_boxes = boxes[i]
                img_scores = scores[i]
                img_labels = labels[i]

                # Filter low confidence
                valid_mask = img_scores > confidence_threshold
                img_boxes = img_boxes[valid_mask]
                img_scores = img_scores[valid_mask]
                img_labels = img_labels[valid_mask]

                pred_dict = {
                    'boxes': img_boxes.detach().cpu(),
                    'scores': img_scores.detach().cpu(),
                    'labels': img_labels.detach().cpu().to(torch.int64),
                }
                predictions.append(pred_dict)

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
                targets.append(target_dict)

                # --- Debug drawing ---
                if debug and drawn < debug_max_images:
                    # Denormalize image from CLIP stats
                    img = pixel_values[i].detach().to('cpu', dtype=torch.float32)  # [C,H,W]
                    mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
                    std = torch.tensor(OPENAI_CLIP_STD, dtype=torch.float32).view(3, 1, 1)
                    img = (img * std + mean).clamp(0.0, 1.0)
                    img_np = img.permute(1, 2, 0).numpy()  # [H,W,3], float32

                    # Create figure
                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.imshow(img_np)
                    ax.set_title(f"Debug view — preds (lime) vs GT (red)")
                    ax.axis('off')

                    # Draw predictions
                    if img_boxes.numel() > 0:
                        for b, s, lab in zip(img_boxes.detach().cpu(), img_scores.detach().cpu(), img_labels.detach().cpu()):
                            x1, y1, x2, y2 = [float(v) for v in b]
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            #ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=2, edgecolor='lime'))
                            #ax.text(x1, max(0, y1 - 3), f"{int(lab)}:{s:.2f}", fontsize=8,
                            #        color='white', bbox=dict(facecolor='lime', alpha=0.6, edgecolor='none', pad=1))

                    # Draw GT
                    if gt_boxes_xyxy.numel() > 0:
                        for b, lab in zip(gt_boxes_xyxy.detach().cpu(), gt_labels.detach().cpu()):
                            x1, y1, x2, y2 = [float(v) for v in b]
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=2, edgecolor='red'))
                            ax.text(x1, max(0, y1 - 3), f"{int(lab)}", fontsize=8,
                                    color='white', bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1))

                    plt.tight_layout()
                    try:
                        plt.show(block=True)
                        plt.pause(0.001)
                    except Exception as e:
                        print(f"[debug] plt.show failed: {e}")
                    drawn += 1
                    # Close figures to avoid memory bloat if many batches
                    plt.close(fig)

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

    return metrics

def main():
    # MLFlow setup
    mlflow.set_experiment("OwlV2-PrototypeDetector-Training")
    
    # Configuration
    config = {
        "device": "cuda",
        "C": 1,
        "K": 4,
        "model_type": "large",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "batch_size": 16,
        "num_epochs": 5,
        "confidence_threshold": 0.05,
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
        det = PrototypeDetector(model, bank)
        det = det.to(device)
        opt = torch.optim.AdamW(bank.parameters(), 
                               lr=config["learning_rate"], 
                               weight_decay=config["weight_decay"])
        
        # Log model architecture info
        mlflow.log_param("text_dim", model.text_dim)
        mlflow.log_param("total_params", sum(p.numel() for p in bank.parameters()))
        mlflow.log_param("trainable_params", sum(p.numel() for p in bank.parameters() if p.requires_grad))
        
        # Setup datasets
        train_dataset = CocoDetection(
            annFile=config["train_dataset_path"], 
            root=config["train_images_path"],
            transform=model.image_transform
        )
        val_dataset = CocoDetection(
            annFile=config["val_dataset_path"], 
            root=config["val_images_path"],
            transform=model.image_transform
        )
        
        # Log dataset info
        mlflow.log_param("train_dataset_size", len(train_dataset))
        mlflow.log_param("val_dataset_size", len(val_dataset))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=partial(coco_collate_fn, id2size=train_dataset.coco.imgs, letterbox=False),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            collate_fn=partial(coco_collate_fn, id2size=val_dataset.coco.imgs, letterbox=False),
            shuffle=False
        )
        
        # Training loop
        best_map = 0.0
        global_step = 0
        
        for epoch in range(config["num_epochs"]):
            print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")
            
            # Training phase
            det.train()
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch["images"].to(device)
                with torch.autocast(device):
                    outputs = det(pixel_values)
                losses = compute_losses(outputs, batch["targets"], bank)
                
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
            mlflow.log_metric("learning_rate", opt.param_groups[0]['lr'], step=epoch)
            
            print(f"Average training loss: {avg_epoch_loss:.4f}")
            
            # Evaluation phase
            print("Running evaluation...")
            metrics = coco_eval(det, val_loader, device, debug=False)
            
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
                
                # Save best model
                model_path = f"best_model_epoch_{epoch}"
                """ torch.save({
                    'epoch': epoch,
                    'model_state_dict': det.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_map': best_map,
                    'config': config
                }, f"{model_path}.pth") """
                
                # Log model to MLFlow
                #mlflow.pytorch.log_model(det, "best_model")
                #mlflow.log_artifact(f"{model_path}.pth")
                
                print(f"New best model saved with mAP: {best_map:.4f}")
        
        # Log final metrics
        mlflow.log_metric("final_mAP", current_map)
        mlflow.log_metric("final_train_loss", avg_epoch_loss)
        
        # Save final model
        final_model_path = "final_model.pth"
        torch.save({
            'epoch': config["num_epochs"] - 1,
            'model_state_dict': det.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'final_map': current_map,
            'config': config
        }, final_model_path)
        mlflow.log_artifact(final_model_path)
        
        print(f"\nTraining completed!")
        print(f"Best mAP achieved: {best_map:.4f}")
        print(f"Final mAP: {current_map:.4f}")

if __name__ == "__main__":
    main()