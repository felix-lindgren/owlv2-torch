"""Distil a class-agnostic token scorer for zero-shot pruning from OWLv2's own objectness.

The tower stays frozen with its original weights. On unlabelled images, a scorer
on the input of block ``--after`` (what ``Encoder._prune`` sees) learns BCE
against sigmoid(objectness) of the unpruned model's final tokens. No labels and
no class names are involved, so the pruned model stays open-vocabulary; mixing
domains keeps the scorer general. Half the images are fed with their padding
cropped off (``OwlV2.padding_free_grid``), as padding-free inference sees them.

The scorer has ``OwlV2.enable_token_pruning``'s architecture; load the output
with ``OwlV2.load_token_scorer(path)``. One run trains a scorer per ``--after``.

python tools/arch_speedup/distill_scorer.py --after 4 6 --out-dir text_checkpoints/zeroshot_scorers
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from OWLv2torch.torch_version.owlv2 import OwlV2

DEFAULT_SOURCES = [
    "/mnt/datasets/coco/train2017:2000",
    "/mnt/datasets/fashion/fashionpedia_coco/train/images:600",
    "/mnt/datasets/fashion/lv_mhp_coco/train/images:400",
    "/mnt/datasets/sat/dior_coco/train/images:600",
    "/mnt/datasets/sat/xview_aitod_v2/train/images:400",
    "/mnt/datasets/uav/auair/images:400",
]


class ImageFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)


def sample_images(sources, seed):
    rng = random.Random(seed)
    paths = []
    for source in sources:
        folder, count = source.rsplit(":", 1)
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        paths += rng.sample(files, min(int(count), len(files)))
    rng.shuffle(paths)
    return paths


def make_scorer(model):
    model.enable_token_pruning(0, 1.0)  # builds the scorer architecture; pruning is switched off below
    scorer = model.vision_model.encoder.token_scorer
    del model.vision_model.encoder.token_scorer
    model.vision_model.encoder.prune = None
    return scorer.float()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--after", type=int, nargs="+", default=[4, 6])
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="folder:count")
    p.add_argument("--val-images", type=int, default=200, help="held out from the sampled images")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--keep", type=float, default=0.5, help="default keep stored with the scorer")
    p.add_argument("--registers", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=Path("text_checkpoints/zeroshot_scorers"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = OwlV2("base").eval().cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    scorers = {a: make_scorer(model).cuda() for a in args.after}
    captured = {}
    for a in args.after:
        model.vision_model.encoder.layers[a].register_forward_pre_hook(
            lambda module, inputs, a=a: captured.__setitem__(a, inputs[0])
        )

    paths = sample_images(args.sources, args.seed)
    val_paths, train_paths = paths[: args.val_images], paths[args.val_images:]
    print(f"{len(train_paths)} train / {len(val_paths)} val images")

    def batches(files, shuffle):
        loader = DataLoader(ImageFiles(files), batch_size=None, shuffle=shuffle, num_workers=6)
        rng = random.Random(args.seed)
        for image in loader:
            height, width = image.shape[-2:]
            pixels = model.preprocess_image(image, fast=True, device="cuda")
            if rng.random() < 0.5 or not shuffle:
                rows, cols = model.padding_free_grid([(height, width)])
                pixels = pixels[..., : rows * model.patch_size, : cols * model.patch_size]
            grid = (pixels.shape[-2] // model.patch_size, pixels.shape[-1] // model.patch_size)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                _, vision_full = model.vision_model(pixels)
                feats, _ = model._detection_image_features_from_vision(vision_full, grid)
                target = torch.sigmoid(model.objectness_head(feats)[..., 0].float())
            yield {a: captured[a][:, 1:].float() for a in args.after}, target

    total = args.epochs * len(train_paths)
    optimizers = {a: torch.optim.AdamW(s.parameters(), lr=args.lr, weight_decay=1e-4) for a, s in scorers.items()}
    schedules = {a: torch.optim.lr_scheduler.OneCycleLR(o, args.lr, total_steps=total, pct_start=0.05)
                 for a, o in optimizers.items()}
    for epoch in range(args.epochs):
        losses = {a: 0.0 for a in args.after}
        for step, (feats, target) in enumerate(batches(train_paths, shuffle=True)):
            for a, scorer in scorers.items():
                loss = F.binary_cross_entropy_with_logits(scorer(feats[a]).squeeze(-1), target)
                optimizers[a].zero_grad()
                loss.backward()
                optimizers[a].step()
                schedules[a].step()
                losses[a] += loss.item()
        print(f"epoch {epoch}: " + ", ".join(f"block {a} loss {losses[a] / (step + 1):.4f}" for a in args.after))

    # Held-out recall: share of the unpruned model's confident tokens inside the scorer's top-keep.
    hits = {(a, k): 0 for a in args.after for k in (0.25, 0.5)}
    positives = 0
    with torch.no_grad():
        for feats, target in batches(val_paths, shuffle=False):
            confident = target > 0.1
            positives += int(confident.sum())
            for a, scorer in scorers.items():
                scores = scorer(feats[a]).squeeze(-1)
                for k in (0.25, 0.5):
                    count = max(1, math.ceil(k * scores.shape[1]))
                    top = torch.zeros_like(confident).scatter_(1, scores.topk(count, dim=1).indices, True)
                    hits[(a, k)] += int((top & confident).sum())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for a, scorer in scorers.items():
        recall = {k: hits[(a, k)] / max(1, positives) for k in (0.25, 0.5)}
        print(f"block {a}: recall of objectness>0.1 tokens in top 25% {recall[0.25]:.4f}, top 50% {recall[0.5]:.4f}")
        out = args.out_dir / f"objectness_scorer_b{a}.pt"
        torch.save({
            "scorer": {k: v.cpu() for k, v in scorer.state_dict().items()},
            "config": {
                "after": a, "keep": args.keep, "registers": args.registers, "threshold": None,
                "target": "objectness", "images": len(train_paths), "epochs": args.epochs,
                "sources": args.sources, "val_recall": recall,
            },
        }, out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
