import torch, sys
from arch_probe import *
model, _ = load_model(None)
for name, ann, imgs in [("aitod", AITOD / "subsets/vehicle_val200.json", AITOD / "val/images"),
                        ("coco", "/mnt/datasets/coco/annotations/instances_val2017_mini500.json", "/mnt/datasets/coco/val2017")]:
    val = CachedVal(model, ann, imgs, batch_size=4, limit=4)
    x = val.batches[0]["images"].cuda().float()
    vm = model.vision_model
    with torch.no_grad():
        pe = vm.patch_embedding(x).flatten(2).transpose(1, 2)
        h = torch.cat([vm.class_embedding.expand(x.shape[0], 1, -1), pe], 1) + vm.position_embedding(vm.position_ids)
        h = vm.pre_layernorm(h)
        print(name, "layer: median patch norm | p99.9 | max | #tokens > 10x median (per image) | CLS norm")
        for li, layer in enumerate(vm.encoder.layers):
            h = run_layer(layer, h, li, Cfg(), 60)
            n = h[:, 1:].norm(dim=-1)
            med = n.median(dim=1).values
            out = (n > 10 * med[:, None]).sum(1).float().mean().item()
            print(f"  {li:2d}: {med.mean():7.1f} | {n.quantile(0.999).item():7.1f} | {n.max().item():7.1f} | {out:5.1f} | {h[:, 0].norm(dim=-1).mean():.1f}")
        # attention mass the patch queries give to the outlier tokens in the last layer with outliers
