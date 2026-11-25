"""Train SurfaceGuard's small defect segmenter on the official KolektorSDD2 split."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

WIDTH, HEIGHT = 96, 256
SEED = 184
ROOT = Path(__file__).resolve().parents[1]


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


class SurfaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail = conv(1, 16, 2)
        self.mid = nn.Sequential(conv(16, 24, 2), conv(24, 24))
        self.context = nn.Sequential(conv(24, 32, 2), conv(32, 32), conv(32, 24))
        self.fuse = nn.Sequential(conv(64, 24), conv(24, 16), nn.Conv2d(16, 1, 1))

    def forward(self, pixels):
        detail = self.detail(pixels)
        mid = self.mid(detail)
        context = self.context(mid)
        features = torch.cat([
            detail, F.interpolate(mid, scale_factor=2, mode="nearest"),
            F.interpolate(context, scale_factor=4, mode="nearest"),
        ], dim=1)
        return F.interpolate(self.fuse(features), scale_factor=2, mode="bilinear", align_corners=False)



class SurfaceNetV2(nn.Module):
    """Joint defect recognition and localization with shallow detail skips."""

    def __init__(self):
        super().__init__()
        self.detail = nn.Sequential(conv(1, 20, 2), conv(20, 20))
        self.mid = nn.Sequential(conv(20, 36, 2), conv(36, 36))
        self.context = nn.Sequential(conv(36, 56, 2), conv(56, 56), conv(56, 36))
        self.up_mid = conv(72, 32)
        self.up_detail = nn.Sequential(conv(52, 24), conv(24, 16))
        self.pixel = nn.Conv2d(17, 1, 3, padding=1)
        self.image = nn.Sequential(nn.Linear(72, 24), nn.ReLU(), nn.Linear(24, 1))

    def outputs(self, pixels):
        detail = self.detail(pixels)
        mid = self.mid(detail)
        context = self.context(mid)
        pooled = torch.cat([context.amax((2, 3)), context.mean((2, 3))], dim=1)
        image = self.image(pooled)
        mid = self.up_mid(torch.cat([mid, F.interpolate(context, scale_factor=2, mode="nearest")], dim=1))
        fine = self.up_detail(torch.cat([detail, F.interpolate(mid, scale_factor=2, mode="nearest")], dim=1))
        fine = F.interpolate(fine, scale_factor=2, mode="bilinear", align_corners=False)
        # Original luminance provides a skip around the three downsampling stages.
        dense = self.pixel(torch.cat([fine, pixels], dim=1))
        return dense + .20 * image[:, :, None, None], image

    def forward(self, pixels):
        return self.outputs(pixels)[0]


def relight_training(pixels, masks):
    pixels, masks = augment(pixels, masks)
    if random.random() < .3:
        pixels, masks = pixels.flip(-2), masks.flip(-2)
    if random.random() < .7:
        yy = torch.linspace(-1, 1, HEIGHT, device=pixels.device).reshape(1, 1, HEIGHT, 1)
        xx = torch.linspace(-1, 1, WIDTH, device=pixels.device).reshape(1, 1, 1, WIDTH)
        controls = torch.rand(5, len(pixels), 1, 1, 1, device=pixels.device)
        x, y, angle, amplitude, width = controls.unbind()
        along = yy - (y * 1.6 - .8)
        cross = xx - (x * 1.6 - .8) + (angle - .5) * along
        reflection = torch.exp(-(cross / (.13 + .22 * width)).square() - (along / .55).square())
        pixels = (pixels + .55 * amplitude * reflection).clamp(-1, 1)
    return pixels, masks


def read_split(directory):
    inputs, masks, sources = [], [], []
    # The archive also contains a few unannotated file copies.
    for label in sorted(directory.glob("*_GT.png")):
        path = label.with_name(label.name.removesuffix("_GT.png") + ".png")
        with Image.open(path) as image, Image.open(label) as mask:
            inputs.append(np.asarray(image.convert("L").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)))
            masks.append(np.asarray(mask.convert("L").resize((WIDTH, HEIGHT), Image.Resampling.BOX)) > 0)
            sources.append(path)
    if not inputs:
        raise ValueError(f"No images and masks found in {directory}")
    return torch.from_numpy(np.stack(inputs))[:, None], torch.from_numpy(np.stack(masks))[:, None], sources


def held_out_indices(masks):
    rng = np.random.default_rng(SEED)
    positive = masks.flatten(1).any(1).numpy()
    train, validation = [], []
    for indices in (np.flatnonzero(positive), np.flatnonzero(~positive)):
        rng.shuffle(indices)
        split = max(1, len(indices) // 5)
        validation.extend(indices[:split])
        train.extend(indices[split:])
    return train, validation


def augment(pixels, masks):
    if random.random() < .5:
        pixels, masks = pixels.flip(-1), masks.flip(-1)
    gain = torch.rand(len(pixels), 1, 1, 1, device=pixels.device) * .38 + .82
    bias = torch.rand_like(gain) * .34 - .17
    return (pixels * gain + bias + .025 * torch.randn_like(pixels)).clamp(-1, 1), masks


def predict(model, pixels, device):
    with torch.inference_mode():
        return torch.cat([
            model(batch.to(device, dtype=torch.float32) / 127.5 - 1).sigmoid().cpu()
            for batch in pixels.split(32)
        ])


def score_maps(maps):
    return F.avg_pool2d(maps, 2, stride=1).flatten(1).amax(1).numpy()


def evaluate(scores, labels, threshold):
    selected = scores > threshold
    tp, fp = (selected & labels).sum(), (selected & ~labels).sum()
    positives, negatives = labels.sum(), (~labels).sum()
    return {
        "images": len(labels), "positives": int(positives), "negatives": int(negatives),
        "auroc": round(roc_auc_score(labels, scores), 4),
        "average_precision": round(average_precision_score(labels, scores), 4),
        "recall": round(float(tp / positives), 4), "false_alert_rate": round(float(fp / negatives), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(negatives - fp), "fn": int(positives - tp),
    }


def pixel_ap(maps, truth):
    # A histogram bounds memory while keeping score order at 1/2048 resolution.
    resolution = 2048
    positive = np.zeros(resolution, dtype=np.int64)
    total = np.zeros(resolution, dtype=np.int64)
    for scores, masks in zip(maps.split(32), truth.split(32)):
        bins = (scores.numpy().ravel() * (resolution - 1)).astype(np.int32)
        total += np.bincount(bins, minlength=resolution)
        positive += np.bincount(bins[masks.numpy().ravel()], minlength=resolution)
    tp = np.cumsum(positive[::-1])
    detections = np.cumsum(total[::-1]).clip(1)
    return round(float(np.sum((tp / detections) * positive[::-1]) / tp[-1]), 4)


def export(model, assets, example):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    program = to_edge_transform_and_lower(
        torch.export.export(model, (example,)), partitioner=[XnnpackPartitioner()]
    )
    destination = assets / "surface.pte"
    with destination.open("wb") as file:
        program.to_executorch().write_to_file(file)
    runtime = Runtime.get().load_program(str(destination)).load_method("forward")
    with torch.inference_mode():
        expected = model(example)
        observed = runtime.execute([example])[0]
    torch.testing.assert_close(observed, expected, atol=5e-4, rtol=5e-4)
    return float((observed - expected).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted folder containing train/ and test/")
    parser.add_argument("--architecture", choices=["v1", "v2"], default="v1")
    parser.add_argument("--validation-only", action="store_true", help="Select a checkpoint without opening the official test data")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--assets", type=Path, default=ROOT / "android/app/src/main/assets")
    parser.add_argument("--checkpoint", type=Path, help="Cache and reuse the selected model state")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(6)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}", flush=True)

    x, y, _ = read_split(args.dataset / "train")
    train, val = held_out_indices(y)
    xv, yv = x[val], y[val].flatten(1).any(1).numpy()
    x, y = x[train], y[train]
    labels = y.flatten(1).any(1)
    weights = torch.where(labels, 1.0 / int(labels.sum()), 1.0 / int((~labels).sum()))
    sampler = WeightedRandomSampler(weights, len(labels), replacement=True)
    second_generation = args.architecture == "v2"
    loader = DataLoader(TensorDataset(x, y), batch_size=24 if second_generation else 32, sampler=sampler)

    model = (SurfaceNetV2() if second_generation else SurfaceNet()).to(device)
    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0003)
        best = -1
        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0
            for images, masks in loader:
                image_label = masks.flatten(1).any(1).to(device, dtype=torch.float32)
                augmentation = relight_training if second_generation else augment
                images, masks = augmentation(images.to(device, dtype=torch.float32) / 127.5 - 1,
                                             masks.to(device, dtype=torch.float32))
                if second_generation:
                    logits, image_logits = model.outputs(images)
                else:
                    logits = model(images)
                prob = logits.sigmoid()
                positive = (masks * prob).sum()
                dice = (2 * positive + 1) / (masks.sum() + prob.sum() + 1)
                bce = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
                focal = (bce * (masks * 5 + 1) * (masks - prob).abs().square()).mean()
                loss = focal * 3 + (1 - dice)
                if second_generation:
                    worst_background = (bce * (1 - masks)).flatten(1).topk(24, dim=1).values.mean()
                    classification = F.binary_cross_entropy_with_logits(image_logits.flatten(), image_label)
                    loss = loss + .32 * classification + .10 * worst_background
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            scheduler.step()
            model.eval()
            scores = score_maps(predict(model, xv, device))
            ap = average_precision_score(yv, scores)
            print(f"epoch {epoch + 1:02d} loss {loss_sum / len(loader):.4f} validation AP {ap:.4f}", flush=True)
            if ap > best:
                best = ap
                checkpoint = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.checkpoint)
    model.cpu().load_state_dict(checkpoint)
    model.eval()
    sv = score_maps(predict(model, xv, "cpu"))
    threshold = float(np.quantile(sv[~yv], .98, method="higher"))

    validation_metrics = evaluate(sv, yv, threshold)
    if args.validation_only:
        print(json.dumps(validation_metrics, indent=2), flush=True)
        return
    xt, yt, sources = read_split(args.dataset / "test")
    maps = predict(model, xt, "cpu")
    scores = score_maps(maps)
    truth = yt.flatten(1).any(1).numpy()
    args.assets.mkdir(parents=True, exist_ok=True)
    max_error = export(model, args.assets, xt[:1].float() / 127.5 - 1)
    specimens = {}
    for is_defect, name in ((False, "clean"), (True, "defect")):
        candidates = np.flatnonzero(truth & (scores > threshold) if is_defect else ~truth & (scores <= threshold))
        if not len(candidates):
            raise ValueError(f"No correctly classified {name} image for the demo")
        # The positive specimen is selected for a readable region. It is a demo sample.
        index = max(candidates, key=lambda idx: float(scores[idx])) if is_defect else candidates[len(candidates) // 2]
        Image.fromarray(xt[index, 0].numpy()).save(args.assets / f"{name}.png")
        specimens[name] = sources[index].name
    metrics = {
        "dataset": "KolektorSDD2", "architecture": args.architecture, "width": WIDTH, "height": HEIGHT, "threshold": threshold,
        "epochs": args.epochs, "seed": SEED, "training_device": device,
        "train_images": len(x), "training_positives": int(labels.sum()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "pte_bytes": (args.assets / "surface.pte").stat().st_size,
        "export_max_abs_error": max_error, "pixel_average_precision": pixel_ap(maps, yt),
        "demo_specimens": specimens,
        "validation": validation_metrics,
        "official_test": evaluate(scores, truth, threshold),
    }
    (args.assets / "vision.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
