"""Fine-tune SurfaceNet on paired photometric views with no runtime model expansion."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train import HEIGHT, SEED, WIDTH, SurfaceNetV2, evaluate, held_out_indices, pixel_ap, predict, read_split, score_maps


def view(pixels, xx, yy):
    shape = (len(pixels), 1, 1, 1)
    random = lambda: torch.rand(shape, device=pixels.device)
    gain, offset = .80 + .48 * random(), -.19 + .38 * random()
    exposure = pixels * gain + offset
    rise = yy - (random() * 1.8 - .9)
    run = xx - (random() * 1.6 - .8) + (random() - .5) * rise
    streak = torch.exp(-(run / (.10 + .20 * random())).square() - (rise / .52).square())
    return (exposure + .58 * random() * streak + .015 * torch.randn_like(pixels)).clamp(-1, 1)


def supervised(logits, mask):
    probs = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    focal = (bce * (1 + 5 * mask) * (mask - probs).abs().square()).mean()
    dice = (2 * (mask * probs).sum() + 1) / (mask.sum() + probs.sum() + 1)
    hard = (bce * (1 - mask)).flatten(1).topk(24, dim=1).values.mean()
    return 3 * focal + (1 - dice) + .09 * hard


def validation(model, images, truth, device):
    maps = predict(model, images, device)
    scores = score_maps(maps)
    labels = truth.flatten(1).any(1).numpy()
    threshold = float(np.quantile(scores[~labels], .98, method="higher"))
    result = evaluate(scores, labels, threshold)
    result["pixel_ap"] = pixel_ap(maps, truth)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(6)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SurfaceNetV2().to(device)
    teacher = SurfaceNetV2().to(device).eval()
    state = torch.load(args.baseline, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    teacher.load_state_dict(state)
    x, y, _ = read_split(args.dataset / "train")
    train_ids, valid = held_out_indices(y)
    baseline = validation(teacher, x[valid], y[valid], device)
    print("FP32 baseline", baseline, flush=True)
    positive = y[train_ids].flatten(1).any(1)
    weights = torch.where(positive, 1.0 / positive.sum(), 1.0 / (~positive).sum())
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(TensorDataset(x[train_ids], y[train_ids]), batch_size=20, sampler=sampler)
    xx = torch.linspace(-1, 1, WIDTH, device=device).reshape(1, 1, 1, WIDTH)
    yy = torch.linspace(-1, 1, HEIGHT, device=device).reshape(1, 1, HEIGHT, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00008)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = -1
    for epoch in range(args.epochs):
        model.train()
        for pixels, masks in loader:
            labels = masks.flatten(1).any(1).to(device, dtype=torch.float32)
            pixels = pixels.to(device, dtype=torch.float32) / 127.5 - 1
            mask = masks.to(device, dtype=torch.float32)
            if random.random() < .5:
                pixels, mask = pixels.flip(-1), mask.flip(-1)
            if random.random() < .25:
                pixels, mask = pixels.flip(-2), mask.flip(-2)
            a, b = view(pixels, xx, yy), view(pixels, xx, yy)
            with torch.no_grad():
                anchor = teacher(pixels).sigmoid()
            logits, image = model.outputs(torch.cat((a,b)))
            la, lb = logits.chunk(2)
            prob_a, prob_b = la.sigmoid(), lb.sigmoid()
            visible = ((a.abs() < .96) & (b.abs() < .96)).float()
            weights = visible * (1 + 8 * mask + 2 * anchor)
            consistent = ((prob_a - prob_b).abs() * weights).sum() / weights.sum().clamp_min(1)
            anchor_loss = (F.smooth_l1_loss(prob_a, anchor, reduction="none") * (1 + 6 * mask)).mean()
            image_loss = F.binary_cross_entropy_with_logits(image.flatten(), labels.repeat(2))
            loss = (supervised(la, mask) + supervised(lb, mask)) * .5 + .27 * image_loss + .45 * consistent + .24 * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        metrics = validation(model, x[valid], y[valid], device)
        score = metrics["average_precision"] + .18 * metrics["pixel_ap"] + .10 * metrics["recall"]
        if score > best:
            best = score
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, args.output)
            args.output.with_suffix(".json").write_text(json.dumps({"epoch":epoch+1,"validation":metrics,"baseline":baseline},indent=2))
        print(f"epoch {epoch + 1} · AP {metrics['average_precision']:.4f} · pixel AP {metrics['pixel_ap']:.4f} · recall {metrics['recall']:.4f}", flush=True)


if __name__ == "__main__":
    main()
