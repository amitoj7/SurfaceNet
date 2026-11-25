"""Select and measure the four-orientation detector with the exact INT8 export."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from executorch.runtime import Runtime
from sklearn.metrics import precision_recall_curve

from benchmark_edge import export_int8, host, predict, timings
from reflection import ReflectedSurfaceNet
from train import HEIGHT, ROOT, SEED, WIDTH, SurfaceNetV2, evaluate, export, held_out_indices, pixel_ap, read_split, score_maps


def profile(method, images, masks, threshold=None):
    maps = predict(method, images)
    scores = score_maps(maps)
    labels = masks.flatten(1).any(1).numpy()
    if threshold is None:
        threshold = float(np.quantile(scores[~labels], .98, method="higher"))
    return {**evaluate(scores, labels, threshold), "pixel_ap": pixel_ap(maps, masks),
            "threshold": threshold}, scores


def open_method(path):
    return Runtime.get().load_program(str(path)).load_method("forward")


def score_curve(scores, labels):
    precision, recall, _ = precision_recall_curve(labels, scores)
    # Keep the full step curve so the downloadable data reproduces the plot.
    return [[round(float(r), 5), round(float(p), 5)] for r, p in zip(recall, precision)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "android/app/src/main/assets")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, _ = read_split(args.dataset / "train")
    train, valid = held_out_indices(y)
    labels = y.flatten(1).any(1).numpy()
    rng = np.random.default_rng(SEED)
    train = np.asarray(train)
    calibration = np.concatenate([rng.choice(train[labels[train]], 96, replace=False),
                                  rng.choice(train[~labels[train]], 288, replace=False)])
    rng.shuffle(calibration)
    backbone = SurfaceNetV2().eval()
    backbone.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    baseline_path = args.baseline / "surface-int8.pte"
    baseline_method = open_method(baseline_path)
    baseline_config = json.loads((args.baseline / "edge-local.json").read_text())
    baseline, _ = profile(baseline_method, x[valid], y[valid], baseline_config["threshold"])
    print("Baseline", baseline, flush=True)
    candidates = []
    for weight in (0.0, 0.25):
        path = args.output / f"candidate-{weight}.pte"
        export_int8(ReflectedSurfaceNet(backbone, weight).eval(), x[calibration], path)
        method = open_method(path)
        result, _ = profile(method, x[valid], y[valid])
        candidates.append((weight, path, result))
        print("Validation", weight, result, flush=True)

    eligible = [item for item in candidates if item[2]["average_precision"] >= baseline["average_precision"] + .01
                and item[2]["pixel_ap"] >= baseline["pixel_ap"] and item[2]["tp"] >= baseline["tp"]
                and item[1].stat().st_size <= 256 * 1024]
    if not eligible:
        raise SystemExit("Reflection did not pass validation. Official test data unopened.")
    weight, candidate_path, validation = max(eligible, key=lambda item: item[2]["average_precision"])
    native = open_method(candidate_path)
    reflected = ReflectedSurfaceNet(backbone, weight).eval()
    error = export(reflected, args.output, x[valid[:1]].float() / 127.5 - 1)
    fp32_path = args.output / "surface.pte"
    fp32 = open_method(fp32_path)
    warm = [x[i:i+1].float() / 127.5 - 1 for i in valid[:120]]
    runtime = timings([baseline_method, fp32, native], warm)
    print("Warmed timings", runtime, flush=True)
    if runtime[-1]["p95_ms"] > 15:
        raise SystemExit("Reflection exceeded the 15 ms host forward budget. Official test data unopened.")
    fp32_valid, _ = profile(fp32, x[valid], y[valid])
    print("Selected weight", weight, "FP32 validation", fp32_valid, flush=True)

    xt, yt, sources = read_split(args.dataset / "test")
    test_labels = yt.flatten(1).any(1).numpy()
    comparisons = []
    curves = []
    for label, path, method, valid_stats, speed in zip(("Single-view INT8", "FP32", "INT8"),
            (baseline_path, fp32_path, candidate_path), (baseline_method, fp32, native),
            (baseline, fp32_valid, validation), runtime):
        test, scores = profile(method, xt, yt, valid_stats["threshold"])
        comparisons.append({"precision":label, "bytes":path.stat().st_size,
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                            "threshold":valid_stats["threshold"], "validation":valid_stats,
                            "official_test":test, "pixel_average_precision":test["pixel_ap"], **speed})
        if label in ("Single-view INT8", "INT8"):
            curves.append({"label":label, "points":score_curve(scores, test_labels)})
    original, full, quant = comparisons
    (args.output / "surface-int8.pte").write_bytes(candidate_path.read_bytes())
    (args.output / "edge-local.json").write_text(json.dumps({"precision":"INT8", "threshold":quant["threshold"]}, indent=2)+"\n")
    previous = json.loads((args.baseline / "vision.json").read_text())
    vision = {**previous, "architecture":"v2 · 4 views", "context_weight":weight, "threshold":full["threshold"],
              "pte_bytes":full["bytes"], "export_max_abs_error":error, "validation":full["validation"],
              "official_test":full["official_test"], "pixel_average_precision":full["pixel_average_precision"]}
    (args.output / "vision.json").write_text(json.dumps(vision, indent=2)+"\n")
    result = {"host":host(), "calibration_images":len(calibration), "int8_release_gate_passed":True,
              "context_weight":weight, "baseline":original, "variants":[full,quant],
              "selection_candidates":[{"context_weight":w,"validation":v} for w, _, v in candidates],
              "curves":curves}
    (args.output / "edge-benchmark.json").write_text(json.dumps(result,indent=2)+"\n")
    for item in comparisons:
        print(item["precision"], item["bytes"], "bytes", item["p95_ms"], "ms p95", item["official_test"], flush=True)


if __name__ == "__main__":
    main()
