"""Add a positive-tail review band without retraining or changing the alert cutoff."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from executorch.runtime import Runtime

from benchmark_edge import predict
from train import ROOT, held_out_indices, read_split, score_maps


def decisions(scores, labels, clear, hold):
    is_hold = scores > hold
    is_clear = scores < clear
    is_review = ~(is_hold | is_clear)
    return {
        "defect": {"clear": int((labels & is_clear).sum()), "review": int((labels & is_review).sum()),
                   "hold": int((labels & is_hold).sum()), "total": int(labels.sum())},
        "clean": {"clear": int((~labels & is_clear).sum()), "review": int((~labels & is_review).sum()),
                  "hold": int((~labels & is_hold).sum()), "total": int((~labels).sum())},
    }


def predict_onnx(session, images):
    logits = np.stack([session.run(["logits"], {"pixels": image[None].float().numpy() / 127.5 - 1})[0][0]
                       for image in images])
    return torch.from_numpy(logits).sigmoid()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--android", type=Path, default=ROOT / "android/app/src/main/assets")
    parser.add_argument("--onnx", type=Path, default=ROOT / "website/dist/assets/surface.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "website/dist/assets/review-calibration.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    alpha = .05
    x, y, _ = read_split(args.dataset / "train")
    _, valid = held_out_indices(y)
    validation_labels = y[valid].flatten(1).any(1).numpy()
    xt, yt, _ = read_split(args.dataset / "test")
    test_labels = yt.flatten(1).any(1).numpy()
    native = json.loads((args.android / "edge-local.json").read_text())
    web = json.loads((args.android / "vision.json").read_text())
    profiles = {}
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), session_options, providers=["CPUExecutionProvider"])
    program = Runtime.get().load_program(str(args.android / "surface-int8.pte"))
    method = program.load_method("forward")
    for name, hold, infer, runtime in [
        ("fp32", web["threshold"], lambda images: predict_onnx(session, images), "ONNX Runtime CPU"),
        ("int8", native["threshold"], lambda images: predict(method, images), "ExecuTorch XNNPACK"),
    ]:
        validation_scores = score_maps(infer(x[valid]))
        positive = np.sort(validation_scores[validation_labels])
        # p(defective) = (1 + number of defect calibration scores <= s) / (n+1).
        # Strict < handles ties conservatively. Cap to keep active alerts intact.
        rank = math.floor(alpha * (len(positive) + 1)) - 1
        clear = min(hold, float(positive[rank])) if rank >= 0 else 0.0
        test = score_maps(infer(xt))
        profiles[name] = {
            "calibration_runtime": runtime,
            "clear_below": clear, "alert_above": hold,
            "calibration_defects": len(positive), "calibration_rank": rank,
            "validation": decisions(validation_scores, validation_labels, clear, hold),
            "official_test": decisions(test, test_labels, clear, hold),
            "binary_test": decisions(test, test_labels, hold, hold),
        }
    args.output.write_text(json.dumps({"alpha": alpha, "profiles": profiles},indent=2)+"\n")
    native["clear_below"] = profiles["int8"]["clear_below"]
    (args.android / "edge-local.json").write_text(json.dumps(native,indent=2)+"\n")
    print(json.dumps(profiles,indent=2), flush=True)


if __name__ == "__main__":
    main()
