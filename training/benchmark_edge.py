"""Compare SurfaceNet FP32 and calibrated INT8 on the same ExecuTorch CPU host."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch

from train import ROOT, SEED, SurfaceNetV2, evaluate, held_out_indices, pixel_ap, read_split, score_maps


def export_int8(model, calibration, target):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    from executorch.backends.xnnpack.utils.configs import get_transform_passes
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    example = calibration[:1].float() / 127.5 - 1
    graph = torch.export.export(model, (example,)).module()
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for batch in calibration.split(1):
            prepared(batch.float() / 127.5 - 1)
    quantized = convert_pt2e(prepared)
    program = to_edge_transform_and_lower(
        torch.export.export(quantized, (example,)), partitioner=[XnnpackPartitioner()],
        transform_passes=get_transform_passes(),
    ).to_executorch()
    with target.open("wb") as stream:
        program.write_to_file(stream)


def predict(method, images):
    with torch.inference_mode():
        return torch.stack([method.execute([image[None].float() / 127.5 - 1])[0][0].clone()
                            for image in images]).sigmoid()


def timings(methods, inputs):
    samples = [[] for _ in methods]
    with torch.inference_mode():
        for _ in range(16):
            for method in methods:
                method.execute([inputs[0]])
        for index, frame in enumerate(inputs):
            for j in (range(len(methods)) if index % 2 == 0 else reversed(range(len(methods)))):
                start = time.perf_counter_ns()
                methods[j].execute([frame])
                samples[j].append((time.perf_counter_ns() - start) / 1e6)
    return [{"p50_ms": round(float(np.median(s)), 3),
             "p95_ms": round(float(np.percentile(s, 95)), 3), "runs": len(s)} for s in samples]


def host():
    try:
        name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        name = platform.processor() or platform.machine()
    return {"cpu": name, "os": f"{platform.system()} {platform.release()}",
            "runtime": "ExecuTorch 1.5.0 / XNNPACK", "scope": "Warm single-ROI forward call"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "website/dist/assets")
    parser.add_argument("--release-to", type=Path,
                        help="Copy the INT8 program and matching calibration after the validation gate")
    args = parser.parse_args()
    torch.set_num_threads(4)
    x, y, _ = read_split(args.dataset / "train")
    train, val = held_out_indices(y)
    label = y.flatten(1).any(1).numpy()
    rng = np.random.default_rng(SEED)
    training = np.asarray(train)
    calibration_ids = np.concatenate([rng.choice(training[label[training]], 96, replace=False),
                                      rng.choice(training[~label[training]], 288, replace=False)])
    rng.shuffle(calibration_ids)
    model = SurfaceNetV2().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    args.output.mkdir(parents=True, exist_ok=True)
    int8 = args.output / "surface-int8.pte"
    export_int8(model, x[calibration_ids], int8)
    print(f"Exported {int8.stat().st_size} bytes", flush=True)
    from executorch.runtime import Runtime

    paths = [args.fp32, int8]
    programs = [Runtime.get().load_program(str(p)) for p in paths]
    methods = [program.load_method("forward") for program in programs]
    warm_inputs = [x[i:i+1].float() / 127.5 - 1 for i in val[:120]]
    measured = timings(methods, warm_inputs)
    variants = []
    for name, path, timing, method in zip(("FP32", "INT8"), paths, measured, methods):
        val_scores = score_maps(predict(method, x[val]))
        threshold = float(np.quantile(val_scores[~label[val]], .98, method="higher"))
        valid = evaluate(val_scores, label[val], threshold)
        variants.append({"precision":name, "bytes":path.stat().st_size, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "threshold":threshold,
                         "validation":valid, **timing})
        print(name, valid, timing, flush=True)
    f, q = variants
    # Release criterion is fixed before opening the official test split.
    eligible = (q["validation"]["average_precision"] >= f["validation"]["average_precision"] - .02
                and q["validation"]["recall"] >= f["validation"]["recall"] - .04
                and q["bytes"] <= .65 * f["bytes"] and q["p95_ms"] <= f["p95_ms"])
    xt, yt, _ = read_split(args.dataset / "test")
    for variant, method in zip(variants, methods):
        maps = predict(method, xt)
        variant["official_test"] = evaluate(score_maps(maps), yt.flatten(1).any(1).numpy(), variant["threshold"])
        variant["pixel_average_precision"] = pixel_ap(maps, yt)
    report = {"host":host(), "calibration_images":len(calibration_ids),
              "int8_release_gate_passed":bool(eligible), "variants":variants}
    (args.output / "edge-benchmark.json").write_text(json.dumps(report,indent=2) + "\n")
    print(json.dumps(report,indent=2), flush=True)
    if args.release_to and eligible:
        args.release_to.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(int8, args.release_to / "surface-int8.pte")
        config = {"precision": "INT8", "threshold": q["threshold"]}
        (args.release_to / "edge-local.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
