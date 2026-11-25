"""Export the selected checkpoint for the website and evaluate the Java lighting gate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch

from train import HEIGHT, ROOT, WIDTH, SurfaceNetV2, read_split
from reflection import ReflectedSurfaceNet

JAVA = ROOT / "android/app/src/main/java/io/surfaceguard/android"
ASSETS = ROOT / "android/app/src/main/assets"


def relight(images):
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    alternate = np.empty_like(images)
    for index, image in enumerate(images):
        dx, dy = (2 if index % 2 else -2), (index % 5) - 2
        image = image[np.clip(yy - dy, 0, HEIGHT - 1), np.clip(xx - dx, 0, WIDTH - 1)]
        shifted = image.astype(float) * 1.08 + 15
        if index % 3 == 0:
            cx, cy = 20 + (index * 17) % 57, 42 + (index * 29) % 166
            streak = np.exp(-((xx - cx + .17 * (yy - cy)) / 10) ** 2 - ((yy - cy) / 43) ** 2)
            shifted += 35 * streak
        alternate[index] = np.clip(np.rint(shifted), 0, 255).astype(np.uint8)
    return alternate


def logits(model, images, device):
    with torch.inference_mode():
        return torch.cat([model(block.to(device, dtype=torch.float32) / 127.5 - 1).cpu()
                          for block in torch.from_numpy(images[:, None]).split(24)]).numpy()


def judge_pairs(x, f, a, b, threshold, java_home):
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        examples = folder / "captures.bin"
        with examples.open("wb") as stream:
            stream.write(struct.pack(">fi", threshold, len(x)))
            for ambient, relit, raw_a, raw_b in zip(x, f, a, b):
                stream.write(ambient.tobytes())
                stream.write(relit.tobytes())
                stream.write(raw_a.astype(">f4").tobytes())
                stream.write(raw_b.astype(">f4").tobytes())
        subprocess.run([str(java_home / "bin/javac"), "-d", temp,
                        str(JAVA / "Inspection.java"), str(JAVA / "Lighting.java"),
                        str(ROOT / "training/checks/PairAudit.java")], check=True)
        with examples.open("rb") as stream:
            result = subprocess.check_output([str(java_home / "bin/java"), "-cp", temp,
                                              "io.surfaceguard.android.PairAudit"], stdin=stream, text=True)
    fields = ("ambient_score", "relit_score", "ambient_alert", "relit_alert", "status", "dx",
              "dy", "registration", "light_change", "overlap", "ambient_support", "relit_support", "mask_match", "elongation")
    rows = list(csv.DictReader(result.splitlines(), fieldnames=fields))
    if len(rows) != len(x):
        raise ValueError("Incomplete lighting results")
    for row in rows:
        row["initial"] = row["ambient_alert"] == "true"
        row["box_confirmed"] = row["initial"] and row["status"] == "USABLE" and float(row["overlap"]) >= .25
        row["confirmed"] = row["initial"] and row["status"] == "USABLE" and row["mask_match"] == "true"
    return rows


def write_cases(output, x, f, masks, sources, rows):
    labels = masks.reshape(len(masks), -1).any(1)
    matches = {
        "persistent": [i for i, row in enumerate(rows) if labels[i] and row["confirmed"]],
        "line": [i for i, row in enumerate(rows) if labels[i] and row["confirmed"]
                 and float(row["elongation"]) >= 3.2 and masks[i].sum() >= 40],
        "transient": [i for i, row in enumerate(rows) if not labels[i] and row["initial"]
                      and row["status"] == "USABLE" and not row["confirmed"]],
        "clean": [i for i, row in enumerate(rows) if not labels[i] and not row["initial"]
                  and row["relit_alert"] == "false" and row["status"] == "USABLE"],
        "missed": [i for i, row in enumerate(rows) if labels[i] and not row["initial"]
                   and row["relit_alert"] == "false" and row["status"] == "USABLE"],
    }
    conflicting = [i for i in matches["transient"] if rows[i]["box_confirmed"]]
    if conflicting:
        matches["transient"] = conflicting
    info = {
        "persistent": ("Persistent defect", "An annotated surface mark that remains detectable after a controlled illumination change."),
        "line": ("Thin mark", "An annotated defect with an elongated learned mask. The same line-like region persists after relighting."),
        "transient": ("Unconfirmed alert", "A clean specimen with a detector alert. The second exposure tests whether the actual pixels agree after alignment."),
        "clean": ("Clean control", "A held-out normal specimen with a low surface score in both exposures."),
        "missed": ("Missed defect", "The detector drew no mask here. Turn on Annotation to see the defect marked by the annotator."),
    }
    cases = []
    for key, candidates in matches.items():
        if not candidates:
            continue
        if key in {"persistent", "missed", "line"}:
            index = max(candidates, key=lambda i: int(masks[i].sum()))
        elif key == "clean":
            index = min(candidates, key=lambda i: max(float(rows[i]["ambient_score"]), float(rows[i]["relit_score"])))
        else:
            index = candidates[len(candidates) // 2]
        dx, dy = (2 if index % 2 else -2), (index % 5) - 2
        yy, xx = np.mgrid[:HEIGHT, :WIDTH]
        alternate = masks[index][np.clip(yy - dy, 0, HEIGHT - 1), np.clip(xx - dx, 0, WIDTH - 1)]
        for field, image in (("ambient", x[index]), ("relit", f[index]),
                             ("truth", masks[index].astype(np.uint8) * 255),
                             ("relit_truth", alternate.astype(np.uint8) * 255)):
            Image.fromarray(image).save(output / f"{key}-{field}.png")
        label, note = info[key]
        cases.append({"id": key, "label": label, "note": note, "source": sources[index].stem,
                      "annotated_defect": bool(labels[index]), "lab_measurement": rows[index]})
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def summarize_pairs(rows, masks):
    labels = masks.reshape(len(masks), -1).any(1)
    counts = {}
    for tag, label in [("defective", True), ("clean", False)]:
        subset = [row for row, positive in zip(rows, labels) if positive == label]
        counts[tag] = {"images": len(subset), "initial": sum(row["initial"] for row in subset),
                       "confirmed": sum(row["confirmed"] for row in subset),
                       "box_confirmed": sum(row["box_confirmed"] for row in subset),
                       "usable": sum(row["status"] == "USABLE" for row in subset)}
    return counts


def native_logits(method, images):
    with torch.inference_mode():
        return torch.cat([method.execute([image[None,None].float() / 127.5 - 1])[0][0].clone()
                          for image in torch.from_numpy(images)]).reshape(len(images),1,HEIGHT,WIDTH).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--java-home", required=True, type=Path)
    parser.add_argument("--assets", type=Path, default=ROOT / "website/dist/assets")
    parser.add_argument("--android", type=Path, default=ASSETS, help="Directory with the target vision metadata")
    parser.add_argument("--edge-report", type=Path, help="Measured FP32/INT8 report to add the native paired result to")
    parser.add_argument("--edge-pte", type=Path, help="Selected calibrated INT8 ExecuTorch program")
    args = parser.parse_args()
    if bool(args.edge_report) != bool(args.edge_pte):
        parser.error("--edge-report and --edge-pte must be passed together")
    output = args.assets
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "training/baselines/surfacenet-v1.json").read_text())
    model_metrics = json.loads((args.android / "vision.json").read_text())
    model = SurfaceNetV2().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    if "context_weight" in model_metrics:
        model = ReflectedSurfaceNet(model, model_metrics["context_weight"]).eval()
    x, truths, paths = read_split(args.dataset / "test")
    example = x[:1].float() / 127.5 - 1
    onnx_file = output / "surface.onnx"
    torch.onnx.export(model, (example,), str(onnx_file), dynamo=False, opset_version=17,
                      input_names=["pixels"], output_names=["logits"])
    with torch.inference_mode():
        expected = model(example).numpy()
    web = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    difference = float(np.abs(web.run(None, {"pixels": example.numpy()})[0] - expected).max())
    if difference > .0005:
        raise ValueError(f"ONNX parity error {difference}")
    images, masks = x[:, 0].numpy(), truths[:, 0].numpy()
    alternate = relight(images)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    rows = judge_pairs(images, alternate, logits(model, images, device), logits(model, alternate, device),
                       model_metrics["threshold"], args.java_home)
    counts = summarize_pairs(rows, masks)
    if args.edge_report:
        from executorch.runtime import Runtime
        report = json.loads(args.edge_report.read_text())
        method = Runtime.get().load_program(str(args.edge_pte)).load_method("forward")
        threshold = next(item["threshold"] for item in report["variants"] if item["precision"] == "INT8")
        pairs = judge_pairs(images, alternate, native_logits(method, images), native_logits(method, alternate),
                            threshold, args.java_home)
        report["paired_int8"] = summarize_pairs(pairs, masks)
        (output / "edge-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["paired_int8"], indent=2), flush=True)
    write_cases(output, images, alternate, masks, paths, rows)
    result = {"baseline": baseline, "deployed": model_metrics, "onnx_max_error": difference,
              "onnx_sha256": hashlib.sha256(onnx_file.read_bytes()).hexdigest(),
              "paired": counts, "scope": "One controlled simulated alternate capture per official test image"}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    icon = Image.new("RGBA", (48, 48), "#0a1011")
    draw = ImageDraw.Draw(icon)
    for x0, y0, dx, dy in [(8,8,1,1),(39,8,-1,1),(8,39,1,-1),(39,39,-1,-1)]:
        draw.line([(x0+9*dx,y0),(x0,y0),(x0,y0+9*dy)], fill="#c8f478", width=3)
    draw.ellipse((21,21,26,26), fill="#c8f478")
    icon.save(output / "icon.png")
    print(json.dumps(result["paired"], indent=2), flush=True)


if __name__ == "__main__":
    main()
