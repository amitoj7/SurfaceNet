"""Build the guided lab illustration from the audited thin-mark specimen."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import binary_dilation, label, shift

ASSETS = Path(__file__).resolve().parents[1] / "website/dist/assets"


def main():
    portfolio = json.loads((ASSETS / "cases.json").read_text())
    item = next(case for case in portfolio if case["id"] == "line")
    measure = item["lab_measurement"]
    threshold = json.loads((ASSETS / "benchmark.json").read_text())["deployed"]["threshold"]
    session = ort.InferenceSession(str(ASSETS / "surface.onnx"), providers=["CPUExecutionProvider"])
    captures = [np.asarray(Image.open(ASSETS / f"line-{stage}.png")) for stage in ("ambient", "relit")]
    regions = []
    for frame in captures:
        logits = session.run(None, {"pixels": frame[None, None].astype(np.float32) / 127.5 - 1})[0][0, 0]
        probs = 1 / (1 + np.exp(-logits))
        pooled = (probs[:-1, :-1] + probs[1:, :-1] + probs[:-1, 1:] + probs[1:, 1:]) / 4
        y, x = np.unravel_index(pooled.argmax(), pooled.shape)
        labels, _ = label(probs > .85 * threshold, np.ones((3, 3)))
        candidates = set(labels[y:y+2, x:x+2].flatten()) - {0}
        region = labels == max(candidates, key=lambda k: (labels == k).sum())
        regions.append(region)
    dx, dy = int(measure["dx"]), int(measure["dy"])
    support = regions[1] & shift(binary_dilation(regions[0], structure=np.ones((3, 3))),
                                (dy, dx), order=0, prefilter=False)
    colored = np.repeat(captures[1][..., None], 3, axis=2)
    colored[regions[1] & ~support] = colored[regions[1] & ~support] * .28 + np.array([243, 164, 86]) * .72
    colored[support] = colored[support] * .25 + np.array([195, 242, 115]) * .75
    for tag, frame in zip(("ambient", "relit", "agreement"),(*captures,colored)):
        image = Image.fromarray(frame).transpose(Image.Transpose.ROTATE_90)
        image.resize((1024,384), Image.Resampling.BICUBIC).save(ASSETS / f"story-{tag}.webp", quality=88, method=5)
    print(item["source"], measure["ambient_support"], measure["relit_support"])


if __name__ == "__main__":
    main()
