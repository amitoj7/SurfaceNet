# SurfaceNet jury briefing

## Project overview

SurfaceNet turns an Arm Android phone into an offline industrial inspection station. A held finding can trigger a second exposure with the rear torch. The app checks the lighting change, registers the captures and measures whether the same defect pixels persist. Llama 3.2 runs through ExecuTorch on the same phone and selects an allowed operator procedure from the frozen evidence.

The Android workflow covers detection, morphology and severity estimates, operator guidance, and evidence capture. [Source code](README.md#source-code) · [PS5 requirement coverage](README.md#ps5-requirement-coverage)

## Verification status

The offline Android workflow was verified on an Arm64 emulator with Wi-Fi disabled. The walkthrough shows native app footage and a simulated second exposure. The official image split and the host inference benchmark provide the quantitative results below. [Android walkthrough](output/walkthrough/SurfaceNet-Android-Walkthrough.mp4) · [Technical and venture report](output/pdf/SurfaceNet-PS5-Technical-and-Venture-Report.pdf)

## Physical validation protocol

Use distinct labeled clean and defective parts in the same fixture. Set aside a setup set for fixture and exposure adjustments. After locking the profile and thresholds, collect one terminal Android record per test part. Save the label sheet separately as `part,truth`, with `truth` equal to `clean` or `defect`. Include reviews and missing records when scoring the trace.

```sh
python3 training/score_fixture.py part-labels.csv surfacenet-inspections.csv
```

In **Device → Export live timings**, save a separate JSON after the camera has been analyzing parts for at least a minute. Report analyzer-to-decision latency and decisions per second. Report sensor-to-decision latency only when the device reports a compatible sensor clock. Repeat the test with altered lighting and score physical rear-light comparisons separately from simulated relighting.

## Benchmark results

| Official test or measured cost | Single-view INT8 | Four-view INT8 |
|---|---:|---:|
| Image AUROC | 97.29% | **97.45%** |
| Image average precision | 91.48% | **92.36%** |
| Pixel average precision | **73.45%** | 69.16% |
| Defects with an alert | 91/110 | **96/110** |
| Clean images with an alert | **13/894** | 24/894 |
| Defects routed to alert or review | 105/110 | 105/110 |
| Clean images routed to alert or review | 165/894 | **130/894** |
| Warm forward p95 on Apple M3 | **1.06 ms** | 6.63 ms |
| Program size | **153.4 KiB** | 159.0 KiB |

The test split contains 1,004 images. Counts come from single-frame scoring. Live disposition also uses enrollment, reference alignment and temporal evidence. Runtime figures come from 120 warmed forwards per profile on the host CPU. Camera-to-decision latency, torch repeatability and sustained thermal behavior still require measurements on the physical handset.

## Live demo

1. [Install the Arm64 build and import the pinned Llama files](README.md#build-and-reproduce). Enable airplane mode.
2. Open **Inspect → Specimens → Annotated defect**. Inspect the mask and severity, choose **Ask local Llama**, save with a part ID, and reopen the evidence in **Trace log**.
3. On a mounted phone, open **Rear camera**, select **Enroll clean** with a verified clean part, then inspect a matching test part. On a stable hold, select **Verify with rear light**. **Inspect → Lighting replay** demonstrates the comparison using a simulated alternate capture.
4. Open **Device → Run benchmark** to compare the vision profiles. Run the [browser evidence lab locally](README.md#browser-evidence-lab) to explore precision–recall and compare the two exposures in the specimen replay.

## Design decisions

| Question | Answer |
|---|---|
| What changes with four views? | The model flips one exposure across both image axes in a four-input batch. Inverse transforms restore the pixel coordinates before the model averages the logits. The same weights process all four orientations, with a context contribution selected on validation. |
| What changes with a second light? | The torch provides a separate physical observation. On 1,004 simulated pairs, the four-view pixel gate confirmed 88 of 96 initial true alerts and 11 of 24 false alerts. Other initial alerts stay under review. |
| How was the model selected? | The four-view candidates scored 94.05% and 94.39% image AP on training validation, versus 90.82% for the existing INT8 model. The release criteria also considered pixel AP, defect recall, program size and host latency. The official test split measured the selected profile. |
| Which profile should be shown? | Single view is the default and had better pixel AP and lower p95. Four views had stronger image ranking and higher alert recall in this experiment. The browser and app can run either. |

Detailed calibration, ablations, and reproduction commands are in the engineering notes. [Engineering and reproducibility](docs/engineering.md)
