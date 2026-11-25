# SurfaceNet jury briefing

## One-minute pitch

SurfaceNet turns an Arm Android phone into an offline industrial inspection station. A held finding can trigger a second exposure with the rear torch. The app checks the lighting change, registers the captures and measures whether the same defect pixels persist. Llama 3.2 runs through ExecuTorch on the same phone and selects an allowed operator procedure from the frozen evidence.

The PS5 demo starts on Android. Show the defect mask, morphology and estimated severity, then generate a local operator recommendation and reopen the saved record. The browser walkthrough and accuracy plots support the native demo. [PS5 requirement coverage](README.md#ps5-requirement-coverage)

## Jury access and competition readiness

The browser path demonstrates live surface segmentation, repeated-frame review, rule-based operator guidance, and evidence capture with a part ID. Judges can export a JSON record containing the crop, mask, model hash, and decision. The required Llama family model and ExecuTorch advisory run in the Android APK. Have both model files imported on the demonstration phone before the session. A private Sites URL needs a verified external viewer grant or a public HTTPS deployment before it can be used as a jury link. Verify it on a separate phone with a fresh browser session, including camera permissions and APK download.

The main evidence gap is a blind physical run. Use distinct labeled clean and defective parts in the same fixture. Preserve a small setup set for fixture and exposure adjustments. After locking the profile and thresholds, collect one terminal Android record per test part. Save the label sheet separately as `part,truth`, with `truth` equal to `clean` or `defect`. Score the trace without discarding reviews or missing records.

```sh
python3 training/score_fixture.py part-labels.csv surfacenet-inspections.csv
```

In **Device → Export live timings**, save a separate JSON after the camera has been analyzing parts for at least a minute. Report analyzer-to-decision latency and decisions per second. Report sensor-to-decision latency only when the device reports a compatible sensor clock. Repeat the test with altered lighting and score actual rear-light comparisons separately from simulated relighting. The results table below uses the official image split and host inference measurements.

EfficientAD is a relevant latency and anomaly-detection comparator for a future matched benchmark on the same images and device. Anomalib provides a reproducible implementation. MVTec AD 2 motivates holding out a physical illumination condition during validation. Terminal Industries informed the presentation's emphasis on showing the operational workflow first. [EfficientAD paper](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) · [Anomalib repository](https://github.com/open-edge-platform/anomalib) · [MVTec AD 2 paper](https://arxiv.org/abs/2503.21622) · [Terminal Industries case study](https://www.rejouice.com/work/terminal-industries)

The four-view INT8 profile scores **97.45% image AUROC** and **92.36% image average precision** on the official 1,004-image KolektorSDD2 test split. It alerts on **96 of 110 defects**. With the calibrated review band, both profiles route **105 of 110 defects** to alert or review. The four-view version sends **130 clean images** to an operator, down from **165** for the single-view version.

## Results to show

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

All image counts come from single-frame scoring. Live disposition also uses enrollment, reference alignment and temporal evidence. Runtime figures come from 120 warmed forwards per profile on the host CPU. Camera-to-decision latency, torch repeatability and sustained thermal behavior remain unmeasured on the iQOO.

## Live demo

1. Open Android in airplane mode with the pinned Llama files installed. Run **Inspect → Specimens → Annotated defect**, show the mask and severity, choose **Ask local Llama**, and reopen the saved evidence in **Trace log**. The offline path passed on an Arm64 Android emulator with Wi-Fi disabled.
2. On the iQOO, run **Device → Run benchmark** and inspect a fixtured part. Show a stable hold and trigger **Verify with rear light**. Use **Inspect → Lighting replay** to demonstrate the comparison when the physical fixture is unavailable. Its alternate capture is simulated.
3. Open [the evidence lab](https://surfaceguard-evidence-lab.amitojxsingh2.chatgpt.site). Drag the exposure wipe, run the lighting check and open **Agreement**. In **Validation**, move the recall slider to compare precision at the same target recall for both Android INT8 profiles.

## Defensible answers

| Question | Answer |
|---|---|
| What changes with four views? | The model flips one exposure across both image axes in a four-input batch. Inverse transforms restore the pixel coordinates before the model averages the logits. The same weights process all four orientations, with a context contribution selected on validation. |
| What changes with a second light? | The torch provides a separate physical observation. On 1,004 simulated pairs, the four-view pixel gate confirmed 88 of 96 initial true alerts and 11 of 24 false alerts. Other initial alerts stay under review. |
| How was the model selected? | The four-view candidates scored 94.05% and 94.39% image AP on training validation, versus 90.82% for the existing INT8 model. The release criteria also considered pixel AP, defect recall, program size and host latency. The official test split measured the selected profile. |
| Which profile should be shown? | Single view is the default and had better pixel AP and lower p95. Four views had stronger image ranking and higher alert recall in this experiment. The browser and app can run either. |

The visual direction takes cues from industrial storytelling in Terminal Industries and the interaction-led opening of Lusion v3. SurfaceNet uses an original optical bench built around an actual specimen and an interactive precision–recall record. [Terminal Industries on Awwwards](https://www.awwwards.com/sites/terminal-industries) · [Lusion v3 on Awwwards](https://www.awwwards.com/sites/lusion-v3)

[Print-ready technical brief](https://surfaceguard-evidence-lab.amitojxsingh2.chatgpt.site/assets/technical-report.html)
