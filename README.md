# SurfaceNet

SurfaceNet is an offline visual inspection station for a fixed phone mount and a repeatable part fixture. The Arm64 Android app detects and localizes surface defects, checks whether a finding persists under different lighting, and logs the operator's evidence. Llama 3.2 provides operator guidance on-device through ExecuTorch. [Bharat AI-SoC PS5 problem statement](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)

## Start here

| Review | Files |
|---|---|
| Android source | [App code](android/app/src/main) · [Build and model setup](#build-and-reproduce) |
| Vision and evaluation | [Training, calibration and benchmarking code](training) · [Validation results](#validation) |
| Browser evidence lab | [Client and model worker](website/dist) · [Local launch](#evidence-lab-website) |
| Jury materials | [Briefing](JURY.md) · [Full technical report (PDF)](output/pdf/SurfaceNet-PS5-Technical-and-Venture-Report.pdf) · [82-second Android walkthrough (MP4)](output/walkthrough/SurfaceNet-Android-Walkthrough.mp4) · [Transcript](output/walkthrough/SurfaceNet-Android-Walkthrough-Transcript.md) |
| Try the app | [Arm64 debug APK](website/dist/assets/surfaceguard.apk) · [Offline demo setup](#what-to-demonstrate) |

Four-view INT8 reached **97.45% image AUROC** on the 1,004-image KolektorSDD2 test split. The recorded Android walkthrough uses an Arm64 emulator and a simulated second exposure. Sustained throughput and rear-light repeatability on the physical handset are pending.

## PS5 requirement coverage

| Original requirement | SurfaceNet implementation and evidence |
|---|---|
| Detect surface defects, cracks and anomalies | SurfaceNet localizes learned defects with a pixel mask. The clean-reference branch detects unfamiliar changes in a fixed fixture. KolektorSDD2 quantifies defect detection and localization. Crack-like findings are line-shaped regions. |
| Classify type and severity | The app reports line-like, compact or diffuse morphology and estimates severity from mask area and elongation. Category-level accuracy and part-specific reject tolerances require separately labeled and calibrated physical samples. Unknown changes remain unrated. |
| Alert and recommend an action | The live gate presents a hold, review or reacquire state with a reason. Llama 3.2 1B Instruct, quantized with SpinQuant, runs through ExecuTorch on the Arm CPU and selects a permitted procedure from the frozen evidence. |
| Log inspection results | SQLite stores the part ID, decision, finding, mask evidence, lighting result and advisory provenance on the phone. Operators can reopen saved captures and export the metadata as CSV. |
| Operate in real time at the edge without cloud dependence | CameraX keeps the latest frame and schedules vision on a dedicated worker. The Android app processes both vision and advisory locally. Host forward timings and the phone's runtime benchmark measure inference. Sustained camera-to-decision latency and torch repeatability still need the physical iQOO fixture. |

The chosen hardware path is an Arm Android phone with the rear camera and torch. Separate ExecuTorch programs handle INT8 vision and local Llama advisory on the Arm CPU. The jury demo uses the Android build for the required runtime. The evidence lab provides an additional browser walkthrough.

## Evidence lab website

The redesigned lab is available as a private Site. [Open the evidence lab](https://surfaceguard-evidence-lab.amitojxsingh2.chatgpt.site)

Serve a local copy with the following command.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory website/dist
```

Open http://127.0.0.1:8765. The launch page links directly to the live camera, Android setup, and a recorded specimen. The recorded inspection wipe shows ambient and simulated alternate exposure over the same surface. Open **Evidence** for benchmark data and reports. Running the check aligns the alternate to the original in wipe view. The Agreement layer shows which defect pixels each exposure corroborates. The register also includes a thin line, an unconfirmed alert, a low-score clean control and a defect the model failed to localize. The browser runs FP32 ONNX Runtime Web on a worker and processes uploaded images locally. The Android APK uses INT8 ExecuTorch with reference enrollment, repeated-frame decisions and local Llama. Both support single-view and four-view inference. The validation section compares the native INT8 checkpoints and includes the empirical precision–recall curves and a print-ready evidence brief. The evidence JSON includes capture crops, masks, both thresholds, registration measurements and a SHA-256 digest.

### Live camera in the browser

Use **Live camera → Start live camera** and grant camera access. The browser selects the rear camera when available, shows the model crop and runs the single-view FP32 detector locally on successive frames. Use **Stop live camera** to release the camera. On the phone, open the page in an HTTPS browser tab. The APK provides fixture enrollment, torch validation, and ExecuTorch advisory.

The browser checks whether a defect region repeats across the most recent analyzed frames. **Freeze inspection** stops the camera and captures the analyzed crop and mask. Enter a part ID and save. Recent browser records are retained on that device with a JSON evidence export. Browser recommendations come from inspection rules. The Android trace identifies the source of the advisory.

## What to demonstrate

**A repeatable mark is stronger evidence than a single-frame prediction.** SurfaceNet records both the original and rear-light evidence when the operator runs a lighting check. The phone changes the light, reacquires the surface, and measures whether the same region still triggers. A disappearing or inconclusive highlight leaves the original hold in place for review.

1. Install the Arm64 debug APK. Copy `models/llama.pte` and `models/tokenizer.model` to the phone. Import each under **Device → Manage model files**. The app verifies the SHA-256 of both files during import. Use `./scripts/fetch-models.sh` on a development computer if the model files are missing.
2. Turn on airplane mode. Use **Specimens → Annotated defect** to show a real held-out production image, localized mask and morphology. Choose **Ask local Llama** and verify that the response identifies **Llama 3.2 1B · ExecuTorch / XNNPACK**.
3. Mount the phone so the same inspection area fills the preview. With a verified clean part and consistent illumination, choose **Rear camera → Enroll clean** and hold still while it captures eight reference frames. The reference applies to the current live session.
4. Tap **Device → Run benchmark** to compare the two INT8 profiles on the actual handset. Use **Device → Switch profile** to change the detector. Switching stops the camera and clears enrollment for the next session. It exports 120 warmed timings per profile and the initial and final Android thermal statuses. Place a matching test part in the fixture. For a stable hold, choose **Verify with rear light** when supported. The app freezes the original result after checking the alternate illumination. For a regular hold or pass, select **Freeze result**.
5. Enter a part or lot ID and **Save inspection**. From **Trace log**, review the original grayscale ROI and mask and, when available, the second exposure. CSV exports record metadata and advisory provenance.

### Android inspection experience

**Inspect** opens on an annotated production specimen. **Lighting replay** runs both captures through the selected model and freezes a review record. Touch the image or use the exposure slider to compare A and B. **Hide mask** switches between model evidence and the underlying image. Replay uses a simulated alternate exposure and records its source separately. A physical torch check attaches both captures to the existing held finding.

**Trace log** opens saved captures in the same comparison viewer. **Device** groups the vision profile, warm-forward timings and Llama file imports. The focusable controls keep text labels and Android's motion setting controls transitions. The first specimen, inspection decision and tab changes have brief native transitions.

Terminal Industries informed the layout's industrial hierarchy. Lando Norris's limited palette makes primary actions easy to pick out. Lusion v3's responsive hero informed the touch-controlled capture comparison. SurfaceNet uses the actual KolektorSDD2 sample as its visual subject. [Terminal Industries](https://www.awwwards.com/sites/terminal-industries) · [Lando Norris](https://www.awwwards.com/sites/lando-norris) · [Lusion v3](https://www.awwwards.com/sites/lusion-v3) · [Android accessibility guidance](https://developer.android.com/guide/topics/ui/accessibility/views/apps-views)

`SurfaceGuard-Replay.html` is the original browser replay workflow. Its historical replay and v1 benchmark are preserved. The Android app contains the learned inspection and the local Llama path.

## Engineering design

| Stage | Implementation |
|---|---|
| Camera | CameraX rear camera. Rotation-aware center ROI, 96 × 256 luminance samples. Latest-frame backpressure and a 100 ms sampling interval cap the live analysis request rate. |
| Learned surface inspection | SurfaceNet v2 couples a multi-resolution decoder and input skip with an image-level recognition head. Training uses photometric and specular-streak augmentation, focal BCE, Dice and hard-background mining. ExecuTorch 1.5.0 delegates compatible ops to XNNPACK on the Arm CPU. The live Android path uses calibrated INT8. The selectable four-view program tests both input axes and re-registers model coordinates in one batched call. |
| Unknown surface change | Eight clean frames estimate a per-pixel mean and variance. A bounded translation search and robust brightness-offset compensation align the reference. Connected residuals can reveal a new change even when the learned score is low. |
| Evidence gate | Poor exposure or poor alignment triggers reacquisition. A live hold requires three spatially agreeing findings in four observations. A live pass requires enrollment and four observations without a finding, all below the calibrated clear threshold. Intermediate scores require operator review. A still always requires operator review or reacquisition. |
| Active verification | The rear light changes illumination. The app waits for exposure to settle and registers high-frequency detail over a ±4 px search. Usable frames require correlation ≥0.58, motion short of the search boundary, and an aligned 75th-percentile luminance change of at least seven levels. Three same-branch matches across five usable frames mark a finding as persistent. Each match requires box IoU ≥0.25 and at least 60% of each aligned mask within one pixel of the other mask. Captures and alignment measurements are saved with the hold. |
| Type and severity | Learned-mask geometry provides line-like, compact and diffuse morphology. Area fraction and elongation provide an ordinal severity proxy. An unknown mark remains unrated. Crack depth and tolerances require physical metrology. |
| Advisory | Llama 3.2 1B Instruct, 4-bit SpinQuant, runs separately through ExecuTorch/XNNPACK. It receives the frozen evidence and chooses one of two allowed procedures. If the lighting check is inconclusive, both procedures keep the part segregated for visual confirmation. App logic validates the choice and owns the disposition. Invalid generation falls back to the documented station procedure. |
| Audit | Local SQLite records link the part ID, finding, severity proxy, mask, confirmation count, lighting result, measured analysis time, vision checkpoint and advisory source. The app requests camera permission and uses Android’s document picker for imports and CSV export. |

The distinctive component is an on-device lighting experiment with an explicit recapture path. Low optical change, unreliable registration or poor exposure produces inconclusive evidence. An inconclusive pair retains the original hold. EfficientAD motivates computationally small industrial models. PNI motivates using spatial context for nominal appearance. Mixed supervision motivates the joint classification and segmentation loss. MVTec AD 2 motivates evaluating illumination as an independent variable. SurfaceNet implements its own model and inspection protocol. [EfficientAD paper](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) · [PNI paper](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html) · [KolektorSDD2 paper](https://arxiv.org/abs/2104.06064) · [MVTec AD 2 lighting protocol](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

## Build and reproduce

Prerequisites are JDK 17, Android SDK Platform 35 and Android Build Tools 35.0.0. The APK targets Android 14 and supports Arm64 Android 9 or later.

```sh
./scripts/fetch-models.sh --runtime-only
cd android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

For retraining, download and unzip the [official KolektorSDD2 archive](https://www.vicos.si/resources/kolektorsdd2/) and pass the folder containing `train/` and `test/` to the trainer. Use Python 3.12 and a matching ExecuTorch/PyTorch environment.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/website_requirements.txt
python training/train.py /path/to/KolektorSDD2 --architecture v2 --epochs 40 --checkpoint checkpoints/surface-v2.pt
python training/benchmark_edge.py /path/to/KolektorSDD2 checkpoints/surface-v2.pt \
  --fp32 android/app/src/main/assets/surface.pte --output checkpoints/edge \
  --release-to android/app/src/main/assets
python training/publish_evidence.py /path/to/KolektorSDD2 checkpoints/surface-v2.pt \
  --java-home "$JAVA_HOME" --edge-report checkpoints/edge/edge-benchmark.json \
  --edge-pte checkpoints/edge/surface-int8.pte
python training/calibrate_review.py /path/to/KolektorSDD2
```

The training command exports the PTE, checks numerical parity against eager PyTorch, and writes the operating-point statistics and example IDs to `android/app/src/main/assets/vision.json`. The threshold comes from the official training split's held-out validation negatives. The official test split is held out from model selection and threshold calibration. Pixel AP is approximate, using 2,048 score bins, at the model's resized resolution.

## Validation

SurfaceNet v2 has 124,455 parameters and a 502.1 KiB ExecuTorch program. It was selected on validation AP, which rose from 88.22% to 91.40%. The v2 checkpoint comes from epoch 29 of a 40-epoch schedule. The image-alert threshold is 0.571171. Export parity was checked against eager PyTorch on the same sample for the native PTE and the browser ONNX graph.

| Official KolektorSDD2 test | SurfaceNet v1 | SurfaceNet v2 |
|---|---:|---:|
| Images | 1,004 | 1,004 |
| Image AUROC | 95.57% | 97.11% |
| Image average precision | 87.88% | 90.54% |
| Defect recall at the deployed threshold | 79.09% (87/110) | 80.91% (89/110) |
| False alerts on clean images | 1.79% (16/894) | 1.45% (13/894) |
| Pixel average precision | 65.02% | 72.60% |

For the paired study, each original test image was paired with a deterministic exposure gain, luminance offset and translation of up to two pixels. Every third alternate also received a localized synthetic highlight. With box overlap alone, the Java gate confirmed **81 of 89 initial true alerts** and **9 of 13 initial false alerts**. Adding bidirectional pixel support confirmed **78 of 89 true alerts** and **5 of 13 false alerts**. Four additional false alerts and three additional true alerts stayed in review. The vision checkpoint and alert threshold were held fixed for this comparison. The study represents a controlled optical perturbation with one alternate per image. Factory release decisions and the camera's five-frame capture sequence require handset measurements.

### Android INT8 release

Static per-channel post-training quantization used 384 images from the training partition. INT8 had to satisfy the preset validation limits on image AP, recall, file size, and host p95 before it could ship. Both precision modes use separately calibrated thresholds and the same annotations. [ExecuTorch XNNPACK quantization](https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-quantization.html)

| SurfaceNet v2 | FP32 | INT8 |
|---|---:|---:|
| ExecuTorch program | 502.1 KiB | 153.4 KiB |
| Warm p50 on Apple M3 | 1.729 ms | 0.931 ms |
| Warm p95 on Apple M3 | 1.957 ms | 1.103 ms |
| Official test image AUROC | 97.11% | 97.29% |
| Defects detected | 89/110 | 91/110 |
| False alerts on clean | 13/894 | 13/894 |
| Pixel AP | 72.60% | 73.45% |

The Apple M3 experiment alternated profile order across 120 warmed forwards per precision and excluded camera capture, preprocessing, the lighting gate, and Llama. In the INT8 simulated paired study, box overlap confirmed 81/91 true initial alerts and 6/13 false initial alerts. Adding pixel support confirmed 74/91 and 4/13 respectively. Those seven additional true alerts and two additional false alerts remain for operator review. The primary lab examples use FP32 browser inference and carry their own benchmark. The handset benchmark alternates the two INT8 profiles on two bundled specimens, after warm-up and with camera analysis paused.

### Four-view INT8 profile

Single view is the default. Four views adds flips across both axes and their combination in a single batched forward call. The graph inversely transforms the four dense maps, averages logits and includes a validated image-context term. It uses the existing trained checkpoint with shared weights and is available in Android and the browser.

| Official KolektorSDD2 test · Android INT8 | Single view | Four views |
|---|---:|---:|
| Image AUROC | 97.29% | 97.45% |
| Image average precision | 91.48% | 92.36% |
| Pixel average precision | 73.45% | 69.16% |
| Defect alerts | 91/110 | 96/110 |
| Clean alerts | 13/894 | 24/894 |
| Defect images sent to alert or review | 105/110 | 105/110 |
| Clean images sent to alert or review | 165/894 | 130/894 |
| Apple M3 warm forward p95 | 1.06 ms | 6.63 ms |
| Program size | 153.4 KiB | 159.0 KiB |

On training validation, the selected INT8 profile scored 94.39% image AP versus 90.82% for single view. Candidate selection used two context weights and fixed minimums for validation image AP, pixel AP, alert recall and a maximum program size. A host p95 limit of 15 ms also applied before test evaluation. Test localization AP declined, and the active alert cutoff flagged more clean images. The separate review cutoff reduced total clean alerts plus reviews by 35, keeping the same 105 defective images routed to an operator. Measurements are per image. The test did not exercise the live phone disposition.

Reproduce the comparison from the v2 checkpoint and a directory containing the original single-view INT8 program and metadata.

```sh
python training/benchmark_reflection.py /path/to/KolektorSDD2 checkpoints/surface-v2.pt checkpoints/reflection
python training/publish_evidence.py /path/to/KolektorSDD2 checkpoints/surface-v2.pt \
  --java-home "$JAVA_HOME" --android checkpoints/reflection --assets checkpoints/reflection/web \
  --edge-report checkpoints/reflection/edge-benchmark.json --edge-pte checkpoints/reflection/surface-int8.pte
python training/calibrate_review.py /path/to/KolektorSDD2 --android checkpoints/reflection \
  --onnx checkpoints/reflection/web/surface.onnx --output checkpoints/reflection/web/review-calibration.json
```

The four-view publication uses `surface-int8.pte` and `edge-local.json` from the reflection checkpoint directory under the Android asset names `surface-reflection-int8.pte` and `edge-reflection.json`. Browser artifacts use `surface-reflection.onnx`, `reflection-browser.json`, `reflection-edge.json` and `reflection-review.json` from the reflection website export. The shared specimen register uses the single-view study's images to allow direct comparison.

### Single-view calibrated review band

The Android INT8 alert cutoff remains 0.45331. A strict clear cutoff of 0.04878 comes from the second-lowest of 49 defective scores in the held-out training validation split, using a 5% lower-tail rank and conservative tie handling. An enrolled four-frame window can pass only when every score is below this cutoff and no localized or reference finding is present. The browser calibrates its FP32 ONNX graph separately and exposes the same review action.

| INT8 single-frame test disposition | Alert cutoff only | With review band |
|---|---:|---:|
| Defective images scoring clear | 19/110 | 5/110 |
| Defective images sent to review | 0/110 | 14/110 |
| Clean images sent to review | 0/894 | 152/894 |
| Clean images with an alert | 13/894 | 13/894 |

This is a score-only ablation. The calibration split also selected the detector checkpoint. The official test split was reserved for evaluation. The live gate also uses temporal, fixture and reference evidence. Device and factory calibration require captures from the intended fixture.

We also finetuned SurfaceNet v2 for 24 epochs with paired lighting perturbations, a symmetric pixel-consistency loss and an anchor to the original model. The best validation checkpoint raised recall from 39/49 to 40/49 at the same false-alert count. Image AP fell from 91.40% to 89.83% and pixel AP fell from 73.30% to 72.95%. The release keeps the original checkpoint. The held-out official test split was unused in that model selection.

The signed Arm64 APK was verified on an API 35 Arm64 emulator in airplane mode with Wi-Fi disabled. The operator imported the pinned Llama model and tokenizer through the app, inspected an annotated defect, generated local advice, and reopened the saved record with Llama provenance. The workflow returned to vision and ran the clean control in the same session. The host advisory check exercised clean, persistent-line, inconclusive-light, compact-mark and clean-still prompts. The Java checks exercised enrollment, registration, novel local changes and temporal decisions. Physical iQOO validation is pending.

The camera luma domain and defect mix can differ from the training data.

For a comparable fixture study, capture clean, scratched and glare-only passes with a fixed phone mount. Assign entire physical parts to one test split. Compare single-frame model alerts to the same sequence with the reference and temporal gate, then test rear-light verification on held parts. Report false holds per clean part, recall per defective part, reacquisition rate, rear-light outcomes and median/p95 latency after warm-up. Measure the first Llama response and steady-state responses separately. The on-screen latency is local vision analysis time, excluding sensor acquisition, and summarizes the latest 120 observations.

## Dependencies and attribution

- The KSDD2 specimens are grayscale resized derivatives of Kolektor Group images supplied through the University of Ljubljana's ViCoS Lab. KSDD2 is CC BY-NC-SA 4.0. The bundled KSDD2-derived materials are provided for the non-commercial challenge prototype. [Dataset and citation](https://www.vicos.si/resources/kolektorsdd2/) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- The separately downloaded quantized model and matching tokenizer follow the Llama 3.2 Community License and usage terms. Built with Llama. [Pinned ExecuTorch model](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/tree/911221e0f49d3354cbd4697a9c09feee25cadec7)
- The runtime is the official XNNPACK Android AAR. Camera acquisition uses Jetpack CameraX. [ExecuTorch Android](https://docs.pytorch.org/executorch/1.5/using-executorch-android.html) · [CameraX analysis](https://developer.android.com/media/camera/camerax/analyze)

- Website inference uses ONNX Runtime Web 1.22.0. The bundled fonts are Barlow Condensed, Instrument Serif, Space Grotesk and IBM Plex Mono. React Bits Splash Cursor responds to pointer and touch in the specimen hero and Android download section. It respects reduced-motion settings. [ONNX Runtime Web deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html) · [React Bits Splash Cursor](https://reactbits.dev/animations/splash-cursor)
