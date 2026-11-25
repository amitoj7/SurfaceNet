# SurfaceNet walkthrough

Narration generated locally with the Kokoro-82M Heart voice. App footage was recorded in the Arm64 Android emulator. The rear-camera input is a simulated fixture feed, and the second-light sequence uses an in-app simulation. The held-out results are from the KolektorSDD2 evaluation.

## 00:00 · SurfaceNet

Can a phone flag a defect and tell an operator what to do? Here’s SurfaceNet, working offline.

## 00:05 · Rear camera and clean reference

In Inspect, open Rear camera. A mounted phone can enroll a clean part as its reference. The app checks image quality and asks for a new capture if detail is too low.

## 00:16 · Defect mask and severity

Open the annotated defect. The app highlights the mark, estimates its severity, and sends it to review.

## 00:22 · Second-light verification

For a live hold, the rear light can test whether the same defect pixels persist. Here, the phone replay uses simulated lighting. Drag the divider to compare the registered captures.

## 00:33 · Offline Llama advisory

Tap Ask local Llama. Llama 3.2 runs on the Arm CPU through ExecuTorch. For this mark, it recommends keeping the part segregated and checking pit depth.

## 00:44 · Save and reopen the trace

Enter the part ID and save. Trace log reopens the mask, finding and Llama recommendation. Records stay on the device and can be exported as CSV.

## 00:53 · Device tools

Under Device, compare both vision profiles and export separate live-camera timings.

## 00:59 · Held-out results

Across 1,004 held-out images, four-view INT8 reached 97.45% image AUROC. Both profiles flagged 105 of 110 defects. Four views flagged 35 fewer clean images.

## 01:15 · Physical validation

Next we validate sustained throughput and rear-light repeatability on the physical handset.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) · [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/)
