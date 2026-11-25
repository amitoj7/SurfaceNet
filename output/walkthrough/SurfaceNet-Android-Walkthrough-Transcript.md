# SurfaceNet walkthrough

Narration generated locally with the Kokoro-82M Heart voice. App footage was recorded in the Arm64 Android emulator. The rear-camera input is a simulated fixture feed, and the second-light sequence uses an in-app simulation. The held-out results are from the KolektorSDD2 evaluation.

## 00:00 · From defect to decision

Can a phone flag a defect and tell an operator what to do? Here’s SurfaceNet, working offline.

## 00:05 · The factory problem

On a production line, a scratch and a reflection can look alike. Miss the scratch and a bad part can ship. Hold the good part and an operator has to inspect it again.

## 00:14 · Rear camera and reference

SurfaceNet turns a mounted phone into an offline inspection station. Open Rear camera and enroll a verified clean part as a reference. If the image lacks detail, the app asks for a new capture.

## 00:26 · Defect mask and severity

Here’s an annotated production specimen from our held-out test set. SurfaceNet outlines the mark and estimates severity from its shape and size.

## 00:34 · Check under a second light

For a held part, the operator can check the same mark under a second light. The app aligns the two captures and checks whether the same defect pixels persist. This replay simulates the second exposure. An inconclusive check keeps the part on hold.

## 00:47 · Local Llama advisory

Tap Ask local Llama. Llama 3.2 runs on the Arm CPU through ExecuTorch. Here it recommends keeping the part segregated and asking quality engineering to verify pit depth.

## 00:59 · Save and review the record

Enter a part ID and save the inspection. Trace log reopens the mask, decision and advice. Export the record as CSV.

## 01:07 · Measure on the device

Under Device, compare the vision profiles and export live camera timings.

## 01:11 · Held-out performance

On 1,004 held-out images, four-view INT8 reached 97.45% image AUROC. Both profiles sent 105 of 110 defects to alert or review. Four views reduced clean referrals from 165 to 130. That’s 21% fewer.

## 01:31 · Offline from inspection to record

From the first inspection to a saved decision, the whole workflow runs offline.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) · [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/)
