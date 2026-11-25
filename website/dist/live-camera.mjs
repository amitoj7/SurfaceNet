import {overlap} from "./inspection.mjs";

const TRACE_KEY = "surfacenet.browser-trace.v1";

export function assessLive(window, clearBelow) {
  const {observation} = window.at(-1);
  if (!observation.quality) return {label:"Reacquire", action:"Improve focus and lighting, then center the part."};
  if (observation.region) {
    const matches = window.filter(frame => frame.observation.quality &&
      overlap(observation.region,frame.observation.region,{dx:0,dy:0}) >= .25).length;
    return matches >= 3
      ? {label:"Repeat finding · Hold for review", action:"Segregate the part and confirm the marked area under a second light."}
      : {label:"Checking repeatability",action:"Keep the part in position for another capture."};
  }
  if (window.some(frame => frame.observation.region || frame.observation.score >= clearBelow)) {
    return {label:"Review",action:"Inspect the part manually before release."};
  }
  if (window.length < 4 || window.some(frame => !frame.observation.quality)) {
    return {label:"Checking repeatability",action:"Keep the part in position for another capture."};
  }
  return {label:"No localized finding", action:"Confirm the part against the inspection plan."};
}

export function connectLiveCamera({width, height, samples, infer, modelHash}) {
  const toggle = document.getElementById("camera-toggle");
  const freeze = document.getElementById("camera-freeze");
  const stage = document.getElementById("camera-stage");
  const video = document.getElementById("camera-video");
  const mask = document.getElementById("camera-mask");
  const still = document.getElementById("camera-still");
  const status = document.getElementById("camera-status");
  const evidence = document.getElementById("camera-evidence");
  const form = document.getElementById("camera-record");
  const part = document.getElementById("camera-part");
  const trace = document.getElementById("camera-trace");
  const records = document.getElementById("camera-records");
  const count = document.getElementById("camera-record-count");
  const context = mask.getContext("2d");
  let stream, cameraSession = 0, requestId = 0, timer, active = false;
  let observations = [], latest, frozen;
  let saved = [];
  try { saved = JSON.parse(localStorage.getItem(TRACE_KEY) || "[]"); } catch { /* Storage can be blocked. */ }

  function drawRecords() {
    count.textContent = saved.length ? `(${saved.length})` : "";
    records.replaceChildren(...saved.map(record => {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${record.partId} · ${record.decision} · ${new Date(record.capturedAt).toLocaleString()}`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button";
      button.textContent = "Download evidence";
      button.addEventListener("click",() => {
        const blob = new Blob([JSON.stringify(record,null,2)],{type:"application/json"});
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "surfacenet-live-inspection.json";
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href),1000);
      });
      row.append(label,button);
      return row;
    }));
    if (!saved.length) {
      const empty = document.createElement("li");
      empty.textContent = "No browser inspections saved.";
      records.append(empty);
    }
  }

  function stop(message = "Camera off. Select Start live camera to request access.") {
    cameraSession++;
    active = false;
    clearTimeout(timer);
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    stage.hidden = evidence.hidden = freeze.hidden = form.hidden = true;
    stage.classList.remove("camera-frozen");
    still.hidden = true;
    video.hidden = false;
    latest = frozen = null;
    observations = [];
    toggle.textContent = "Start live camera";
    status.textContent = message;
  }

  function placeMask() {
    if (video.hidden) return;
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return;
    const cropWidth = Math.min(w, h * width / height);
    const cropHeight = cropWidth * height / width;
    const scale = Math.min(video.clientWidth / w, video.clientHeight / h);
    Object.assign(mask.style, {
      width:`${cropWidth * scale}px`, height:`${cropHeight * scale}px`,
      left:`${(video.clientWidth - cropWidth * scale) / 2}px`,
      top:`${(video.clientHeight - cropHeight * scale) / 2}px`,
    });
  }

  async function sampleFrame(session) {
    if (session !== cameraSession) return;
    if (video.readyState >= 2 && video.videoWidth) {
      try {
        const start = performance.now();
        const pixels = samples(video, video.videoWidth, video.videoHeight);
        const capturedAt = new Date().toISOString();
        const result = await infer(--requestId,pixels);
        if (session !== cameraSession) return;
        const observation = result.a;
        const region = observation.quality ? observation.region : null;
        const image = context.createImageData(width,height);
        if (region) for (const i of region.pixels) image.data.set([255,105,52,158],i * 4);
        context.putImageData(image,0,0);
        if (observations.length && start - observations.at(-1).time > 1200) observations = [];
        if (!observation.quality) observations = [];
        observations.push({observation,time:start});
        if (observations.length > 4) observations.shift();
        const decision = assessLive(observations,result.clearBelow);
        latest = {pixels,capturedAt,observation,decision,inferenceMs:result.elapsed};
        evidence.textContent = `${decision.label} · ${region ? observation.shape + " · " + observation.severity + " severity · " : ""}Score ${observation.score.toFixed(3)} · ${Math.round(result.elapsed)} ms inference. ${decision.action}`;
        evidence.hidden = false;
        freeze.hidden = !observation.quality || decision.label === "Checking repeatability";
        if (status.textContent !== "Live camera · local vision") status.textContent = "Live camera · local vision";
        timer = setTimeout(() => sampleFrame(session),Math.max(30,150 - (performance.now() - start)));
      } catch {
        if (session === cameraSession) stop("Local vision could not start. Reload the page or try the Android app.");
      }
    } else {
      timer = setTimeout(() => sampleFrame(session),150);
    }
  }

  toggle.disabled = false;
  drawRecords();
  freeze.addEventListener("click",() => {
    if (!latest) return;
    const snapshot = latest, message = evidence.textContent;
    const defectMaskPng = snapshot.observation.region ? mask.toDataURL("image/png") : null;
    stop("Inspection frozen. Add a part ID to save the record.");
    frozen = {...snapshot,defectMaskPng};
    const stillContext = still.getContext("2d"), image = stillContext.createImageData(width,height);
    for (let i=0;i<width*height;i++) {
      const gray = snapshot.pixels[i];
      image.data.set([gray,gray,gray,255],i * 4);
    }
    stillContext.putImageData(image,0,0);
    stage.classList.add("camera-frozen");
    video.hidden = true;
    stage.hidden = still.hidden = evidence.hidden = form.hidden = false;
    evidence.textContent = message;
    part.value = "";
    frozen.cropPng = still.toDataURL("image/png");
    if (defectMaskPng) stillContext.drawImage(mask,0,0);
  });
  form.addEventListener("submit",event => {
    event.preventDefault();
    const partId = part.value.trim();
    if (!frozen || !partId) return;
    const {observation,decision,cropPng,defectMaskPng,capturedAt,inferenceMs} = frozen;
    const record = {schema:1,partId,capturedAt,source:"Live browser camera",model:"SurfaceNet v2 FP32 single view",
      modelHash,decision:decision.label,action:decision.action,advisorySource:"Browser inspection rules",
      type:observation.shape,severity:observation.severity,score:observation.score,inferenceMs,cropPng,defectMaskPng};
    saved = [record,...saved].slice(0,12);
    try {
      localStorage.setItem(TRACE_KEY,JSON.stringify(saved));
      status.textContent = "Saved in this browser. The last 12 inspections are retained.";
    } catch {
      status.textContent = "Storage is unavailable. Download the evidence before closing this tab.";
    }
    form.hidden = true;
    drawRecords();
    trace.open = true;
  });
  toggle.addEventListener("click",async () => {
    if (active) return stop();
    if (!navigator.mediaDevices?.getUserMedia) {
      status.textContent = "Open this HTTPS page in Chrome or another browser with camera access.";
      return;
    }
    stop();
    const session = ++cameraSession;
    active = true;
    toggle.textContent = "Stop live camera";
    status.textContent = "Waiting for camera permission…";
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio:false, video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},
      });
      if (session !== cameraSession) {
        media.getTracks().forEach(track => track.stop());
        return;
      }
      stream = media;
      stream.getVideoTracks()[0].addEventListener("ended",() => {
        if (session === cameraSession) stop("Camera disconnected. Start it again to continue.");
      });
      context.clearRect(0,0,width,height);
      stage.hidden = false;
      video.srcObject = stream;
      await video.play();
      if (session !== cameraSession) return;
      placeMask();
      status.textContent = "Starting local vision…";
      sampleFrame(session);
    } catch (error) {
      if (session !== cameraSession) return;
      const message = error.name === "NotAllowedError" ? "Camera access was blocked. Allow it in your browser’s site settings and try again." :
        error.name === "NotFoundError" ? "No camera was found on this device." :
        "Could not start the camera. Close other apps using it and try again.";
      stop(message);
    }
  });
  new ResizeObserver(placeMask).observe(stage);
  video.addEventListener("resize",placeMask);
  document.addEventListener("visibilitychange",() => { if (document.hidden && active) stop(); });
  window.addEventListener("pagehide",() => { if (active) stop(); });
}
