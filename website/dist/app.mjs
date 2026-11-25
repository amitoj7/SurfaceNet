import {height, width} from "./inspection.mjs";
import {revealFinding, revealSpecimen} from "./motion.mjs";
import {connectLiveCamera} from "./live-camera.mjs";
import {renderCurve} from "./performance.mjs";

const el = (id) => document.getElementById(id);
const write = (id, value) => { el(id).textContent = value; };
const asset = (name) => new URL("./assets/" + name, import.meta.url).href;
const format = (value) => (value * 100).toFixed(2) + "%";
const scoreText = (value) => value.toFixed(3);
let shelf, report, reflectionMetrics, comparison, calibration, reflectedCalibration, worker, current, result;
const pending = new Map();
let mode = "raw";
let layout = "wipe";
let ticket = 0;

async function json(name) {
  const response = await fetch(asset(name));
  if (!response.ok) throw new Error(name);
  return response.json();
}

function loadImage(source, local = false) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Image could not be decoded"));
    image.src = local ? source : asset(source);
  });
}

function samples(image, sourceWidth = image.width, sourceHeight = image.height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", {willReadFrequently:true});
  const regionWidth = Math.min(sourceWidth, sourceHeight * width / height);
  const regionHeight = regionWidth * height / width;
  context.drawImage(image, (sourceWidth - regionWidth)/2, (sourceHeight - regionHeight)/2,
    regionWidth, regionHeight, 0, 0, width, height);
  const source = context.getImageData(0, 0, width, height).data;
  const pixels = new Uint8Array(width * height);
  for (let i = 0; i < pixels.length; i++) {
    const offset = i * 4;
    pixels[i] = (77 * source[offset] + 150 * source[offset + 1] + 29 * source[offset + 2] + 128) >> 8;
  }
  return pixels;
}

function imagesFromState() {
  if (!current) return;
  paint("ambient", current.a, mode === "truth" ? current.truthA : result?.a.region,
    mode, result?.agreement?.ambientMatches);
  paint("relit", current.b, mode === "truth" ? current.truthB : result?.b?.region,
    mode, result?.agreement?.relitMatches);

  const aligned = result?.pair?.status === "USABLE";
  for (const [id, stage] of [["wipe-ambient","ambient"],["wipe-relit","relit"]]) {
    const canvas = el(id), context = canvas.getContext("2d");
    context.clearRect(0,0,canvas.width,canvas.height);
    if (stage === "ambient" || current.b) {
      const x = aligned && stage === "relit" ? -result.pair.dy * canvas.width / height : 0;
      const y = aligned && stage === "relit" ? result.pair.dx * canvas.height / width : 0;
      context.drawImage(el(stage),x,y,canvas.width,canvas.height);
    }
  }
  write("plate-registration",aligned && layout === "wipe" ? "B aligned to A" : "Captured position");
}

function setLayout(next) {
  layout = next;
  el("plate").dataset.layout = next;
  el("split-mode").setAttribute("aria-pressed",next === "split");
  el("wipe-mode").setAttribute("aria-pressed",next === "wipe");
  el("wipe-panel").hidden = next !== "wipe";
  el("split-panel").hidden = next !== "split";
  imagesFromState();
  revealSpecimen();
}

function paint(id, pixels, overlay, view, matched) {
  const canvas = el(id), context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!pixels) return;
  const specimen = document.createElement("canvas");
  specimen.width = width;
  specimen.height = height;
  const target = specimen.getContext("2d");
  const image = target.createImageData(width, height);
  const mask = new Uint8Array(width * height);
  if (overlay?.pixels && (view === "model" || view === "agreement")) for (const i of overlay.pixels) mask[i] = 1;
  if (view === "agreement" && matched) for (const i of matched) mask[i] = 2;
  if (view === "truth" && overlay) for (let i = 0; i < mask.length; i++) mask[i] = overlay[i] > 127 ? 2 : 0;
  const warm = [255,105,52], cool = [195,242,115];
  for (let i = 0; i < mask.length; i++) {
    const offset = i*4, value = pixels[i], strength = mask[i] ? .70 : 0;
    const color = mask[i] === 2 ? cool : warm;
    for (let channel = 0; channel < 3; channel++) image.data[offset+channel] = value*(1-strength) + color[channel]*strength;
    image.data[offset+3] = 255;
  }
  target.putImageData(image,0,0);
  if (overlay?.pixels && view === "model") {
    target.strokeStyle = "#ffae85";
    target.lineWidth = .75;
    target.strokeRect(overlay.x0-1.5,overlay.y0-1.5,overlay.x1-overlay.x0+4,overlay.y1-overlay.y0+4);
  }
  context.save();
  context.translate(0, canvas.height);
  context.rotate(-Math.PI/2);
  context.drawImage(specimen, 0,0,canvas.height,canvas.width);
  context.restore();
}

function status(title, state, reason) {
  write("verdict",title);
  el("verdict").dataset.state = state;
  write("explanation",reason);
}
function reset() {
  result = null;
  el("run-feedback").hidden = true;
  phase("ready");
  el("agreement-toggle").disabled = true;
  for (const id of ["ambient-score","relit-score","surface-scores","shape","severity","registration","light","overlap"]) write(id,"—");
  el("local-trace").disabled = true;
  status("Ready to inspect","","Run the lighting check to evaluate the selected images.");
  write("runtime","Ready for local inference");
}
function chooseLayer(next) {
  mode = next;
  for (const button of el("layers").querySelectorAll("button")) button.setAttribute("aria-pressed", button.dataset.layer === mode);
  el("agreement-legend").hidden = mode !== "agreement";
  imagesFromState();
}
function selected(id) {
  for (const button of el("cases").querySelectorAll("button")) button.setAttribute("aria-pressed", button.dataset.case === id);
}
function setCase(details) {
  current = details;
  selected(details.id);
  write("case-title",details.label);
  write("case-source",details.source);
  write("case-note",details.note);
  write("capture-origin",details.id === "upload" ? "Local upload" : "KolektorSDD2 · Simulated second exposure");
  write("ambient-caption",details.ambientCaption);
  write("relit-caption",details.relitCaption);
  el("truth-toggle").disabled = !details.truthA;
  el("split-mode").disabled = !details.b;
  el("wipe-panel").dataset.single = String(!details.b);
  if (!details.b && layout !== "wipe") setLayout("wipe");
  el("run").disabled = false;
  el("vision-profile").disabled = false;
  chooseLayer("raw");
  revealSpecimen({replace:true});
  el("lab-shell").setAttribute("aria-busy","false");
}

async function selectPreset(details) {
  const turn = ++ticket;
  reset();
  el("run").disabled = true;
  el("vision-profile").disabled = true;
  el("lab-shell").setAttribute("aria-busy","true");
  selected(details.id);
  try {
    const [a,b,truthA,truthB] = await Promise.all(
      ["ambient","relit","truth","relit_truth"].map(async (part) => samples(await loadImage(details.id+"-"+part+".png"))));
    if (turn !== ticket) return;
    setCase({...details, a, b, truthA, truthB, source:"KolektorSDD2 · " + details.source,
      ambientCaption:"Original held-out image", relitCaption:"Simulated relight and shift"});
  } catch {
    if (turn !== ticket) return;
    status("Specimen unavailable","review","Choose another specimen or load images from this device.");
    el("lab-shell").setAttribute("aria-busy","false");
  }
}

async function upload(files) {
  const turn = ++ticket;
  reset();
  el("run").disabled = true;
  el("vision-profile").disabled = true;
  el("lab-shell").setAttribute("aria-busy","true");
  try {
    if (files.length < 1 || files.length > 2) throw new Error("Select one image or an ambient and alternate image pair.");
    if (files.some(file => file.size > 25 * 1024 * 1024)) throw new Error("Each image must be 25 MB or smaller.");
    const [a,b] = await Promise.all(files.map(async file => {
      const url = URL.createObjectURL(file);
      try { return samples(await loadImage(url,true)); } finally { URL.revokeObjectURL(url); }
    }));
    if (turn !== ticket) return;
    setCase({id:"upload",a,b,label:"Uploaded specimen",source:"Local image" + (b ? " pair" : ""),
      note:"The browser center-crops the selected files to the inspection region. Both images stay in this browser.",
      ambientCaption:files[0].name, relitCaption:files[1]?.name || "Load a second capture to verify"});
  } catch (error) {
    if (turn !== ticket) return;
    status("Image input unavailable","review",error.message || "Choose a supported image.");
    el("lab-shell").setAttribute("aria-busy","false");
    el("run").disabled = !current;
    el("vision-profile").disabled = !current;
  }
}

function phase(name) {
  const meter = el("workflow");
  meter.dataset.phase = name;
  el("lab-shell").dataset.phase = name;
  el("inspection-measurements").hidden = name !== "result";
  for (const step of meter.querySelectorAll("li")) {
    if (step.dataset.step === name) step.setAttribute("aria-current","step");
    else step.removeAttribute("aria-current");
  }
}
function inspect(id, sample, profile, config) {
  if (!worker) {
    worker = new Worker(new URL("./model-worker.mjs",import.meta.url),{type:"module"});
    worker.onmessage = ({data}) => {
      const run = pending.get(data.id);
      if (!run) return;
      if (data.stage) {
        if (data.id === ticket) {
          phase(data.stage);
          const names = {loading:"Loading local vision model…",segment:"Segmenting the captures…",compare:"Aligning and comparing defect masks…"};
          write("runtime",names[data.stage]);
        }
        return;
      }
      pending.delete(data.id);
      if (data.error) run.reject(new Error(data.error));
      else run.resolve(data.result);
    };
    worker.onerror = event => {
      for (const run of pending.values()) run.reject(new Error(event.message || "Worker unavailable"));
      pending.clear();
      worker.terminate();
      worker = null;
    };
  }
  return new Promise((resolve,reject) => {
    pending.set(id,{resolve,reject});
    worker.postMessage({id,profile,frames:{a:sample.a,b:sample.b},threshold:config.alert_above,clearBelow:config.clear_below});
  });
}

async function run() {
  if (!current) return;
  const turn = ++ticket, sample = current;
  const profile = el("vision-profile").value;
  const config = (profile === "reflection" ? reflectedCalibration : calibration).profiles.fp32;
  reset();
  el("run").disabled = true;
  el("vision-profile").disabled = true;
  el("lab-shell").setAttribute("aria-busy","true");
  status("Checking evidence","","Running the segmenter and comparing the available captures.");
  try {
    const next = await inspect(turn,sample,profile,config);
    if (turn !== ticket) return;
    const {a,b,pair,agreement,assessment,elapsed} = next;
    result = {...next,profile,config};
    status(assessment.title,assessment.state,assessment.reason);
    write("run-feedback", assessment.title + " ↓");
    el("run-feedback").hidden = false;
    phase("result");
    write("ambient-score", scoreText(a.score));
    write("relit-score", b ? scoreText(b.score) : "—");
    write("surface-scores", b ? `${scoreText(a.score)} / ${scoreText(b.score)}` : scoreText(a.score));
    write("shape", a.region ? a.shape : b?.region ? b.shape : a.shape);
    write("severity",a.region ? a.severity : b?.severity || "—");
    write("registration",pair ? pair.correlation.toFixed(2) + " · " + pair.dx + "," + pair.dy + " px" : "—");
    write("light",pair ? "Δ" + pair.lightChange + " levels" : "—");
    write("overlap",agreement ? Math.round(agreement.ambient*100)+"% / "+Math.round(agreement.relit*100)+"%" : "—");
    const {alert_above, clear_below} = config;
    write("runtime",`WASM worker · ${elapsed.toFixed(0)} ms for ${b ? "2 images" : "1 image"} · Clear below ${scoreText(clear_below)} · Alert above ${scoreText(alert_above)}`);
    el("local-trace").disabled = false;
    el("agreement-toggle").disabled = !agreement;
    chooseLayer("model");
    revealSpecimen();
    revealFinding();
  } catch (error) {
    if (turn !== ticket) return;
    console.error(error);
    phase("ready");
    status("Inference unavailable","review","The local model could not start. Retry the check in a browser with WebAssembly SIMD support.");
    write("runtime","Runtime unavailable");
  } finally {
    if (turn === ticket) {
      el("run").disabled = false;
      el("vision-profile").disabled = false;
      el("lab-shell").setAttribute("aria-busy","false");
    }
  }
}

function raster(pixels, region) {
  if (!pixels) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d"), image = context.createImageData(width,height);
  const values = region ? new Uint8Array(width*height) : pixels;
  if (region) for (const index of region.pixels) values[index] = 255;
  for (let i=0;i<values.length;i++) image.data.set([values[i],values[i],values[i],255],i*4);
  context.putImageData(image,0,0);
  return canvas.toDataURL("image/png");
}
async function exportEvidence() {
  if (!result || !current) return;
  const summarize = (frame, observation) => observation && {
    score:observation.score, shape:observation.shape, severity:observation.severity,
    capturePng:raster(frame), modelMaskPng:observation.region ? raster(frame,observation.region) : null,
  };
  const evidence = {
    version:4, profile:result.profile, capturedAt:new Date().toISOString(), model:"SurfaceNet v2 · ONNX Runtime Web",
    source:current.source, simulatedAlternate:current.id !== "upload",
    modelSha256:(result.profile === "reflection" ? reflectionMetrics : report).onnx_sha256,
    threshold:result.config.alert_above, clearBelow:result.config.clear_below,
    verdict:result.assessment.title,
    ambient:summarize(current.a,result.a), alternate:summarize(current.b,result.b),
    registration:result.pair, agreement:result.agreement && {
      boxOverlap:result.agreement.box, ambientPixelSupport:result.agreement.ambient,
      alternatePixelSupport:result.agreement.relit, confirmed:result.agreement.confirmed,
    }, inferenceMs:result.elapsed,
  };
  let sha256 = null;
  if (globalThis.crypto?.subtle) {
    const digest = await crypto.subtle.digest("SHA-256",new TextEncoder().encode(JSON.stringify(evidence)));
    sha256 = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2,"0")).join("");
  }
  const blob = new Blob([JSON.stringify({evidence,sha256},null,2)],{type:"application/json"});
  const url = URL.createObjectURL(blob);
  const download = document.createElement("a");
  download.href = url;
  download.download = "surfacenet-evidence.json";
  download.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}

function tableRows(id, rows) {
  el(id).replaceChildren(...rows.map(([label,left,right]) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    tr.append(th);
    for (const value of [left,right]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    return tr;
  }));
}

function renderEdge() {
  const single = comparison.baseline, q = comparison.variants[1], test = q.official_test;
  write("hero-auc",format(test.auroc));
  const kib = (value) => Math.round(value/1024)+" KiB";
  write("hero-size",kib(q.bytes));
  const ms = (value) => value.toFixed(2)+" ms";
  write("edge-context","Measured on "+comparison.host.cpu+" using "+q.runs+
    " warmed forwards per profile. The APK includes the same A/B benchmark for handset measurements.");
  tableRows("edge-rows",[
    ["Program size",kib(single.bytes),kib(q.bytes)],
    ["Median forward",ms(single.p50_ms),ms(q.p50_ms)],
    ["P95 forward",ms(single.p95_ms),ms(q.p95_ms)],
  ]);
  if (comparison.paired_int8) {
    const {defective,clean} = comparison.paired_int8;
    write("edge-pairs","Four-view INT8 · "+defective.confirmed+"/"+defective.initial+" true alerts and "+
      clean.confirmed+"/"+clean.initial+" false alerts confirmed on simulated paired captures.");
  }
}

function renderReport() {
  const single = comparison.baseline, four = comparison.variants[1], a = single.official_test, b = four.official_test;
  const rows = [
    ["Image AUROC",format(a.auroc),format(b.auroc)],
    ["Image average precision",format(a.average_precision),format(b.average_precision)],
    ["Pixel average precision",format(single.pixel_average_precision),format(four.pixel_average_precision)],
    ["Defects detected",a.tp+" / "+a.positives,b.tp+" / "+b.positives],
    ["Defects missed",a.fn,b.fn],
    ["False alerts on clean",a.fp+" / "+a.negatives,b.fp+" / "+b.negatives],
  ];
  tableRows("benchmark-rows",rows);
  const pairs = report.paired;
  for (const [label, item] of [["Defective parts",pairs.defective],["Clean parts with an alert",pairs.clean]]) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = '<div class="bar-title"><span></span><strong></strong></div><div class="bar-outer"><div class="bar-inner"></div></div><small></small>';
    row.querySelector("span").textContent = label;
    row.querySelector("strong").textContent = item.confirmed + " / " + item.initial;
    row.querySelector(".bar-inner").style.setProperty("--fill",(item.confirmed / item.initial * 100) + "%");
    row.querySelector("small").textContent = item.box_confirmed + " passed box overlap · " +
      (item.initial-item.confirmed) + " stayed in review with the pixel check";
    el("stress-chart").append(row);
  }
  write("stress-note","Single-view browser profile with one simulated second exposure per image. Unconfirmed alerts require review.");
}

function renderReview() {
  const {binary_test:binary, official_test:band, calibration_defects:count} = reflectedCalibration.profiles.int8;
  const single = calibration.profiles.int8.official_test;
  const referred = band.defect.review + band.defect.hold;
  write("hero-recall",referred + " / " + band.defect.total);
  const rows = [
    ["Defects scoring clear", binary.defect.clear, band.defect.clear],
    ["Defects sent to review", binary.defect.review, band.defect.review],
    ["Clean parts sent to review", binary.clean.review, band.clean.review],
    ["Alerts on defective / clean", `${binary.defect.hold} / ${binary.clean.hold}`, `${band.defect.hold} / ${band.clean.hold}`],
  ];
  tableRows("review-rows",rows);
  write("review-context",`Both profiles sent ${referred} of ${band.defect.total} defective images to alert or review. The four-view model sent ${band.clean.review + band.clean.hold} clean images, down from ${single.clean.review + single.clean.hold}. These are single-image scores. The cutoff used ${count} defective validation images.`);
}

el("run").addEventListener("click",run);
el("vision-profile").addEventListener("change",() => { reset(); chooseLayer("raw"); });
el("wipe-mode").addEventListener("click",() => setLayout("wipe"));
el("split-mode").addEventListener("click",() => setLayout("split"));
el("exposure-split").addEventListener("input",event => {
  const p = event.target.value + "%";
  el("wipe-panel").style.setProperty("--position",p);
  event.target.setAttribute("aria-valuetext",p+" ambient exposure");
});
el("local-trace").addEventListener("click",exportEvidence);
el("upload").addEventListener("change",event => {
  const files = Array.from(event.target.files);
  event.target.value = "";
  if (files.length) upload(files);
});
el("layers").addEventListener("click",event => {
  const layer = event.target.closest("[data-layer]");
  if (layer) chooseLayer(layer.dataset.layer);
});

try {
  [shelf,report,reflectionMetrics,comparison,calibration,reflectedCalibration] = await Promise.all([
    json("cases.json"),json("benchmark.json"),json("reflection-browser.json"),json("reflection-edge.json"),
    json("review-calibration.json"),json("reflection-review.json"),
  ]);

  connectLiveCamera({width,height,samples,modelHash:report.onnx_sha256,infer:async (id,pixels) => {
    const config = calibration.profiles.fp32;
    return {...await inspect(id,{a:pixels},"single",config),clearBelow:config.clear_below};
  }});
  renderReport();
  renderEdge();
  renderReview();
  renderCurve(comparison);
  for (const specimen of shelf) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.case = specimen.id;
    const thumbnail = new Image(width,height);
    thumbnail.src = asset(specimen.id + "-ambient.png");
    thumbnail.alt = "";
    thumbnail.loading = "lazy";
    const label = document.createElement("span");
    label.className = "case-label";
    label.textContent = specimen.label;
    const source = document.createElement("span");
    source.className = "case-code";
    source.textContent = specimen.source;
    source.setAttribute("aria-hidden","true");
    label.append(source);
    button.append(thumbnail,label);
    button.addEventListener("click",() => selectPreset(specimen));
    el("cases").append(button);
  }
  await selectPreset(shelf[0]);
} catch(error) {
  console.error(error);
  status("Lab assets unavailable","review","Reload the page to reconnect the model and its benchmark data.");
  el("lab-shell").setAttribute("aria-busy","false");
}
