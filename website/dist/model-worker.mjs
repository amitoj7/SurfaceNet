import {agree, height, measure, observe, verdict, width} from "./inspection.mjs";

const models = new Map();
const modelFiles = {single:"surface.onnx",reflection:"surface-reflection.onnx"};
let queue = Promise.resolve();

function model(profile) {
  if (!models.has(profile)) {
    const loading = (async () => {
      const ort = await import("./vendor/ort.wasm.min.mjs");
      ort.env.wasm.wasmPaths = new URL("./vendor/",import.meta.url).href;
      ort.env.wasm.numThreads = 1;
      const modelUrl = new URL(`./assets/${modelFiles[profile]}`,import.meta.url).href;
      const session = await ort.InferenceSession.create(modelUrl,{executionProviders:["wasm"]});
      return {session, Tensor:ort.Tensor};
    })().catch(error => {
      models.delete(profile);
      throw error;
    });
    models.set(profile,loading);
  }
  return models.get(profile);
}

self.onmessage = ({data}) => {
  queue = queue.then(() => inspect(data)).catch(error => {
    self.postMessage({id:data.id,error:error.message || "Inference failed"});
  });
};

async function inspect({id,frames,profile,threshold,clearBelow}) {
  const stage = (name) => self.postMessage({id,stage:name});
  stage("loading");
  const {session,Tensor} = await model(profile);
  async function infer(frame) {
    const values = Float32Array.from(frame, value => value/127.5-1);
    const prediction = await session.run({pixels:new Tensor("float32",values,[1,1,height,width])});
    return observe(frame,prediction.logits.data,threshold);
  }
  stage("segment");
  const start = performance.now();
  const a = await infer(frames.a);
  const b = frames.b ? await infer(frames.b) : null;
  const elapsed = performance.now() - start;
  stage("compare");
  const pair = b ? measure(frames.a,frames.b) : null;
  const agreement = pair ? agree(a.region,b.region,pair) : null;
  const assessment = verdict(a,b,pair,agreement,clearBelow);
  self.postMessage({id,result:{a,b,pair,agreement,assessment,elapsed}});
}
