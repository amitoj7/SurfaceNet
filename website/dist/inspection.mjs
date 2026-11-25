export const width = 96;
export const height = 256;
const size = width * height;
const travel = 4;

function quality(frame) {
  let total = 0, edges = 0, clipped = 0;
  for (let y = 1; y < height; y++) {
    for (let x = 1; x < width; x++) {
      const i = y * width + x, value = frame[i];
      total += value;
      edges += Math.abs(value - frame[i - 1]) + Math.abs(value - frame[i - width]);
      if (value <= 9 || value >= 248) clipped++;
    }
  }
  const count = (width - 1) * (height - 1);
  const mean = total / count;
  return mean >= 24 && mean <= 234 && clipped / count <= .24 && edges / (2 * count) >= .65;
}

function detail(frame) {
  const response = new Int16Array(size);
  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      const i = y * width + x;
      const contrast = 4 * frame[i] - frame[i - 2] - frame[i + 2] - frame[i - 2 * width] - frame[i + 2 * width];
      response[i] = Math.max(-72, Math.min(72, contrast));
    }
  }
  return response;
}

export function measure(ambient, relit) {
  if (!quality(ambient) || !quality(relit)) return {status:"CAPTURE_QUALITY", dx:0, dy:0, correlation:0, lightChange:0};
  const a = detail(ambient), b = detail(relit);
  let best = -2, dx = 0, dy = 0, correlation = 0;
  for (let sy = -travel; sy <= travel; sy++) {
    for (let sx = -travel; sx <= travel; sx++) {
      let ab = 0, aa = 0, bb = 0, count = 0;
      for (let y = 8; y < height - 8; y += 4) {
        for (let x = 8; x < width - 8; x += 3) {
          const av = a[y * width + x], bv = b[(y + sy) * width + x + sx];
          ab += av * bv;
          aa += av * av;
          bb += bv * bv;
          count++;
        }
      }
      const value = Math.min(aa, bb) < 36 * count ? 0 : ab / Math.sqrt(aa * bb);
      const score = value - .002 * (Math.abs(sx) + Math.abs(sy));
      if (score > best) {
        best = score;
        dx = sx;
        dy = sy;
        correlation = value;
      }
    }
  }
  const changes = [];
  for (let y = 8; y < height - 8; y += 4) {
    for (let x = 8; x < width - 8; x += 3) {
      changes.push(Math.abs(ambient[y * width + x] - relit[(y + dy) * width + x + dx]));
    }
  }
  changes.sort((a,b) => a-b);
  const lightChange = changes[Math.floor((changes.length - 1) * 3 / 4)];
  let status = "USABLE";
  if (correlation < .58 || Math.abs(dx) === travel || Math.abs(dy) === travel) status = "ALIGNMENT";
  else if (lightChange < 7) status = "INSUFFICIENT_LIGHT";
  return {status, dx, dy, correlation, lightChange};
}

export function overlap(base, alternate, {dx, dy}) {
  if (!base || !alternate) return 0;
  const w = Math.max(0, Math.min(base.x1 + dx, alternate.x1) - Math.max(base.x0 + dx, alternate.x0) + 1);
  const h = Math.max(0, Math.min(base.y1 + dy, alternate.y1) - Math.max(base.y0 + dy, alternate.y0) + 1);
  const area = (r) => (r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1);
  return w * h / (area(base) + area(alternate) - w * h);
}

function support(source, target, dx, dy) {
  const dilated = new Uint8Array(size);
  for (const index of target) {
    const x = index % width, y = Math.floor(index / width);
    for (let yy = Math.max(0,y-1); yy <= Math.min(height-1,y+1); yy++) {
      for (let xx = Math.max(0,x-1); xx <= Math.min(width-1,x+1); xx++) dilated[yy * width + xx] = 1;
    }
  }
  const matched = [];
  for (const index of source) {
    const x = index % width + dx, y = Math.floor(index / width) + dy;
    if (x >= 0 && x < width && y >= 0 && y < height && dilated[y * width + x]) matched.push(index);
  }
  return {fraction:matched.length / source.length, matched:Int32Array.from(matched)};
}

export function agree(ambient, relit, pair) {
  if (!ambient || !relit || pair?.status !== "USABLE") return null;
  const box = overlap(ambient, relit, pair);
  const a = support(ambient.pixels, relit.pixels, pair.dx, pair.dy);
  const b = support(relit.pixels, ambient.pixels, -pair.dx, -pair.dy);
  return {box, ambient:a.fraction, relit:b.fraction, ambientMatches:a.matched, relitMatches:b.matched,
    confirmed:box >= .25 && Math.min(a.fraction,b.fraction) >= .60};
}

export function observe(frame, logits, threshold) {
  const p = Float32Array.from(logits, (value) => 1 / (1 + Math.exp(-value)));
  let score = 0, peak = -1;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const i = y * width + x;
      const value = (p[i] + p[i + 1] + p[i + width] + p[i + width + 1]) * .25;
      if (value > score) {
        score = value;
        peak = i;
      }
    }
  }
  const region = score > threshold ? findRegion(p, threshold, peak) : null;
  const shape = region ? region.elongation >= 3.2 ? "Line-like" : region.fill > .48 ? "Compact mark" : "Diffuse mark" : "No localized finding";
  const fraction = region ? region.pixels.length / size : 0;
  const severity = region ? fraction >= .012 || region.elongation >= 4.5 ? "High" : fraction >= .002 ? "Moderate" : "Low" : "—";
  return {score, region, shape, severity, quality:quality(frame)};
}

function findRegion(map, threshold, peak) {
  const active = Uint8Array.from(map, (value) => value > .85 * threshold);
  const queue = new Int32Array(size);
  const px = peak % width, py = Math.floor(peak / width);
  let largest = null;
  for (let first = 0; first < size; first++) {
    if (!active[first]) continue;
    active[first] = 0;
    queue[0] = first;
    let count = 1, x0 = width, x1 = 0, y0 = height, y1 = 0, covers = false;
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let head = 0; head < count; head++) {
      const i = queue[head], x = i % width, y = Math.floor(i / width);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y;
      covers ||= x >= px && x <= px + 1 && y >= py && y <= py + 1;
      for (let yy = Math.max(0,y-1); yy <= Math.min(height-1,y+1); yy++) {
        for (let xx = Math.max(0,x-1); xx <= Math.min(width-1,x+1); xx++) {
          const next = yy * width + xx;
          if (active[next]) {
            active[next] = 0;
            queue[count++] = next;
          }
        }
      }
    }
    if (!covers || (largest && largest.pixels.length >= count)) continue;
    const vx = sxx / count - (sx/count)**2, vy = syy / count - (sy/count)**2;
    const radius = Math.hypot(vx - vy, 2 * (sxy / count - (sx/count) * (sy/count)));
    largest = {x0,x1,y0,y1,fill:count / ((x1-x0+1)*(y1-y0+1)),pixels:queue.slice(0,count),
      elongation:Math.sqrt((vx + vy + radius + .5)/(vx + vy - radius + .5))};
  }
  return largest;
}

export function verdict(a, b, pair, agreement, clearBelow) {
  if (!a.quality || (pair && pair.status !== "USABLE")) {
    const messages = {
      CAPTURE_QUALITY:"Reacquire with stable focus and usable exposure.",
      ALIGNMENT:"Reseat the part. The two frames could not be registered reliably.",
      INSUFFICIENT_LIGHT:"Increase the lighting difference and capture again. These images are too similar for an independent check.",
    };
    return {title:"Reacquire", state:"review", reason:messages[pair?.status] || messages.CAPTURE_QUALITY};
  }
  if (!b) return {title:"Single-capture review", state:"review", reason:"Load a second capture under different illumination to check persistence."};
  if (agreement?.confirmed) return {title:"Repeat finding · Hold", state:"held", reason:"At least 60% of each defect mask has matching pixels after alignment. Segregate the part for operator verification."};
  if (a.region || b.region) return {title:"Unconfirmed · Review", state:"review", reason:"The captures provided insufficient support for the same defect pixels. Keep the part for visual confirmation."};
  if (Math.max(a.score, b.score) >= clearBelow) return {title:"Uncertain · Review", state:"review", reason:"A capture scored in the review range. Ask an operator to check the part before releasing it."};
  return {title:"No repeat finding", state:"clear", reason:"Both captures scored below the calibrated clear threshold. The operator confirms the disposition."};
}
