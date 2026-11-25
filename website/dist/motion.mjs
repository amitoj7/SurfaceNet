const page = document.documentElement;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const hover = matchMedia("(hover: hover) and (pointer: fine)");
const tabs = Array.from(document.querySelectorAll(".masthead nav a, .nav-download"), link => ({
  link, destination:document.querySelector(link.getAttribute("href")),
}));
const splashFields = new Map(Array.from(document.querySelectorAll(".splash-field"),
  field => [field, {visible:false, loading:false, close:null}]));
const moving = new Map();
const settle = "cubicBezier(.18,.73,.24,1)";
let motionLibrary;
let scheduled = false;

async function play(element, keyframes) {
  if (!element || reduced.matches || !element.animate) return;
  motionLibrary ??= import("./vendor/anime-motion.mjs").catch(() => null);
  const library = await motionLibrary;
  if (!library || reduced.matches || !element.isConnected || element.closest("[hidden]")) return;

  moving.get(element)?.revert();
  const animation = library.waapi.animate(element, {
    ease: settle,
    ...keyframes,
    onComplete(completed) {
      if (moving.get(element) !== completed) return;
      completed.revert();
      moving.delete(element);
    },
  });
  moving.set(element, animation);
}

function arrive(element, {distance = 10, opacity = .75, duration = 400, delay = 0} = {}) {
  play(element, {
    transform: [`translateY(${distance}px)`, "none"],
    opacity: [opacity, 1],
    duration,
    delay,
  });
}

function revealCover(cover) {
  arrive(cover.querySelector("#cover-title"), {distance:14, opacity:.82, duration:550});
  arrive(cover.querySelector(".cover-actions"), {distance:7, opacity:.87, duration:380, delay:70});
  play(cover.querySelector(".art-scan"), {
    transform: ["translateX(-115%)", "translateX(430%)"],
    opacity: [0, .78, .52, 0],
    ease: "linear",
    duration: 920,
    delay: 120,
  });
  play(cover.querySelector(".art-target"), {
    scale: ["1.16", "1"],
    opacity: [.64, 1],
    duration: 380,
    delay: 610,
  });
}

export function revealSpecimen({replace = false} = {}) {
  const frame = document.querySelector("#plate > :is(.wipe, .captures):not([hidden])");
  if (replace) {
    play(frame, {
      clipPath: ["inset(0 14% 0 0)", "inset(0)"],
      opacity: [.42, 1],
      duration: 345,
    });
  } else {
    arrive(frame, {distance:3, opacity:.7, duration:250});
  }
}

const artwork = document.querySelector(".cover-art");
const glint = artwork?.querySelector(".art-glint");
if (glint) {
  let frame, pointer;
  artwork.addEventListener("pointermove", event => {
    if (reduced.matches || !hover.matches || event.pointerType === "touch") return;
    pointer = event;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      const bounds = artwork.getBoundingClientRect();
      const x = Math.round(pointer.clientX - bounds.left);
      const y = Math.round(pointer.clientY - bounds.top);
      glint.style.transform = `translate3d(${x}px,${y}px,0) translate(-50%,-50%)`;
      artwork.dataset.glint = "visible";
    });
  }, {passive:true});
  artwork.addEventListener("pointerleave", () => {
    cancelAnimationFrame(frame);
    frame = null;
    delete artwork.dataset.glint;
  });
}

export function revealFinding() {
  arrive(document.querySelector("#verdict"), {distance:6, duration:330});
  arrive(document.querySelector(".readout .measurements"), {distance:6, duration:330, delay:55});
}

function readingPosition() {
  scheduled = false;
  const height = page.scrollHeight - window.innerHeight;
  page.style.setProperty("--reading",height <= 0 ? "0" : String(Math.max(0,Math.min(1,scrollY / height))));
  let active;
  for (const entry of tabs) if (entry.destination.getBoundingClientRect().top <= 140) active = entry;
  for (const entry of tabs) {
    if (entry === active) entry.link.setAttribute("aria-current","location");
    else entry.link.removeAttribute("aria-current");
  }
}

function schedule() {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(readingPosition);
  }
}

async function syncSplash(field, state) {
  if (!state.visible || reduced.matches) {
    state.close?.();
    state.close = null;
    return;
  }
  if (state.close || state.loading) return;
  state.loading = true;
  try {
    const {mountSplash} = await import("./vendor/hero-splash.js");
    if (state.visible && !reduced.matches) state.close = mountSplash(field);
  } catch (error) {
    console.warn("Fluid decoration unavailable",error);
  } finally {
    state.loading = false;
  }
}

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const state = splashFields.get(entry.target);
      state.visible = entry.isIntersecting;
      syncSplash(entry.target,state);
    }
  });
  for (const field of splashFields.keys()) observer.observe(field);

  const chapters = new IntersectionObserver(entries => {
    for (const {isIntersecting, target} of entries) {
      if (!isIntersecting) continue;
      chapters.unobserve(target);
      if (target.matches(".edition-cover")) revealCover(target);
      else arrive(target);
    }
  }, {threshold:.25});
  for (const section of document.querySelectorAll(".edition-cover, .chapter-head h2, #lab-title, #android-title, .headline-stats")) {
    chapters.observe(section);
  }
} else {
  const cover = document.querySelector(".edition-cover");
  if (cover) revealCover(cover);
}

const benchmarks = document.querySelector(".evidence-disclosure");
benchmarks?.addEventListener("toggle", () => {
  if (benchmarks.open) arrive(benchmarks.querySelector(".plot-heading"), {distance:7, duration:360});
});
window.addEventListener("scroll",schedule,{passive:true});
window.addEventListener("resize",schedule,{passive:true});
reduced.addEventListener("change",() => {
  for (const [field,state] of splashFields) syncSplash(field,state);
  if (reduced.matches) {
    for (const animation of moving.values()) animation.revert();
    moving.clear();
  }
});
schedule();
