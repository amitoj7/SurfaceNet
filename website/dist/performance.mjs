export function renderCurve(report) {
  const plot = document.getElementById("comparison");
  const canvas = document.getElementById("precision-curve");
  const slider = document.getElementById("recall-slider");
  const cursor = document.getElementById("recall-cursor");
  const readout = document.getElementById("curve-readout");
  const summary = document.getElementById("curve-summary");
  const percent = value => Math.round(value * 100) + "%";
  const ap = value => (value * 100).toFixed(2) + "%";
  const series = report.curves.map((curve,index) => ({
    ...curve, name:index ? "Four views" : "Single view", ink:index ? "#2544c5" : "#5e687a",
  }));
  summary.textContent = "Test average precision: " + ap(report.baseline.official_test.average_precision) +
    " single view · " + ap(report.variants[1].official_test.average_precision) + " four views.";
  plot.hidden = false;

  function render() {
    const {width,height} = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const scale = Math.min(window.devicePixelRatio || 1,2);
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale,scale);
    const left = 40, right = width - 22, top = 47, bottom = height - 36;
    const x = recall => left + recall * (right - left);
    const y = precision => bottom - precision * (bottom - top);
    ctx.font = "11px Plex, monospace";
    ctx.strokeStyle = "#bec9dd";
    ctx.fillStyle = "#4c5769";
    for (let i=0;i<=5;i++) {
      const tick = i * .2;
      ctx.beginPath();
      ctx.moveTo(left,y(tick));
      ctx.lineTo(right,y(tick));
      ctx.moveTo(x(tick),top);
      ctx.lineTo(x(tick),bottom);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(percent(tick),left - 6,y(tick)+4);
      ctx.textAlign = "center";
      ctx.fillText(percent(tick),x(tick),bottom+18);
    }
    ctx.textAlign = "left";
    ctx.fillText("Precision",left,top-11);
    ctx.textAlign = "right";
    ctx.fillText("Recall",right,height-3);
    for (const line of series) {
      ctx.beginPath();
      ctx.strokeStyle = line.ink;
      ctx.lineWidth = 2;
      for (const [index, [r,p]] of line.points.entries()) {
        if (index === 0) ctx.moveTo(x(r),y(p));
        else ctx.lineTo(x(r),y(p));
      }
      ctx.stroke();
    }
    const target = Number(slider.value) / 100;
    cursor.value = percent(target);
    slider.setAttribute("aria-valuetext",percent(target) + " recall");
    ctx.setLineDash([3,4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#21201d";
    ctx.beginPath();
    ctx.moveTo(x(target),top);
    ctx.lineTo(x(target),bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    const measurements = series.map(line => {
      const optimum = line.points.reduce((best, point) => point[0] + .00001 >= target && point[1] > best[1] ? point : best);
      ctx.fillStyle = line.ink;
      ctx.beginPath();
      ctx.arc(x(optimum[0]),y(optimum[1]),4,0,Math.PI*2);
      ctx.fill();
      return line.name + " " + percent(optimum[1]);
    });
    readout.textContent = "Best observed precision at or above " + percent(target) + " recall · " + measurements.join(" · ") + ".";
  }

  slider.addEventListener("input",render);
  const resize = new ResizeObserver(render);
  resize.observe(canvas);
  render();
}
