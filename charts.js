const CHART_TASK_LABELS = {
  relative_depth: "Depth", surface_normals: "Normals", semantic_segmentation: "SemSeg",
  instance_segmentation: "InstSeg", panoptic_segmentation: "Panoptic", edge_detection: "Edges",
  saliency_detection: "Saliency", human_pose_2d: "Pose", face_landmarks: "Face",
  object_keypoints: "ObjKpt", colorization: "Color", denoising: "Denoise",
  deblurring: "Deblur", inpainting: "Inpaint", super_resolution: "SR",
};

function chartEscape(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function chartMetricLabel(metric) {
  const labels = {
    lpips_alex: "LPIPS", pck_bbox: "PCK", pck_bbox_reject: "PCK",
    nme_bbox: "NME", nme_bbox_matched: "NME", pq_novoid: "PQ", pq: "PQ",
    ois_f: "OIS F-score", max_fbeta: "Max Fβ", miou: "mIoU",
    delta1_aligned_ssc: "δ₁", mean_deg: "MAE (degrees)",
  };
  return labels[metric] || String(metric).replaceAll("_", " ");
}

function benefitChart(rows) {
  const width = 900;
  const left = 160;
  const right = 32;
  const top = 74;
  const rowHeight = 46;
  const plotWidth = width - left - right;
  const colors = { scrambled: "#4169e1", grey: "#55bca9", copy: "#f26b38" };
  const labels = { scrambled: "Real beats scrambled", grey: "Real beats grey", copy: "Real beats copy input" };
  let svg = `<svg class="score-chart" viewBox="0 0 ${width} ${top + rows.length * rowHeight + 45}" role="img" aria-labelledby="benefit-svg-title benefit-svg-desc">
    <title id="benefit-svg-title">Tasks where real prompts outperform controls</title>
    <desc id="benefit-svg-desc">Grouped horizontal bars show task win counts out of fifteen for seven evaluated models.</desc>`;
  let legendX = left;
  Object.keys(colors).forEach((key) => {
    svg += `<rect x="${legendX}" y="18" width="12" height="12" fill="${colors[key]}"></rect><text x="${legendX + 18}" y="29" class="chart-legend">${labels[key]}</text>`;
    legendX += key === "scrambled" ? 190 : 160;
  });
  [0, 5, 10, 15].forEach((tick) => {
    const x = left + (tick / 15) * plotWidth;
    svg += `<line x1="${x}" y1="${top - 9}" x2="${x}" y2="${top + rows.length * rowHeight}" class="chart-grid"></line><text x="${x}" y="${top + rows.length * rowHeight + 25}" text-anchor="middle" class="chart-axis">${tick}</text>`;
  });
  rows.forEach((row, index) => {
    const y = top + index * rowHeight;
    svg += `<text x="${left - 14}" y="${y + 19}" text-anchor="end" class="chart-model">${chartEscape(row.model)}</text>`;
    ["scrambled", "grey", "copy"].forEach((key, barIndex) => {
      const value = Number(row[key]);
      const barY = y + barIndex * 10;
      const barW = (value / 15) * plotWidth;
      svg += `<rect x="${left}" y="${barY}" width="${barW}" height="7" rx="1" fill="${colors[key]}"><title>${chartEscape(row.model)}: ${value}/15 tasks — ${labels[key]}</title></rect><text x="${left + barW + 6}" y="${barY + 7}" class="chart-value">${value}</text>`;
    });
  });
  return `${svg}</svg>`;
}

function outcomeMatrix(matrix) {
  const cell = 49;
  const left = 165;
  const top = 120;
  const rowHeight = 38;
  const width = left + matrix.tasks.length * cell + 26;
  const height = top + matrix.rows.length * rowHeight + 52;
  let svg = `<svg class="score-chart matrix-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="matrix-svg-title matrix-svg-desc">
    <title id="matrix-svg-title">Real versus scrambled task outcomes</title>
    <desc id="matrix-svg-desc">Green cells show tasks where real prompts beat scrambled prompts. Orange cells show the opposite result.</desc>`;
  matrix.tasks.forEach((task, index) => {
    const x = left + index * cell + 21;
    svg += `<text x="${x}" y="${top - 13}" transform="rotate(-52 ${x} ${top - 13})" text-anchor="start" class="matrix-task">${chartEscape(CHART_TASK_LABELS[task] || task)}</text>`;
  });
  matrix.rows.forEach((row, rowIndex) => {
    const y = top + rowIndex * rowHeight;
    svg += `<text x="${left - 14}" y="${y + 21}" text-anchor="end" class="chart-model">${chartEscape(row.model)}</text>`;
    matrix.tasks.forEach((task, index) => {
      const result = row.tasks[task];
      const x = left + index * cell;
      const fill = result.real_beats_scrambled ? "#bceee5" : "#ffd7c8";
      const mark = result.real_beats_scrambled ? "✓" : "×";
      const label = result.real_beats_scrambled ? "real wins" : "scrambled wins";
      svg += `<rect x="${x}" y="${y}" width="42" height="29" rx="2" fill="${fill}" stroke="#9ba8b7"><title>${chartEscape(row.model)} · ${chartEscape(CHART_TASK_LABELS[task] || task)} · ${label} · ${chartEscape(chartMetricLabel(result.metric))}</title></rect><text x="${x + 21}" y="${y + 20}" text-anchor="middle" class="matrix-mark">${mark}</text>`;
    });
  });
  svg += `<rect x="${left}" y="${height - 27}" width="13" height="13" fill="#bceee5" stroke="#9ba8b7"></rect><text x="${left + 19}" y="${height - 16}" class="chart-legend">Real wins</text><rect x="${left + 105}" y="${height - 27}" width="13" height="13" fill="#ffd7c8" stroke="#9ba8b7"></rect><text x="${left + 124}" y="${height - 16}" class="chart-legend">Scrambled wins</text>`;
  return `${svg}</svg>`;
}

function scalingChart(series) {
  const width = 900;
  const height = 440;
  const left = 88;
  const right = 42;
  const top = 40;
  const bottom = 92;
  const plotHeight = height - top - bottom;
  const x = { 1: 170, 2: 450, 4: 730 };
  const y = (value) => top + ((130 - value) / 130) * plotHeight;
  const colors = { VisualCloze: "#f26b38", LVM: "#4169e1" };
  let svg = `<svg class="score-chart scaling-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="scaling-svg-title scaling-svg-desc">
    <title id="scaling-svg-title">Prompt-count scaling for VisualCloze and LVM</title>
    <desc id="scaling-svg-desc">P equals two is the task-normalized baseline at one hundred. P equals four shows the median direction-adjusted score index across fifteen tasks. P equals one is not included in this normalized index.</desc>`;
  [0, 25, 50, 75, 100, 125].forEach((tick) => {
    const yy = y(tick);
    svg += `<line x1="${left}" y1="${yy}" x2="${width - right}" y2="${yy}" class="chart-grid${tick === 100 ? " baseline" : ""}"></line><text x="${left - 13}" y="${yy + 4}" text-anchor="end" class="chart-axis">${tick}</text>`;
  });
  [1, 2, 4].forEach((p) => {
    svg += `<text x="${x[p]}" y="${height - 50}" text-anchor="middle" class="chart-prompt">P=${p}</text>`;
  });
  svg += `<text x="18" y="${top + plotHeight / 2}" transform="rotate(-90 18 ${top + plotHeight / 2})" text-anchor="middle" class="chart-axis-title">Median direction-adjusted score index</text>`;

  series.forEach((model) => {
    const p2 = model.points.find((point) => point.p === 2);
    const p4 = model.points.find((point) => point.p === 4);
    const color = colors[model.model];
    svg += `<line x1="${x[2]}" y1="${y(p2.median_index)}" x2="${x[4]}" y2="${y(p4.median_index)}" stroke="${color}" stroke-width="4"></line>`;
    svg += `<line x1="${x[4]}" y1="${y(p4.q1)}" x2="${x[4]}" y2="${y(p4.q3)}" stroke="${color}" stroke-width="2"></line><line x1="${x[4] - 8}" y1="${y(p4.q1)}" x2="${x[4] + 8}" y2="${y(p4.q1)}" stroke="${color}" stroke-width="2"></line><line x1="${x[4] - 8}" y1="${y(p4.q3)}" x2="${x[4] + 8}" y2="${y(p4.q3)}" stroke="${color}" stroke-width="2"></line>`;
    [p2, p4].forEach((point) => {
      svg += `<circle cx="${x[point.p]}" cy="${y(point.median_index)}" r="7" fill="${color}" stroke="#fff" stroke-width="2"><title>${chartEscape(model.model)} P=${point.p}: median index ${point.median_index.toFixed(1)}</title></circle>`;
    });
    svg += `<text x="${x[4] + 14}" y="${y(p4.median_index) + 4}" class="scaling-label" fill="${color}">${chartEscape(model.model)} ${p4.median_index.toFixed(1)}</text>`;
    svg += `<circle cx="${x[1]}" cy="${height - 79}" r="7" fill="#fff" stroke="${color}" stroke-width="2" stroke-dasharray="3 2"></circle><text x="${x[1]}" y="${height - 62}" text-anchor="middle" class="missing-label" fill="${color}">${chartEscape(model.model)} · not in index</text>`;
  });
  svg += `<text x="${x[2]}" y="${y(100) - 11}" text-anchor="middle" class="baseline-label">P=2 baseline</text>`;
  return `${svg}</svg>`;
}

document.addEventListener("DOMContentLoaded", async () => {
  const benefit = document.querySelector("#prompt-benefit-chart");
  if (!benefit) return;
  try {
    const response = await fetch("data/chart-summary.json");
    if (!response.ok) throw new Error(`chart summary request failed: ${response.status}`);
    const data = await response.json();
    benefit.innerHTML = benefitChart(data.prompt_benefit);
    document.querySelector("#task-outcome-matrix").innerHTML = outcomeMatrix(data.task_prompt_benefit);
    document.querySelector("#prompt-scaling-chart").innerHTML = scalingChart(data.prompt_scaling);
    const vc = data.prompt_scaling.find((item) => item.model === "VisualCloze");
    const lvm = data.prompt_scaling.find((item) => item.model === "LVM");
    document.querySelector("#prompt-scaling-summary").textContent =
      `LVM P=1 is now available in the exact-score ledger and LPIPS check. This normalized scaling plot still begins at P=2 because LVM P=1 and the older P=2 artifact do not yet share one complete primary-metric set. From P=2 to P=4, VisualCloze improved on ${vc.tasks_improved_p4_vs_p2}/15 tasks and LVM on ${lvm.tasks_improved_p4_vs_p2}/15.`;
  } catch (error) {
    ["prompt-benefit-chart", "task-outcome-matrix", "prompt-scaling-chart"].forEach((id) => {
      const target = document.getElementById(id);
      if (target) target.textContent = "Chart data could not be loaded.";
    });
    console.error(error);
  }
});
