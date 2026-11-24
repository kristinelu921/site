const benchV2TaskLabels = {
  colorization: "Colorization",
  deblurring: "Deblurring",
  denoising: "Denoising",
  edge_detection: "Edge detection",
  face_landmarks: "Face landmarks",
  human_pose_2d: "Human pose 2D",
  inpainting: "Inpainting",
  instance_segmentation: "Instance segmentation",
  relative_depth: "Relative depth",
  saliency_detection: "Saliency detection",
  semantic_segmentation: "Semantic segmentation",
  super_resolution: "Super-resolution",
  surface_normals: "Surface normals"
};

const benchV2MetricLabels = {
  lpips: "LPIPS",
  lpips_alex: "LPIPS Alex",
  ois_f: "OIS F-score",
  nme: "NME",
  nme_bbox_matched: "matched bbox NME",
  "pck_torso_0.2": "PCK torso @ 0.2",
  pck_bbox_reject: "bbox-reject PCK",
  pq: "PQ",
  pq_novoid: "PQ no-void",
  delta1: "δ₁",
  delta1_aligned_ssc: "δ₁ aligned SSC",
  miou: "mIoU",
  max_fbeta: "max Fβ",
  mean_deg: "mean degrees"
};

const benchV2RelationLabels = {
  same_metric_family: "Same metric family · different pack",
  same_metric_different_protocol: "Same primary name · corrected protocol",
  metric_changed_not_comparable: "Metric definition changed",
  same_metric_family_corrected_composite: "Same family · exact-hole composite",
  protocol_changed_similar_scale: "Related PQ · corrected protocol",
  alignment_protocol_changed_not_comparable: "Depth alignment changed"
};

const benchV2OurModelColumns = [
  { model: "LVM", protocol: "Native P=2", label: "LVM · P=2", note: "64-query pack" },
  { model: "LVM", protocol: "Native P=4", label: "LVM · P=4", note: "64-query pack" },
  { model: "JFlux T3EZ", protocol: "TTT · P=16 · 200 steps", label: "JFlux T3EZ", note: "q0 only" }
];

const benchV2PaperOurs = {
  edge_detection: 0.79,
  relative_depth: 0.51,
  colorization: 0.22,
  inpainting: 0.07,
  deblurring: 0.11,
  denoising: 0.17,
  human_pose_2d: 0.06,
  face_landmarks: 0.42
};

function benchV2Format(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(6).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

function benchV2Arrow(direction) {
  return direction === "higher" ? "↑" : "↓";
}

function benchV2RenderFacts(data) {
  const progress = data.progress;
  document.querySelector("#bench-v2-facts").innerHTML = `
    <div><dt>Scored packs</dt><dd>${progress.completed_packs} / ${progress.total_packs}</dd></div>
    <div><dt>Scored queries</dt><dd>${progress.completed_queries.toLocaleString()}</dd></div>
    <div><dt>Generated queries</dt><dd>${progress.inference_queries_complete.toLocaleString()}</dd></div>
    <div><dt>Prompt count</dt><dd>P=${data.model.prompt_count}</dd></div>`;
}

function benchV2RenderRows(data) {
  document.querySelector("#bench-v2-results-body").innerHTML = data.rows.map((row) => {
    const reference = row.site_reference;
    const relation = benchV2RelationLabels[reference.relation] || "Different evaluation contract";
    return `<tr>
      <th scope="row">
        <span class="task-name">${benchV2TaskLabels[row.task] || row.task}</span>
        <span class="metric-name">${row.dataset} · ${row.n.toLocaleString()} queries</span>
      </th>
      <td><span class="bench-v2-score">${benchV2Format(row.mean)}</span><span class="cell-note">${benchV2MetricLabels[row.primary_metric] || row.primary_metric} ${benchV2Arrow(row.direction)}</span></td>
      <td><span class="bench-v2-reference">${benchV2Format(reference.mean)}</span><span class="cell-note">${benchV2MetricLabels[reference.metric] || reference.metric}</span></td>
      <td><span class="bench-v2-relation">${relation}</span></td>
    </tr>`;
  }).join("");
}

function benchV2ParseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function benchV2RenderOurModelRows(data, scoreRows) {
  const tasks = [...new Set(data.rows.map((row) => row.task))];
  document.querySelector("#bench-v2-our-models-body").innerHTML = tasks.map((task) => {
    const matches = benchV2OurModelColumns.map((column) => scoreRows.find((row) =>
      row.model === column.model &&
      row.protocol === column.protocol &&
      row.task === task &&
      row.condition === "real"
    ));
    const metricRow = matches.find(Boolean);
    const metric = metricRow ? (benchV2MetricLabels[metricRow.primary_metric] || metricRow.primary_metric) : "site-native metric";
    const direction = metricRow ? benchV2Arrow(metricRow.direction) : "";
    const cells = matches.map((row, index) => {
      if (!row) return '<td class="paper-missing">—</td>';
      const column = benchV2OurModelColumns[index];
      const note = column.model === "JFlux T3EZ" ? "1 query · q0 subset" : `${Number(row.n).toLocaleString()} scored queries`;
      return `<td><span class="our-model-score">${benchV2Format(Number(row.mean))}</span><span class="cell-note">${note}</span></td>`;
    }).join("");
    const paperValue = benchV2PaperOurs[task];
    const paperCell = Number.isFinite(paperValue)
      ? `<td><span class="our-model-score paper-reference-score">${benchV2Format(paperValue)}</span><span class="cell-note">reported precision</span></td>`
      : '<td class="paper-missing">—</td>';
    return `<tr>
      <th scope="row"><span class="task-name">${benchV2TaskLabels[task] || task}</span><span class="metric-name">${metric} ${direction}</span></th>
      ${cells}${paperCell}
    </tr>`;
  }).join("");
}

function benchV2RenderProvenance(data) {
  const captured = new Date(data.captured_at_utc);
  document.querySelector("#bench-v2-captured").textContent = captured.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short"
  });
  document.querySelector("#bench-v2-model-revision").textContent = `${data.model.revision.slice(0, 12)}…`;
  document.querySelector("#bench-v2-release").textContent = `${data.scoring.release_sha256.slice(0, 12)}…`;
  document.querySelector("#bench-v2-provenance-packs").textContent = `${data.progress.completed_packs} authoritative pack manifests`;
}

function benchV2RenderValidationComparison(data) {
  const body = document.querySelector("#bench-v2-validation-body");
  const facts = document.querySelector("#bench-v2-validation-facts");
  if (!body || !facts) return;
  body.closest("table").querySelector("thead").innerHTML = `<tr>
    <th scope="col">Validation task / metric</th>
    ${data.models.map((model) => `<th scope="col">${model.label}<br><span>P=${model.prompt_pairs}</span>${model.key === "instamanip-p2" ? "<br><span>Task instructions · updated</span>" : model.key === "visualcloze-p2" ? "<br><span>Task prompts · verified rerun</span>" : ""}</th>`).join("")}
    <th scope="col">Best score</th>
  </tr>`;
  const modelByKey = Object.fromEntries(data.models.map((model) => [model.key, model]));
  facts.innerHTML = `
    <div><dt>Models</dt><dd>${data.models.length}</dd></div>
    <div><dt>Tasks</dt><dd>${data.tasks}</dd></div>
    <div><dt>Queries / task</dt><dd>${data.queries_per_task}</dd></div>
    <div><dt>Episodes / model</dt><dd>${data.episodes_per_model}</dd></div>`;
  body.innerHTML = data.rows.map((row) => {
    const leader = modelByKey[row.leader];
    const rankedModels = data.models.map((model, index) => ({
      key: model.key,
      index,
      score: Number(row[model.value_field])
    })).sort((left, right) => {
      const difference = row.direction === "higher"
        ? right.score - left.score
        : left.score - right.score;
      return difference || left.index - right.index;
    });
    const rankByModel = Object.fromEntries(rankedModels.map((model, index) => [model.key, index + 1]));
    const modelCells = data.models.map((model) => {
      const leads = row.leader === model.key;
      const reported = model.key === "ours-p16";
      const rank = rankByModel[model.key];
      const classes = [`validation-rank-${rank}`, leads ? "validation-best" : "", reported ? "validation-reported" : ""].filter(Boolean).join(" ");
      const note = reported ? `#${rank} · TA · source-rounded` : model.key === "visualcloze-p2" ? `#${rank} · task prompts · P=2` : `#${rank} · native P=${model.prompt_pairs}`;
      return `<td class="${classes}"><span class="bench-v2-score">${benchV2Format(row[model.value_field])}</span><span class="cell-note">${note}</span></td>`;
    }).join("");
    return `<tr>
      <th scope="row">
        <span class="task-name">${benchV2TaskLabels[row.task] || row.task}</span>
        <span class="metric-name">${benchV2MetricLabels[row.metric] || row.metric} ${benchV2Arrow(row.direction)} · ${row.n} queries</span>
      </th>
      ${modelCells}
      <td><span class="validation-leader">${leader.label}</span><span class="cell-note">best corrected score</span></td>
    </tr>`;
  }).join("");
}

async function loadBenchV2ValidationComparison() {
  try {
    const response = await fetch("data/bench-v2-validation-models.json?v=67");
    if (!response.ok) throw new Error("validation comparison unavailable");
    benchV2RenderValidationComparison(await response.json());
  } catch (error) {
    const body = document.querySelector("#bench-v2-validation-body");
    if (body) body.innerHTML = '<tr><td colspan="14" class="loading-cell">The shared validation comparison could not be loaded.</td></tr>';
  }
}

async function loadBenchV2Partial() {
  try {
    const response = await fetch("data/bench-v2-visualcloze-p2-complete.json");
    if (!response.ok) throw new Error("bench_v2 complete data unavailable");
    const data = await response.json();
    benchV2RenderFacts(data);
    benchV2RenderRows(data);
    benchV2RenderProvenance(data);
    try {
      const scoreResponse = await fetch("data/all-eval-scores.csv");
      if (!scoreResponse.ok) throw new Error("site score ledger unavailable");
      benchV2RenderOurModelRows(data, benchV2ParseCsv(await scoreResponse.text()));
    } catch (error) {
      document.querySelector("#bench-v2-our-models-body").innerHTML = '<tr><td colspan="5" class="loading-cell">Our archived model references could not be loaded. The complete score ledger remains available below.</td></tr>';
    }
  } catch (error) {
    document.querySelector("#bench-v2-results-body").innerHTML = '<tr><td colspan="4" class="loading-cell">The bench_v2 partial snapshot could not be loaded. Historical site results remain available below.</td></tr>';
    document.querySelector("#bench-v2-our-models-body").innerHTML = '<tr><td colspan="5" class="loading-cell">The task-aligned model references could not be loaded.</td></tr>';
  }
}

loadBenchV2ValidationComparison();
loadBenchV2Partial();
