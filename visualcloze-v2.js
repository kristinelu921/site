const taskLabels = {
  colorization: "Colorization",
  deblurring: "Deblurring",
  denoising: "Denoising",
  edge_detection: "Edge detection",
  face_landmarks: "Face landmarks",
  human_pose_2d: "Human pose 2D",
  inpainting: "Inpainting",
  instance_segmentation: "Instance segmentation",
  object_keypoints: "Object keypoints",
  panoptic_segmentation: "Panoptic segmentation",
  relative_depth: "Relative depth",
  saliency_detection: "Saliency detection",
  semantic_segmentation: "Semantic segmentation",
  super_resolution: "Super-resolution",
  surface_normals: "Surface normals"
};

const metricLabels = {
  lpips: "LPIPS",
  lpips_alex: "LPIPS Alex",
  ois_f: "OIS F-score",
  pooled_ois: "pooled OIS",
  nme: "NME",
  nme_bbox_matched: "matched bbox NME",
  "pck_torso_0.2": "PCK torso @ 0.2",
  pck_bbox_reject: "bbox-reject PCK",
  pq: "PQ",
  pq_novoid: "PQ no-void",
  delta1: "δ₁",
  delta1_aligned_ssc: "δ₁ aligned SSC",
  max_fbeta: "max Fβ",
  miou: "mIoU",
  mean_deg: "mean degrees"
};

const taskOrder = Object.keys(taskLabels);

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toFixed(6).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

function directionArrow(direction) {
  return direction === "higher" ? "↑" : "↓";
}

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function renderTaskScoreRows(selector, rows) {
  const taskSummary = aggregateNewPackByTask(rows);
  const summaryTasks = taskOrder.filter((task) => taskSummary[task]);
  document.querySelector(selector).innerHTML = summaryTasks
    .map((task) => {
      const row = taskSummary[task];
      const metric = metricLabels[row.primary_metric] || row.primary_metric;
      return `<tr>
        <th scope="row">${taskLabels[task] || task}</th>
        <td>
          <strong>${formatScore(row.mean)}</strong>
          <span class="task-score-meta">${metric} ${directionArrow(row.direction)} · n=${row.n.toLocaleString()}</span>
        </td>
      </tr>`;
    }).join("");
}

function renderNewPack(data) {
  const progress = data.progress;
  document.querySelector("#run-facts").children[0].querySelector("dd").textContent = `${progress.completed_packs} / ${progress.total_packs} packs`;
  document.querySelector("#new-pack-status").children[0].querySelector("strong").textContent = `${progress.completed_packs} / ${progress.total_packs}`;
  document.querySelector("#new-pack-status").children[1].querySelector("strong").textContent = `${progress.completed_queries.toLocaleString()} / ${progress.total_queries.toLocaleString()}`;
  renderTaskScoreRows("#new-pack-test-task-rows", data.rows);
  document.querySelector("#new-pack-body").innerHTML = [...data.rows]
    .sort((a, b) => taskOrder.indexOf(a.task) - taskOrder.indexOf(b.task) || a.dataset.localeCompare(b.dataset))
    .map((row) => `<tr>
      <th scope="row"><span class="task-name">${taskLabels[row.task] || row.task}</span><span class="dataset-name">${row.dataset}</span></th>
      <td>${row.n.toLocaleString()}</td>
      <td><span class="metric-pill">${metricLabels[row.primary_metric] || row.primary_metric} ${directionArrow(row.direction)}</span></td>
      <td class="score-cell"><strong>${formatScore(row.mean)}</strong></td>
    </tr>`).join("");

  const captured = new Date(data.captured_at_utc);
  document.querySelector("#snapshot-time").textContent = `Snapshot ${captured.toLocaleString(undefined, {year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short"})}`;
}

function renderValidationComparison(data) {
  const modelByKey = Object.fromEntries(data.models.map((model) => [model.key, model]));
  document.querySelector("#validation-comparison-body").closest("table").querySelector("thead").innerHTML = `<tr>
    <th scope="col">Task / metric</th>
    ${data.models.map((model) => `<th scope="col">${model.label}<br><span>${model.recipe.split(" · ")[0]}</span></th>`).join("")}
  </tr>`;
  document.querySelector("#validation-model-status").innerHTML = data.models.map((model) => `
    <article class="validation-status-card ${model.status_tone}">
      <strong>${model.label}</strong>
      <span>${model.recipe}</span>
      <em>${model.status}</em>
    </article>`).join("");

  document.querySelector("#validation-comparison-body").innerHTML = data.rows
    .sort((a, b) => taskOrder.indexOf(a.task) - taskOrder.indexOf(b.task))
    .map((row) => {
      const rankedModels = data.models.map((model, index) => ({
        key: model.key,
        index,
        score: Number(row[model.key])
      })).sort((left, right) => {
        const difference = row.direction === "higher"
          ? right.score - left.score
          : left.score - right.score;
        return difference || left.index - right.index;
      });
      const rankByModel = Object.fromEntries(rankedModels.map((model, index) => [model.key, index + 1]));
      const metric = metricLabels[row.metric] || row.metric;
      const aggregation = row.aggregation === "corrected_pooled_dataset_readout" ? "pooled dataset" : "mean per query";
      return `<tr>
        <th scope="row">
          <span class="task-name">${taskLabels[row.task] || row.task}</span>
          <span class="dataset-name">${metric} ${directionArrow(row.direction)} · ${aggregation}</span>
        </th>
        ${data.models.map((model) => {
          const value = Number(row[model.key]);
          const rank = rankByModel[model.key];
          return `<td class="validation-score validation-rank-${rank} ${model.key}-validation-score${rank === 1 ? " best-score" : ""}">
            <strong>${formatScore(value)}</strong>
            <span>#${rank}${rank === 1 ? " · best" : ""} · n=${row.n}</span>
          </td>`;
        }).join("")}
      </tr>`;
    }).join("");

  const lvm = modelByKey.lvm;
  document.querySelector("#validation-comparison-title").title = lvm.status;
}

function aggregateNewPackByTask(rows) {
  const groups = {};
  rows.forEach((row) => {
    const n = Number(row.n);
    const mean = Number(row.mean);
    if (!groups[row.task]) {
      groups[row.task] = {
        task: row.task,
        primary_metric: row.primary_metric,
        direction: row.direction,
        n: 0,
        weightedSum: 0,
        dataset_count: 0
      };
    }
    const group = groups[row.task];
    if (group.primary_metric !== row.primary_metric || group.direction !== row.direction) {
      throw new Error(`new-pack metric contract differs within ${row.task}`);
    }
    group.n += n;
    group.weightedSum += mean * n;
    group.dataset_count += 1;
  });
  return Object.fromEntries(Object.entries(groups).map(([task, group]) => [task, {
    ...group,
    mean: group.weightedSum / group.n
  }]));
}

function comparisonCell(row, className, scope) {
  if (!row) {
    return '<td class="comparison-result comparison-empty"><span>Not evaluated</span></td>';
  }
  const metric = metricLabels[row.primary_metric] || row.primary_metric;
  return `<td class="comparison-result ${className}">
    <strong>${formatScore(row.mean)}</strong>
    <span class="cell-metric">${metric} ${directionArrow(row.direction)}</span>
    <span class="cell-scope">${scope}</span>
  </td>`;
}

function renderEstablishedPack(newPackData, rows) {
  const visualClozeRows = rows.filter((row) =>
    row.model === "VisualCloze" &&
    row.protocol === "Native P=2" &&
    row.condition === "real" &&
    row.split === "Test" &&
    row.scope === "15 tasks × 64 queries"
  );
  const jfluxRows = rows.filter((row) =>
    row.model === "JFlux T3EZ" &&
    row.protocol === "TTT · P=16 · 200 steps" &&
    row.condition === "real" &&
    row.split === "Test · q0 subset"
  );
  const newByTask = aggregateNewPackByTask(newPackData.rows);
  const vcByTask = Object.fromEntries(visualClozeRows.map((row) => [row.task, row]));
  const jfByTask = Object.fromEntries(jfluxRows.map((row) => [row.task, row]));

  document.querySelector("#old-pack-body").innerHTML = taskOrder
    .filter((task) => newByTask[task] || vcByTask[task] || jfByTask[task])
    .map((task) => {
      const newPack = newByTask[task];
      const vc = vcByTask[task];
      const jf = jfByTask[task];
      const newScope = newPack
        ? `${newPack.dataset_count} ${newPack.dataset_count === 1 ? "dataset" : "datasets"} · n=${newPack.n.toLocaleString()}`
        : "";
      return `<tr>
        <th scope="row"><span class="task-name">${taskLabels[task] || task}</span></th>
        ${comparisonCell(newPack, "new-vc-score", newScope)}
        ${comparisonCell(vc, "vc-score", vc ? `old v5 · n=${Number(vc.n).toLocaleString()}` : "")}
        ${comparisonCell(jf, "jf-score", jf ? `old v5 · n=${Number(jf.n).toLocaleString()}` : "")}
      </tr>`;
    }).join("");
}

async function loadResults() {
  try {
    const [newResponse, oldResponse, validationResponse] = await Promise.all([
      fetch("data/bench-v2-visualcloze-p2-complete.json"),
      fetch("data/all-eval-scores.csv"),
      fetch("data/bench-v2-validation-comparison.json?v=66")
    ]);
    if (!newResponse.ok || !oldResponse.ok || !validationResponse.ok) throw new Error("result source unavailable");
    const newPackData = await newResponse.json();
    renderNewPack(newPackData);
    renderValidationComparison(await validationResponse.json());
    renderEstablishedPack(newPackData, parseCsv(await oldResponse.text()));
  } catch (error) {
    document.querySelector("#new-pack-body").innerHTML = '<tr><td colspan="4" class="loading-cell">The new-pack result snapshot could not be loaded.</td></tr>';
    document.querySelector("#validation-comparison-body").innerHTML = '<tr><td colspan="11" class="loading-cell">The aligned validation table could not be loaded.</td></tr>';
    document.querySelector("#old-pack-body").innerHTML = '<tr><td colspan="4" class="loading-cell">The side-by-side score table could not be loaded.</td></tr>';
  }
}

loadResults();
