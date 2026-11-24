const paperModels = [
  { name: "VisualCloze", slug: "visualcloze", href: "#test", protocol: "Native P=2" },
  { name: "LVM", slug: "lvm", href: "lvm/", protocol: "Native P=2" },
  { name: "Painter", slug: "painter", href: "painter/", protocol: "P=1 ×2 mean" },
  { name: "GenLV", slug: "genlv", href: "genlv/", protocol: "P=1 ×2 mean" },
  { name: "Prompt Diffusion", slug: "prompt-diffusion", href: "prompt-diffusion/", protocol: "P=1 ×2 mean" },
  { name: "IMProv", slug: "improv", href: "improv/", protocol: "P=1 ×2 mean" },
  { name: "RelationAdapter", slug: "relation-adapter", href: "relation-adapter/", protocol: "P=1 ×2 mean" }
];

const paperTasks = [
  { key: "edge_detection", label: "Edge detection", metric: "OIS F-score", direction: "higher" },
  { key: "saliency_detection", label: "Saliency", metric: "max Fβ", direction: "higher" },
  { key: "semantic_segmentation", label: "Semantic segmentation", metric: "mIoU", direction: "higher" },
  { key: "relative_depth", label: "Relative depth", metric: "δ₁", direction: "higher" },
  { key: "surface_normals", label: "Surface normals", metric: "MAE (°)", direction: "lower" },
  { key: "colorization", label: "Colorization", metric: "LPIPS", direction: "lower" },
  { key: "inpainting", label: "Inpainting", metric: "LPIPS", direction: "lower" },
  { key: "deblurring", label: "Deblurring", metric: "LPIPS", direction: "lower" },
  { key: "denoising", label: "Denoising", metric: "LPIPS", direction: "lower" },
  { key: "super_resolution", label: "Super-resolution", metric: "LPIPS", direction: "lower" },
  { key: "human_pose_2d", label: "Human pose", metric: "PCK", direction: "higher" },
  { key: "face_landmarks", label: "Face landmarks", metric: "NME", direction: "lower" },
  { key: "object_keypoints", label: "Object keypoints", metric: "PCK", direction: "higher" },
  { key: "panoptic_segmentation", label: "Panoptic segmentation", metric: "PQ", direction: "higher" }
];

const suppliedOurs = {
  edge_detection: 0.79,
  saliency_detection: 0.78,
  semantic_segmentation: 0.17,
  relative_depth: 0.51,
  surface_normals: 45.10,
  colorization: 0.22,
  inpainting: 0.07,
  deblurring: 0.11,
  denoising: 0.17,
  super_resolution: 0.17,
  human_pose_2d: 0.06,
  face_landmarks: 0.42,
  object_keypoints: 0.16,
  panoptic_segmentation: 0.15
};

function parsePaperCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function paperFormat(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(6).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

function paperHeader() {
  return `<tr><th scope="col">Method</th>${paperTasks.map((task) => `
    <th scope="col"><span class="paper-task-name">${task.label}</span><span class="paper-metric-name"><i>${task.metric}</i> ${task.direction === "higher" ? "↑" : "↓"}</span></th>`).join("")}</tr>`;
}

function scoreRank(values, direction) {
  const unique = [...new Set(values.filter(Number.isFinite))].sort((a, b) => direction === "higher" ? b - a : a - b);
  return { best: unique[0], second: unique[1] };
}

function near(a, b) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= Math.max(1e-12, Math.abs(b) * 1e-12);
}

function renderPaperTable(scoreRows, vict) {
  document.querySelector("#paper-score-head").innerHTML = paperHeader();
  const rows = paperModels.map((model) => {
    const cells = new Map(scoreRows
      .filter((row) => row.model_slug === model.slug && row.split === "Test" && row.condition === "real" && row.prompt_count === "2")
      .map((row) => [row.task, { value: Number(row.mean), exact: row.mean, rankable: true }]));
    return { ...model, cells };
  });

  const victCells = new Map();
  Object.entries(vict.cells || {}).forEach(([task, cell]) => {
    if (!paperTasks.some((item) => item.key === task)) return;
    victCells.set(task, {
      value: Number(cell.vict_mean),
      exact: String(cell.vict_mean),
      rankable: Boolean(cell.complete),
      partial: !cell.complete,
      n: cell.n,
      expectedN: cell.expected_n
    });
  });
  rows.push({ name: "VICT · legacy test snapshot", slug: "vict", href: "vict/data/vict-partial-summary.json", protocol: "P=1 TTT · partial test", cells: victCells });
  rows.push({
    name: "Ours",
    slug: "ours",
    href: null,
    protocol: "Supplied reference row",
    cells: new Map(Object.entries(suppliedOurs).map(([task, value]) => [task, { value, exact: String(value), rankable: true, supplied: true }]))
  });

  const ranks = new Map(paperTasks.map((task) => {
    const values = rows.map((row) => row.cells.get(task.key)).filter((cell) => cell?.rankable).map((cell) => cell.value);
    return [task.key, scoreRank(values, task.direction)];
  }));

  document.querySelector("#paper-score-body").innerHTML = rows.map((row) => {
    const method = row.href ? `<a href="${row.href}">${row.name}</a>` : row.name;
    const cells = paperTasks.map((task) => {
      const cell = row.cells.get(task.key);
      if (!cell) return `<td class="paper-missing" aria-label="${row.name}, ${task.label}: unavailable">—</td>`;
      const rank = ranks.get(task.key);
      const className = near(cell.value, rank.best) ? "paper-best" : near(cell.value, rank.second) ? "paper-second" : "";
      const partial = cell.partial ? `<sup title="Incomplete: ${cell.n} of ${cell.expectedN} queries">†</sup>` : "";
      const source = cell.supplied ? "; supplied reference value" : "";
      const completeness = cell.partial ? `; incomplete ${cell.n}/${cell.expectedN}, excluded from highlighting` : "";
      return `<td class="${className}" title="Exact stored mean: ${cell.exact}${source}${completeness}"><span>${paperFormat(cell.value)}${partial}</span></td>`;
    }).join("");
    return `<tr class="paper-model-${row.slug}"><th scope="row"><span class="paper-method-name">${method}</span><span class="paper-protocol">${row.protocol}</span></th>${cells}</tr>`;
  }).join("");
}

async function loadPaperTable() {
  try {
    const [scoresResponse, victResponse] = await Promise.all([
      fetch("data/all-eval-scores.csv"),
      fetch("vict/data/vict-partial-summary.json")
    ]);
    if (!scoresResponse.ok || !victResponse.ok) throw new Error("score data unavailable");
    renderPaperTable(parsePaperCsv(await scoresResponse.text()), await victResponse.json());
  } catch (error) {
    document.querySelector("#paper-score-body").innerHTML = '<tr><td colspan="15" class="loading-cell">The corrected method × task table could not be loaded. The exact score matrix below remains available.</td></tr>';
  }
}

loadPaperTable();
