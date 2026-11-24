const taskLabels = {
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

const metricLabels = {
  lpips: "LPIPS",
  ois_f: "OIS F-score",
  nme: "NME",
  "pck_torso_0.2": "PCK torso @ 0.2",
  pq: "PQ",
  delta1: "δ₁",
  max_fbeta: "max Fβ",
  miou: "mIoU",
  mean_deg: "mean degrees"
};

const demoTaskLabels = taskLabels;

const prefixAssetPath = (path) => path.startsWith("assets/") ? `../${path}` : path;

function formatScore(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(6).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

async function loadValidationTable() {
  const head = document.querySelector("#validation-head");
  const body = document.querySelector("#validation-body");

  try {
    const response = await fetch("../data/bench-v2-validation-models.json?v=67");
    if (!response.ok) throw new Error("validation table unavailable");
    const data = await response.json();

    head.innerHTML = `<th scope="col">Task / metric</th>${data.models.map((model) =>
      `<th scope="col">${model.label}<span>P=${model.prompt_pairs}</span>${['prompt-diffusion-p1', 'visualcloze-p2'].includes(model.key) ? '<span>Task prompts · verified rerun</span>' : model.key === 'instamanip-p2' ? '<span>Task instructions · updated</span>' : ''}</th>`
    ).join("")}`;

    const taskOrder = Object.keys(taskLabels);
    const orderedRows = [
      ...taskOrder.map((task) => data.rows.find((row) => row.task === task)).filter(Boolean),
      ...data.rows.filter((row) => !taskOrder.includes(row.task))
    ];

    body.innerHTML = orderedRows.map((row) => {
      const rankedModels = data.models.map((model, index) => ({
        key: model.key,
        index,
        score: row[model.value_field]
      })).sort((left, right) => {
        const scoreDifference = row.direction === "higher"
          ? right.score - left.score
          : left.score - right.score;
        return scoreDifference || left.index - right.index;
      });
      const rankByModel = Object.fromEntries(
        rankedModels.map((model, index) => [model.key, index + 1])
      );
      const cells = data.models.map((model) => {
        const rank = rankByModel[model.key];
        const note = model.key === "ours-p16"
          ? "source-rounded"
          : model.key === "analogist-p1"
            ? "8-candidate mean"
            : model.key === "prompt-diffusion-p1"
            ? "task prompts · P=1"
            : model.key === "visualcloze-p2"
            ? "task prompts · P=2"
            : model.key === "instamanip-p2"
            ? "task instructions · P=2"
            : `native P=${model.prompt_pairs}`;
        const rankName = rank === 1 ? " Best" : rank === data.models.length ? " Worst" : "";
        return `<td class="validation-score rank-${rank}">
          <span class="rank-label"><b>#${rank}</b>${rankName}</span>
          <strong>${formatScore(row[model.value_field])}</strong>
          <span>${note}</span>
        </td>`;
      }).join("");
      return `<tr data-task="${row.task}">
        <th scope="row">
          <span class="task-name">${taskLabels[row.task] || row.task}</span>
          <span class="metric-name">${metricLabels[row.metric] || row.metric} ${row.direction === "higher" ? "↑" : "↓"} · ${row.n} queries</span>
        </th>
        ${cells}
      </tr>`;
    }).join("");
  } catch (error) {
    body.innerHTML = '<tr><td colspan="13" class="loading-cell">The validation table could not be loaded.</td></tr>';
  }
}

let demoData = null;
let demoTaskIndex = 0;
let demoModelIndex = 1;

function selectModelFromHash() {
  if (!demoData) return false;
  const requestedModel = window.location.hash.slice(1);
  const requestedIndex = demoData.models.findIndex((model) => model.key === requestedModel);
  if (requestedIndex < 0) return false;
  const task = demoData.tasks[demoTaskIndex];
  const available = demoData.rows.some((row) => row.task === task && row.model === requestedModel);
  if (!available) return false;
  demoModelIndex = requestedIndex;
  return true;
}

function demoFigure(file, alt, label) {
  const source = prefixAssetPath(file);
  return `<figure>
    <button type="button" class="bench-v2-example-image" data-example-src="${source}" data-example-alt="${alt}">
      <img src="${source}" alt="${alt}" width="256" height="256" loading="lazy">
      <span class="zoom-hint">Open ↗</span>
    </button>
    <figcaption>${label}</figcaption>
  </figure>`;
}

function selectTabs(container, selectedIndex) {
  container.querySelectorAll("button").forEach((button, index) => {
    const selected = index === selectedIndex;
    button.setAttribute("aria-selected", selected ? "true" : "false");
    button.tabIndex = selected ? 0 : -1;
  });
}

function escapeDemoText(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function demoTextPrompt(row) {
  const text = row.text_conditioning;
  if (!text) return '<div class="pd-text-prompt"><h4>Text prompt</h4><p>Not recorded in this archive.</p></div>';
  const fields = text.fields.map((field) => `<h4>${escapeDemoText(field.label)}</h4><p>${field.text === "" ? '<em>Empty string ("")</em>' : escapeDemoText(field.text)}</p>`).join("");
  return `<div class="pd-text-prompt" aria-label="Text conditioning used for this example">
    ${fields || '<h4>Text prompt</h4>'}<small>${escapeDemoText(text.note)}</small>
    ${row.model === "prompt-diffusion" ? `<p><a href="#promptdiffusion-examples/${encodeURIComponent(row.task)}/q0">View the task-prompt rerun →</a></p>` : ""}
  </div>`;
}

function renderDemo() {
  const task = demoData.tasks[demoTaskIndex];
  let model = demoData.models[demoModelIndex];
  let row = demoData.rows.find((candidate) => candidate.task === task && candidate.model === model.key);
  if (!row) {
    const painterIndex = demoData.models.findIndex((candidate) => candidate.key === "painter");
    const painterAvailable = demoData.rows.some((candidate) =>
      candidate.task === task && candidate.model === "painter"
    );
    const fallbackIndex = painterAvailable ? painterIndex : demoData.models.findIndex((candidate) =>
      demoData.rows.some((candidateRow) => candidateRow.task === task && candidateRow.model === candidate.key)
    );
    demoModelIndex = fallbackIndex;
    model = demoData.models[demoModelIndex];
    row = demoData.rows.find((candidate) => candidate.task === task && candidate.model === model.key);
  }
  const taskLabel = demoTaskLabels[task] || task;

  document.querySelector("#bench-v2-example-task").textContent = taskLabel;
  document.querySelector("#bench-v2-example-dataset").textContent = `bench_v2 validation · shared q0 · ${model.label}`;
  document.querySelector("#bench-v2-example-count").textContent = `${demoTaskIndex + 1} / ${demoData.tasks.length}`;
  const queryResult = `
    <div class="bench-v2-example-triptych">
      ${demoFigure(row.query.input.path, `${model.label}, ${taskLabel}, shared query input`, "Query input")}
      ${demoFigure(row.query.output.path, `${model.label}, ${taskLabel}, model output`, `${model.label} output`)}
      ${demoFigure(row.query.target.path, `${model.label}, ${taskLabel}, target`, "Ground truth")}
    </div>`;
  document.querySelector("#bench-v2-example-stage").innerHTML = `
    <article class="bench-v2-example-row model-comparison-row">
      <header class="bench-v2-example-row-heading">
        <p class="bench-v2-query-label">Query q0</p>
        <strong>${model.label} · P=${row.p}</strong>
        <span>${row.recipe}${row.status ? ` · ${row.status}` : ""}</span>
      </header>
      <div class="bench-v2-example-episode model-comparison-episode">
        <section class="bench-v2-example-block" aria-label="Frozen demonstrations for ${model.label}">
          <p class="bench-v2-example-block-label">Image prompt · ${row.demos.length} demonstration${row.demos.length === 1 ? "" : "s"}</p>
          <div class="model-demo-scroll">
            ${row.demos.map((demo) => `<div class="bench-v2-demo-pair">
              <p>Demo ${demo.position}</p>
              <div class="bench-v2-demo-pair-images">
                ${demoFigure(demo.input.path, `${model.label}, ${taskLabel}, demo ${demo.position} input`, "Input")}
                ${demoFigure(demo.target.path, `${model.label}, ${taskLabel}, demo ${demo.position} target`, "Target")}
              </div>
            </div>`).join("")}
          </div>
          ${demoTextPrompt(row)}
        </section>
        <section class="bench-v2-example-block bench-v2-query-block" aria-label="Shared query and ${model.label} result">
          <p class="bench-v2-example-block-label">Shared query q0 and ${Array.isArray(row.candidates) ? "candidate set" : "result"}</p>
          ${queryResult}
        </section>
      </div>
    </article>`;

  renderModelTabs(task);
  selectTabs(document.querySelector("#bench-v2-example-tabs"), demoTaskIndex);
}

function addTab(container, label, sublabel, onClick, disabled = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.role = "tab";
  button.innerHTML = sublabel ? `<span>${label}</span><small>${sublabel}</small>` : label;
  button.disabled = disabled;
  button.setAttribute("aria-disabled", disabled ? "true" : "false");
  button.addEventListener("click", onClick);
  container.append(button);
}

function renderModelTabs(task) {
  const modelTabs = document.querySelector("#model-example-tabs");
  modelTabs.innerHTML = "";
  demoData.models.forEach((model, index) => {
    const row = demoData.rows.find((candidate) => candidate.task === task && candidate.model === model.key);
    const coverage = row && Array.isArray(row.candidates) ? `P=${row.p} · ${row.candidates.length} candidates` : row ? `P=${row.p}` : "Not archived";
    addTab(modelTabs, model.label, coverage, () => {
      demoModelIndex = index;
      renderDemo();
    }, !row);
  });
  selectTabs(modelTabs, demoModelIndex);
}

async function loadDemos() {
  try {
    const response = await fetch("../data/bench-v2-validation-gallery.json?v=67");
    if (!response.ok) throw new Error("demos unavailable");
    demoData = await response.json();
    const hiddenGalleryModels = new Set(["analogist", "analogist-current-vision"]);
    demoData.models = demoData.models.filter((model) => !hiddenGalleryModels.has(model.key));
    demoData.rows = demoData.rows.filter((row) => !hiddenGalleryModels.has(row.model));

    const taskTabs = document.querySelector("#bench-v2-example-tabs");
    demoData.tasks.forEach((task, index) => {
      addTab(taskTabs, demoTaskLabels[task] || task, "", () => {
        demoTaskIndex = index;
        renderDemo();
      });
    });

    document.querySelector("#bench-v2-example-previous").addEventListener("click", () => {
      demoTaskIndex = (demoTaskIndex - 1 + demoData.tasks.length) % demoData.tasks.length;
      renderDemo();
    });
    document.querySelector("#bench-v2-example-next").addEventListener("click", () => {
      demoTaskIndex = (demoTaskIndex + 1) % demoData.tasks.length;
      renderDemo();
    });
    selectModelFromHash();
    renderDemo();
  } catch (error) {
    document.querySelector("#bench-v2-example-stage").innerHTML = '<p class="gallery-error">The demos could not be loaded.</p>';
  }
}

window.addEventListener("hashchange", () => {
  if (selectModelFromHash()) renderDemo();
});

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-example-src]");
  if (!button) return;
  const dialog = document.querySelector("#image-dialog");
  document.querySelector("#dialog-image").src = button.dataset.exampleSrc;
  document.querySelector("#dialog-image").alt = button.dataset.exampleAlt;
  document.querySelector("#dialog-title").textContent = button.dataset.exampleAlt;
  dialog.showModal();
});

document.querySelector("#dialog-close").addEventListener("click", () => {
  document.querySelector("#image-dialog").close();
});

loadValidationTable();
loadDemoCountTables();
loadDemos();

async function loadDemoCountTables() {
  const container = document.querySelector('#demo-count-tables');
  try {
    const response = await fetch('../data/bench-v2-validation-demo-counts.json?v=68');
    if (!response.ok) throw new Error('Demo-count scores unavailable');
    const data = await response.json();
    container.innerHTML = data.groups.map((group) => {
      const headings = group.models.map((model) => `<th scope="col">${model.label}
        <span>${model.p} demo${model.p === 1 ? '' : 's'}</span>
        ${model.key === 'lvm-p7' ? '<span class="recipe-qualification">Diagnostic snapshot</span>' : ''}
        ${model.model === 'instamanip' ? '<span class="recipe-qualification">Task-text recipe</span>' : ''}
        ${model.model === 'prompt-diffusion' ? '<span class="recipe-qualification">Task prompts · verified rerun</span>' : ''}
        ${model.model === 'visualcloze' && model.provenance?.receipt_file ? `<span class="recipe-qualification">Task prompts · verified rerun</span><span><a href="../data/${model.provenance.receipt_file}">Prompts &amp; provenance ↗</a></span>` : ''}
      </th>`).join('');
      const rows = data.tasks.map((task) => {
        const ranked = group.models.map((model) => model.scores[task.task])
          .filter(Number.isFinite)
          .sort((a, b) => task.direction === 'higher' ? b - a : a - b);
        return `<tr data-task="${task.task}">
          <th scope="row"><span class="task-name">${taskLabels[task.task]}</span>
            <span class="metric-name">${metricLabels[task.metric]} ${task.direction === 'higher' ? '↑' : '↓'} · ${task.n} queries</span>
          </th>
          ${group.models.map((model) => {
            const value = model.scores[task.task];
            if (!Number.isFinite(value)) return '<td class="loading-cell">Not scored</td>';
            const rank = ranked.indexOf(value) + 1;
            const qualifier = model.key === 'lvm-p7' ? 'Diagnostic snapshot'
              : model.model === 'visualcloze' && model.provenance?.receipt_file ? `Task prompts · P=${model.p}`
              : model.model === 'analogist' ? '8-candidate mean' : `${model.p} demo${model.p === 1 ? '' : 's'}`;
            return `<td class="validation-score rank-${rank}" title="${model.note}">
              <span class="rank-label"><b>#${rank}</b>${rank === 1 ? ' Best' : ''}</span>
              <strong>${formatScore(value)}</strong><span>${qualifier}</span>
            </td>`;
          }).join('')}
        </tr>`;
      }).join('');
      const note = group.key === '7-8'
        ? 'LVM 7B at 7 demos, together with VisualCloze and LaVin-DiT at 8 demos.'
        : `${group.models.length} models with completed ${group.demo_counts[0]}-demo scores.`;
      return `<article class="demo-count-group" id="demos-${group.key}" aria-labelledby="demos-${group.key}-title">
        <header class="demo-count-heading"><div><h3 id="demos-${group.key}-title">${group.title}</h3>
          <p>${note}</p></div><a class="download-scores" href="../data/${group.csv}" download>Download CSV ↗</a></header>
        <p class="table-scroll-hint">Scroll horizontally to compare all columns.</p>
        <div class="table-shell validation-table-shell" role="region" aria-label="${group.title} validation scores" tabindex="0">
          <table class="validation-comparison-table focused-validation-table demo-count-table ${group.models.length > 4 ? 'wide-demo-table' : 'compact-demo-table'}">
            <caption class="sr-only">${group.title}: ${group.models.length} models; 13 tasks; 17 queries per task.</caption>
            <thead><tr><th scope="col">Task / metric</th>${headings}</tr></thead><tbody>${rows}</tbody>
          </table>
        </div>
      </article>`;
    }).join('');
    if (/^#demos-(1|2|4|7-8)$/.test(window.location.hash)) {
      document.querySelector(window.location.hash)?.scrollIntoView();
    }
  } catch (error) {
    container.innerHTML = '<p class="gallery-error">The demo-count tables could not be loaded. Reload the page or use the CSV downloads below.</p><p><a href="../data/bench-v2-validation-demos-1.csv">1 demo</a> · <a href="../data/bench-v2-validation-demos-2.csv">2 demos</a> · <a href="../data/bench-v2-validation-demos-4.csv">4 demos</a> · <a href="../data/bench-v2-validation-demos-7-8.csv">7/8 demos</a></p>';
  }
}
