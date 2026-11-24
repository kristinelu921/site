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

let data = null;
let taskIndex = 0;

const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  "'": "&#39;",
  '"': "&quot;"
})[character]);

const assetPath = (path) => path.startsWith("assets/") ? `../${path}` : path;

function figure(descriptor, label, alt) {
  const source = assetPath(descriptor.path);
  return `<figure>
    <button type="button" data-image-src="${escapeHtml(source)}" data-image-alt="${escapeHtml(alt)}">
      <img src="${escapeHtml(source)}" alt="${escapeHtml(alt)}" width="256" height="256" loading="lazy">
      <span>Open ↗</span>
    </button>
    <figcaption>${escapeHtml(label)}</figcaption>
  </figure>`;
}

function render() {
  const row = data.rows[taskIndex];
  const taskLabel = taskLabels[row.task] || row.task;
  document.querySelector("#task-count").textContent = `${taskIndex + 1} / ${data.rows.length}`;
  document.querySelector("#analogist-stage").innerHTML = `
    <header class="analogist-stage-heading">
      <div><p>SHARED VALIDATION QUERY Q0</p><h3>${escapeHtml(taskLabel)}</h3></div>
      <span>P=1 · 50 DDIM steps · 8 candidates</span>
    </header>
    <div class="analogist-context-grid">
      <section>
        <p class="analogist-block-label">Demonstration</p>
        <div class="analogist-image-pair">
          ${figure(row.demos[0].input, "Demo input", `${taskLabel} demonstration input`)}
          ${figure(row.demos[0].target, "Demo target", `${taskLabel} demonstration target`)}
        </div>
      </section>
      <section>
        <p class="analogist-block-label">Query and ground truth</p>
        <div class="analogist-image-pair">
          ${figure(row.query.input, "Query input", `${taskLabel} query input`)}
          ${figure(row.query.target, "Ground truth", `${taskLabel} ground truth`)}
        </div>
      </section>
    </div>
    <aside class="analogist-prompt"><strong>Current-vision prompt</strong><span>${escapeHtml(row.conditioned_prompt)}</span></aside>
    <div class="analogist-candidate-heading"><h3>Eight retained candidates</h3><span>No target-based selection</span></div>
    <div class="analogist-candidate-grid">
      ${row.candidates.map((candidate, index) => figure(candidate, `Candidate ${index + 1}`, `${taskLabel} candidate ${index + 1}`)).join("")}
    </div>`;

  document.querySelectorAll("#task-tabs button").forEach((button, index) => {
    const selected = index === taskIndex;
    button.setAttribute("aria-selected", selected ? "true" : "false");
    button.tabIndex = selected ? 0 : -1;
  });
}

function buildTabs() {
  const tabs = document.querySelector("#task-tabs");
  data.rows.forEach((row, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.textContent = taskLabels[row.task] || row.task;
    button.addEventListener("click", () => {
      taskIndex = index;
      render();
    });
    tabs.append(button);
  });
}

document.querySelector("#previous-task").addEventListener("click", () => {
  taskIndex = (taskIndex - 1 + data.rows.length) % data.rows.length;
  render();
});

document.querySelector("#next-task").addEventListener("click", () => {
  taskIndex = (taskIndex + 1) % data.rows.length;
  render();
});

document.querySelector("#analogist-stage").addEventListener("click", (event) => {
  const button = event.target.closest("[data-image-src]");
  if (!button) return;
  const dialog = document.querySelector("#image-dialog");
  document.querySelector("#dialog-image").src = button.dataset.imageSrc;
  document.querySelector("#dialog-image").alt = button.dataset.imageAlt;
  document.querySelector("#dialog-title").textContent = button.dataset.imageAlt;
  dialog.showModal();
});

document.querySelector("#dialog-close").addEventListener("click", () => {
  document.querySelector("#image-dialog").close();
});

fetch("../data/analogist-current-vision-bench-v2-validation.json?v=52")
  .then((response) => {
    if (!response.ok) throw new Error("Analogist result source unavailable");
    return response.json();
  })
  .then((result) => {
    if (result.status !== "INFERENCE_COMPLETE_SCORES_PENDING" || result.rows.length !== 13) {
      throw new Error("Analogist validation manifest differs");
    }
    data = result;
    buildTabs();
    render();
  })
  .catch((error) => {
    document.querySelector("#analogist-stage").innerHTML = `<p class="analogist-error">Analogist outputs could not be loaded: ${escapeHtml(error.message)}</p>`;
  });
