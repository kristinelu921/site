const validationExampleTaskLabels = {
  semantic_segmentation: "Semantic segmentation",
  relative_depth: "Relative depth",
  denoising: "Denoising"
};

let validationExampleData = null;
let validationExampleTaskIndex = 0;
let validationExampleModelIndex = 0;

function validationModelTabContainer() {
  return document.querySelector("#bench-v2-example-models") || document.querySelector("#model-example-tabs");
}

function validationExampleFigure(file, alt, label) {
  return `<figure>
    <button type="button" class="bench-v2-example-image" data-example-src="${file}" data-example-alt="${alt}">
      <img src="${file}" alt="${alt}" width="256" height="256" loading="lazy">
      <span class="zoom-hint">Open ↗</span>
    </button>
    <figcaption>${label}</figcaption>
  </figure>`;
}

function setSelectedTabs(container, selectedIndex) {
  container.querySelectorAll("button").forEach((button, index) => {
    const selected = index === selectedIndex;
    button.setAttribute("aria-selected", selected ? "true" : "false");
    button.tabIndex = selected ? 0 : -1;
  });
}

function renderValidationExample() {
  const task = validationExampleData.tasks[validationExampleTaskIndex];
  const model = validationExampleData.models[validationExampleModelIndex];
  const row = validationExampleData.rows.find((candidate) => candidate.task === task && candidate.model === model.key);
  const taskLabel = validationExampleTaskLabels[task] || task;
  const detail = document.querySelector("#bench-v2-example-model-detail");

  if (detail) detail.textContent = row.recipe;
  document.querySelector("#bench-v2-example-task").textContent = taskLabel;
  document.querySelector("#bench-v2-example-dataset").textContent = `bench_v2 validation · shared q0 · ${model.label}`;
  document.querySelector("#bench-v2-example-count").textContent = `${validationExampleTaskIndex + 1} / ${validationExampleData.tasks.length}`;
  document.querySelector("#bench-v2-example-stage").innerHTML = `
    <article class="bench-v2-example-row model-comparison-row">
      <header class="bench-v2-example-row-heading">
        <p class="bench-v2-query-label">Query q0</p>
        <strong>${model.label} · P=${row.p}</strong>
        <span>Exact frozen episode</span>
      </header>
      <div class="bench-v2-example-episode model-comparison-episode">
        <section class="bench-v2-example-block" aria-label="Exact in-context demonstrations for ${model.label}">
          <p class="bench-v2-example-block-label">Actual prompt · ${row.demos.length} demonstration${row.demos.length === 1 ? "" : "s"}</p>
          <div class="model-demo-scroll">
            ${row.demos.map((demo) => `<div class="bench-v2-demo-pair">
              <p>Demo ${demo.position}</p>
              <div class="bench-v2-demo-pair-images">
                ${validationExampleFigure(demo.input.path, `${model.label}, ${taskLabel}, demo ${demo.position} input`, "Input")}
                ${validationExampleFigure(demo.target.path, `${model.label}, ${taskLabel}, demo ${demo.position} target`, "Target")}
              </div>
            </div>`).join("")}
          </div>
        </section>
        <section class="bench-v2-example-block bench-v2-query-block" aria-label="Shared query and ${model.label} result">
          <p class="bench-v2-example-block-label">Shared query q0 and result</p>
          <div class="bench-v2-example-triptych">
            ${validationExampleFigure(row.query.input.path, `${model.label}, ${taskLabel}, shared query input`, "Query input")}
            ${validationExampleFigure(row.query.output.path, `${model.label}, ${taskLabel}, model output`, `${model.label} output`)}
            ${validationExampleFigure(row.query.target.path, `${model.label}, ${taskLabel}, episode-coded target`, "Ground truth")}
          </div>
        </section>
      </div>
    </article>`;

  setSelectedTabs(validationModelTabContainer(), validationExampleModelIndex);
  setSelectedTabs(document.querySelector("#bench-v2-example-tabs"), validationExampleTaskIndex);
}

function makeTab(container, label, sublabel, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.role = "tab";
  button.innerHTML = sublabel ? `<span>${label}</span><small>${sublabel}</small>` : label;
  button.addEventListener("click", onClick);
  container.append(button);
}

async function loadValidationExamples() {
  try {
    const response = await fetch("data/bench-v2-validation-model-examples.json");
    if (!response.ok) throw new Error("gallery source unavailable");
    validationExampleData = await response.json();

    const modelTabs = validationModelTabContainer();
    validationExampleData.models.forEach((model, index) => {
      const row = validationExampleData.rows.find((candidate) => candidate.model === model.key);
      makeTab(modelTabs, model.label, `P=${row.p}`, () => {
        validationExampleModelIndex = index;
        renderValidationExample();
      });
    });

    const taskTabs = document.querySelector("#bench-v2-example-tabs");
    validationExampleData.tasks.forEach((task, index) => {
      makeTab(taskTabs, validationExampleTaskLabels[task] || task, "", () => {
        validationExampleTaskIndex = index;
        renderValidationExample();
      });
    });

    document.querySelector("#bench-v2-example-previous").addEventListener("click", () => {
      validationExampleTaskIndex = (validationExampleTaskIndex - 1 + validationExampleData.tasks.length) % validationExampleData.tasks.length;
      renderValidationExample();
    });
    document.querySelector("#bench-v2-example-next").addEventListener("click", () => {
      validationExampleTaskIndex = (validationExampleTaskIndex + 1) % validationExampleData.tasks.length;
      renderValidationExample();
    });
    renderValidationExample();
  } catch (error) {
    document.querySelector("#bench-v2-example-stage").innerHTML = '<p class="gallery-error">The aligned validation examples could not be loaded.</p>';
  }
}

document.querySelector("#bench-v2-example-stage").addEventListener("click", (event) => {
  const button = event.target.closest("[data-example-src]");
  if (!button) return;
  const imageDialog = document.querySelector("#image-dialog");
  document.querySelector("#dialog-image").src = button.dataset.exampleSrc;
  document.querySelector("#dialog-image").alt = button.dataset.exampleAlt;
  document.querySelector("#dialog-title").textContent = button.dataset.exampleAlt;
  imageDialog.showModal();
});

loadValidationExamples();
