(() => {
  const category = document.querySelector('#pd-category');
  const query = document.querySelector('#pd-query');
  const previous = document.querySelector('#pd-previous');
  const next = document.querySelector('#pd-next');
  const stage = document.querySelector('#pd-example-stage');
  const status = document.querySelector('#pd-example-status');
  let data;
  let taskIndex = 0;
  let queryIndex = 0;
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  function selectFromHash() {
    const match = window.location.hash.match(/^#promptdiffusion-examples\/([a-z0-9_]+)\/q(\d+)$/);
    if (!match) return false;
    const index = data.groups.findIndex(group => group.task === match[1]);
    if (index < 0) return false;
    const sampleIndex = data.groups[index].samples.findIndex(sample => sample.q === Number(match[2]));
    if (sampleIndex < 0) return false;
    taskIndex = index;
    queryIndex = sampleIndex;
    return true;
  }

  function render(updateHash = false) {
    const group = data.groups[taskIndex];
    const sample = group.samples[queryIndex];
    category.value = group.task;
    query.innerHTML = group.samples.map((row, index) => `<option value="${index}">${index + 1} of ${group.samples.length} · q${row.q}</option>`).join('');
    query.value = String(queryIndex);
    previous.disabled = queryIndex === 0;
    next.disabled = queryIndex === group.samples.length - 1;
    status.textContent = `${group.label} · example ${queryIndex + 1} of ${group.samples.length} · 221 examples total`;
    const figure = (key, label) => demoFigure(sample.images[key].path,
      escape(`PromptDiffusion, ${group.label}, q${sample.q}, ${label}`), label);
    const metric = `${escape(metricLabels[group.metric] || group.metric)} ${group.direction === 'lower' ? '↓' : '↑'}`;
    const columns = data.variants.map(variant => {
      const output = sample.outputs[variant.key];
      return `<article class="pd-output-column" aria-label="${escape(variant.label)}">
        <header><h3>${escape(variant.label)}</h3><small>${escape(variant.subtitle)}</small></header>
        ${demoFigure(output.image.path, escape(`PromptDiffusion, ${group.label}, q${sample.q}, ${variant.label} output`), `${escape(variant.label)} output`)}
        <p class="pd-score">${metric}: <strong>${formatScore(output.score)}</strong> <span>this example</span><br>Category average: ${formatScore(group.means[variant.key])}</p>
        <div class="pd-text-prompt"><h4>Exact text prompt</h4><p>${output.text_prompt === '' ? '<em>Empty string ("")</em>' : escape(output.text_prompt)}</p><small>${escape(output.prompt_source)}</small></div>
      </article>`;
    }).join('');
    stage.innerHTML = `<div class="pd-comparison">
      <section class="pd-shared" aria-label="Shared demonstration pair, query, and ground truth">
        <h3>Same demonstration and query for every column</h3>
        <div class="pd-shared-images">${figure('demo_input', 'Demo input')}${figure('demo_target', 'Demo output')}${figure('query', 'Query')}${figure('ground_truth', 'Ground truth')}</div>
      </section>
      <p class="pd-scroll-hint">Compare the three outputs below. On a narrow screen, scroll sideways.</p>
      <div class="pd-output-scroll" tabindex="0" role="region" aria-label="Three prompt and output columns; scroll horizontally on narrow screens">
        <div class="pd-output-columns">${columns}</div>
      </div>
    </div>`;
    if (updateHash) {
      window.history.replaceState(null, '', `#promptdiffusion-examples/${group.task}/q${sample.q}`);
    }
  }

  async function load() {
    try {
      const response = await fetch('../data/promptdiffusion-prompt-comparison.json?v=64');
      if (!response.ok) throw new Error('Examples unavailable');
      data = await response.json();
      if (data.total_examples !== 221 || data.groups.length !== 13 || data.variants.length !== 3 ||
          data.groups.some(group => group.samples.length !== 17 || group.samples.some(sample =>
            data.variants.some(variant => !sample.outputs[variant.key])))) throw new Error('Incomplete examples');
      category.innerHTML = data.groups.map(group => `<option value="${escape(group.task)}">${escape(group.label)}</option>`).join('');
      category.disabled = false;
      query.disabled = false;
      category.addEventListener('change', () => {
        taskIndex = data.groups.findIndex(group => group.task === category.value);
        queryIndex = 0;
        render(true);
      });
      query.addEventListener('change', () => { queryIndex = Number(query.value); render(true); });
      previous.addEventListener('click', () => { if (queryIndex > 0) { queryIndex -= 1; render(true); } });
      next.addEventListener('click', () => {
        if (queryIndex + 1 < data.groups[taskIndex].samples.length) { queryIndex += 1; render(true); }
      });
      window.addEventListener('hashchange', () => {
        if (selectFromHash()) { render(); document.querySelector('#promptdiffusion-examples').scrollIntoView(); }
      });
      selectFromHash();
      render();
      if (window.location.hash.startsWith('#promptdiffusion-examples')) {
        document.querySelector('#promptdiffusion-examples').scrollIntoView();
      }
    } catch (error) {
      category.disabled = true;
      query.disabled = true;
      previous.disabled = true;
      next.disabled = true;
      status.textContent = 'The examples could not be loaded. Reload the page to try again.';
      stage.innerHTML = '';
    }
  }
  load();
})();
