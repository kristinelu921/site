(() => {
  const category = document.querySelector('#ra-category');
  const stage = document.querySelector('#ra-example-stage');
  const status = document.querySelector('#ra-example-status');
  const previous = document.querySelector('#ra-previous');
  const next = document.querySelector('#ra-next');
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let data, index = 0;
  const metric = row => `${escape(metricLabels[row.metric] || row.metric)} ${row.lower_is_better ? '↓' : '↑'}`;

  function fromHash() {
    const match = location.hash.match(/^#relationadapter-prompts\/([a-z0-9_]+)$/);
    if (!match) return false;
    const found = data.rows.findIndex(row => row.task === match[1]);
    if (found < 0) return false;
    index = found;
    return true;
  }

  function render(updateHash = false) {
    const row = data.rows[index];
    category.value = row.task;
    previous.disabled = index === 0;
    next.disabled = index === data.rows.length - 1;
    status.textContent = `${row.label} · q0 · task ${index + 1} of 13 · one query, two prompt versions`;
    const figure = (key, label) => demoFigure(row.images[key].path, escape(`RelationAdapter, ${row.label}, q0, ${label}`), label);
    stage.innerHTML = `<div class="pd-comparison">
      <section class="pd-shared" aria-label="Shared demonstration pair, query, and ground truth"><h3>Same demonstration and query for both prompts</h3>
        <div class="pd-shared-images">${figure('demo_input','Demo input')}${figure('demo_target','Demo output')}${figure('query_input','Query')}${figure('query_target','Ground truth')}</div>
      </section>
      <p class="pd-scroll-hint">Compare both outputs below. On a narrow screen, scroll sideways.</p>
      <div class="pd-output-scroll" tabindex="0" role="region" aria-label="Two prompt and output columns; scroll horizontally on narrow screens">
        <div class="pd-output-columns ra-output-columns">${data.variants.map(variant => {
          const output = row.outputs[variant.key];
          return `<article class="pd-output-column" aria-label="${escape(variant.label)}">
            <header><h3>${escape(variant.label)}</h3><small>${escape(variant.subtitle)}</small></header>
            ${demoFigure(output.image.path,escape(`RelationAdapter, ${row.label}, q0, ${variant.label} output`),escape(variant.label))}
            <p class="pd-score">${metric(row)}: <strong>${formatScore(output.score)}</strong> <span>this example · N = 1</span></p>
            <div class="pd-text-prompt"><h4>Exact text prompt</h4><p>${escape(output.prompt)}</p></div>
          </article>`;
        }).join('')}</div>
      </div>
    </div>`;
    if (updateHash) history.replaceState(null, '', `#relationadapter-prompts/${row.task}`);
  }

  async function load() {
    try {
      const response = await fetch('../data/relationadapter-prompt-comparison.json?v=65');
      if (!response.ok) throw new Error('Unavailable comparison');
      data = await response.json();
      if (data.total_examples !== 13 || data.total_outputs !== 26 || data.n_per_task !== 1 || data.rows.length !== 13 || data.variants.length !== 2 ||
          new Set(data.rows.map(r => r.task)).size !== 13 || data.rows.some(r => r.q !== 0 || r.n !== 1 || data.variants.some(v => !Number.isFinite(r.outputs[v.key]?.score)))) {
        throw new Error('Incomplete diagnostic');
      }
      document.querySelector('#ra-score-body').innerHTML = data.rows.map(row => `<tr>
        <th scope="row"><a href="#relationadapter-prompts/${row.task}" data-ra-task="${row.task}">${escape(row.label)} ↗</a><small>${metric(row)}</small></th>
        <td>${formatScore(row.outputs.generic.score)}</td><td>${formatScore(row.outputs.demo_instruction.score)}</td>
        <td>${row.change > 0 ? '+' : ''}${formatScore(row.change)}</td><td class="ra-result ra-${row.verdict.toLowerCase()}">${escape(row.verdict)}</td>
      </tr>`).join('');
      const counts = verdict => data.rows.filter(row => row.verdict === verdict).length;
      document.querySelector('#ra-score-summary').textContent = `On these 13 individual examples: ${counts('Improved')} improved, ${counts('Worse')} worsened, and ${counts('Tied')} tied. This does not establish performance across each category.`;
      category.innerHTML = data.rows.map(row => `<option value="${row.task}">${escape(row.label)}</option>`).join('');
      category.disabled = false;
      category.addEventListener('change', () => { index = data.rows.findIndex(row => row.task === category.value); render(true); });
      previous.addEventListener('click', () => { if (index > 0) { index--; render(true); } });
      next.addEventListener('click', () => { if (index < 12) { index++; render(true); } });
      document.querySelector('#ra-score-body').addEventListener('click', event => {
        const link = event.target.closest('[data-ra-task]');
        if (!link) return;
        event.preventDefault();
        index = data.rows.findIndex(row => row.task === link.dataset.raTask);
        render(true);
        document.querySelector('#ra-viewer').scrollIntoView({behavior: 'smooth', block: 'start'});
      });
      window.addEventListener('hashchange', () => {
        if (fromHash()) { render(); document.querySelector('#ra-viewer').scrollIntoView(); }
      });
      fromHash(); render();
      if (location.hash.startsWith('#relationadapter-prompts')) {
        document.querySelector(location.hash.includes('/') ? '#ra-viewer' : '#relationadapter-prompts').scrollIntoView();
      }
    } catch (error) {
      category.disabled = previous.disabled = next.disabled = true;
      document.querySelector('#ra-score-summary').textContent = 'The comparison could not be loaded. Reload the page or download the CSV.';
      document.querySelector('#ra-score-body').innerHTML = '<tr><td colspan="5">Scores unavailable.</td></tr>';
      status.textContent = 'Examples unavailable. Please reload the page.';
      stage.innerHTML = '';
    }
  }
  load();
})();
