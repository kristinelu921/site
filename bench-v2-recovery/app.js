(() => {
  'use strict';
  const labels = {edge_detection:'Edge detection', saliency_detection:'Saliency', semantic_segmentation:'Semantic segmentation', instance_segmentation:'Instance segmentation', relative_depth:'Depth', surface_normals:'Surface normals', human_pose_2d:'Human pose', face_landmarks:'Face landmarks', colorization:'Colorization', inpainting:'Inpainting', deblurring:'Deblurring', denoising:'Denoising', super_resolution:'Super-resolution'};
  const metrics = {ois_f:'OIS F', max_fbeta:'max Fβ', miou:'mIoU', pq:'PQ', delta1:'δ1', mean_deg:'Mean degrees', 'pck_torso_0.2':'PCK@0.2', nme:'NME', lpips:'LPIPS'};
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
  const score = value => Number.isFinite(value) ? `<span title="${value}">${value.toFixed(6)}</span>` : '<span class="unmeasured">—</span>';
  const delta = value => Number.isFinite(value) ? `<span class="${value > 0 ? 'good' : value < 0 ? 'bad' : ''}" title="${value}">${value > 0 ? '+' : ''}${value.toFixed(6)} · ${value > 0 ? 'better' : value < 0 ? 'worse' : 'tie'}</span>` : '<span class="unmeasured">Unmeasured</span>';
  const percent = (value, measured) => Number.isFinite(value) ? `<span class="${value > 0 ? 'good' : value < 0 ? 'bad' : ''}">${value > 0 ? '+' : ''}${value.toFixed(2)}%</span>` : `<span class="unmeasured">${measured ? 'Undefined · zero control' : '—'}</span>`;
  let data, model, task = 'relative_depth', query = 0;
  const label = task => labels[task] || task;
  const metric = row => `${metrics[row.metric] || row.metric} ${row.direction === 'lower' ? '↓' : '↑'}`;
  const missing = () => /pending|awaiting|running/i.test(model.status) ? 'Pending' : 'Not rerun';

  function renderModel(key, updateHash = true) {
    model = data.models.find(m => m.key === key) || data.models.find(m => m.key === 'prompt_diffusion');
    if (updateHash) history.replaceState(null, '', `#${model.key}/${task}`);
    $('model-list').innerHTML = data.models.map(m => `<button type="button" data-model="${m.key}" aria-pressed="${m.key === model.key}"><strong>${esc(m.name)}</strong><span>${esc(m.status)}</span></button>`).join('');
    $('model-title').textContent = model.name;
    $('model-status').textContent = model.status;
    $('baseline-demos').textContent = `Published baseline · P=${model.baseline_demos}`;
    $('model-change').textContent = model.change;
    $('model-evidence').textContent = model.evidence;
    $('model-limit').textContent = model.limitation;
    $('variant-heading').textContent = model.comparison_label;
    const measured = model.rows.filter(r => r.updated !== null && Number.isFinite(r.relative_improvement_pct));
    $('magnitude-summary').hidden = measured.length === 0;
    if (measured.length) {
      const largest = [...measured].sort((a,b)=>Math.abs(b.relative_improvement_pct)-Math.abs(a.relative_improvement_pct)).slice(0,3);
      $('magnitude-summary').innerHTML = `<strong>Largest relative category changes</strong><ul>${largest.map(r=>`<li>${label(r.task)} · ${metric(r)}: ${score(r.control)} → ${score(r.updated)}; ${delta(r.improvement)} (${percent(r.relative_improvement_pct,true)}).</li>`).join('')}</ul><p>Percentages use each category’s control mean. Read the native-unit change alongside them: a small baseline can produce a large percentage. Different metrics are not averaged into an overall score.</p>`;
    }
    $('historical-text').hidden = model.key !== 'prompt_diffusion';
    if (model.key === 'prompt_diffusion') {
      const history = data.promptdiffusion_historical_text;
      $('historical-text').innerHTML = `<summary>Earlier task-prompt changes · historical comparison</summary><p>${esc(history.note)} These differ from the fresh range comparison below; hardware also changed, so this does not isolate text alone.</p><div class="recovery-table-wrap" tabindex="0"><table class="recovery-table"><thead><tr><th>Category</th><th>Generic text</th><th>Short task text</th><th>Signed change</th><th>Relative change</th></tr></thead><tbody>${history.tasks.map(r=>{const d=(r.means.previous-r.means.original)*(r.direction==='lower'?-1:1);return `<tr><th>${label(r.task)}<span class="metric">${metric(r)}</span></th><td>${score(r.means.original)}</td><td>${score(r.means.previous)}</td><td>${delta(d)}</td><td>${percent(r.means.original===0?null:100*d/Math.abs(r.means.original),true)}</td></tr>`;}).join('')}</tbody></table></div>`;
    }
    $('category-caption').textContent = `${model.name} · 13 categories · 17 queries per category · published baseline and separately measured conditions`;
    $('task-select').innerHTML = model.rows.map(r => `<option value="${r.task}">${label(r.task)} · ${metric(r)}</option>`).join('');
    renderAlternatives();
    renderTask(task, false);
  }

  function renderCategories() {
    $('category-rows').innerHTML = model.rows.map(r => `<tr class="${r.task === task ? 'selected' : ''}"><th scope="row"><button type="button" class="task-button" data-task="${r.task}">${label(r.task)}</button><span class="metric">${metric(r)} · n=17</span></th><td>${score(r.published)}</td><td>${score(r.control)}</td><td>${r.updated === null ? `<span class="unmeasured">${missing()}</span>` : score(r.updated)}</td><td>${delta(r.improvement)}</td><td>${percent(r.relative_improvement_pct,r.updated!==null)}</td><td>${r.wins === null ? '—' : `${r.wins} / ${r.ties} / ${r.losses}`}</td></tr>`).join('');
  }

  function renderTask(key, updateHash = true) {
    task = model.rows.some(r => r.task === key) ? key : model.rows[0].task;
    $('task-select').value = task;
    if (updateHash) history.replaceState(null, '', `#${model.key}/${task}`);
    const row = model.rows.find(r => r.task === task);
    renderCategories();
    $('query-context').textContent = `${model.name} · ${label(task)} · ${metric(row)}. Queries q0–q16 are shown in fixed order. ${row.updated === null ? 'This category has no new measured recovery score.' : 'Signed improvement compares the measured variant with its fresh paired control.'}`;
    $('query-rows').innerHTML = row.queries.map(q => `<tr><th scope="row">q${q.q}</th><td>${score(q.published)}</td><td>${score(q.control)}</td><td>${score(q.updated)}</td><td>${delta(q.improvement)}</td><td>${percent(q.relative_improvement_pct,q.updated!==undefined)}</td></tr>`).join('');
    $('query-magnitude').hidden = !row.magnitude;
    if (row.magnitude) {
      const m=row.magnitude;
      $('query-magnitude').innerHTML = `<strong>How the individual changes add up</strong><p>Gains contribute ${score(m.gain_contribution)} and losses ${score(m.loss_contribution)} to the category mean, producing ${delta(row.improvement)}. Median query change: ${delta(m.median_query_improvement)}.</p><p>Largest signed query change: q${m.best_query}, ${delta(m.best_query_improvement)}. Smallest signed query change: q${m.worst_query}, ${delta(m.worst_query_improvement)}. Leaving out any one query gives a mean signed change between ${score(m.mean_without_one_query_min)} and ${score(m.mean_without_one_query_max)}. This sensitivity range is not a confidence interval.</p>`;
    }
    $('paired-images').hidden = model.key !== 'prompt_diffusion';
    $('painter-secondary').hidden = model.key !== 'painter' || task !== 'relative_depth';
    if (!$('painter-secondary').hidden) {
      const p = data.painter_secondary;
      $('painter-secondary').innerHTML = `<strong>Depth-only adaptation.</strong> AbsRel ↓: ${score(p.generic_control.abs_rel)} → ${score(p.native_depth_bridge.abs_rel)}. Zero-scale queries: 17 → 0. Extra demonstration depth endpoints are used to approximate the native representation from clipped and quantized depth codes. CPU FP32; no original-paper or TPU numerical-parity claim.`;
    }
    if (model.key === 'prompt_diffusion') {
      $('query-select').innerHTML = row.queries.map(q => `<option value="${q.q}">Query ${q.q + 1} / 17 · q${q.q}</option>`).join('');
      $('query-select').value = String(query);
      renderImages();
    }
  }

  function renderImages() {
    const row = model.rows.find(r => r.task === task);
    const q = row.queries.find(r => r.q === query);
    const images = [['demo_input','Demonstration input'],['demo_output','Demonstration output'],['query_input','Query input'],['S1','S1 · control [0,1]'],['A1','A1 · range [−1,1]']];
    $('image-grid').innerHTML = images.map(([key,title]) => `<figure><a href="${esc(q.images[key])}" target="_blank" rel="noopener" aria-label="Open ${esc(title)} for ${label(task)} q${query}"><img src="${esc(q.images[key])}" alt="${esc(title)} · ${label(task)} · q${query}" width="256" height="256" loading="lazy"></a><figcaption>${title}${key === 'S1' || key === 'A1' ? `<span>${metric(row)} ${score(q[key === 'S1' ? 'control' : 'updated'])}</span>` : ''}</figcaption></figure>`).join('');
  }

  function renderAlternatives() {
    const node = $('alternatives');
    node.hidden = model.alternatives.length === 0;
    if (node.hidden) { node.innerHTML = ''; return; }
    node.innerHTML = `<h3>Previously completed native alternatives</h3><p class="table-help">These are verified archived results, each with 221 queries. They are separate demonstration-count conditions, not new correction reruns. LaVin’s pending RNG comparison is at P=8; its published baseline above is P=4.</p><div class="recovery-table-wrap" tabindex="0" role="region" aria-label="Archived native alternatives"><table class="recovery-table"><thead><tr><th>Category / metric</th>${model.alternatives.map(a=>`<th>${esc(a.label)}</th>`).join('')}</tr></thead><tbody>${model.rows.map(r=>`<tr><th>${label(r.task)}<span class="metric">${metric(r)}</span></th>${model.alternatives.map(a=>`<td>${score(a.scores[r.task])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  async function start() {
    try {
      const response = await fetch('data.json?v=76');
      if (!response.ok) throw new Error('Snapshot could not be loaded');
      data = await response.json();
      if (data.models.length !== 10 || data.models.some(m => m.rows.length !== 13)) throw new Error('Incomplete snapshot');
      const goal=data.resolution_goal;
      if(!goal || goal.models.length!==11) throw new Error('Incomplete resolution goal');
      const failure=data.confirmed_numerical_failure;
      $('numerical-failure').innerHTML=`<strong>VisualCloze: numerical failure confirmed.</strong> The guarded replay completed ${failure.completed_validation_queries} validation queries, then rejected ${failure.nan_count.toLocaleString()} NaNs before image conversion on the next saliency batch. ${esc(failure.next_step)} Archived score differences remain diagnostic.`;
      $('runtime-scope').textContent=data.runtime_diagnosis.scope;
      $('runtime-mechanism').textContent=data.runtime_diagnosis.mechanism_probe;
      $('runtime-rows').innerHTML=data.runtime_diagnosis.models.map(m=>`<tr><th scope="row">${esc(m.name)}</th><td>${esc(m.setup)}</td><td>${esc(m.bottleneck)}</td><td>${esc(m.warm)}<span class="metric">${esc(m.limit)}</span></td></tr>`).join('');
      $('completion-summary').textContent=`${goal.resolved_count} of ${goal.model_count} models fully resolved. Goal ${goal.status}.`;
      $('completion-models').innerHTML=goal.models.map(m=>`<article><h3>${esc(m.name)}</h3><p class="status-label">${esc(m.status.replaceAll('_',' ').toLowerCase())}</p><p>${esc(m.next_action)}</p><details><summary>Evidence required to close</summary><p>${esc(m.required_evidence)}</p></details></article>`).join('');
      $('snapshot').textContent = `Status snapshot: ${new Date(data.updated_utc).toLocaleString(undefined, {dateStyle:'long', timeStyle:'short', timeZone:'UTC'})} UTC. Job status is captured at publication; this page is not a live monitor.`;
      $('provenance').innerHTML = [['Pack','bench_v2_val · supervision v6'],['Pack ID',data.pack_id],['Corrected-v3 scorer release',data.release_sha256],['Preserved published table SHA-256',data.published_baseline_sha256]].map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
      const [key, hashTask] = location.hash.slice(1).split('/');
      if (labels[hashTask]) task = hashTask;
      renderModel(key, false);
      $('model-list').addEventListener('click', event => { const button = event.target.closest('button[data-model]'); if(button) renderModel(button.dataset.model); });
      $('category-rows').addEventListener('click', event => { const button = event.target.closest('button[data-task]'); if(button) {renderTask(button.dataset.task); $('query-view').scrollIntoView({behavior:'smooth'});} });
      $('task-select').addEventListener('change', event => renderTask(event.target.value));
      $('query-select').addEventListener('change', event => {query=Number(event.target.value);renderImages();});
      document.querySelectorAll('[data-select-model]').forEach(button => button.addEventListener('click', () => {renderModel(button.dataset.selectModel);$('explorer').scrollIntoView({behavior:'smooth'});}));
    } catch (error) {
      $('load-error').hidden = false;
      $('load-error').textContent = 'The recovery snapshot could not be loaded. The original tables and downloadable scores remain available. Please reload this page.';
      console.error(error);
    }
  }
  start();
})();
