(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = value => Number.isFinite(value) ? value.toFixed(6).replace(/\.?0+$/, '') : '—';
  const signed = value => `${value > 0 ? '+' : ''}${number(value)}`;
  const taskName = key => ({human_pose_2d:'Human pose 2D',super_resolution:'Super-resolution',surface_normals:'Surface normals'}[key] || key.replaceAll('_',' ').replace(/^./, c => c.toUpperCase()));
  const metricName = key => ({lpips:'LPIPS',ois_f:'OIS F-score',nme:'NME','pck_torso_0.2':'PCK torso @ 0.2',pq:'PQ',delta1:'δ₁',delta1_signed_scale:'δ₁ · signed scale',max_fbeta:'max Fβ',miou:'mIoU',mean_deg:'Mean degrees'}[key] || key);
  let data;
  let datasets;
  const signedProtocol = () => Boolean(data.depth_evaluation_change);
  let chosen = {};
  let currentDetail = null;
  const candidate = (model, task, recipe) => data.candidates.find(c => c.model === model && c.task === task && c.recipe === recipe);

  function selectedCell(model, task) {
    const original = data.selected.consistent.find(t => t.task === task).cells.find(c => c.model === model).original;
    const c = candidate(model, task, chosen[model]);
    const improvement = signedProtocol() && task === 'relative_depth' ? null : c.direction === 'lower' ? original-c.score : c.score-original;
    return {...c, original, improvement, relative_improvement_pct: original && improvement !== null ? 100*improvement/Math.abs(original) : null};
  }

  function tradeoffs(model) {
    const cells = data.tasks.map(t => selectedCell(model,t.task)).filter(c=>c.improvement !== null);
    const better = cells.filter(c=>c.improvement>1e-12).length;
    const worse = cells.filter(c=>c.improvement< -1e-12).length;
    return `${better} better · ${worse} worse · ${cells.length-better-worse} tied vs old table${signedProtocol()?" · other 12 tasks":""}`;
  }

  // Rank only comparable finite measurements; unrescored references have no rank.
  function rank(value, values, direction='higher') {
    return 1 + values.filter(v => direction === 'higher' ? v > value : v < value).length;
  }
  const rankLabel = n => `<span class="rank-label"><b>#${n}</b>${n===1?' Best':''}</span>`;
  function render() {
    const models = data.models.filter(m => el('best-model').value === 'all' || m.key === el('best-model').value);
    const signedDepth = signedProtocol();
    const rows = data.selected.consistent.filter(t => el('best-task').value === 'all' || t.task === el('best-task').value);
    el('best-table').style.minWidth = `${270 + models.length * 130}px`;
    el('best-head').innerHTML = '<th scope="col">Category / metric</th>' + models.map(m => `<th scope="col">${escape(m.name)}<label class="best-recipe-label">One recipe · all categories<select data-recipe-model="${escape(m.key)}" aria-label="${escape(m.name)} recipe for all 13 categories">${m.complete_recipes.map(r=>`<option value="${escape(r.recipe)}"${chosen[m.key]===r.recipe?' selected':''}>${escape(r.label)}${r.recipe===m.default_recipe?' · Selected':''}</option>`).join('')}</select></label><small>${tradeoffs(m.key)}</small>${m.key === 'visualcloze' ? '<small>Archived · numerics unresolved</small>' : ''}</th>`).join('') + `<th scope="col">JFLUX (ours) / TA<small>P=16 · ${signedDepth?'Depth verified; other tasks reported':'Reported reference'}</small></th>`;
    el('best-body').innerHTML = rows.map(row => {const referenceComparable=row.reference_comparable !== false && Number.isFinite(row.reported_reference);const values=[...(referenceComparable?[row.reported_reference]:[]),...data.models.map(m=>selectedCell(m.key,row.task).score)].filter(Number.isFinite);return `<tr data-task="${row.task}"><th scope="row">${escape(taskName(row.task))}${row.task === 'surface_normals' ? ' <a href="../bench-v2-surface-normals/" aria-label="Surface-normal evaluation audit">Audit ↗</a>' : row.task === 'relative_depth' ? (signedDepth ? ' <a href="../bench-v2-signed-depth/" aria-label="Signed-scale depth evaluation">Rescore ↗</a>' : ' <a href="../bench-v2-depth-rescore/" aria-label="Depth decoding correction">Correction ↗</a>') : ''}<small>${escape(metricName(row.metric))} ${row.direction === 'higher' ? '↑' : '↓'} · 17 queries</small></th>${models.map(m => {
      const cell = selectedCell(m.key,row.task);
      const position=rank(cell.score,values,row.direction);
      return `<td class="validation-score rank-${position}"><button type="button" class="best-score-button" data-best-model="${escape(m.key)}" data-best-task="${escape(row.task)}" aria-label="${escape(m.name)}, ${escape(taskName(row.task))}: ${escape(String(cell.score))}. Inspect all recipes and query scores.">${rankLabel(position)}<strong title="${escape(String(cell.score))}">${number(cell.score)}</strong></button></td>`;
    }).join('')}${Number.isFinite(row.reported_reference) ? referenceComparable ? `<td class="best-reference validation-score rank-${rank(row.reported_reference,values,row.direction)}">${rankLabel(rank(row.reported_reference,values,row.direction))}<strong title="${escape(String(row.reported_reference))}">${number(row.reported_reference)}</strong><small>${row.reference_kind==='rescored'?'<a href="../bench-v2-jflux-depth/">Verified · 17 queries ↗</a>':'Reported reference'}</small></td>` : `<td class="best-reference"><strong>${number(row.reported_reference)}</strong><small>Previous depth rule<br>Shown, not ranked</small></td>` : '<td class="best-reference"><strong>—</strong><small>Reference unavailable</small></td>'}</tr>`;}).join('');
    el('best-caption').textContent = signedDepth ? 'Signed-scale depth · 12 models × 17 queries, including JFLUX/Ours verified from its saved predictions. One previously selected recipe per model; other 12 tasks unchanged. Other JFLUX task values remain source-rounded references. Use the scoring-rule switch for previous scores and recipe alternatives. Colors compare scores within each task, including filtered-out models.' : 'Previous depth rule · negative fitted scales become zero. Full comparison · 12 columns · 13 categories. Colors compare rank within each row; ties share rank. Change a column selector to apply one recipe to every category. Task ranks include all models, even when filtered.';
    el('best-head').querySelectorAll('[data-recipe-model]').forEach(select => select.addEventListener('change',()=>{
      chosen[select.dataset.recipeModel]=select.value;
      render();
    }));
    el('best-body').querySelectorAll('[data-best-model]').forEach(button => button.addEventListener('click', () => {
      showDetail(button.dataset.bestModel, button.dataset.bestTask, true);
    }));
    if (currentDetail) showDetail(...currentDetail, false);
  }

  function showDetail(model, task, moveFocus) {
    currentDetail = [model, task];
    const t = data.tasks.find(t => t.task === task);
    const cell = selectedCell(model,task);
    const all = data.candidates.filter(c => c.model === model && c.task === task).sort((a,b) => t.direction === 'lower' ? a.score-b.score : b.score-a.score);
    if (signedProtocol() && task === 'relative_depth') all.push({...cell,recipe:cell.recipe+'::previous',label:cell.label+' · previous scoring rule',score:cell.previous_score,queries:cell.previous_queries,comparison_only:true,note:'Same predictions under the previous nonnegative-scale rule; shown for comparison, not eligible for the new depth ranking.'});
    const name = data.models.find(m => m.key === model).name;
    el('best-detail').hidden = false;
    el('best-detail-title').textContent = `${name} · ${taskName(task)}`;
    const m = data.models.find(m=>m.key===model);
    el('best-detail-summary').textContent = signedProtocol() && task === 'relative_depth' ? `Previous rule: ${number(cell.previous_score)}. Signed scale: ${number(cell.score)}. Same predictions; only the evaluation rule changed. The answer key determines scale, shift and near/far direction. This is not an inference-time improvement. Other complete recipes remain in the previous-protocol view.` : `Old table: ${number(cell.original)}. Selected: ${number(cell.score)}. Direction-adjusted change: ${signed(cell.improvement)}${cell.relative_improvement_pct === null ? ' (percentage undefined from zero)' : ` (${signed(cell.relative_improvement_pct)}%)`}. Positive means better. Across comparable categories: ${tradeoffs(model)}. The selected recipe applies to the entire column. Default choice: ${m.selection_reason} Recipe differences can include changed target palettes, checkpoints and backends; they do not isolate one correction.`;
    el('best-recipes').innerHTML = all.map(c => {
      const selected = c.recipe === cell.recipe;
      const status = c.comparison_only ? 'Comparison only · previous evaluation rule' : c.superseded_by ? 'Superseded: previous depth decoder' : !c.eligible ? 'Excluded: legacy diagnostic' : c.metadata_assisted ? 'Separate: extra demo metadata' : !c.complete_recipe ? 'Separate: category-only experiment' : selected ? 'Selected for all 13 categories' : 'Completed full-recipe alternative';
      return `<tr class="${selected ? 'best-selected-row' : ''}"><th scope="row">${escape(c.label)}<small>${escape(status)}</small></th><td title="${escape(String(c.score))}">${number(c.score)}</td><td>${escape(c.note)} <a href="${escape(c.detail)}">Evidence ↗</a></td></tr>`;
    }).join('');
    el('best-query-head').innerHTML = '<th scope="col">Query</th>' + all.map(c => `<th scope="col">${escape(c.label)}${c.recipe === cell.recipe ? '<small>Selected</small>' : ''}</th>`).join('');
    el('best-query-body').innerHTML = Array.from({length:17}, (_, q) => `<tr><th scope="row">q${q}</th>${all.map(c => `<td class="${c.recipe === cell.recipe ? 'best-selected-row' : ''}" title="${escape(String(c.queries.find(x => x.q === q).score))}">${number(c.queries.find(x => x.q === q).score)}</td>`).join('')}</tr>`).join('');
    if (moveFocus) {
      el('best-detail-title').focus({preventScroll:true});
      el('best-detail').scrollIntoView({behavior:'smooth',block:'start'});
    }
  }

  function activateProtocol() {
    data = datasets[el('best-protocol').value];
    chosen = Object.fromEntries(data.models.map(m=>[m.key,m.default_recipe]));
    currentDetail = null;
    el('best-detail').hidden = true;
    el('best-updated').textContent = `Updated ${new Date(data.updated_utc).toISOString().slice(0,16).replace('T',' ')} UTC · ${signedProtocol()?'Signed-scale depth · 204 query means verified, including JFLUX/Ours':'Previous scoring rule · all recipe alternatives'}`;
    el('best-model').innerHTML = '<option value="all">All models</option>' + data.models.map(m => `<option value="${escape(m.key)}">${escape(m.name)}</option>`).join('');
    el('best-task').innerHTML = '<option value="all">All categories</option>' + data.tasks.map(t => `<option value="${escape(t.task)}">${escape(taskName(t.task))}</option>`).join('');
    el('best-model').value='all';el('best-task').value='all';
    el('best-model-status').innerHTML = data.models.map(m => `<article><h3>${escape(m.name)}</h3><p>${escape(m.status)}</p><p><strong>Default:</strong> ${escape(m.selection_reason)}</p></article>`).join('');
    for (const [id,file] of [['best-download','best-results.csv'],['recipes-download','all-recipes.csv'],['queries-download','all-query-scores.csv'],['json-download','best-results.json']]) el(id).href=signedProtocol()?file:file.replace(/\.(csv|json)$/,'-previous.$1');
    render();
  }

  async function load() {
    try {
      const fetchData = async name => {
        const response=await fetch(name+'?v=106');
        if(!response.ok)throw new Error('Results unavailable');
        return response.json();
      };
      const [signed,previous]=await Promise.all([fetchData('best-results.json'),fetchData('best-results-previous.json')]);
      datasets={signed,previous};
      el('best-protocol').value='signed';
      el('best-protocol').disabled=false;
      el('best-protocol').addEventListener('change',activateProtocol);
      ['best-model','best-task'].forEach(id => {el(id).disabled=false;el(id).addEventListener('change',render);});
      activateProtocol();
    } catch (error) {
      el('best-body').innerHTML='<tr><td colspan="13">The updated scores could not load. Download the CSV below or reload this page.</td></tr>';
      el('best-updated').textContent='Updated results temporarily unavailable.';
    }
  }
  load();
})();
