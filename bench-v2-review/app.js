(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const label = task => task.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
  const num = value => Number.isFinite(value) ? value.toFixed(6) : '—';
  let data, model, recipe, task;
  const rows = () => model.categories.filter(r => r.recipe === recipe);
  function renderTask() {
    const row = rows().find(r=>r.task===task);
    el('task-select').value=task;
    el('category-paper').textContent=row.paper_comparison;
    el('category-method').textContent=row.method;
    el('query-caption').textContent=`${recipe} · ${label(task)} · ${row.metric} (${row.direction} is better) · ${row.n_candidates===8?'mean of 8 independently scored candidates per query':'single output per query'}`;
    el('query-rows').innerHTML=row.queries.map(q=>`<tr><td>${q.q}</td><td>${esc(q.query_record)}</td><td class="number">${num(q.score)}</td><td class="number">${num(q.copy_score)}</td></tr>`).join('');
  }
  function renderRecipe() {
    const rs=rows();
    el('recipe-select').value=recipe;
    el('recipe-status').textContent=data.recipe_status[recipe];
    el('category-caption').textContent=`${recipe} · 13 categories × 17 queries · published baseline; corrections remain separate`;
    el('category-rows').innerHTML=rs.map(r=>`<tr><td><button class="task-button" type="button" data-task="${esc(r.task)}">${esc(label(r.task))}</button></td><td>${esc(r.metric)} ${r.direction==='lower'?'↓':'↑'}</td><td class="number">${num(r.mean)}</td><td class="number">${num(r.min_query)}–${num(r.max_query)}</td><td class="number">${num(r.copy_mean)}</td><td class="flags">${esc(r.flags||'No automated symptom flag; see the method review.')}</td></tr>`).join('');
    el('task-select').innerHTML=rs.map(r=>`<option value="${esc(r.task)}">${esc(label(r.task))}</option>`).join('');
    if(!rs.some(r=>r.task===task)) task=rs[0].task;
    renderTask();
  }
  function renderModel() {
    el('model-select').value=model.key;
    el('model-title').textContent=model.name;
    el('observation').textContent=model.observation;
    el('paper-facts').textContent=model.paper.facts;
    el('paper-link').href=model.paper.url;
    el('qualification').textContent=model.settings.qualification;
    el('recipe-select').innerHTML=model.recipes.map(r=>`<option value="${esc(r)}">${esc(r)}</option>`).join('');
    el('settings').innerHTML=[['checkpoint','Checkpoint'],['demonstrations','Demonstrations'],['native_size','Native resolution'],['sampling','Sampling'],['required','Required handling']].map(([key,name])=>`<dt>${name}</dt><dd>${esc(model.settings[key])}</dd>`).join('');
    el('remaining').innerHTML=model.settings.remaining.map(s=>`<li>${esc(s)}</li>`).join('');
    recipe=model.recipes[0];renderRecipe();
  }
  async function start() {
    try {
      if(globalThis.__REVIEW_DATA) data=globalThis.__REVIEW_DATA;
      else {const response=await fetch('data.json');if(!response.ok)throw new Error('Review data unavailable');data=await response.json();}
      const authorized=data.authorization_progress ? ` · authorized reruns: ${data.authorization_progress.updated_utc.replace('T',' ').slice(0,19)} UTC` : '';
      el('snapshot').textContent=`Review: ${data.updated_utc.replace('T',' ').slice(0,19)} UTC · original run-status snapshot: ${data.execution_snapshot.updated_utc.replace('T',' ').slice(0,19)} UTC${authorized} · this page is not a live job monitor.`;
      el('model-select').innerHTML=data.models.map(m=>`<option value="${esc(m.key)}">${esc(m.name)} · ${m.recipes.length} recipe${m.recipes.length===1?'':'s'}</option>`).join('');
      el('ours-rows').innerHTML=data.ours.rows.map(r=>`<tr><td>${esc(label(r.task))}</td><td>${esc(r.metric)} ${r.direction==='lower'?'↓':'↑'}</td><td>${r.mean_reported.toFixed(2)}</td></tr>`).join('');
      el('model-select').addEventListener('change',e=>{model=data.models.find(m=>m.key===e.target.value);renderModel();});
      el('recipe-select').addEventListener('change',e=>{recipe=e.target.value;renderRecipe();});
      el('task-select').addEventListener('change',e=>{task=e.target.value;renderTask();});
      el('category-rows').addEventListener('click',e=>{const b=e.target.closest('[data-task]');if(b){task=b.dataset.task;renderTask();el('queries').scrollIntoView({behavior:'smooth',block:'start'});}});
      model=data.models[0];renderModel();
    } catch(error) {console.error(error);el('load-error').hidden=false;el('snapshot').textContent='Review unavailable.';}
  }
  start();
})();
