(() => {
  'use strict';
  const el=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const label=v=>v.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
  const num=v=>Number.isFinite(v)?(v!==0&&Math.abs(v)<1e-5?v.toExponential(2):v.toFixed(6)):'Undefined';
  const signed=v=>Number.isFinite(v)?`${v>0?'+':''}${num(v)}`:'Undefined';
  const pct=v=>Number.isFinite(v)?`${v>0?'+':''}${v!==0&&Math.abs(v)<.01?v.toExponential(2):v.toFixed(2)}%`:'Undefined';
  const color=v=>v>1e-12?'good':v< -1e-12?'bad':'';
  let data,model,task;
  function renderTask(){
    const row=model.categories.find(r=>r.task===task);el('task-select').value=task;
    const gain=row.largest_gain.improvement>1e-12?`Largest gain: query ${row.largest_gain.q}, ${signed(row.largest_gain.improvement)}.`:'No query improved.';
    const loss=row.largest_regression.improvement< -1e-12?`Largest regression: query ${row.largest_regression.q}, ${signed(row.largest_regression.improvement)}.`:'No query regressed.';
    el('effect-summary').textContent=`${label(task)}: gains contribute ${signed(row.gain_contribution)} and losses contribute ${signed(row.loss_contribution)} to the category mean, for a net ${signed(row.improvement)} ${row.metric}. ${gain} ${loss}`;
    el('query-caption').textContent=`${model.name} · ${label(task)} · ${row.metric} (${row.lower_is_better?'lower':'higher'} is better) · all 17 queries in fixed order`;
    el('query-rows').innerHTML=[...row.queries].sort((a,b)=>a.q-b.q).map(q=>`<tr><th scope="row">${q.q}<span class="metric">${esc(q.query_record)}</span></th><td title="${q.archived_tpu}">${num(q.archived_tpu)}</td><td title="${q.cpu_fp32}">${num(q.cpu_fp32)}</td><td class="${color(q.improvement)}" title="${q.improvement}">${signed(q.improvement)}</td><td>${pct(q.relative_improvement_pct)}</td></tr>`).join('');
  }
  function renderModel(){
    el('model-select').value=model.key;el('model-title').textContent=model.name;
    el('condition').textContent=model.recipe_difference;
    el('reference-note').textContent=model.reference_note;
    el('category-rows').innerHTML=model.categories.map(r=>`<tr><th scope="row"><button class="task-button" type="button" data-task="${esc(r.task)}">${esc(label(r.task))}</button><span class="metric">${esc(r.metric)} ${r.lower_is_better?'↓':'↑'}</span></th><td title="${r.archived_tpu}">${num(r.archived_tpu)}</td><td title="${r.cpu_fp32}">${num(r.cpu_fp32)}</td><td class="${color(r.improvement)}" title="${r.improvement}">${signed(r.improvement)}</td><td>${pct(r.relative_improvement_pct)}</td><td>${r.wins} / ${r.ties} / ${r.losses}</td></tr>`).join('');
    el('task-select').innerHTML=model.categories.map(r=>`<option value="${esc(r.task)}">${esc(label(r.task))}</option>`).join('');
    if(!model.categories.some(r=>r.task===task))task=model.categories[0].task;
    const e=model.edge_pooled_ois_supplement;
    el('edge-note').textContent=`Separate pooled edge OIS supplement: ${num(e.archived_tpu)} → ${num(e.cpu_fp32)} (${signed(e.improvement)}). The main category table above uses the mean of per-query scores; pooled OIS and ODS are separate statistics.`;
    renderTask();
  }
  async function start(){
    try{
      if(globalThis.__CPU_REFERENCE_DATA)data=globalThis.__CPU_REFERENCE_DATA;
      else{const r=await fetch('data.json');if(!r.ok)throw new Error('Comparison unavailable');data=await r.json();}
      if(data.models.length!==2||data.models.some(m=>m.status!=='COMPLETE_PAIRED_CPU_REFERENCE_COMPARISON'))throw new Error('Incomplete comparison');
      el('snapshot').textContent=`Verified snapshot: ${data.updated_utc.replace('T',' ').slice(0,19)} UTC · ${data.models.length} models × 13 categories × 17 queries`;
      el('host-evidence').textContent=`The rebuilt original metric core reproduced all ${data.host_qualification.native_calls.toLocaleString()} recorded native matching calls across four reference cases exactly, including complete pairs, costs and all 99 threshold counts. Independent saved-result verification passed.`;
      el('model-select').innerHTML=data.models.map(m=>`<option value="${esc(m.key)}">${esc(m.name)}</option>`).join('');
      el('model-select').addEventListener('change',e=>{model=data.models.find(m=>m.key===e.target.value);renderModel();});
      el('task-select').addEventListener('change',e=>{task=e.target.value;renderTask();});
      el('category-rows').addEventListener('click',e=>{const b=e.target.closest('[data-task]');if(b){task=b.dataset.task;renderTask();el('queries').scrollIntoView({behavior:'smooth',block:'start'});}});
      model=data.models[0];renderModel();
    }catch(error){console.error(error);el('load-error').hidden=false;el('snapshot').textContent='Comparison unavailable.';}
  }
  start();
})();
