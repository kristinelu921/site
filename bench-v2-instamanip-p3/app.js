(() => {
  'use strict';
  const el=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const label=v=>v.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
  const num=v=>Number.isFinite(v)?v.toFixed(6):'Undefined';
  const signed=v=>Number.isFinite(v)?`${v>0?'+':''}${num(v)}`:'Undefined';
  const pct=v=>Number.isFinite(v)?`${v>0?'+':''}${v.toFixed(2)}%`:'Undefined';
  const color=v=>v>1e-12?'good':v< -1e-12?'bad':'';
  let data,task;
  function renderTask(){
    const row=data.categories.find(r=>r.task===task);el('task-select').value=task;
    const gain=row.largest_gain,loss=row.largest_regression;
    el('effect-summary').textContent=`${label(task)}: gains contribute ${signed(row.gain_contribution)} and losses contribute ${signed(row.loss_contribution)} to the mean, for a net ${signed(row.improvement)} ${row.metric}. Largest gain: query ${gain.q}, ${signed(gain.improvement)}. Largest regression: query ${loss.q}, ${signed(loss.improvement)}. ${row.identical_targets}/17 targets are identical.`;
    el('query-caption').textContent=`${label(task)} · ${row.metric} (${row.direction} is better) · all 17 queries`;
    el('query-rows').innerHTML=[...row.queries].sort((a,b)=>a.q-b.q).map(q=>`<tr><th scope="row">${q.q}<span class="metric">${esc(q.query_record)}</span></th><td title="${q.p2}">${num(q.p2)}</td><td title="${q.p3}">${num(q.p3)}</td><td class="${color(q.improvement)}" title="${q.improvement}">${signed(q.improvement)}</td><td>${pct(q.relative_improvement_pct)}</td><td>${q.query_target_identical?'Yes':'Changed'}</td></tr>`).join('');
  }
  async function start(){
    try{
      const response=await fetch('data.json');if(!response.ok)throw Error('Comparison unavailable');data=await response.json();
      if(data.status!=='RECOVERED_COMPLETE_P3_ARCHIVE_AND_ALL_CATEGORY_COMPARISON'||data.categories.length!==13||data.categories.some(r=>r.queries.length!==17))throw Error('Incomplete comparison');
      el('snapshot').textContent=`Verified snapshot: ${data.updated_utc.replace('T',' ').slice(0,19)} UTC · 13 categories · 221 queries · ${data.category_summary.better} means improve / ${data.category_summary.worse} worsen`;
      el('category-rows').innerHTML=data.categories.map(r=>`<tr><th scope="row"><button class="task-button" type="button" data-task="${esc(r.task)}">${esc(label(r.task))}</button><span class="metric">${esc(r.metric)} ${r.direction==='lower'?'↓':'↑'}</span></th><td title="${r.p2}">${num(r.p2)}</td><td title="${r.p3}">${num(r.p3)}</td><td class="${color(r.improvement)}" title="${r.improvement}">${signed(r.improvement)}</td><td>${pct(r.relative_improvement_pct)}</td><td>${r.wins} / ${r.ties} / ${r.losses}</td><td>${r.identical_targets} / 17</td></tr>`).join('');
      el('task-select').innerHTML=data.categories.map(r=>`<option value="${esc(r.task)}">${esc(label(r.task))}</option>`).join('');
      el('task-select').addEventListener('change',e=>{task=e.target.value;renderTask();});
      el('category-rows').addEventListener('click',e=>{const button=e.target.closest('[data-task]');if(button){task=button.dataset.task;renderTask();el('queries').scrollIntoView({behavior:'smooth',block:'start'});}});
      task='saliency_detection';renderTask();
    }catch(error){console.error(error);el('load-error').hidden=false;el('snapshot').textContent='Comparison unavailable.';}
  }
  start();
})();
