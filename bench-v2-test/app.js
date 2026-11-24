'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const title = s => s.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
const metricNames={lpips:'LPIPS',ois_f:'OIS F',nme:'NME', 'pck_torso_0.2':'PCK torso 0.2',pq:'PQ',delta1:'δ1',max_fbeta:'Max Fβ',miou:'mIoU',mean_deg:'Mean angle °'};
const fmt = n => Number(n).toFixed(6);
const gain = r => r.lower ? r.copy-r.score : r.score-r.copy;
let data;
function render(){
 const mode=$('view').value, selected=$('task').value, models=data.models.filter(m=>$('model').value==='all'||m.model===$('model').value);
 const all=data[mode], rows=all.filter(r=>selected==='all'||r.task===selected), keys=[...new Set(rows.map(r=>mode==='tasks'?r.task:r.pack_rel))];
 $('caption').textContent=`${keys.length} ${mode==='tasks'?'tasks':'datasets'} · ${models.length} models · ${$('values').value==='score'?'Primary metric means':'Direction-adjusted improvement over copy input'}`;
 let html='<thead><tr><th scope="col">Task / metric</th>'+models.map(m=>`<th scope="col">${esc(m.name)}<small>P=${m.prompt_pairs} · 17,511 queries</small></th>`).join('')+'</tr></thead><tbody>';
 for(const key of keys){
  const group=rows.filter(r=>(mode==='tasks'?r.task:r.pack_rel)===key), first=group[0], best=(first.lower?Math.min:Math.max)(...group.map(r=>r.score));
  html+=`<tr><th scope="row">${esc(title(first.task))}${mode==='packs'?`<small>${esc(first.dataset)}</small>`:''}<small>${esc(metricNames[first.metric]||first.metric)} ${first.lower?'↓':'↑'} · n=${first.n.toLocaleString()}</small></th>`;
  for(const m of models){const r=group.find(x=>x.model===m.model);const g=gain(r);const value=$('values').value==='score'?r.score:g;
   html+=`<td class="${r.score===best?'winner':''}"><button class="best-score-button" data-model="${esc(m.model)}" data-task="${esc(r.task)}" aria-label="Inspect ${esc(m.name)} ${esc(title(r.task))}${mode==='packs'?' '+esc(r.dataset):''}"><strong class="${$('values').value==='gain'?(g<0?'negative':'positive'):''}">${$('values').value==='gain'&&value>0?'+':''}${fmt(value)}</strong><small>copy ${fmt(r.copy)}</small></button></td>`;
  }html+='</tr>';
 }$('scores').innerHTML=html+'</tbody>';
 $('scores').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>detail(b.dataset.model,b.dataset.task)));
}
function detail(mid,task){
 const m=data.models.find(x=>x.model===mid), rows=data.packs.filter(r=>r.model===mid&&r.task===task), total=data.tasks.find(r=>r.model===mid&&r.task===task);
 $('detail').innerHTML=`<h3>${esc(m.name)} · ${esc(title(task))}</h3><p>${esc(metricNames[total.metric]||total.metric)} ${total.lower?'↓ lower is better':'↑ higher is better'} · query-weighted task mean <strong>${fmt(total.score)}</strong></p><div class="table-shell"><table><thead><tr><th>Dataset</th><th>Queries</th><th>Model</th><th>Copy input</th><th>Improvement</th></tr></thead><tbody>${rows.map(r=>`<tr><th scope="row">${esc(r.dataset)}</th><td>${r.n.toLocaleString()}</td><td>${fmt(r.score)}</td><td>${fmt(r.copy)}</td><td>${gain(r)>0?'+':''}${fmt(gain(r))}</td></tr>`).join('')}</tbody></table></div><p class="best-compact">${esc(m.qualification)}</p>`;
 $('detail').tabIndex=-1;$('detail').focus({preventScroll:true});$('detail').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});
}
async function init(){
 try{const response=await fetch('results.json');if(!response.ok)throw Error('Score data could not be loaded');data=await response.json();
 $('updated').textContent=`Snapshot ${new Date(data.generated_utc).toLocaleString('en-US',{timeZone:'America/Los_Angeles',dateStyle:'medium',timeStyle:'short'})} Pacific · all six sets verified complete`;
 for(const task of [...new Set(data.tasks.map(r=>r.task))])$('task').add(new Option(title(task),task));
 for(const m of data.models)$('model').add(new Option(m.name,m.model));
 $('recipe-cards').innerHTML=data.models.map(m=>`<article><h3>${esc(m.name)}</h3><p class="recipe-pairs">${m.prompt_pairs} demonstration ${m.prompt_pairs===1?'pair':'pairs'} · 26/26 datasets</p><code>${esc(m.recipe)}</code><p>${esc(m.qualification)}</p><small>Verified ${new Date(m.verified_utc).toLocaleDateString('en-US',{timeZone:'America/Los_Angeles',dateStyle:'medium'})} Pacific</small></article>`).join('');
 $('pending').innerHTML=data.pending.map(m=>`<span>${esc(m.name)}</span>`).join('');
 $('provenance').innerHTML=`<dt>Evaluation contract</dt><dd>${esc(data.evaluation_contract)}</dd><dt>Frozen manifest SHA-256</dt><dd>${esc(data.manifest_sha256)}</dd><dt>Supervision version</dt><dd>${data.supervision_version}</dd>`;
 for(const id of ['view','task','model','values'])$(id).addEventListener('change',render);render();
 }catch(error){$('updated').textContent=`Results unavailable: ${error.message}. Please reload the page.`;}
}
init();
