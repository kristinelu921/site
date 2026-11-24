/* Saved benchmark images only. No model inference or image transformation. */
'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let catalog, datasets, page = 1;
const PAGE_SIZE = 10;
const modelLabel = id => catalog.models.find(m => m.id === id).label;
const image = (src, label, caption) => `<figure class="image-tile"><button class="image-button" type="button" data-image="${esc(src)}" data-label="${esc(label)}" aria-label="Enlarge ${esc(label)}"><img src="${esc(src)}" width="256" height="256" loading="lazy" decoding="async" alt="${esc(label)}"></button><figcaption>${esc(caption)}</figcaption></figure>`;
function populateDatasets(value = 'all') {
  const options = catalog.datasets.filter(d => $('task').value === 'all' || d.task === $('task').value);
  $('dataset').innerHTML = '<option value="all">All datasets</option>' + options.map(d => `<option value="${d.id}">${esc(d.taskLabel)} · ${esc(d.label)}</option>`).join('');
  $('dataset').value = options.some(d => d.id === value) ? value : 'all';
}
function filteredQueries() {
  const text = $('search').value.trim().toLowerCase();
  return catalog.queries.filter(q => ($('task').value === 'all' || q.task === $('task').value) && ($('dataset').value === 'all' || q.dataset === $('dataset').value) && (!text || `${q.q} ${q.record} ${datasets[q.dataset].label} ${datasets[q.dataset].taskLabel}`.toLowerCase().includes(text)));
}
function syncURL() {
  const params = new URLSearchParams();
  for (const key of ['task','dataset','model']) if ($(key).value !== 'all') params.set(key,$(key).value);
  if ($('search').value.trim()) params.set('query',$('search').value.trim());
  if (page > 1) params.set('page',page);
  if (!$('show-demos').checked) params.set('demos','0');
  if (!$('show-targets').checked) params.set('targets','0');
  const query = params.toString();
  history.replaceState(null,'',location.pathname + (query ? '?' + query : '') + location.hash);
}
function referenceSection(d) {
  return `<div class="dataset-heading"><h3>${esc(d.taskLabel)} <span>/</span> ${esc(d.label)}</h3><span>10 selected queries</span></div><details class="reference-demos"><summary>Original eight reference demo pairs</summary><div class="reference-body"><p>These are the eight pairs from your original selection folder. The context pairs actually used in each run appear on the model cards below.</p><div class="reference-grid">${d.references.map(r => `<div><h4>${r.id}</h4><div class="pair">${image(r.input,`${d.label} reference ${r.id} input`,'Input')}${image(r.target,`${d.label} reference ${r.id} target`,'Target')}</div></div>`).join('')}</div></div></details>`;
}
function modelCard(q, id) {
  const m = q.models[id], label = modelLabel(id), prefix = `${datasets[q.dataset].label} · ${q.q} · ${label}`;
  const result = `<div class="${$('show-targets').checked ? 'pair' : 'single'}">${image(m.output,prefix+' output','Saved output')}${$('show-targets').checked ? image(m.target,prefix+' matching target','Matching target') : ''}</div>`;
  const demos = $('show-demos').checked ? m.demos.map((d,i) => `<div class="demo-set"><p class="demo-heading">Context demo ${i+1} <span>· ${d.id}</span></p><div class="pair">${image(d.input,prefix+` demo ${i+1} input`,'Demo input')}${image(d.target,prefix+` demo ${i+1} target`,'Demo target')}</div></div>`).join('') : '';
  return `<article class="model-card" data-model="${id}"><div class="model-heading"><h4>${label}</h4><span>${m.demos.length} context ${m.demos.length === 1 ? 'pair' : 'pairs'}</span></div>${result}${demos}</article>`;
}
function render() {
  const matches = filteredQueries(), maxPage = Math.max(1,Math.ceil(matches.length/PAGE_SIZE));
  page = Math.min(Math.max(1,page),maxPage);
  const start = (page-1)*PAGE_SIZE, shown = matches.slice(start,start+PAGE_SIZE);
  const models = $('model').value === 'all' ? catalog.models.map(m => m.id) : [$('model').value];
  let previousDataset = '', html = '';
  for (const q of shown) {
    if (q.dataset !== previousDataset) { html += referenceSection(datasets[q.dataset]); previousDataset = q.dataset; }
    html += `<section class="query" data-query="${esc(q.id)}"><header class="query-header">${image(q.input,`${datasets[q.dataset].label} · ${q.q} query input`,'Query input')}<div><h3>${q.q}</h3><p>${esc(datasets[q.dataset].taskLabel)} · ${esc(datasets[q.dataset].label)}</p><p>${esc(q.record)}</p><button type="button" class="text-button" data-image="${esc(q.referenceTarget)}" data-label="${esc(datasets[q.dataset].label)} · ${q.q} · original eight-demo reference target">Original reference target</button></div><button type="button" class="copy-link" data-copy="${esc(q.id)}">Copy query link</button></header><div class="query-models ${models.length===1 ? 'one-model' : ''}">${models.map(m => modelCard(q,m)).join('')}</div></section>`;
  }
  if (!shown.length) html = '<div class="empty"><h3>No matching queries</h3><p>Try another record ID or reset the filters to browse all 260 selections.</p></div>';
  $('gallery').innerHTML = html;
  $('gallery').setAttribute('aria-busy','false');
  const caption = matches.length ? `Showing ${start+1}–${Math.min(start+PAGE_SIZE,matches.length)} of ${matches.length} queries · ${models.length} ${models.length===1?'model':'models'}` : '0 matching queries';
  $('caption').textContent = $('bottom-caption').textContent = caption;
  $('page-count').textContent = $('page-count-bottom').textContent = `${page} / ${maxPage}`;
  for (const id of ['previous','previous-bottom']) $(id).disabled = page<=1;
  for (const id of ['next','next-bottom']) $(id).disabled = page>=maxPage;
  syncURL();
}
function notify(text) { $('toast').textContent=text; $('toast').classList.add('visible'); clearTimeout(notify.timer); notify.timer=setTimeout(()=>$('toast').classList.remove('visible'),2500); }
function openImage(button) {
  $('lightbox-title').textContent = button.dataset.label;
  $('lightbox-image').src = button.dataset.image;
  $('lightbox-image').alt = button.dataset.label;
  $('download-image').href = button.dataset.image;
  $('download-image').download = button.dataset.label.replace(/[^a-z0-9_-]+/gi,'_')+'.png';
  $('lightbox').showModal();
}
async function copyQuery(id) {
  const q = catalog.queries.find(q => q.id === id), url = new URL(location.href);
  url.search = new URLSearchParams({dataset:q.dataset,query:q.q}).toString();url.hash='explore';
  try { await navigator.clipboard.writeText(url.href);notify('Query link copied'); }
  catch { window.prompt('Copy this query link:',url.href); }
}
document.addEventListener('click', event => {
  const button = event.target.closest('[data-image]');if(button) openImage(button);
  const copy = event.target.closest('[data-copy]');if(copy) copyQuery(copy.dataset.copy);
});
$('close-lightbox').addEventListener('click',()=>$('lightbox').close());
$('lightbox').addEventListener('click',event=>{if(event.target===$('lightbox'))$('lightbox').close();});
function restoreState() {
  const params = new URLSearchParams(location.search);
  const task = params.get('task');if([...$('task').options].some(o=>o.value===task))$('task').value=task;
  populateDatasets(params.get('dataset')||'all');
  const model=params.get('model');if(catalog.models.some(m=>m.id===model))$('model').value=model;
  $('search').value=params.get('query')||'';
  page=Math.max(1,parseInt(params.get('page')||'1',10)||1);
  $('show-demos').checked=params.get('demos')!=='0';$('show-targets').checked=params.get('targets')!=='0';
}
async function start() {
  try {
    const response=await fetch('catalog.json');if(!response.ok)throw new Error('catalog unavailable');catalog=await response.json();
    datasets=Object.fromEntries(catalog.datasets.map(d=>[d.id,d]));
    const tasks=new Map(catalog.datasets.map(d=>[d.task,d.taskLabel]));
    for(const [id,label] of tasks)$('task').add(new Option(label,id));
    for(const m of catalog.models)$('model').add(new Option(m.label,m.id));
    restoreState();
    $('task').addEventListener('change',()=>{populateDatasets();page=1;render();});
    for(const id of ['dataset','model','show-demos','show-targets'])$(id).addEventListener('change',()=>{page=1;render();});
    $('search').addEventListener('input',()=>{page=1;render();});
    $('reset').addEventListener('click',()=>{for(const id of ['task','model'])$(id).value='all';populateDatasets();$('search').value='';$('show-demos').checked=true;$('show-targets').checked=true;page=1;render();});
    for(const id of ['previous','previous-bottom','next','next-bottom'])$(id).addEventListener('click',()=>{page+=id.startsWith('next')?1:-1;render();$('explore').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});});
    window.addEventListener('popstate',()=>{ $('task').value='all';$('model').value='all';restoreState();render();});
    render();
  } catch(error) {
    $('caption').textContent='The gallery could not load. Please reload the page.';$('gallery').setAttribute('aria-busy','false');
    $('gallery').innerHTML='<p class="note">You can still download the <a href="image-index.csv">complete image index</a>.</p>';console.error(error);
  }
}
start();
