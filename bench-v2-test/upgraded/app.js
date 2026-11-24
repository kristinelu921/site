'use strict';

const state = { data: null };
const labels = {
  colorization:'Colorization', deblurring:'Deblurring', denoising:'Denoising',
  edge_detection:'Edge detection', face_landmarks:'Face landmarks', human_pose_2d:'Human pose 2D',
  inpainting:'Inpainting', instance_segmentation:'Instance segmentation', relative_depth:'Relative depth',
  saliency_detection:'Saliency detection', semantic_segmentation:'Semantic segmentation',
  super_resolution:'Super resolution', surface_normals:'Surface normals'
};
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => Number.isFinite(value) ? value.toFixed(6).replace(/\.?0+$/, '') : '—';
const signed = value => `${value > 0 ? '+' : ''}${fmt(value)}`;
const rankLabel = n => `<span class="rank-label"><b>#${n}</b>${n === 1 ? ' Best' : ''}</span>`;

function render() {
  const policy = $('policy').value;
  const task = $('task').value;
  const modelFilter = $('model').value;
  const scoreKey = `${policy}_score`;
  const allModels = state.data.models;
  const models = allModels.filter(model => modelFilter === 'all' || model.key === modelFilter);
  const rank = (row, rows) => {
    const values = rows.map(candidate => candidate[scoreKey]).filter(Number.isFinite);
    return 1 + values.filter(value => row.lower ? value < row[scoreKey] : value > row[scoreKey]).length;
  };
  const renderTable = (scope, target) => {
    const allRows = state.data[scope].filter(row => task === 'all' || row.task === task);
    const grouped = new Map();
    allRows.forEach(row => {
      const key = scope === 'datasets' ? `${row.task}/${row.dataset}` : row.task;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    });
    const tableRows = [...grouped.values()].sort((a,b) => {
      const left = `${a[0].task}/${a[0].dataset || ''}`;
      const right = `${b[0].task}/${b[0].dataset || ''}`;
      return left.localeCompare(right);
    });
    const reference = state.data.reference;
    const header = '<th scope="col">Category / metric</th>' + models.map(model => `<th scope="col">${esc(model.name)}<small>${policy === 'ceiling' ? 'Best available' : policy === 'upgraded' ? 'Consistent upgrade' : 'Frozen benchmark'}</small></th>`).join('') + `<th scope="col" class="reference-head">${esc(reference.name)}<small>Validation reference · P=16</small></th>`;
    const body = tableRows.map(rows => {
      const first = rows[0];
      const detail = scope === 'datasets' ? `${esc(first.dataset)} · ${first.n.toLocaleString()} queries` : `${first.n.toLocaleString()} queries · 2 test groups combined`;
      const label = `${labels[first.task]}<small>${detail} · ${esc(first.metric)} ${first.lower ? '↓' : '↑'}</small>`;
      const cells = models.map(model => {
        const row = rows.find(candidate => candidate.model === model.key);
        if (!row) return '<td>—</td>';
        const position = rank(row, rows);
        const value = row[scoreKey];
        const change = value - row.frozen_score;
        const beneficial = row.lower ? change < -1e-12 : change > 1e-12;
        const rule = policy === 'frozen' ? 'Frozen benchmark' : (row[`${policy}_policy`] || 'Query-weighted dataset readouts');
        const delta = Math.abs(change) < 1e-12 ? 'Frozen value' : `${signed(change)} vs frozen${beneficial ? ' · better' : ' · different reduction'}`;
        return `<td class="validation-score rank-${position}" title="${esc(rule)}">${rankLabel(position)}<strong>${fmt(value)}</strong><small>${esc(delta)}</small></td>`;
      }).join('');
      const ref = reference.rows.find(candidate => candidate.task === first.task);
      const referenceCell = ref ? `<td class="reference-cell" title="${esc(reference.note)}"><span class="reference-label">VAL</span><strong>${fmt(ref.score)}</strong><small>Separate reference · not ranked</small></td>` : '<td class="reference-cell">—</td>';
      return `<tr><th scope="row">${label}</th>${cells}${referenceCell}</tr>`;
    }).join('');
    $(target).innerHTML = `<div class="best-table-wrap" role="region" tabindex="0" aria-label="Color-coded upgraded test results; scroll horizontally"><table class="best-table focused-validation-table upgraded-matrix"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
    return tableRows.length;
  };

  const categoryCount = renderTable('tasks', 'category-results');
  const datasetCount = renderTable('datasets', 'results');
  $('category-caption').textContent = `${categoryCount} category groups · query-weighted combination of each category's two test groups. Flux 2 is shown separately from the test comparison.`;
  $('caption').textContent = `${datasetCount} individual test groups · ${state.data.policies[policy]} Colors rank all six test models within each row. The Flux 2 validation reference is repeated by category and excluded from ranks.`;
}

fetch('results.json').then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(data => {
  state.data = data;
  [...new Set(data.tasks.map(row => row.task))].sort().forEach(task => $('task').insertAdjacentHTML('beforeend', `<option value="${task}">${labels[task]}</option>`));
  data.models.forEach(model => $('model').insertAdjacentHTML('beforeend', `<option value="${model.key}">${esc(model.name)}</option>`));
  $('qualifications').innerHTML = data.qualifications.map(item => `<li>${esc(item)}</li>`).join('');
  ['policy','task','model'].forEach(id => $(id).addEventListener('change', render));
  render();
}).catch(error => {
  $('results').innerHTML = `<p role="alert">Could not load results: ${esc(error.message)}</p>`;
});
