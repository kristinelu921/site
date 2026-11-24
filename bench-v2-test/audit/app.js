'use strict';
const controls = ['model', 'task', 'dataset'].map(id => document.getElementById(id));
function filterRows() {
  const [model, task, dataset] = controls.map(el => el.value.toLowerCase().trim());
  let shown = 0;
  document.querySelectorAll('[data-model]').forEach(el => {
    const match = (model === 'all' || !el.dataset.model || el.dataset.model === model)
      && (task === 'all' || !el.dataset.task || el.dataset.task === task)
      && (!dataset || !el.dataset.dataset || el.dataset.dataset.toLowerCase().includes(dataset));
    el.hidden = !match;
    if (match && el.dataset.model) shown++;
  });
  document.getElementById('filter-status').textContent = `${shown} matching diagnostic rows and examples. Filters apply to the tables and examples; explanatory text remains visible.`;
}
controls.forEach(el => el.addEventListener(el.type === 'search' ? 'input' : 'change', filterRows));
document.getElementById('reset').addEventListener('click', () => {
  controls[0].value = 'all'; controls[1].value = 'all'; controls[2].value = ''; filterRows();
});
