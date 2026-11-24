(function () {
  const table = document.querySelector('.p16-embedded-table table');
  if (!table) return;

  const base = '../assets/p16-comparison-demos';
  const taskSlugs = {
    'Colorization': 'colorization',
    'Deblurring': 'deblurring',
    'Denoising': 'denoising',
    'Edge detection': 'edge_detection',
    'Face landmarks': 'face_landmarks',
    'Human pose 2D': 'human_pose_2d',
    'Inpainting': 'inpainting',
    'Instance segmentation': 'instance_segmentation',
    'Relative depth': 'relative_depth',
    'Saliency detection': 'saliency_detection',
    'Semantic segmentation': 'semantic_segmentation',
    'Super-resolution': 'super_resolution',
    'Surface normals': 'surface_normals'
  };
  const top = [
    ['Query', 'query_input.webp'],
    ['VisualCloze', 'visualcloze.webp'],
    ['LaVin-DiT', 'lavin_dit.webp'],
    ['InstaManip', 'instamanip.webp'],
    ['Ground truth', 'query_target.webp']
  ];

  function imageCard(taskLabel, label, src, className) {
    const card = document.createElement('figure');
    card.className = className;
    const link = document.createElement('a');
    link.href = src;
    link.target = '_blank';
    link.rel = 'noopener';
    const img = document.createElement('img');
    img.src = src;
    img.alt = `${taskLabel}: ${label}`;
    img.loading = 'lazy';
    link.appendChild(img);
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    card.append(link, caption);
    return card;
  }

  function episodePanel(task, taskLabel) {
    const folder = `${base}/${task}_q0`;
    const panel = document.createElement('div');
    panel.className = 'p16-demo-viewer p16-inline-viewer';

    const intro = document.createElement('div');
    intro.className = 'p16-inline-heading';
    intro.innerHTML = `<div><span>Saved episode · query 1 of 17</span><h3>${taskLabel} · P=16</h3></div><p>All three models received this same query and the same ordered 16 demonstrations.</p>`;

    const queryGrid = document.createElement('div');
    queryGrid.className = 'p16-query-grid';
    queryGrid.append(...top.map(([label, file]) => imageCard(taskLabel, label, `${folder}/${file}`, 'p16-query-card')));

    const subheading = document.createElement('div');
    subheading.className = 'p16-demo-subheading';
    subheading.innerHTML = '<h4>16 demonstrations</h4><span>Input → target, in the exact episode order</span>';

    const demoGrid = document.createElement('div');
    demoGrid.className = 'p16-demo-grid';
    for (let i = 1; i <= 16; i += 1) {
      const n = String(i).padStart(2, '0');
      const pair = document.createElement('article');
      pair.className = 'p16-demo-pair';
      const title = document.createElement('h5');
      title.textContent = `Demo ${i}`;
      const images = document.createElement('div');
      images.append(
        imageCard(taskLabel, 'Input', `${folder}/demo_${n}_input.webp`, 'p16-demo-card'),
        imageCard(taskLabel, 'Target', `${folder}/demo_${n}_target.webp`, 'p16-demo-card')
      );
      pair.append(title, images);
      demoGrid.append(pair);
    }

    const manifest = document.createElement('p');
    manifest.className = 'table-footnote';
    manifest.innerHTML = 'Exact record IDs and source-pixel checksums: <a href="../p16-comparison/demo-manifest.json">open the demo manifest ↗</a>';
    panel.append(intro, queryGrid, subheading, demoGrid, manifest);
    return panel;
  }

  const header = table.querySelector('thead tr');
  const episodeHeader = document.createElement('th');
  episodeHeader.scope = 'col';
  episodeHeader.textContent = 'Saved P=16 episode';
  header.append(episodeHeader);

  let openDetail = null;
  let openButton = null;
  const rows = [...table.querySelectorAll('tbody > tr')];
  rows.forEach((row, index) => {
    const taskLabel = row.querySelector('.task-name').textContent.trim();
    const task = taskSlugs[taskLabel];
    const cell = document.createElement('td');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'p16-show-demos';
    button.textContent = 'Show 16 demos';
    button.setAttribute('aria-expanded', 'false');
    cell.append(button);
    row.append(cell);

    button.addEventListener('click', () => {
      if (openButton === button) {
        openDetail.remove();
        openDetail = null;
        openButton = null;
        button.textContent = 'Show 16 demos';
        button.setAttribute('aria-expanded', 'false');
        return;
      }
      if (openDetail) openDetail.remove();
      if (openButton) {
        openButton.textContent = 'Show 16 demos';
        openButton.setAttribute('aria-expanded', 'false');
      }
      const detail = document.createElement('tr');
      detail.className = 'p16-table-demo-row';
      const detailCell = document.createElement('td');
      detailCell.colSpan = 7;
      detailCell.append(episodePanel(task, taskLabel));
      detail.append(detailCell);
      row.after(detail);
      openDetail = detail;
      openButton = button;
      button.textContent = 'Hide demos';
      button.setAttribute('aria-expanded', 'true');
    });

    if (index === 0) button.click();
  });
}());
