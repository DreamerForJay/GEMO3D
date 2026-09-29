const imageCompare = document.getElementById('imageCompare');
const compareRange = document.getElementById('compareRange');
const rearImage = document.getElementById('rearImage');
compareRange.addEventListener('input', () => {
  imageCompare.style.setProperty('--split', `${compareRange.value}%`);
});

const sceneTabs = [...document.querySelectorAll('[data-scene]')];
sceneTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const showRear = tab.dataset.scene === 'rear';
    imageCompare.hidden = showRear;
    rearImage.hidden = !showRear;
    sceneTabs.forEach(item => {
      const selected = item === tab;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
  });
  tab.addEventListener('keydown', event => {
    if (!['ArrowRight', 'ArrowLeft'].includes(event.key)) return;
    event.preventDefault();
    const index = (sceneTabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + sceneTabs.length) % sceneTabs.length;
    sceneTabs[index].click();
    sceneTabs[index].focus();
  });
});

const exampleTabs = [...document.querySelectorAll('[data-distance]')];
const examplePanels = [...document.querySelectorAll('[data-example]')];
exampleTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    exampleTabs.forEach(item => {
      const selected = item === tab;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    examplePanels.forEach(panel => {
      panel.hidden = panel.dataset.example !== tab.dataset.distance;
    });
  });
  tab.addEventListener('keydown', event => {
    if (!['ArrowRight', 'ArrowLeft'].includes(event.key)) return;
    event.preventDefault();
    const index = (exampleTabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + exampleTabs.length) % exampleTabs.length;
    exampleTabs[index].click();
    exampleTabs[index].focus();
  });
});

let copyTimer;
document.getElementById('copyBib').addEventListener('click', async () => {
  const status = document.getElementById('copyStatus');
  clearTimeout(copyTimer);
  try {
    await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
    status.textContent = 'Copied';
  } catch {
    status.textContent = 'Select the citation text to copy.';
  }
  copyTimer = setTimeout(() => { status.textContent = ''; }, 2500);
});

// Play the silent method loops only while they are on screen.
const loops = document.querySelectorAll('video[autoplay][muted]');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.play().catch(() => {});
      else entry.target.pause();
    });
  }, { threshold: 0.25 });
  loops.forEach(video => observer.observe(video));
}
