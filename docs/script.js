const languageButton = document.getElementById('language');
let language = 'en';
languageButton.addEventListener('click', () => {
  language = language === 'en' ? 'zh' : 'en';
  document.documentElement.lang = language === 'zh' ? 'zh-Hant' : 'en';
  document.querySelectorAll('[data-en][data-zh]').forEach(element => {
    element.innerHTML = element.dataset[language];
  });
  languageButton.textContent = language === 'en' ? '繁中' : 'EN';
  languageButton.setAttribute('aria-label', language === 'en' ? 'Switch to Traditional Chinese' : 'Switch to English');
  document.getElementById('copyStatus').textContent = '';
});

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
    status.textContent = language === 'en' ? 'Copied' : '已複製';
  } catch {
    status.textContent = language === 'en' ? 'Select the citation text to copy.' : '請選取引用文字複製。';
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

// Accessible full-screen viewer for research figures and method animations.
const mediaViewer = document.getElementById('mediaViewer');
const mediaViewerStage = document.getElementById('mediaViewerStage');
const mediaViewerImage = document.getElementById('mediaViewerImage');
const mediaViewerVideo = document.getElementById('mediaViewerVideo');
const mediaViewerCaption = document.getElementById('mediaViewerCaption');
const mediaViewerClose = document.getElementById('mediaViewerClose');
let mediaViewerTrigger = null;

const zoomableMedia = document.querySelectorAll([
  '.method-figure img',
  '.loop-card video',
  '.model-card img',
  '.chart-grid img',
  '.rear-image',
  '.example-panel img'
].join(','));

function mediaCaption(media) {
  const figure = media.closest('figure');
  const caption = figure?.querySelector('figcaption');
  return caption?.textContent.trim() || media.getAttribute('alt') || '';
}

function openMediaViewer(media) {
  mediaViewerTrigger = media;
  mediaViewerCaption.textContent = mediaCaption(media);
  mediaViewer.hidden = false;
  document.body.classList.add('viewer-open');

  if (media instanceof HTMLVideoElement) {
    const source = media.currentSrc || media.querySelector('source')?.src || media.src;
    mediaViewerImage.hidden = true;
    mediaViewerImage.removeAttribute('src');
    mediaViewerVideo.hidden = false;
    mediaViewerVideo.src = source;
    mediaViewerVideo.poster = media.poster;
    mediaViewerVideo.currentTime = media.currentTime;
    mediaViewerVideo.play().catch(() => {});
  } else {
    const linkedSource = media.closest('a')?.href;
    mediaViewerVideo.pause();
    mediaViewerVideo.hidden = true;
    mediaViewerVideo.removeAttribute('src');
    mediaViewerImage.hidden = false;
    mediaViewerImage.classList.remove('is-actual');
    mediaViewerImage.src = linkedSource || media.currentSrc || media.src;
    mediaViewerImage.alt = media.alt;
  }

  mediaViewerStage.scrollTo({ top: 0, left: 0 });
  mediaViewerClose.focus();
}

function closeMediaViewer() {
  if (mediaViewer.hidden) return;
  mediaViewerVideo.pause();
  mediaViewerVideo.removeAttribute('src');
  mediaViewerImage.removeAttribute('src');
  mediaViewerImage.classList.remove('is-actual');
  mediaViewer.hidden = true;
  document.body.classList.remove('viewer-open');
  mediaViewerTrigger?.focus();
  mediaViewerTrigger = null;
}

zoomableMedia.forEach(media => {
  media.classList.add('media-zoomable');
  media.tabIndex = 0;
  media.setAttribute('role', 'button');
  media.setAttribute('aria-label', `${mediaCaption(media) || 'Research media'} — open full-screen viewer`);
  media.addEventListener('click', event => {
    event.preventDefault();
    openMediaViewer(media);
  });
  media.addEventListener('keydown', event => {
    if (!['Enter', ' '].includes(event.key)) return;
    event.preventDefault();
    openMediaViewer(media);
  });
});

mediaViewerImage.addEventListener('click', () => {
  mediaViewerImage.classList.toggle('is-actual');
  if (!mediaViewerImage.classList.contains('is-actual')) {
    mediaViewerStage.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  }
});
mediaViewerClose.addEventListener('click', closeMediaViewer);
mediaViewer.addEventListener('click', event => {
  if (event.target === mediaViewer) closeMediaViewer();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !mediaViewer.hidden) closeMediaViewer();
});
