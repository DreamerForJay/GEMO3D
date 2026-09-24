const imageCompare = document.getElementById('imageCompare');
const compareRange = document.getElementById('compareRange');
const rearImage = document.getElementById('rearImage');
compareRange.addEventListener('input', () => {
  imageCompare.style.setProperty('--split', `${compareRange.value}%`);
});
document.querySelectorAll('[data-scene]').forEach(button => {
  button.addEventListener('click', () => {
    const showRear = button.dataset.scene === 'rear';
    imageCompare.hidden = showRear;
    rearImage.hidden = !showRear;
    document.querySelectorAll('[data-scene]').forEach(tab => {
      const active = tab === button;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
  });
  button.addEventListener('keydown', event => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll('[data-scene]')];
    tabs[(tabs.indexOf(button) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length].click();
    document.querySelector('[data-scene].active').focus();
  });
});
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
});
const copyButton = document.getElementById('copyBib');
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.getElementById('bibtex').innerText);
    copyButton.textContent = language === 'en' ? 'Copied' : '已複製';
    setTimeout(() => copyButton.textContent = language === 'en' ? 'Copy' : '複製', 1800);
  } catch {
    copyButton.textContent = language === 'en' ? 'Select text to copy' : '請選取文字複製';
  }
});
