// Basic interactivity for both index.html and install.html
// – Highlights the current nav link (scroll spy)
// – Adds a shadow / darker background to the nav when scrolling

(function () {
  const nav = document.querySelector('.topnav');
  if (!nav) return;

  // Scroll‑spy: highlight the link whose section is currently in view
  const links = nav.querySelectorAll('nav a[href^="#"]');
  const sections = [...links].map(l => document.querySelector(l.getAttribute('href'))).filter(Boolean);

  function onScroll() {
    const scrollPos = window.scrollY + 120; // offset for sticky nav

    // Toggle shadow + darker BG on nav
    nav.classList.toggle('scrolled', window.scrollY > 10);

    // Determine current section
    let current = sections[0];
    for (const section of sections) {
      if (scrollPos >= section.offsetTop) current = section;
    }

    // Highlight the corresponding link
    links.forEach(l => l.classList.remove('active'));
    const active = nav.querySelector(`nav a[href="#${current.id}"]`);
    if (active) active.classList.add('active');
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  document.addEventListener('DOMContentLoaded', onScroll);
})();


/* ── Copy-to-clipboard for <pre data-copy> blocks ── */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('pre[data-copy]:not(.no-copy)')
    .forEach(block => block.addEventListener('click', () => {
      navigator.clipboard.writeText(block.innerText.trim())
        .then(() => {
          block.classList.add('copied');
          setTimeout(() => block.classList.remove('copied'), 1200);
        });
    }));
});

/* ── Lazy-load Prism.js if any <code class="language-python"> exists ── */
(function () {
  if (!document.querySelector('code.language-python')) return; // nothing to highlight

  // 1) inject Prism core + theme
  const core = document.createElement('script');
  core.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js';
  core.defer = true;

  const theme = document.createElement('link');
  theme.rel = 'stylesheet';
  theme.href = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css';

  document.head.appendChild(theme);
  document.head.appendChild(core);

  // 2) once core loads, add Python component then highlight
  core.onload = () => {
    const py = document.createElement('script');
    py.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js';
    py.defer = true;
    py.onload = () => window.Prism && Prism.highlightAll();
    document.head.appendChild(py);
  };
})();


// --- Appended by API reference integration ---

/* =====================
   API REFERENCE ENHANCEMENTS
   ===================== */
(function () {
  // Build local TOC on the right if #local-toc exists
  const localToc = document.getElementById('local-toc');
  const content = document.querySelector('.content');
  if (localToc && content) {
    const heads = content.querySelectorAll('h2, h3');
    const ol = document.createElement('ol');
    heads.forEach(h => {
      if (!h.id) h.id = h.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-');
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      ol.appendChild(li);

      // Add anchor permalink to headings
      const anchor = document.createElement('a');
      anchor.href = '#' + h.id;
      anchor.className = 'anchor';
      anchor.setAttribute('aria-label', 'Permalink');
      anchor.textContent = '¶';
      h.appendChild(anchor);
    });
    localToc.appendChild(ol);

    // Scroll spy to highlight current heading in TOC
    const tocLinks = localToc.querySelectorAll('a');
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const id = entry.target.getAttribute('id');
        const link = localToc.querySelector(`a[href="#${id}"]`);
        if (entry.isIntersecting) {
          tocLinks.forEach(l => l.classList.remove('active'));
          if (link) link.classList.add('active');
        }
      });
    }, { rootMargin: '-40% 0px -55% 0px', threshold: [0, 1.0] });
    heads.forEach(h => obs.observe(h));
  }

  // Auto-enable copy-to-clipboard on code blocks
  document.querySelectorAll('pre:not(.no-copy)').forEach(pre => pre.setAttribute('data-copy', ''));

  // Smooth anchor scrolling with header offset
  function scrollWithOffset(e) {
    if (this.hash && this.pathname === location.pathname) {
      const target = document.querySelector(this.hash);
      if (target) {
        e.preventDefault();
        const top = target.getBoundingClientRect().top + window.pageYOffset - 80;
        window.scrollTo({ top, behavior: 'smooth' });
        history.replaceState(null, '', this.hash);
      }
    }
  }
  document.querySelectorAll('a[href^="#"]').forEach(a => a.addEventListener('click', scrollWithOffset));
})();
