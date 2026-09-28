/* MINT Lab - theme toggle.
   Dark is the default; 'light' is stored in localStorage as 'mint-theme'.
   A tiny inline script in each page's <head> applies the stored theme
   before first paint; this file just renders and wires the button. */
(function () {
  var KEY = 'mint-theme';

  function current() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function apply(theme) {
    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    try { localStorage.setItem(KEY, theme); } catch (e) { /* private mode */ }
  }

  function render(btn) {
    var dark = current() === 'dark';
    // Label shows the mode the button switches TO.
    btn.innerHTML = dark
      ? '<span class="theme-toggle-icon">☀</span>light'
      : '<span class="theme-toggle-icon">☾</span>dark';
    btn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    btn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
  }

  function init() {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'theme-toggle';

    var slot = document.querySelector('#statusline .statusline-row-top');
    if (slot) {
      slot.appendChild(btn);
    } else {
      btn.className += ' theme-toggle--floating';
      document.body.appendChild(btn);
    }

    render(btn);
    btn.addEventListener('click', function () {
      apply(current() === 'dark' ? 'light' : 'dark');
      render(btn);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
