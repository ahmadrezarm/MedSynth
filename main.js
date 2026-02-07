(() => {
  // Copy BibTeX
  const btn = document.getElementById("copyBibBtn");
  const bib = document.getElementById("bibBlock");
  if (btn && bib) {
    btn.addEventListener("click", async () => {
      const text = bib.innerText.trim();
      try {
        await navigator.clipboard.writeText(text);
        const old = btn.textContent;
        btn.textContent = "✅ Copied";
        setTimeout(() => (btn.textContent = old), 1200);
      } catch {
        // fallback: select text
        const range = document.createRange();
        range.selectNodeContents(bib);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        alert("Copy failed. BibTeX text selected—press Ctrl/Cmd+C.");
      }
    });
  }
})();
