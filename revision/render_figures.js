// Render the SVG figures written by figures_drawio.py to vector PDF and 600-dpi-class PNG.
//   node revision/render_figures.js revision/results/figures_drawio
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const dir = process.argv[2] || 'revision/results/figures_drawio';
  const browser = await chromium.launch();
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.svg'))) {
    const svg = fs.readFileSync(path.join(dir, f), 'utf8');
    const w = +svg.match(/width="([\d.]+)"/)[1];
    const h = +svg.match(/height="([\d.]+)"/)[1];
    const html = `<html><head><style>html,body{margin:0;padding:0;background:#fff}</style></head><body>${svg}</body></html>`;
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 4 });
    await page.setContent(html);
    const base = path.join(dir, f.replace(/\.svg$/, ''));
    await page.screenshot({ path: base + '.png', clip: { x: 0, y: 0, width: w, height: h } });
    await page.pdf({ path: base + '.pdf', width: `${w}px`, height: `${h}px`, printBackground: true,
                     pageRanges: '1', margin: { top: 0, right: 0, bottom: 0, left: 0 } });
    await page.close();
    console.log('rendered', base);
  }
  await browser.close();
})();
