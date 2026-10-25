const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1440, height: 560 }, deviceScaleFactor: 2.5 });
  await p.goto('file:///home/claude/fig1/fig1.svg');
  await p.screenshot({ path: '/home/claude/fig1/fig1_setup.png' });
  await p.goto('about:blank'); await p.setContent(`<html><body style="margin:0">${require('fs').readFileSync('/home/claude/fig1/fig1.svg','utf8')}</body></html>`);
  await p.pdf({ path: '/home/claude/fig1/fig1_setup.pdf', width: '1440px', height: '560px', printBackground: true, pageRanges: '1' });
  await b.close();
})();
