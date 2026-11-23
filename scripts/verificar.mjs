// Verifica slides.html y blog.html con Brave headless por CDP (sin dependencias).
// Uso: node scripts/verificar.mjs [carpeta_capturas]
// - errores de consola y excepciones en los dos documentos
// - slides: desborde de cada slide (1280x720) y una captura por slide
// - blog: una captura por widget a 1280 px y ancho de página a 390 px
// - simulación de colapso: colapsa sin stop-gradient y no colapsa con stop-gradient + EMA
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const raiz = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const outDir = process.argv[2] ? path.resolve(process.argv[2]) : null;
if (outDir) fs.mkdirSync(outDir, { recursive: true });
const brave = process.env.NAVEGADOR || '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const port = 9300 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(process.env.TMPDIR || '/tmp', 'verificar-'));
const proc = spawn(brave, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--disable-gpu', '--hide-scrollbars', '--window-size=1280,720', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let targets;
for (let i = 0; i < 60; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.length) break; } catch {} await sleep(250); }
const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pend = new Map(), errores = [];
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errores.push('EXCEPCIÓN ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) errores.push(m.params.type.toUpperCase() + ' ' + m.params.args.map(a => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise(res => { const i = ++id; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async expr => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result.result.value;
};
const foto = async (nombre, clip) => { if (!outDir) return; const s = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
  fs.writeFileSync(path.join(outDir, nombre + '.png'), Buffer.from(s.result.data, 'base64')); };
await send('Page.enable'); await send('Runtime.enable');
let fallas = 0;

// ------------------------------------------------------------ slides
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: 'file://' + path.join(raiz, 'slides.html') });
await sleep(1500);
await ev(`new Promise(r => { const t = setInterval(() => { if (window.Reveal && Reveal.isReady && Reveal.isReady()) { clearInterval(t); r(true); } }, 100); })`);
await sleep(4000);
const total = await ev(`Reveal.getTotalSlides()`);
console.log('SLIDES');
for (let i = 0; i < total; i++) {
  await ev(`(() => { const idx = Reveal.getIndices(Reveal.getSlides()[${i}]); Reveal.slide(idx.h, idx.v, 99); })()`);
  await sleep(1300);
  const m = await ev(`(() => {
    const s = Reveal.getCurrentSlide(), sc = Reveal.getScale ? Reveal.getScale() : 1, top = s.getBoundingClientRect().top;
    let bottom = 0, right = 0, worst = '';
    s.querySelectorAll('*').forEach(el => {
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') return;
      if (el.closest('aside.notes, .notes, defs, .MathJax_Preview, .MJX_Assistive_MathML')) return;
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return;
      if (r.bottom > bottom) { bottom = r.bottom; worst = el.tagName + '.' + String(el.className.baseVal ?? el.className).split(' ')[0]; }
      if (r.right > right) right = r.right;
    });
    return { bottom: Math.round((bottom - top) / sc), right: Math.round(right / sc), worst, t: (s.querySelector('h2') || s).textContent.trim().replace(/\\s+/g, ' ').slice(0, 50) };
  })()`);
  const flag = m.bottom > 668 ? 'ABAJO' : m.right > 1282 ? 'DERECHA' : '';
  if (flag) fallas++;
  console.log(`${String(i + 1).padStart(2)} ${flag.padEnd(8)} bottom=${String(m.bottom).padStart(4)} right=${String(m.right).padStart(4)} ${m.t}${flag ? '  <' + m.worst + '>' : ''}`);
  await foto('s' + String(i + 1).padStart(2, '0'));
}
// la arquitectura avanza con los fragments
const pasos = await ev(`(() => { const r = document.querySelector('.ot-widget[data-widget=arquitectura]'); const sec = r.closest('section'); const idx = Reveal.getIndices(sec); Reveal.slide(idx.h, idx.v, -1); const n0 = sec.querySelectorAll('.fragment.visible').length; Reveal.nextFragment(); Reveal.nextFragment(); return [n0, sec.querySelectorAll('.fragment.visible').length, sec.querySelectorAll('.fragment.paso').length]; })()`);
console.log('fragments de arquitectura (inicio, tras dos avances, total):', pasos.join(', '));
if (pasos[0] !== 0 || pasos[1] !== 2 || pasos[2] !== 6) fallas++;
// simulación de colapso (está en el blog)
const chequeaColapso = async () => {
  const sim = await ev(`(() => { const r = document.querySelector('.ot-widget[data-widget=colapso]'); const o = {}; for (const [k, c] of [['sin', { sg: false, m: 0 }], ['m0', { sg: true, m: 0 }], ['m09', { sg: true, m: 0.9 }], ['m099', { sg: true, m: 0.99 }], ['m0996', { sg: true, m: 0.996 }]]) o[k] = r._sim(c).varianza; return o; })()`);
  console.log('varianza final de la simulación:', Object.entries(sim).map(([k, v]) => k + '=' + v.toFixed(3)).join('  '));
  if (!(sim.sin < 0.02) || ['m0', 'm09', 'm099', 'm0996'].some(k => !(sim[k] > 0.5))) { fallas++; console.log('  la simulación no se comporta como se describe'); }
  
};

// ------------------------------------------------------------ blog
for (const ancho of [1280, 390]) {
  await send('Emulation.setDeviceMetricsOverride', { width: ancho, height: 900, deviceScaleFactor: 1, mobile: ancho < 500 });
  await send('Page.navigate', { url: 'file://' + path.join(raiz, 'blog.html') });
  await sleep(3500);
  const d = await ev(`({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, n: document.querySelectorAll('.ot-widget').length, sinDibujo: [...document.querySelectorAll('.ot-widget canvas')].filter(c => c.width < 1000).length })`);
  console.log(`BLOG a ${ancho}px: ${d.n} widgets, ancho de página ${d.sw}/${d.cw}${d.sw > d.cw + 1 ? '  DESBORDA' : ''}${d.sinDibujo ? '  ' + d.sinDibujo + ' sin inicializar' : ''}`);
  if (d.sw > d.cw + 1 || d.sinDibujo) fallas++;
  if (ancho === 1280) await chequeaColapso();
  if (d.sw > d.cw + 1) console.log('  más anchos que la página:', await ev(`[...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1 && e.tagName !== 'CANVAS').slice(0, 6).map(e => e.tagName + '.' + String(e.className).split(' ')[0] + ' ' + Math.round(e.getBoundingClientRect().right)).join(' | ')`));
  const n = d.n;
  for (let k = 0; k < n; k++) {
    const r = await ev(`(() => { const el = document.querySelectorAll('.ot-widget')[${k}]; el.scrollIntoView({ block: 'center' }); document.querySelectorAll('.aparece').forEach(e => e.classList.add('visto')); const b = el.getBoundingClientRect(); return { x: Math.max(0, b.left), y: b.top + scrollY, width: Math.min(b.width, innerWidth), height: b.height, w: el.dataset.widget }; })()`);
    await sleep(1500);
    await foto(`blog${ancho}-${String(k + 1).padStart(2, '0')}-${r.w}`, { x: r.x, y: r.y, width: r.width, height: r.height });
  }
  if (ancho === 1280) { await ev(`scrollTo(0, 0)`); await sleep(300); await foto('blog1280-portada'); }
}
console.log(errores.length ? '\nMENSAJES DE CONSOLA:\n' + [...new Set(errores)].join('\n') : '\nsin errores de consola');
console.log(`${fallas + (errores.length ? 1 : 0)} problemas`);
ws.close(); proc.kill(); fs.rmSync(profile, { recursive: true, force: true });
process.exit(0);
