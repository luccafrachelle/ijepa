// Compila a SVG (MathJax) todas las fórmulas $...$ que aparecen en las cadenas de los widgets
// y las escribe al principio de widgets/nucleo.html como window.OTK_TEX = { fuente: { svg, w, h, y } } (unidades: milésimas de em).
// Uso: (cd scripts && npm install) && node scripts/compilar_tex.mjs
import fs from 'node:fs'; import path from 'node:path'; import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH ? path.join(process.env.NODE_PATH, 'x.js') : import.meta.url);
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');
const raiz = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const adaptor = liteAdaptor(); RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', { InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: 'none' }) });
const fuentes = new Set();
// todos los widgets salvo nucleo.html (que guarda el resultado) y lo que está en vendor/
for (const f of fs.readdirSync(path.join(raiz, 'widgets')).filter(f => f.endsWith('.html') && f !== 'nucleo.html').sort()) {
  const src = fs.readFileSync(path.join(raiz, 'widgets', f), 'utf8');
  for (const m of src.matchAll(/\$(?!\{)([^$`'"\n]+?)\$/g)) fuentes.add(m[1].replace(/\\\\/g, '\\'));
}
const out = {};
for (const t of [...fuentes].sort()) {
  const node = doc.convert(t, { display: false });
  let svg = adaptor.outerHTML(adaptor.firstChild(node));
  const vb = svg.match(/viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"/).slice(1).map(Number);
  svg = svg.replace(/ width="[^"]*"/, ` width="${(vb[2] / 1000 * 96).toFixed(1)}"`).replace(/ height="[^"]*"/, ` height="${(vb[3] / 1000 * 96).toFixed(1)}"`)
           .replace(/ style="[^"]*"/, '').replace(/ role="img"| focusable="false"| aria-hidden="true"/g, '');
  out[t] = { svg, w: vb[2], h: vb[3], y: vb[1] };
}
const bloque = '<!--TEX--><script>/* Fórmulas generadas por scripts/compilar_tex.mjs: no editar a mano. */\nwindow.OTK_TEX = ' + JSON.stringify(out) + ';</script><!--/TEX-->';
const nf = path.join(raiz, 'widgets', 'nucleo.html');
fs.writeFileSync(nf, fs.readFileSync(nf, 'utf8').replace(/<!--TEX-->[\s\S]*?<!--\/TEX-->/, () => bloque));
console.log(Object.keys(out).length + ' fórmulas:', Object.keys(out).join('  |  '));
