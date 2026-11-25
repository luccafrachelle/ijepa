# I-JEPA: aprender a ver prediciendo representaciones

Proyecto final de *Redes Neuronales para Visión Artificial* (Udelar, FING, 2026). Lucca Frachelle.

Tiene dos partes:

1. **Charla y blog** sobre I-JEPA (Assran et al., CVPR 2023), con la propuesta de LeCun (2022) como contexto. Las slides duran unos 20 minutos. El blog es la versión larga e interactiva.
2. **Experimento propio**: I-JEPA con los regularizadores de distribución SIGReg (LeJEPA) y SUSReg (SPHERE-JEPA), en lugar del EMA o además de él. Es una grilla 2 × 3 corrida en una RTX 5090 Laptop. Ver [`experimentos/ijepa-sphere/`](experimentos/ijepa-sphere/).

## Estructura

```
slides.qmd, slides.html        charla (Reveal.js, 1280×720), HTML autocontenido
blog.qmd, blog.html            blog interactivo, HTML autocontenido
widgets/                       animaciones (canvas y SVG + GSAP), compartidas por slides y blog
  nucleo.html                  librería: canvas, pasos, controles, fórmulas compiladas
  anim.html                    base SVG + GSAP (A.pasos, helpers); instrucciones al principio del archivo
  vendor/gsap.min.js           GSAP 3.13.0 (cdnjs), incrustado en el HTML: funciona sin internet
  objetivo.html                paso 4, el encoder objetivo (SVG + GSAP)
  ijepa.html                   figura general, máscaras, píxeles vs. representación, energía, resultados…
  pasos.html                   los siete pasos de I-JEPA y la evaluación
  experimento.html             regularizadores, grilla de resultados y rango efectivo
scripts/
  compilar_tex.mjs             compila con MathJax las fórmulas $…$ de los widgets
  verificar.mjs                verificación con un navegador headless (errores, desbordes, capturas)
references.bib, estilo-*.scss, fonts/, img/, title-slide.html
papers/README.md               fuentes (los PDF no se versionan)
experimentos/ijepa-sphere/     código, configuraciones, corridas, resultados e informe del experimento
```

## Compilar

Requisitos: Quarto ≥ 1.7 y Node ≥ 22.

```bash
# fórmulas de los widgets (sólo si se cambian textos con $…$)
(cd scripts && npm install) && node scripts/compilar_tex.mjs

quarto render slides.qmd
quarto render blog.qmd

# verificación: errores de consola, desbordes por slide, blog a 1280 y 390 px
node scripts/verificar.mjs /tmp/capturas
```

`verificar.mjs` usa Brave por CDP. Se puede apuntar a otro navegador Chromium con `NAVEGADOR=/ruta/al/binario`.

Las animaciones del proceso avanzan por pasos. En las slides se avanza con la flecha del teclado, porque cada paso es un *fragment* de Reveal. En el blog se avanza con los botones o haciendo clic en la figura.

## Verificación del contenido

Las afirmaciones sobre el método se contrastaron con el paper (arXiv 2301.08243v3) y con el código oficial, facebookresearch/ijepa en el commit `52c1ae9`. Por ejemplo:

- La pérdida es L2 al cuadrado en el paper y smooth L1 en el código.
- El código no usa *gradient clipping*.
- El tamaño de los bloques de las máscaras se sortea una vez por batch.
- Los resultados con 1 % de etiquetas se obtienen con ajuste fino.

Los números del experimento se contrastaron con las corridas guardadas en `experimentos/ijepa-sphere/runs/`.

## Historial

El historial del experimento (`experimentos/ijepa-sphere/`) viene del repositorio original en el servidor de entrenamiento (`fcea:~/dlviz/ijepa-sphere`), incorporado con `git subtree`. Los repos de terceros (`third_party/`) son submódulos fijados a los commits usados. Después de clonar:

```bash
git submodule update --init
```
