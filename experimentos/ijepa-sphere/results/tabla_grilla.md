# Grilla I-JEPA (ImageNet-100, ViT-Tiny)

Accuracy top-1 en % (media ± desvío estándar muestral entre semillas; `(n=1)` = una sola semilla, sin desvío). Rango efectivo, uniformidad y alineación: finales, embeddings pooled, media entre semillas.

| Celda | Descripción | n semillas | Estado | Lineal | k-NN | Rango ef. | Uniformidad | Alineación | GPU-h/corrida | GPU-h total |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | EMA, sin reg (I-JEPA) | 3 | ok | 43.40 ± 0.22 | 38.05 ± 0.87 | 101.6 | -1.578 | 0.097 | 2.42 | 7.26 |
| C2 | sin EMA, sin reg (control) | 1 | ok | 19.44 (n=1) | 14.70 (n=1) | 27.7 | -0.681 | 0.027 | 2.41 | 2.41 |
| C3 | EMA + SIGReg | 3 | ok | 28.37 ± 0.70 | 23.18 ± 1.00 | 62.6 | -2.306 | 0.170 | 2.46 | 7.38 |
| C4 | sin EMA + SIGReg | 3 | ok | 25.03 ± 0.33 | 21.81 ± 0.24 | 57.0 | -2.356 | 0.138 | 2.45 | 7.35 |
| C5 | EMA + SUSReg | 3 | ok | 26.45 ± 1.48 | 21.26 ± 1.00 | 68.8 | -1.664 | 0.177 | 2.52 | 7.57 |
| C6 | sin EMA + SUSReg | 3 | ok | 21.57 ± 0.66 | 19.28 ± 0.58 | 57.0 | -1.827 | 0.106 | 2.51 | 7.54 |

## Comparaciones (diferencia de medias A − B)

"distinguible" si |diferencia| > 2 desvíos (el mayor de los dos) o si el test t pareado por semilla da p < 0.05; si no, "no distinguible" (PLAN §6).

| Pregunta | A vs B | Métrica | Dif. (pts) | p (t pareado) | Veredicto |
|---|---|---|---|---|---|
| P1 | C6 vs C1 | lineal | -21.83 | 0.000 | distinguible |
| P1 | C6 vs C1 | k-NN | -18.77 | 0.001 | distinguible |
| P2 | C6 vs C4 | lineal | -3.47 | 0.004 | distinguible |
| P2 | C6 vs C4 | k-NN | -2.53 | 0.018 | distinguible |
| P2 | C5 vs C3 | lineal | -1.92 | 0.076 | no distinguible |
| P2 | C5 vs C3 | k-NN | -1.92 | 0.054 | no distinguible |
| P3 | C5 vs C1 | lineal | -16.95 | 0.002 | distinguible |
| P3 | C5 vs C1 | k-NN | -16.79 | 0.001 | distinguible |

## Tabla combinada con la tabla 7 publicada

| Método | Lineal | k-NN | Notas |
|---|---|---|---|
| LeJEPA (publicado) | 70.71 ± 0.38 | 67.39 ± 0.61 | publicado |
| SPHERE-JEPA (publicado) | 71.91 ± 0.90 | 69.03 ± 0.43 | publicado |
| InfoNCE (publicado) | 51.93 | 56.93 | publicado |
| VICReg (publicado) | 57.74 | 56.01 | publicado |
| C1: EMA, sin reg (I-JEPA) | 43.40 ± 0.22 | 38.05 ± 0.87 | n=3, ok |
| C2: sin EMA, sin reg (control) | 19.44 (n=1) | 14.70 (n=1) | n=1, ok |
| C3: EMA + SIGReg | 28.37 ± 0.70 | 23.18 ± 1.00 | n=3, ok |
| C4: sin EMA + SIGReg | 25.03 ± 0.33 | 21.81 ± 0.24 | n=3, ok |
| C5: EMA + SUSReg | 26.45 ± 1.48 | 21.26 ± 1.00 | n=3, ok |
| C6: sin EMA + SUSReg | 21.57 ± 0.66 | 19.28 ± 0.58 | n=3, ok |

Nota: las filas publicadas (tabla 7 de SPHERE-JEPA) usan 200 épocas y 6 vistas por imagen (token CLS); nuestras celdas I-JEPA usan máscaras (no vistas), avg-pool y las épocas indicadas en el protocolo de la grilla. No son estrictamente comparables en costo.
