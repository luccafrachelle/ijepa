# ¿Puede un regularizador de distribución reemplazar al EMA en I-JEPA?

I-JEPA evita el colapso con una asimetría: el encoder objetivo es un promedio móvil (EMA) del encoder de contexto y no recibe gradiente. LeJEPA (SIGReg, gaussiana isótropa) y SPHERE-JEPA (SUSReg, uniforme en la esfera) proponen en cambio un regularizador explícito sobre la distribución de los embeddings, pero lo prueban con vistas aumentadas y no con las máscaras de I-JEPA. Este experimento mete los dos regularizadores dentro de I-JEPA y cruza la presencia del EMA con el tipo de regularizador.

El detalle completo está en [`PLAN.md`](PLAN.md) (diseño, fijado antes de correr) y en [`report/informe.qmd`](report/informe.qmd) / [`report/informe.pdf`](report/informe.pdf) (resultados y discusión). La versión de divulgación está en la sección 11 del blog (`../../blog.qmd`).

## Diseño

|                    | con EMA          | sin EMA        |
|--------------------|------------------|----------------|
| sin regularizador  | **C1** I-JEPA    | **C2** control |
| SIGReg (gaussiana) | **C3**           | **C4**         |
| SUSReg (esfera)    | **C5**           | **C6**         |

- **P1**: ¿el regularizador reemplaza al EMA? C4 y C6 contra C1.
- **P2**: ¿esfera o gaussiana? C6 contra C4, y C5 contra C3.
- **P3**: ¿EMA y regularizador se suman? C3 y C5 contra C1.

**Sin EMA** significa que el encoder objetivo *es* el encoder de contexto (pesos compartidos), aplicado a la imagen completa bajo `no_grad`; equivale a m = 0. El regularizador actúa sobre una cabeza de proyección (MLP 192 → 2048 → 2048 → 256, la misma de SPHERE-JEPA), que se aplica al promedio de los tokens visibles del encoder de contexto. La pérdida es L = L_pred + λ L_reg, con la L_pred de I-JEPA (smooth L1 contra objetivos con layer norm). SIGReg y SUSReg son las implementaciones oficiales (`src/ijepa_sphere/_official_regs.py`).

**Configuración común**:

| Parámetro | Valor |
|---|---|
| Encoder | ViT-Tiny/16, 224 px, sin CLS |
| Predictor | profundidad 6, ancho 192 |
| Datos | ImageNet-100 (`clane9/imagenet-100`) |
| Entrenamiento | 100 épocas, lote 256, AdamW |
| Tasa de aprendizaje | pico 2,5e-4 |
| EMA | 0,996 → 1 |
| λ | 0,05 (barrido sin EMA) |
| Semillas | 3 por celda (C2: 1) |

Para elegir la tasa de aprendizaje y λ se hicieron barridos de 30 épocas sobre un split interno (`data/val_interno.txt`). La evaluación final se hace sobre el val oficial, con el encoder objetivo congelado y el promedio de los tokens:
- **Probe lineal**: SGD, 90 épocas.
- **k-NN**: similitud coseno, k = 20, T = 0,07.

## Resultados

Top-1 en el val oficial (%), media ± desvío entre semillas.

| Celda | EMA | Regularizador | Lineal | k-NN | Rango efectivo final |
|---|---|---|---|---|---|
| C1 | sí | — | **43,40 ± 0,22** | **38,05 ± 0,87** | 101,6 |
| C2 | no | — | 19,44 | 14,70 | 27,7 |
| C3 | sí | SIGReg | 28,37 ± 0,70 | 23,18 ± 1,00 | 62,6 |
| C4 | no | SIGReg | 25,03 ± 0,33 | 21,81 ± 0,24 | 57,0 |
| C5 | sí | SUSReg | 26,45 ± 1,48 | 21,26 ± 1,00 | 68,8 |
| C6 | no | SUSReg | 21,57 ± 0,66 | 19,28 ± 0,58 | 57,0 |

- **P1, no.** Sin EMA, las celdas con regularizador quedan entre 16 y 22 puntos debajo de C1. Pero con regularizador, el EMA sólo suma entre 1,4 y 4,9 puntos (C3 − C4, C5 − C6). La mayor parte de la distancia la pone el regularizador mismo.
- **P2, la esfera no ganó.** Sin EMA gana la gaussiana: 3,5 puntos en la evaluación lineal, diferencia distinguible. Con EMA la diferencia es de 1,9 puntos y no llega al umbral.
- **P3, no se suman.** Agregar el regularizador a I-JEPA le resta entre 15 y 17 puntos, con el λ elegido sin EMA.
- **Colapso.** C2 tuvo un colapso parcial (dimensional): su rango efectivo terminó en ~28, contra ~102 de C1. Los regularizadores lo evitan, pero el rango se estanca entre 57 y 69.

**Calibración.** El código oficial de SPHERE-JEPA, corrido sobre los mismos datos, da 72,8 en la evaluación lineal y 70,6 en k-NN, contra 71,9 y 69,0 publicados. Queda dentro del criterio de ±2 puntos fijado de antemano. Esto valida los datos y la evaluación.

**Costo.**

| Parte | Horas de GPU |
|---|---|
| Grilla (16 corridas) | 39,5 |
| Barridos | ~7 |
| Calibración (SPHERE-JEPA 21,7, LeJEPA 10,5) | ~32 |

Todo corrió en una RTX 5090 Laptop (24 GB).

## Contenido

```
PLAN.md                 diseño, criterios y presupuesto, escrito antes de correr
src/ijepa_sphere/       I-JEPA (adaptado del oficial), regularizadores, métricas de colapso, entrenamiento
configs/                base.yaml + una config por celda y por barrido
queues/                 colas de corridas (fase 2: barridos; fase 3: grilla)
scripts/                train.py, run_queue.py, run_fase1/2.sh, run_rest.sh, aggregate.py, figures.py, …
tests/                  tests (pytest)
runs/<corrida>/         config.yaml, metrics.csv, collapse.csv, knn.csv, evaluation.json, train.log
runs/calib*/            calibración con el código oficial de SPHERE-JEPA y LeJEPA
results/                tablas (grilla, barridos, tiempos) y figuras
report/                 informe (qmd + pdf) y bibliografía
logs/                   logs de las fases (los grandes, comprimidos con gzip)
data/                   manifiesto del dataset y split interno (las imágenes no se versionan)
third_party/            submódulos: facebookresearch/ijepa, galilai-group/lejepa, darktanuki/SPHEREJEPA
```

Los checkpoints (`*.pt`, `*.ckpt`, unos 5,8 GB) quedan sólo en el servidor de entrenamiento.

## Reproducir

```bash
git submodule update --init
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/verify_data.py          # verifica ImageNet-100 (HF) y escribe data/manifest.json
scripts/run_fase1.sh imagenet100_tinyvit_spherejepa   # calibración con el código oficial
scripts/run_fase2.sh                                  # barridos de lr y λ → results/sweeps.md
scripts/run_rest.sh                                   # calibración de LeJEPA y grilla completa
.venv/bin/python scripts/aggregate.py grid && .venv/bin/python scripts/figures.py
```

Una celda suelta: `PYTHONPATH=src .venv/bin/python scripts/train.py --config configs/c1.yaml --seed 0`.
