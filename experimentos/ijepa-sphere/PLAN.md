# Plan de experimentos: I-JEPA con regularizadores de distribución (esfera y gaussiana)

Proyecto final de Redes Neuronales para Visión Artificial. Este documento es el plan completo; todavía no hay código. Está pensado para que una sesión nueva de Claude Code en esta máquina arranque a implementar a partir de acá, en el orden de las fases.

---

## 0. Resumen en un párrafo

I-JEPA evita el colapso con una asimetría (corte de gradiente + encoder objetivo por EMA), sin garantía teórica. SPHERE-JEPA (Nicollier, …, Musé, Facciolo; NeurIPS 2026) lo evita con un regularizador explícito, SUSReg, que empuja los embeddings normalizados hacia la distribución uniforme en la esfera; LeJEPA hace lo mismo con una gaussiana (SIGReg). Ambos trabajan con vistas aumentadas, no con máscaras. **El experimento mete esos regularizadores dentro de I-JEPA** (máscaras multi-bloque, predictor, sin aumentos) y pregunta: (1) ¿puede el regularizador reemplazar al EMA?, (2) ¿importa la geometría (esfera vs. gaussiana)?, (3) ¿EMA y regularizador se suman? Escala: ViT-Tiny/16 en ImageNet-100, la misma escala que la tabla 7 de SPHERE-JEPA, para tener un número publicado contra el cual calibrar.

---

## 1. La máquina (relevado el 17/11/2026)

| Recurso | Valor | Consecuencia |
|---|---|---|
| GPU | **RTX 5090 Laptop**, 24 GB, tope 175 W, sm_120 | ~24 GB, no 32. Más lenta que la de escritorio. Batch 256 de ViT-Tiny entra holgado. |
| Driver / CUDA | 595.91, CUDA 13.2 | PyTorch con wheels cu128 o posteriores funciona. |
| PyTorch existente | `~/dlviz/.venv`: torch 2.11.0+cu128, torchvision 0.26, CUDA disponible | Sirve de base; crear un venv propio para este proyecto. |
| CPU / RAM | Core Ultra 9 275HX (24 hilos), 62 GB | Data loading con 12–16 workers está bien. |
| Disco | **105 GB libres** de 197 GB (se liberó espacio el 18/11: caché de pip, modelo Qwen GGUF, venv de sim_oml, descargas de TFDS) | Alcanza para ImageNet-100 completo, checkpoints y ADE20K opcional. Ver §4. |
| Otros | tmux y screen instalados; `uv` no instalado; máquina con varias sesiones abiertas | Correr todo en tmux. Mirar `nvidia-smi` antes de lanzar, por si alguien más usa la GPU. Laptop: enchufada y con buena ventilación (el thermal throttling cambia los tiempos). |

---

## 2. Material de referencia (en `papers/` de esta carpeta)

- `2605.26900.pdf` — **SPHERE-JEPA** (paper principal). Tabla 7 (apéndice D) es el ancla de comparación.
- `2606.17603.pdf` — *Expanding SPHERE-JEPA* (MMD/KSD/KL deterministas; opcional).
- `2609.21656.pdf` — *Beyond Gaussian Worlds* (teoría de identificabilidad; contexto).
- Código oficial de SPHERE-JEPA: <https://github.com/darktanuki/SPHEREJEPA> (enlazado desde la página del póster de NeurIPS 2026; sin licencia explícita → uso académico).
- Código oficial de I-JEPA: <https://github.com/facebookresearch/ijepa> (CC BY-NC 4.0).
- Código oficial de LeJEPA: <https://github.com/galilai-group/lejepa> (CC BY-NC 4.0).
- Paper I-JEPA: Assran et al., CVPR 2023, arXiv 2301.08243.

### Lo que dice el código oficial de SPHERE-JEPA (leído, importa para la implementación)

- `losses/spherejepa.py`:
  - `SUSReg`: proyecta los embeddings **normalizados** sobre 1024 direcciones aleatorias, **reescala por √d**, y compara la función característica empírica contra la exacta de una coordenada de la esfera (Bessel), con el test de Epps–Pulley; `t_max=5`, `knots` configurable (51 por defecto, 151 en el config de ViT-Tiny).
  - **Si `dim > 256` usa la aproximación gaussiana** en lugar de la Bessel. ⇒ La cabeza de proyección debe tener **d = 256** (o menos), si no SUSReg ≈ SIGReg sobre vectores normalizados y la comparación esfera/gaussiana pierde sentido.
  - Pérdida de invariancia por coseno: `1 − ⟨z, μ⟩`. `Lreg` se multiplica por el tamaño del batch B.
- `losses/lejepa.py`: `SIGReg`, mismo esquema contra `exp(−t²/2)`, sin normalizar, estadístico multiplicado por N.
- `lightningmodules/pretrained.py`: ViT-Tiny = `timm.create_model("vit_tiny_patch16_224", global_pool="token")` (usa token CLS).
- `evaluation.py`: **protocolo de evaluación** — features congeladas del encoder; k-NN ponderado (k=20, T=0.07, similitud coseno); probe lineal con SGD, 90 épocas, lr 0.1·(B/256). **Usar este mismo código para evaluar todo** (es lo que hace comparables los números).
- `dataloaders/imagenet100.py`: dataset HF `clane9/imagenet-100`; soporta `source: hf` (por defecto, baja el dataset entero) y `source: folder` (ImageFolder local). Ver §4.
- `configs/imagenet100_tinyvit_{spherejepa,lejepa}.yaml`: 224 px, 6 vistas (2 globales a 224 con escala (0.4, 1), 4 locales a 96 con escala (0.05, 0.4)), 200 épocas, batch 256, lr 0.05, wd 5e-4, `projection_dim` 256, λ = 0.05.
- Diferencias admitidas en el README respecto del paper: lr constante tras 1 época de warmup (el paper usa coseno), `t_max` fijo. No trae ImageNet-1K ni EMA.
- No fija precisión ⇒ Lightning corre en fp32 por defecto.

---

## 3. Preguntas, hipótesis y lectura de resultados

| Pregunta | Comparación | Hipótesis | Si sale al revés |
|---|---|---|---|
| P1. ¿El regularizador reemplaza al EMA? | C6 (sin EMA + SUSReg) vs. C1 (I-JEPA) | C6 no colapsa y queda cerca de C1 | Si C6 no colapsa pero queda bastante abajo: el regularizador evita el colapso pero no reemplaza la estabilización del objetivo (coherente con lo que el EMA aporta en SPHERE-JEPA). |
| P2. ¿Importa la geometría? | C6 vs. C4 (y C5 vs. C3) | La esfera gana sobre todo en k-NN | Si dan igual: lo que importa es regularizar, no la forma. |
| P3. ¿Se suman? | C5 (EMA + SUSReg) vs. C1 | C5 ≥ C1 | Si C5 < C1: el regularizador pelea con el objetivo de predicción en I-JEPA. |
| Control | C2 (sin EMA, sin regularizador) | Colapsa (varianza → 0) | Si no colapsa, revisar implementación: la grilla no prueba nada sin este control. |

Dato motivador (SPHERE-JEPA, tabla 3, ImageNet-1K, ViT-B/14, 100 ép.): sin EMA LeJEPA 59,2 / 50,4 y SPHERE-JEPA 61,0 / 50,4 (lineal / k-NN); con EMA 68,2 / 61,7 y 69,3 / 62,6. Ahí el regularizador ya evita el colapso, así que los ~8–9 puntos del EMA vienen de otra cosa (objetivo más estable). En I-JEPA el EMA hace las dos cosas a la vez: este experimento permite separarlas.

---

## 4. Datos: ImageNet-100

Hay espacio, así que se usa el dataset **tal como lo usa el repo oficial**, sin redimensionar: es lo más fiel para reproducir la tabla 7.

1. Bajar `clane9/imagenet-100` de Hugging Face con el loader del repo (`source: hf`) o con `datasets.load_dataset`, poniendo la caché de HF dentro de la carpeta del proyecto (`HF_HOME=~/dlviz/ijepa-sphere/data/hf`) para saber qué ocupa y poder borrarlo al final. Anotar el tamaño real en disco.
2. Verificar: 100 clases, ≈ 126,7k imágenes de train (≈ 1300 por clase) y 5000 de val (50 por clase), sin imágenes corruptas. Guardar un `manifest.json` con conteos y hash del listado.
3. Si el data loading desde parquet resulta cuello de botella en la medición de la fase 0 (GPU < 80 % de uso), exportar una vez a ImageFolder (JPEG, resolución original) y usar `source: folder`. Si se redimensiona, hacerlo **para todas las corridas** y documentarlo.
4. Generar el split de validación interno (10 % de train, estratificado, semilla fija, ids en `data/val_interno.txt`).
5. **Mismo conjunto de datos para todas las corridas** (calibración y grilla). Esto es lo que hace comparables los resultados internos.

Presupuesto de disco: dataset (anotar el tamaño real, del orden de decenas de GB con la caché) + checkpoints (ViT-Tiny ≈ 5,7M parámetros: ~25 MB el modelo, ~70 MB con optimizador; guardar sólo `last` y `best` por corrida) + logs. Avisar si el disco baja de 10 GB libres; no borrar nada fuera de esta carpeta sin preguntar.

---

## 5. Diseño experimental

### 5.1 Lo que se mantiene fijo en toda la grilla (I-JEPA)

| Parámetro | Valor | Origen |
|---|---|---|
| Encoder | ViT-Tiny/16 (dim 192, 12 capas, 3 cabezas), **sin token CLS** | Escala de la tabla 7 de SPHERE-JEPA; I-JEPA no usa CLS. |
| Resolución | 224 px (196 parches) | Igual que la tabla 7 ⇒ comparable. Plan B en §8. |
| Aumentos | sólo RandomResizedCrop (escala 0,3–1,0) + flip horizontal | Como I-JEPA (verificar valores en `src/` del repo oficial). |
| Máscaras | 4 bloques objetivo, escala (0,15; 0,2), aspecto (0,75; 1,5); contexto escala (0,85; 1,0), aspecto 1; se le quitan al contexto los parches de los objetivos | Paper I-JEPA, apéndice A. |
| Predictor | ViT angosto, profundidad 6, ancho 192 | El paper usa ancho 384 con encoders grandes; con un encoder de ancho 192, 384 no es un cuello de botella. Decisión a registrar. |
| Pérdida de predicción | **la de I-JEPA sin cambios** en todas las celdas | Así la única variable es el mecanismo anti-colapso. El código oficial normaliza (layer norm) los objetivos y usa smooth L1: verificar en `src/train.py` y copiar exactamente. |
| Optimizador | AdamW; warmup 10 % de las épocas, coseno; wd 0,04 → 0,4 lineal | Paper I-JEPA. |
| lr pico | a fijar en la fase 2 (barrido {2,5e-4; 5e-4; 1e-3}) | El paper usa 1e-3 con batch 2048; con batch 256 hay que barrer. |
| Batch | 256 | Igual que la calibración. |
| Precisión | bf16 autocast (y TF32 activado) | Medir en fase 0 que no cambie resultados vs. fp32 en una corrida corta. |
| EMA (cuando hay) | m de 0,996 → 1,0 lineal | Paper I-JEPA. |
| Evaluación | `evaluation.py` del repo SPHERE-JEPA, features = **promedio de los tokens de salida del encoder objetivo** | Protocolo común; I-JEPA evalúa con avg-pool del encoder objetivo. |

### 5.2 La grilla (2 × 3)

| | **con EMA** | **sin EMA** |
|---|---|---|
| **sin regularizador** | **C1** I-JEPA original (referencia) | **C2** control: debería colapsar |
| **SIGReg** (gaussiana, LeJEPA) | **C3** | **C4** |
| **SUSReg** (esfera, SPHERE-JEPA) | **C5** | **C6** |

Definiciones precisas:

- **"sin EMA"**: el encoder objetivo **es el mismo** encoder de contexto (pesos compartidos), se calcula sobre la imagen completa con `torch.no_grad()` / corte de gradiente (equivale a m = 0).
- **Dónde va el regularizador (diseño principal, R-pool):** cabeza de proyección `h` (MLP 192 → 2048 → 256, como en LeJEPA/SPHERE-JEPA) aplicada al **promedio de los tokens del encoder de contexto** sobre los parches visibles. El regularizador actúa sobre `h(·)` del batch: para SUSReg, vectores normalizados (lo hace la propia función); para SIGReg, sin normalizar. Tiene que ir en una rama con gradiente: en las celdas con EMA el encoder objetivo no se entrena, así que el regularizador no puede ir sólo ahí.
- **Pérdida total:** `L = L_pred + λ · L_reg`. Nota: los papers usan `(1−λ)·L_inv + λ·L_reg`; con λ chico es casi lo mismo, pero dejarlo explícito en el código y en el informe. Usar `SUSReg` y `SIGReg` **copiados del repo oficial sin cambios** (incluido el factor B / N).
- **λ:** barrido en fase 2 sobre C6 y C4: {0,01; 0,05; 0,2}. Elegir uno por regularizador por k-NN en un **split de validación interno** (10 % del train, ver §6), no en el val oficial. Mismo λ en la celda con EMA.

### 5.3 Variantes opcionales (sólo si sobra tiempo; orden de prioridad)

1. **R-tokens:** regularizador sobre cada token de parche (cada token como una muestra; B × N_ctx muestras por paso). Responde la duda del mail.
2. **C6-sg:** sin EMA y **sin** corte de gradiente (el gradiente pasa por las dos ramas, como en LeJEPA). ¿El regularizador solo alcanza?
3. **Predicción sobre la esfera:** normalizar predicción y objetivo y usar pérdida de coseno en C5/C6 (más fiel a SPHERE-JEPA, pero cambia dos cosas a la vez).
4. **MMD esférico determinista** (del paper *Expanding SPHERE-JEPA*) en lugar de SUSReg en C6.
5. **Más épocas** (200 → 300) para C1 y la mejor celda regularizada.

---

## 6. Comparabilidad: cómo se garantiza

Dos niveles:

**Interno (lo principal).** Todas las celdas: mismos datos, misma resolución, mismas épocas, mismo batch, mismo cronograma de lr y wd, mismos aumentos, mismas máscaras, mismo predictor, mismo código de evaluación, mismas semillas (0, 1, 2). La única diferencia entre celdas es EMA sí/no y el regularizador. Cualquier cambio de configuración se aplica a toda la grilla y se re-corre.

**Externo (ancla publicada).** La fase 1 reproduce con el código oficial dos filas de la **tabla 7 de SPHERE-JEPA** (ViT-Tiny, ImageNet-100, 200 épocas, 4 semillas):

| Método | Lineal | k-NN |
|---|---|---|
| LeJEPA | 70,71 ± 0,38 | 67,39 ± 0,61 |
| SPHERE-JEPA (SUSReg) | 71,91 ± 0,90 | 69,03 ± 0,43 |
| (también reportan InfoNCE 51,93 / 56,93 y VICReg 57,74 / 56,01) | | |

Si nuestra reproducción cae dentro de ~2 puntos, los datos y la evaluación están bien, y los números de la grilla I-JEPA se pueden poner **en la misma tabla** que estos (misma escala, mismo protocolo de evaluación), aclarando que I-JEPA usa máscaras en lugar de vistas.

Cuidados para no romper la comparabilidad:
- **Pooling:** la tabla 7 usa el token CLS; I-JEPA no tiene CLS y evalúa con avg-pool. Para las corridas de calibración, evaluar **con los dos** (CLS y avg-pool de los tokens) y reportar ambos.
- **Épocas efectivas:** LeJEPA/SPHERE-JEPA procesan 6 vistas por imagen, I-JEPA una. Reportar también **horas de GPU** por corrida, que es la comparación justa de costo.
- **Selección de hiperparámetros:** lr y λ se eligen en un split de validación interno (10 % de train, fijo y guardado en un archivo con los ids). El val oficial (5000 imágenes) se usa **sólo** para los números finales.
- **Semillas:** 3 por celda (0, 1, 2). Reportar media ± desvío estándar muestral. Para afirmar que una celda supera a otra, pedir diferencia de medias > 2 desvíos (o test t pareado por semilla, p < 0,05). Si no, decir "no distinguible".
- **No reutilizar** el número de un paper con otra resolución, otras épocas u otro protocolo como si fuera comparable.

---

## 7. Qué medir

### Durante el entrenamiento (cada época, a CSV + TensorBoard o W&B offline)
- `L_pred`, `L_reg`, `L` total; lr; m del EMA; norma de gradiente.
- **Indicadores de colapso** sobre un subconjunto fijo de 10 000 imágenes de train (sin aumentos), cada 5 épocas:
  - desvío estándar medio por dimensión de los embeddings normalizados (colapso ⇒ → 0);
  - **rango efectivo** (exp de la entropía de los valores singulares normalizados) de los embeddings pooled y, aparte, de los tokens de parche;
  - **uniformidad** (Wang & Isola 2020): log E[exp(−2‖zᵢ − zⱼ‖²)] sobre embeddings normalizados; y **alineación** entre dos recortes de la misma imagen.
- k-NN rápido (k=20) sobre el split de validación interno cada 10 épocas (curva de aprendizaje barata).
- Throughput (img/s), tiempo por época, memoria pico, temperatura y clock de la GPU (`nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv -l 60` a un log).

### Al final de cada corrida (val oficial)
- **Probe lineal** top-1 (90 épocas, protocolo de `evaluation.py`).
- **k-NN** top-1 (k = 20, T = 0,07).
- Los indicadores de colapso finales.
- Horas de GPU totales.
- Opcional si sobra tiempo: probe lineal de segmentación en ADE20K (la tabla 7 también lo reporta: LeJEPA 11,76 ± 0,24 y SPHERE-JEPA 12,88 ± 0,39 de mIoU). Hay espacio en disco para bajarlo.

### Salida
- Un `results/<corrida>/evaluation.json` por corrida con config completa, commit del código, semilla, métricas y horas de GPU.
- Script de agregación → `results/tabla_grilla.csv` y `results/tabla_grilla.md` (media ± desvío por celda) y figuras:
  1. barras de la grilla 2 × 3 (lineal y k-NN);
  2. rango efectivo vs. época para las 6 celdas (muestra el colapso de C2);
  3. uniformidad vs. alineación al final (dónde cae cada método);
  4. tabla combinada con las filas de la tabla 7 + nuestras celdas.

---

## 8. Fases, tiempos y criterios de avance

> **Todos los tiempos son estimaciones a corregir con la medición de la fase 0.** La GPU es la versión laptop (más lenta que la de escritorio) y el throughput real depende de bf16, del data loading y de la temperatura.

### Fase 0 — Entorno, datos y medición (≈ 1 día)
1. Venv propio para el proyecto (copiando la versión de torch que ya funciona: 2.11.0+cu128, o la que pida el repo si anda en sm_120). Instalar `timm`, `lightning`, `datasets`, `lightly`, `kornia`, `scipy`, `pyyaml`. Fijar versiones en `requirements.txt` / `uv.lock`.
2. Clonar los dos repos oficiales en `third_party/` (SPHERE-JEPA e I-JEPA), anotar el commit.
3. Bajar y verificar ImageNet-100 (§4). Generar el split de validación interno (10 % de train, estratificado, semilla fija).
4. `--smoke` del repo SPHERE-JEPA.
5. **Medir throughput** (img/s, memoria) con 200 pasos de:
   - SPHERE-JEPA ViT-Tiny 6 vistas, fp32 y bf16;
   - I-JEPA ViT-Tiny 224, bf16, con y sin EMA;
   - I-JEPA ViT-Tiny 128 px (por si hay que bajar resolución).
6. Con esos números completar la tabla de tiempos de abajo y **decidir épocas** (ver regla de decisión).

**Criterio de salida:** datos verificados, smoke OK, tabla de tiempos real.

### Fase 1 — Calibración contra la tabla 7 (≈ 1–2 noches)
- Correr con el código oficial, sin modificar salvo paths: `imagenet100_tinyvit_lejepa.yaml` y `imagenet100_tinyvit_spherejepa.yaml`, semilla 0, 200 épocas.
- Evaluar con `evaluation.py` (CLS y avg-pool).
- **Criterio:** dentro de ~2 puntos de 70,71 / 67,39 y 71,91 / 69,03 (lineal / k-NN), y SPHERE-JEPA ≥ LeJEPA en k-NN.
- Si no da: revisar datos, fp32 vs. bf16, número de workers/augmentations. **No seguir a la grilla hasta resolverlo** o hasta documentar la diferencia.
- Si el tiempo por corrida supera ~15 h: correr una sola de las dos a 200 épocas y la otra a 100, y comparar sólo el orden.

### Fase 2 — Implementación I-JEPA y barridos cortos (≈ 1–2 días)
1. Implementar I-JEPA ViT-Tiny partiendo del repo oficial (máscaras, predictor, EMA) + los flags `--ema {on,off}` y `--reg {none,sigreg,susreg}` + `--lambda`.
2. **Tests de correctitud antes de entrenar:**
   - las máscaras reproducen las proporciones del paper (contar parches de contexto y objetivo en 1000 muestras);
   - con `--ema off` el encoder objetivo comparte pesos y no recibe gradiente (chequear `.grad is None` en la rama objetivo);
   - `SUSReg` da ~0 para vectores uniformes en la esfera y > 0 para una mezcla concentrada; `SIGReg` ~0 para N(0, I);
   - una corrida de 5 épocas de C2 muestra la varianza cayendo (si no cae, algo está mal).
3. **Barrido de lr** en C1: {2,5e-4; 5e-4; 1e-3}, 30 épocas, semilla 0, k-NN en val interno. Fijar el mejor para **toda** la grilla.
4. **Barrido de λ** en C4 y C6: {0,01; 0,05; 0,2}, 30 épocas, semilla 0. Fijar uno por regularizador.

### Fase 3 — Grilla principal (≈ 2–3 días de GPU)
- 5 celdas × 3 semillas = 15 corridas a E épocas (E según fase 0; objetivo **100**, ideal 200).
- **C2 (control):** 1 semilla, cortar en cuanto el rango efectivo caiga a ≈ 1 durante 3 evaluaciones seguidas (probablemente < 20 épocas). Registrar en qué época colapsa.
- Orden de corrida: primero semilla 0 de las 6 celdas (ya da una tabla completa), después semillas 1 y 2.
- Correr en tmux, una corrida por vez (la GPU es una), con un script de cola que lea una lista de configs y salte las que ya tienen `evaluation.json`.

### Fase 4 — Opcionales (§5.3) y análisis (≈ 1–2 días)
- Agregación, figuras, tabla combinada con la tabla 7.
- Variantes opcionales según tiempo restante.

### Tabla de tiempos (rellenar en fase 0)

| Corrida | img/s medido | min/época | épocas | horas/corrida | corridas | horas totales |
|---|---|---|---|---|---|---|
| Calibración LeJEPA (6 vistas, 224) | ? | ? | 200 | ? | 1 | ? |
| Calibración SPHERE-JEPA (6 vistas, 224) | ? | ? | 200 | ? | 1 | ? |
| Barridos I-JEPA (lr, λ) | ? | ? | 30 | ? | 9 | ? |
| Grilla I-JEPA (C1, C3–C6) | ? | ? | E | ? | 15 | ? |
| Control C2 | ? | ? | ≤ 20 | ? | 1 | ? |

Estimación a priori, sólo para dimensionar (no usar como dato): I-JEPA ViT-Tiny a 224 procesa una vista por imagen y el encoder de contexto ve ~25–45 % de los parches, así que cada época debería costar del orden de un minuto o dos en esta GPU; las corridas de calibración, con 6 vistas, varias veces más por época. Con 100 épocas, la grilla completa entraría en 1–3 días de GPU.

**Regla de decisión de épocas:** elegir el E más grande tal que la grilla principal (15 corridas) entre en **≤ 60 h de GPU**. Si ni con E = 100 entra, bajar a 128 px (64 parches, ~3× más barato) **para toda la grilla**, y en ese caso la calibración de la fase 1 también se repite a 128 px para mantener la comparación interna (el ancla de la tabla 7 queda sólo como referencia cualitativa).

---

## 9. Riesgos y qué hacer

| Riesgo | Señal | Acción |
|---|---|---|
| Disco lleno | < 10 GB libres | Parar, avisar al usuario; no borrar nada fuera de esta carpeta. |
| La calibración no reproduce la tabla 7 | > 2 puntos abajo | Revisar datos/precisión/augmentations; documentar; no seguir sin entender la diferencia. |
| C2 no colapsa | rango efectivo alto tras 20 épocas | Bug probable en el modo sin EMA (¿el objetivo recibe gradiente o usa otros pesos?). |
| C6 colapsa | rango efectivo → 1 | Subir λ; si persiste, es un resultado (el regularizador no alcanza con máscaras) y se reporta. |
| Regularizador domina y la predicción no aprende | `L_pred` no baja | Bajar λ; revisar escala de `L_reg` (el factor B del repo). |
| I-JEPA ViT-Tiny aprende poco en 100 épocas | k-NN muy bajo en todas las celdas | Es esperable que I-JEPA necesite más épocas que los métodos con vistas; priorizar 200 épocas en C1 y la mejor celda antes que más variantes. |
| Thermal throttling de la laptop | clock SM cae, img/s varía > 15 % | Registrar; comparar costo en horas sólo entre corridas con condiciones similares. |
| GPU compartida | otro proceso en `nvidia-smi` | Esperar o coordinar; no matar procesos ajenos. |

---

## 10. Estructura de carpetas propuesta

```
~/dlviz/ijepa-sphere/
  PLAN.md                 este documento
  papers/                 los tres PDFs de referencia
  third_party/            repos oficiales clonados (SPHEREJEPA, ijepa) con el commit anotado
  data/imagenet100/       dataset (caché de HF o ImageFolder), manifest.json, val_interno.txt
  src/                    código propio (I-JEPA + flags de EMA y regularizador, métricas de colapso)
  configs/                un YAML por celda y por barrido
  scripts/                descarga y verificación de datos, cola de corridas, agregación, figuras
  runs/<nombre>_seed<k>/  logs, checkpoints (last, best), evaluation.json
  results/                tablas agregadas y figuras para la slide
```

---

## 11. Entregable para la charla

Una slide «Experimento propio: ¿I-JEPA sin promedio móvil?»:
- izquierda: diagrama de I-JEPA con la flecha del EMA tachada y una caja «regularizador: embeddings uniformes en la esfera»;
- derecha: barras de la grilla 2 × 3 (lineal y k-NN, media ± desvío, 3 semillas), C1 resaltada como referencia y C2 marcada «colapsa»;
- abajo: una frase de conclusión según el resultado (ver §3);
- en las notas: escala (ViT-Tiny, ImageNet-100, 224 px, E épocas), horas de GPU, y la fila de calibración contra la tabla 7.

Duración en la charla ≈ 1:30.
