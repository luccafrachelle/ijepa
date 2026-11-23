# Tabla de tiempos (fase 0, medida)

Horas/corrida = entrenamiento (img/s medido) + estimación de overhead de evaluación (colapso cada 5 ép. x 20 s, k-NN rápido cada 10 ép. x 40 s, evaluación final 600 s; estimado, no medido). Throughput: bs 256, 200 pasos (20 de warmup excluidos), SDPA, GPU sola.

| Corrida | img/s medido | min/época | épocas | horas/corrida | corridas | horas totales |
|---|---|---|---|---|---|---|
| Calibración LeJEPA (6 vistas, 224) | pendiente: requiere permiso para ejecutar third_party | | 200 | | 1 | |
| Calibración SPHERE-JEPA (6 vistas, 224) | pendiente: requiere permiso para ejecutar third_party | | 200 | | 1 | |
| Barrido lr (C1, split interno 114020) | 1442 | 1.32 | 30 | 0.89 | 3 | 2.7 |
| Barrido λ (C4/C6, sin EMA + reg; medido con SUSReg) | 1362 | 1.39 | 30 | 0.93 | 6 | 5.6 |
| **Barridos I-JEPA (lr, λ) total** | | | 30 | | 9 | **8.3** |
| Grilla I-JEPA (C1 con img/s=1442; C3–C6 con img/s=1362, conservador) | 1362 | 1.55 | 100 | 2.97 (C1: 2.83) | 15 | 44.2 |
| Grilla I-JEPA (C1 con img/s=1442; C3–C6 con img/s=1362, conservador) | 1362 | 1.55 | 150 | 4.37 (C1: 4.16) | 15 | 65.0 |
| Grilla I-JEPA (C1 con img/s=1442; C3–C6 con img/s=1362, conservador) | 1362 | 1.55 | 200 | 5.78 (C1: 5.49) | 15 | 85.8 |
| Grilla I-JEPA (C1 con img/s=1442; C3–C6 con img/s=1362, conservador) | 1362 | 1.55 | 300 | 8.58 (C1: 8.16) | 15 | 127.5 |
| Control C2 (sin EMA, sin reg; early-stop) | 1439 | 1.47 | ≤ 20 | ≤ 0.70 | 1 | ≤ 0.7 |

Minutos/época: split interno N=114020, entrenamiento completo N=126689.
Las filas de grilla repiten una fila por cada E candidato. Filas de calibración: pendiente: requiere permiso para ejecutar third_party.

## Otras mediciones

| Caso | img/s | memoria pico (GB) | min/época (N completo) |
|---|---|---|---|
| I-JEPA 224 bf16, EMA on (C1) | 1442 | 5.60 | 1.46 |
| I-JEPA 224 bf16, EMA off (C2) | 1439 | 5.64 | 1.47 |
| I-JEPA 224 bf16, EMA off + SUSReg (C6) | 1362 | 6.41 | 1.55 |
| I-JEPA 224 fp32, EMA on | 923 | 9.39 | 2.29 |
| I-JEPA 128 px bf16, EMA on (min_keep=4; official min_keep=10 hangs) | 4584 | 1.70 | 0.46 |

128 px (C1 como proxy): grilla de 15 corridas a E=100 = 17.3 h.

## Regla de decisión de épocas

Mayor E en [100, 150, 200, 300] con la grilla de 15 corridas <= 60 h de GPU (incluye overhead de evaluación estimado):

| E | horas grilla (15 corridas) | entra |
|---|---|---|
| 100 | 44.2 | sí |
| 150 | 65.0 | no |
| 200 | 85.8 | no |
| 300 | 127.5 | no |

**Recomendación: E = 100** (44.2 h <= 60 h).
