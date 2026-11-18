# Barridos fase 2 (30 épocas, semilla 0, k-NN en val_interno)

## lr sweep (C1)

| lr | k-NN val_interno (%) | estado | corrida |
|---|---|---|---|
| 0.00025 | 29.21 **(mejor)** | ok | sweep_lr2.5e-4_seed0 |
| 0.0005 | 28.68 | ok | sweep_lr5e-4_seed0 |
| 0.001 | 26.06 | ok | sweep_lr1e-3_seed0 |

Elegido: lr = 0.00025 (k-NN 29.21%).

## λ sweep (C4)

| λ | k-NN val_interno (%) | estado | corrida |
|---|---|---|---|
| 0.01 | 15.98 | ok | sweep_lambda_c4_0.01_seed0 |
| 0.05 | 17.83 **(mejor)** | ok | sweep_lambda_c4_0.05_seed0 |
| 0.2 | 10.69 | ok | sweep_lambda_c4_0.2_seed0 |

Elegido: λ = 0.05 (k-NN 17.83%).

## λ sweep (C6)

| λ | k-NN val_interno (%) | estado | corrida |
|---|---|---|---|
| 0.01 | 16.58 | ok | sweep_lambda_c6_0.01_seed0 |
| 0.05 | 18.44 **(mejor)** | ok | sweep_lambda_c6_0.05_seed0 |
| 0.2 | 7.38 | ok | sweep_lambda_c6_0.2_seed0 |

Elegido: λ = 0.05 (k-NN 18.44%).

## Para la cola de la grilla

`--var LR=0.00025 --var LAMBDA_SIGREG=0.05 --var LAMBDA_SUSREG=0.05` (obligatorio: la cola no tiene valores por defecto)
