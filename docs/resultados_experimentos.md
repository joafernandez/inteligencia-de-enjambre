# Resultados de los experimentos

Generado por `python experimentos.py` el 2026-11-22 (Python 3.12.10, NumPy 2.5.5, pandas 3.1.1).

Configuración PSO: `{'n_particulas': 30, 'iteraciones': 100, 'inercia': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'vmax_frac': 0.25, 'semilla': 42}`. Semillas para la robustez: [1, 2, 3, 42, 123]. Monte Carlo: 50 000 muestras, semilla = semilla de PSO.

## 1. Matriz de decisión (calculada por el prototipo)

| | nl | sac | bic_nl | bic_sac | dp | lp |
|---|---|---|---|---|---|---|
| CSO | 109.2500 | 0.0291 | 102.9286 | 0.0331 | 12.0000 | 0.1406 |
| PSO | 111.5000 | 0.0286 | 110.2857 | 0.0279 | 6.0000 | 0.1094 |
| ANO | 107.0000 | 0.0283 | 104.2143 | 0.0355 | 10.0000 | 0.1484 |
| TLBO | 103.7500 | 0.0310 | 104.3571 | 0.0340 | 10.0000 | 0.1328 |
| ABC | 109.2500 | 0.0269 | 104.2857 | 0.0348 | 8.0000 | 0.1250 |
| GA | 108.0000 | 0.0322 | 103.3571 | 0.0369 | 10.0000 | 0.1406 |
| FA | 107.5000 | 0.0369 | 104.3571 | 0.0342 | 10.0000 | 0.1328 |


Criterios: nl = NL promedio (↑), sac = desvío SAC de 0.5 (↓), bic_nl = BIC-NL promedio (↑), bic_sac = desvío BIC-SAC de 0.5 (↓), dp = uniformidad diferencial (↓), lp = probabilidad lineal (↓).

## 2. Matriz normalizada (1 = mejor, 0 = peor)

| | nl | sac | bic_nl | bic_sac | dp | lp |
|---|---|---|---|---|---|---|
| CSO | 0.710 | 0.780 | 0.000 | 0.419 | 0.000 | 0.200 |
| PSO | 1.000 | 0.829 | 1.000 | 1.000 | 1.000 | 1.000 |
| ANO | 0.419 | 0.854 | 0.175 | 0.155 | 0.333 | 0.000 |
| TLBO | 0.000 | 0.585 | 0.194 | 0.326 | 0.333 | 0.400 |
| ABC | 0.710 | 1.000 | 0.184 | 0.233 | 0.667 | 0.600 |
| GA | 0.548 | 0.463 | 0.058 | 0.000 | 0.333 | 0.200 |
| FA | 0.484 | 0.000 | 0.194 | 0.302 | 0.333 | 0.400 |


**Dominancia de Pareto:** CSO dominada por PSO; ANO dominada por ABC; TLBO dominada por PSO; GA dominada por PSO, ABC; FA dominada por PSO.

**Frecuencia de victoria con 100 000 pesos uniformes al azar (semilla 0):** CSO 0.00 %, PSO 99.90 %, ANO 0.00 %, TLBO 0.00 %, ABC 0.10 %, GA 0.00 %, FA 0.00 %.

## Prueba A — prioridad en no linealidad

Pesos: `{'nl': 0.5, 'sac': 0.1, 'bic_nl': 0.1, 'bic_sac': 0.1, 'dp': 0.1, 'lp': 0.1}`

**Recomendada: PSO** (puntaje 0.9829); segunda: ABC (margen 0.3597).

| | puntaje | posicion |
|---|---|---|
| PSO | 0.9829 | 1 |
| ABC | 0.6232 | 2 |
| CSO | 0.4947 | 3 |
| GA | 0.3797 | 4 |
| FA | 0.3649 | 5 |
| ANO | 0.3614 | 6 |
| TLBO | 0.1838 | 7 |



Cambio mínimo de pesos (variación total) para que **ABC** pase a ganar:

| | PSO dist | PSO evals | MC dist | MC evals | pesos PSO |
|---|---|---|---|---|---|
| 1 | 0.5299 | 3030 | 0.5338 | 50000 | nl=0.370, sac=0.630 |
| 2 | 0.5299 | 3030 | 0.5325 | 50000 | nl=0.370, sac=0.630 |
| 3 | 0.5299 | 3030 | 0.5308 | 50000 | nl=0.370, sac=0.630 |
| 42 | 0.5299 | 3030 | 0.5395 | 50000 | nl=0.370, sac=0.630 |
| 123 | 0.5299 | 3030 | 0.5387 | 50000 | nl=0.370, sac=0.630 |


PSO: media 0.5299, mín 0.5299, máx 0.5299. Monte Carlo: media 0.5350, mín 0.5308. PSO ≤ Monte Carlo en 5/5 semillas, con 3030 evaluaciones frente a 50000.

PSO aplicado directamente a elegir la S-box (10×20, semillas 0–29): acertó 28/30 veces con 210 evaluaciones por corrida; la búsqueda exhaustiva acierta siempre con 7.

## Prueba B — prioridad en avalancha y resistencia diferencial

Pesos: `{'nl': 0.075, 'sac': 0.35, 'bic_nl': 0.075, 'bic_sac': 0.075, 'dp': 0.35, 'lp': 0.075}`

**Recomendada: PSO** (puntaje 0.9402); segunda: ABC (margen 0.2274).

| | puntaje | posicion |
|---|---|---|
| PSO | 0.9402 | 1 |
| ABC | 0.7128 | 2 |
| ANO | 0.4716 | 3 |
| TLBO | 0.3905 | 4 |
| CSO | 0.3728 | 5 |
| GA | 0.3394 | 6 |
| FA | 0.2202 | 7 |



Cambio mínimo de pesos (variación total) para que **ABC** pase a ganar:

| | PSO dist | PSO evals | MC dist | MC evals | pesos PSO |
|---|---|---|---|---|---|
| 1 | 0.3051 | 3030 | 0.3086 | 50000 | nl=0.075, sac=0.655, dp=0.270 |
| 2 | 0.3115 | 3030 | 0.3115 | 50000 | sac=0.661, dp=0.339 |
| 3 | 0.3115 | 3030 | 0.3105 | 50000 | sac=0.661, dp=0.339 |
| 42 | 0.3115 | 3030 | 0.3140 | 50000 | sac=0.661, dp=0.339 |
| 123 | 0.3053 | 3030 | 0.3152 | 50000 | nl=0.075, sac=0.655, dp=0.270 |


PSO: media 0.3090, mín 0.3051, máx 0.3115. Monte Carlo: media 0.3120, mín 0.3086. PSO ≤ Monte Carlo en 4/5 semillas, con 3030 evaluaciones frente a 50000.

PSO aplicado directamente a elegir la S-box (10×20, semillas 0–29): acertó 28/30 veces con 210 evaluaciones por corrida; la búsqueda exhaustiva acierta siempre con 7.

## Prueba C (control) — sólo avalancha

Pesos: `{'nl': 0.0, 'sac': 1.0, 'bic_nl': 0.0, 'bic_sac': 0.0, 'dp': 0.0, 'lp': 0.0}`

**Recomendada: ABC** (puntaje 1.0000); segunda: ANO (margen 0.1463).

| | puntaje | posicion |
|---|---|---|
| ABC | 1.0000 | 1 |
| ANO | 0.8537 | 2 |
| PSO | 0.8293 | 3 |
| CSO | 0.7805 | 4 |
| TLBO | 0.5854 | 5 |
| GA | 0.4634 | 6 |
| FA | 0.0000 | 7 |


La segunda (ANO) está dominada por ABC: no puede ganar con ningún peso. El rival alcanzable mejor ubicado es **PSO**.


Cambio mínimo de pesos (variación total) para que **PSO** pase a ganar:

| | PSO dist | PSO evals | MC dist | MC evals | pesos PSO |
|---|---|---|---|---|---|
| 1 | 0.1821 | 3030 | 0.1750 | 50000 | sac=0.818, bic_sac=0.182 |
| 2 | 0.1819 | 3030 | 0.1749 | 50000 | sac=0.818, bic_nl=0.003, bic_sac=0.179 |
| 3 | 0.1821 | 3030 | 0.1735 | 50000 | sac=0.818, bic_sac=0.182 |
| 42 | 0.1732 | 3030 | 0.1743 | 50000 | sac=0.827, bic_nl=0.173 |
| 123 | 0.1820 | 3030 | 0.1736 | 50000 | sac=0.818, bic_nl=0.003, bic_sac=0.179 |


PSO: media 0.1803, mín 0.1732, máx 0.1821. Monte Carlo: media 0.1742, mín 0.1735. PSO ≤ Monte Carlo en 1/5 semillas, con 3030 evaluaciones frente a 50000.

PSO aplicado directamente a elegir la S-box (10×20, semillas 0–29): acertó 30/30 veces con 210 evaluaciones por corrida; la búsqueda exhaustiva acierta siempre con 7.

## Referencia — pesos iguales

Pesos: `{'nl': 0.166667, 'sac': 0.166667, 'bic_nl': 0.166667, 'bic_sac': 0.166667, 'dp': 0.166667, 'lp': 0.166667}`

**Recomendada: PSO** (puntaje 0.9715); segunda: ABC (margen 0.4060).

| | puntaje | posicion |
|---|---|---|
| PSO | 0.9715 | 1 |
| ABC | 0.5656 | 2 |
| CSO | 0.3515 | 3 |
| ANO | 0.3227 | 4 |
| TLBO | 0.3064 | 5 |
| FA | 0.2856 | 6 |
| GA | 0.2672 | 7 |



Cambio mínimo de pesos (variación total) para que **ABC** pase a ganar:

| | PSO dist | PSO evals | MC dist | MC evals | pesos PSO |
|---|---|---|---|---|---|
| 1 | 0.4831 | 3030 | 0.5199 | 50000 | nl=0.166, sac=0.650, dp=0.166, lp=0.018 |
| 2 | 0.5023 | 3030 | 0.5186 | 50000 | nl=0.167, sac=0.669, lp=0.164 |
| 3 | 0.4829 | 3030 | 0.5316 | 50000 | nl=0.167, sac=0.649, dp=0.167, lp=0.017 |
| 42 | 0.5000 | 3030 | 0.4866 | 50000 | nl=0.169, sac=0.661, dp=0.170 |
| 123 | 0.4831 | 3030 | 0.5132 | 50000 | nl=0.167, sac=0.650, dp=0.164, lp=0.019 |


PSO: media 0.4903, mín 0.4829, máx 0.5023. Monte Carlo: media 0.5140, mín 0.4866. PSO ≤ Monte Carlo en 4/5 semillas, con 3030 evaluaciones frente a 50000.

PSO aplicado directamente a elegir la S-box (10×20, semillas 0–29): acertó 28/30 veces con 210 evaluaciones por corrida; la búsqueda exhaustiva acierta siempre con 7.

## Reproducción del EDAS del artículo (Tablas 10–16)

| | AS reproducido | AS artículo | SP reproducido | SN reproducido |
|---|---|---|---|---|
| CSO | 0.9705 | 0.6083 | 0.0929 | 0.0003 |
| PSO | 1.0000 | 1.0000 | 0.0986 | 0.0000 |
| ANO | 0.9634 | 0.6386 | 0.0914 | 0.0002 |
| TLBO | 0.9341 | 0.6952 | 0.0857 | 0.0001 |
| ABC | 0.9356 | 0.8015 | 0.0859 | 0.0000 |
| GA | 0.9910 | 0.6609 | 0.0969 | 0.0002 |
| FA | 0.0000 | 0.0000 | 0.0000 | 0.5506 |


AV, PDA y NDA coinciden con las Tablas 12–14. SP, SN y AS no coinciden con las Tablas 15–16; ver `docs/referencia_cientifica.md`.
