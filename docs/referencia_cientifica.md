# Referencia científica y trazabilidad de los datos

## Artículo de referencia

Khan, M. (2025). *Intelligent decision-making framework for selecting optimized substitution boxes in modern block cipher design*. **Scientific Reports, 15**, 41261. https://doi.org/10.1038/s41598-025-25173-3

- Versión consultada: PubMed Central, PMCID PMC12638994 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12638994/
- Licencia: CC BY-NC-ND 4.0.
- Fecha de extracción: 2026-11-22.

### Qué hace el artículo

Compara siete S-boxes de 8×8 bits generadas en trabajos previos con distintas metaheurísticas. Las evalúa con seis criterios (NL, BIC-SAC, BIC-NL, SAC, DP, LP) y las ordena con un método híbrido **EDAS + pesos de entropía de Shannon**. También contrasta ese orden con TOPSIS y SAW. Su orden final es PSO > ABC > TLBO > GA > ANO > CSO > FA.

| ID | Técnica de generación | Tabla del artículo | Ref. en el artículo |
|---|---|---|---|
| CSO | Cuckoo Search | 1 | [7] |
| PSO | Particle Swarm Optimization | 2 | [9] |
| ANO | Ant Colony Optimization (el artículo la rotula «ANO») | 3 | [10] |
| TLBO | Teaching-Learning-Based Optimization | 4 | [11] |
| ABC | Artificial Bee Colony | 5 | [13] |
| GA | Genetic Algorithm | 6 | [15] |
| FA | Firefly Algorithm | 7 | [16] |

> La S-box «PSO» fue generada con PSO por sus autores originales. **No tiene relación con el PSO de este prototipo**, que no genera S-boxes: analiza los pesos de decisión.

## Origen de cada dato

| Dato | Origen | Archivo |
|---|---|---|
| 7 S-boxes (256 valores c/u) | **Artículo**, Tablas 1–7. Extraídas automáticamente de las celdas HTML de PMC, sin transcripción manual. | `data/sboxes_articulo.json` |
| NL, SAC, BIC-NL, BIC-SAC, DP, LP reportados | **Artículo**, Tablas 8–9 (transcriptos sin corregir erratas) | `data/metricas_reportadas.csv` |
| Puntajes EDAS / TOPSIS / SAW y rankings | **Artículo**, Tablas 16–18 | `data/metricas_reportadas.csv` |
| Pesos de entropía | **Artículo**, Tabla 11 | `src/selector.py` (`ART_TABLA11_PESOS`) |
| Todas las métricas usadas para decidir | **Calculadas por el prototipo** a partir de los 256 valores | `src/metricas.py` |
| Normalización, puntajes, robustez, PSO | **Calculados por el prototipo** | `src/selector.py`, `src/pso.py` |

Controles de integridad (en `tests/test_sboxes.py`):

- Cada tabla tiene exactamente 256 celdas hexadecimales.
- Cada S-box es una permutación de 0–255 (biyectiva).
- Los enteros coinciden con los hexadecimales publicados.

Las filas de las Tablas 9–18 tienen el rótulo como imagen. Se asume el mismo orden que la Tabla 8, y eso está verificado: la columna NL de la Tabla 10 coincide fila por fila con el NL promedio de la Tabla 8.

## Validación de las métricas implementadas

Primero se validaron con la S-box de AES, generada algorítmicamente en los tests (no copiada). Resultados: NL = 112 en las 8 coordenadas, DU = 4, LP = 16/256, SAC promedio = 0.504883 y BIC-NL mínimo = 112.

Después se compararon con los valores del artículo:

| Métrica | Coincide con el artículo | Diferencias |
|---|---|---|
| Uniformidad diferencial (DU) | 7/7 | — |
| BIC-NL mínimo | 7/7 | — |
| BIC-NL promedio | 7/7 (el artículo redondea a entero) | — |
| NL mín/máx/prom | 5/7 | **TLBO**: calc. 100/106/103.75, art. 104/110/106.5. **FA**: prom. calc. 107.5, art. 107. |
| LP | 5/7 | **TLBO**: calc. 34/256 = 0.1328, art. 0.1172. **ABC**: calc. 32/256 = 0.125, art. 0.1172. |
| SAC promedio | 5/7 (±2·10⁻⁴) | **TLBO**: 0.4973 vs 0.4995. **FA**: 0.4944 vs 0.42892. |
| BIC-SAC promedio | 5/7 | **TLBO**: 0.4992 vs 0.4983. **FA**: 0.4978 vs **0.01276** (errata evidente: el mismo artículo da mín 0.46484 y máx 0.525391). |
| BIC-SAC mín | 7/7 | Se usa la convención del artículo: mín/máx sobre el promedio de cada par de bits de salida. |

**Interpretación.** Las diferencias se concentran en TLBO y FA. Hay dos explicaciones posibles: la Tabla 4 del artículo contiene erratas de transcripción, aunque la S-box sigue siendo biyectiva, o el artículo calculó esas métricas sobre otra versión de la S-box. Con los datos disponibles no se puede distinguir entre ambas. El prototipo **no corrige** los datos publicados: usa sus propias métricas calculadas y muestra las diferencias.

## Reproducción del método EDAS del artículo

Partiendo de la Tabla 10 (matriz de decisión) y la Tabla 11 (pesos), `src/selector.py::edas` reproduce:

- el vector promedio AV (Tabla 12): coincide exactamente;
- las matrices PDA y NDA (Tablas 13 y 14): coinciden exactamente;
- el primer y el último puesto: PSO y FA.

**No** se reproducen SP, SN, NSP y NSN (Tabla 15) ni los puntajes AS (Tabla 16). Ejemplos:

- SP(PSO) con los pesos de la Tabla 11 = 0.0986; el artículo informa 0.2306.
- Con los propios valores NSP y NSN de la Tabla 15 para CSO, (0.4063 + 0.1897)/2 = 0.298, pero la Tabla 16 informa 0.6083.

Los pesos de entropía de la Tabla 11 tampoco salen de la fórmula de entropía estándar aplicada a la Tabla 10. Por ejemplo, BIC-SAC da una entropía ≈ 0.93; el artículo informa 0.1294. Esto se documenta como una **inconsistencia interna del artículo**. No invalida que PSO y FA ocupen los extremos, pero sí los puestos intermedios.

## Diferencias metodológicas del prototipo respecto del artículo

1. **SAC y BIC-SAC se tratan como desvío respecto de 0.5 (menor = mejor).** El artículo los trata como criterios de beneficio («mayor es mejor»), pero el ideal criptográfico es 0.5, no el máximo.
2. **Normalización min-max orientada.** En cada criterio, 1 = mejor y 0 = peor del conjunto. Los puntajes se combinan por suma ponderada (SAW).
3. **Los pesos los define el usuario.** En el artículo son pesos objetivos de entropía. La entropía premia el criterio con más dispersión entre alternativas, no el más importante para la seguridad. En el artículo, la errata de FA hace que BIC-SAC y SAC concentren el 99 % del peso.
4. **Se añade dominancia de Pareto y análisis de robustez con PSO** (ver README).

## Alcance

- Las métricas evalúan la **S-box aislada**. Una buena S-box es necesaria, pero no suficiente, para un cifrador seguro. Falta evaluar la capa de difusión, el key schedule, el número de rondas, los ataques algebraicos y de canal lateral, y la implementación.
- Este prototipo **no** es un algoritmo de cifrado ni genera S-boxes nuevas.
- PSO no aumenta la seguridad de ninguna S-box. Solo se usa para analizar cómo depende la decisión de los pesos elegidos.
