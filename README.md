# Selector criptográfico adaptativo (IA para criptografía)

**Sistema de selección criptográfica adaptativa mediante inteligencia de enjambre y decisión multicriterio.**

Prototipo académico del proyecto de investigación *«Criptografía e Inteligencia Artificial»*, en la línea **inteligencia computacional para fortalecer la criptografía**.

La aplicación compara las siete S-boxes publicadas en Khan (2025). Calcula sus propiedades criptográficas y recomienda la más adecuada según las prioridades que defina el usuario. Usa **Particle Swarm Optimization (PSO)** para medir cuánto depende esa recomendación de la elección de prioridades.

> Las métricas describen la **S-box aislada**, no la seguridad de un cifrador completo. Esto no es un algoritmo de cifrado nuevo.

## Antecedente científico

Khan, M. (2025). *Intelligent decision-making framework for selecting optimized substitution boxes in modern block cipher design*. **Scientific Reports, 15**, 41261. https://doi.org/10.1038/s41598-025-25173-3 (acceso abierto: [PMC12638994](https://pmc.ncbi.nlm.nih.gov/articles/PMC12638994/)).

El artículo ordena siete S-boxes (generadas con CSO, PSO, ACO, TLBO, ABC, GA y Firefly) con un método EDAS y pesos de entropía. El origen de cada dato, la validación de las métricas y las inconsistencias encontradas están en [docs/referencia_cientifica.md](docs/referencia_cientifica.md).

## Qué agrega este prototipo respecto del artículo

| Artículo | Prototipo |
|---|---|
| Usa las métricas tal como las reportan los trabajos originales. | **Recalcula** todas las métricas a partir de los 256 valores. Las valida contra AES y las compara con lo publicado, con lo que detecta erratas (TLBO, ABC, FA). |
| Pesos fijos (entropía). | **Pesos definidos por el usuario** con controles deslizantes: la selección se adapta a la prioridad del diseñador. |
| Un único ranking. | Ranking con **aporte de cada criterio** y una explicación en lenguaje llano. |
| — | **Dominancia de Pareto**: identifica qué S-boxes no pueden ganar con ningún peso. |
| — | **Robustez con PSO**: cambio mínimo de prioridades necesario para que otra S-box pase a ser la recomendada. |
| — | **Reproducibilidad**: semillas, parámetros y registro de cada ejecución. Script que regenera todos los resultados. |
| SAC tratado como «mayor es mejor». | SAC y BIC-SAC medidos como **desvío del ideal 0.5**. |

### El papel de PSO (y por qué no se usa para «elegir» la S-box)

Con 7 alternativas, la **búsqueda exhaustiva** (evaluar las 7 y quedarse con la mejor) es exacta y requiere 7 evaluaciones. Es el **método de referencia**. Para demostrarlo, la app incluye también un PSO que elige la S-box directamente: en 30 semillas acierta 28/30 veces y necesita 210 evaluaciones. Para este problema no aporta.

PSO se usa donde la enumeración no es posible: en el **espacio continuo de pesos**. Para cada S-box rival, PSO busca los pesos más parecidos a los elegidos con los que ese rival quedaría primero:

```
minimizar   d(w', w)  +  λ · max(0, max_j S_j(w') − S_rival(w') + ε)
con         w' = x / Σx,  x ∈ [0,1]^6        d = variación total = ½·Σ|w' − w|
```

El resultado se interpreta así: «habría que reasignar X % de la importancia para cambiar la recomendación». Es una medida de **cuán robusta es la decisión**. Se contrasta con una búsqueda aleatoria Monte Carlo de 50 000 muestras.

## Instalación

Requiere Python ≥ 3.11.

```bash
git clone <url-del-repositorio>
cd selector-criptografico-ia
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py          # interfaz web en http://localhost:8501
python experimentos.py        # regenera docs/resultados_experimentos.md
pytest                        # 61 pruebas automatizadas
```

Los gráficos se hacen con Altair, que ya viene incluido con Streamlit. Así no hace falta sumar Matplotlib o Plotly y los gráficos tienen tooltips interactivos.

### Demostración sugerida (5 minutos)

1. **Prueba A**: botón *Prueba A* (no linealidad 50 %, el resto 10 %) y luego **▶ Ejecutar selección**.
2. **Prueba B**: botón *Prueba B* (avalancha 35 %, resistencia diferencial 35 %, el resto 7.5 %) y luego **▶ Ejecutar selección**.
3. Pestaña **Comparar ejecuciones**: pesos, puntajes y robustez lado a lado.
4. **Prueba C (control)**: *Prueba C* (sólo avalancha) para ver un caso en que la recomendación **sí cambia**.

### Resultados obtenidos (semilla 42; detalle en [docs/resultados_experimentos.md](docs/resultados_experimentos.md))

| Configuración | Recomendada | 2.ª | Margen | Cambio mínimo de pesos para que gane el rival (PSO) |
|---|---|---|---|---|
| A · no linealidad | **PSO** | ABC | 0.360 | 53.0 % (rival ABC) |
| B · avalancha + diferencial | **PSO** | ABC | 0.227 | 31.1 % (rival ABC) |
| C · sólo avalancha (control) | **ABC** | ANO | 0.146 | 17.3 % (rival PSO; ANO está dominada) |
| Pesos iguales | **PSO** | ABC | 0.406 | 50.0 % (rival ABC) |

**Lectura de los resultados:**

- Entre A y B **la recomendación no cambia**. No se forzó un resultado distinto. La S-box PSO es la mejor del conjunto en 5 de los 6 criterios calculados (NL, BIC-NL, BIC-SAC, DU y LP).
- Lo que cambia es la **robustez**. Con la prioridad de B, a ABC le bastaría una reasignación del 31 % de la importancia para pasar al primer lugar. Con A necesitaría el 53 %. Según PSO, ABC gana solo si el efecto avalancha pesa alrededor de dos tercios del total.
- 5 de las 7 S-boxes (CSO, ANO, TLBO, GA, FA) están **dominadas en sentido de Pareto**: ningún conjunto de pesos las hace ganar. Con 100 000 pesos al azar, PSO gana el 99.9 % de las veces y ABC el 0.1 %.
- PSO frente a Monte Carlo: en 20 corridas (4 configuraciones × 5 semillas), PSO igualó o mejoró a Monte Carlo en 14, con 3 030 evaluaciones contra 50 000. No es superior en todos los casos: en la Prueba C, Monte Carlo fue mejor en 4 de 5 semillas.

## Estructura

```
selector-criptografico-ia/
├── app.py                     # interfaz Streamlit
├── experimentos.py            # reproduce las pruebas A/B/C y escribe docs/resultados_experimentos.md
├── src/
│   ├── sboxes.py              # carga de datos del artículo
│   ├── metricas.py            # NL (Walsh), SAC, BIC-NL, BIC-SAC, DDT/DU, LAT/LP
│   ├── pso.py                 # PSO genérico y reproducible
│   └── selector.py            # criterios, normalización, búsqueda exhaustiva, Pareto, robustez con PSO, EDAS
├── data/
│   ├── sboxes_articulo.json   # 7 S-boxes (Tablas 1–7), con metadatos de la fuente
│   └── metricas_reportadas.csv# valores publicados en el artículo (Tablas 8–9, 16–18)
├── tests/                     # pytest: métricas (AES), datos, PSO, selector, EDAS, interfaz
└── docs/
    ├── referencia_cientifica.md
    └── resultados_experimentos.md
```

## Criterios

| Criterio | Métrica calculada | Mejor |
|---|---|---|
| No linealidad | NL promedio de las 8 funciones coordenadas (espectro de Walsh-Hadamard) | mayor |
| Efecto avalancha (SAC) | promedio de \|SAC − 0.5\| en la matriz 8×8 | menor |
| Independencia de bits (BIC-NL) | NL promedio de f_j ⊕ f_k (28 pares) | mayor |
| Independencia de bits (BIC-SAC) | promedio de \|SAC(f_j ⊕ f_k) − 0.5\| | menor |
| Resistencia diferencial | uniformidad diferencial (máximo de la DDT, a ≠ 0) | menor |
| Resistencia lineal | LP = max \|#{a·x = b·S(x)} − 128\| / 256 | menor |

## Limitaciones

- **Alcance criptográfico.** Las métricas son propiedades de la S-box aislada. No miden la seguridad de un cifrador ni la resistencia a ataques algebraicos, de canal lateral o de implementación.
- **Conjunto pequeño y fijo.** Son 7 alternativas. La normalización min-max es relativa a ese conjunto: si se agrega una S-box, cambian todos los puntajes.
- **Modelo de agregación lineal (SAW).** Supone criterios compensables entre sí. Otros métodos (EDAS, TOPSIS, AHP) podrían ordenar distinto los puestos intermedios.
- **PSO es estocástico.** No garantiza el óptimo global: en algunas semillas Monte Carlo encontró una distancia menor. Por eso se registran las semillas y se recomienda comparar varias.
- **El problema de robustez tiene solución exacta** (programación cuadrática o lineal sobre el simplex). PSO se eligió porque es el objetivo del proyecto y porque admite cambiar el modelo de puntuación sin reformular el problema. Un solver exacto sería una buena validación adicional.
- **Datos del artículo.** Hay discrepancias entre las S-boxes publicadas y las métricas que el mismo artículo reporta (TLBO, ABC, FA). Además, sus Tablas 15–16 no se pueden reproducir. Ver [docs/referencia_cientifica.md](docs/referencia_cientifica.md).
- **Los pesos son subjetivos.** La herramienta hace visible esa subjetividad, pero no la elimina.

## Licencia y datos

Código: uso académico. Las S-boxes y los valores reportados provienen de Khan (2025), publicado bajo CC BY-NC-ND 4.0, y se citan con fines de investigación.
