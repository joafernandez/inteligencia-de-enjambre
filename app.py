"""Interfaz Streamlit: selección adaptativa de S-boxes con decisión multicriterio y PSO.

Ejecutar con:  streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import selector as S
from src.pso import ConfigPSO
from src.sboxes import cargar_metricas_reportadas, cargar_sboxes, fuente

st.set_page_config(page_title="Selector criptográfico IA", page_icon="🔐", layout="wide")

# Paleta categórica validada (slots fijos por criterio; el color sigue al criterio, no al rango)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
COLOR_CRIT = dict(zip([S.NOMBRES[k] for k in S.CLAVES], COLORES))
RESULTADOS = Path(__file__).parent / "resultados"


# ------------------------------------------------------------------ datos (cacheados)
@st.cache_data
def datos():
    sboxes = cargar_sboxes()
    props = S.tabla_propiedades(sboxes)
    matriz = S.matriz_decision(props)
    norm = S.normalizar(matriz)
    return sboxes, props, matriz, norm


@st.cache_data
def frecuencias(_norm, n=100_000):
    return S.frecuencia_victoria(_norm, n=n)


sboxes, props, matriz, norm = datos()
reportadas = cargar_metricas_reportadas()
TECNICA = {s.id: s.tecnica for s in sboxes}

# ------------------------------------------------------------------ estado
PRESETS = {"Prueba A · no linealidad": S.PESOS_PRUEBA_A,
           "Prueba B · avalancha + diferencial": S.PESOS_PRUEBA_B,
           "Prueba C · sólo avalancha": S.PESOS_PRUEBA_C,
           "Pesos iguales": S.PESOS_IGUALES}
for k in S.CLAVES:
    st.session_state.setdefault(f"w_{k}", float(S.PESOS_IGUALES[k]))
st.session_state.setdefault("ejecuciones", [])


def aplicar_preset(nombre):
    for k, v in PRESETS[nombre].items():
        st.session_state[f"w_{k}"] = float(v)
    st.session_state["etiqueta"] = nombre


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("1 · Prioridades")
    st.caption("Indicá cuánto importa cada propiedad. Los pesos se normalizan para sumar 100 %.")
    c1, c2 = st.columns(2)
    c1.button("Prueba A", on_click=aplicar_preset, args=("Prueba A · no linealidad",), width="stretch",
              help="Prioridad en no linealidad (50 %).")
    c2.button("Prueba B", on_click=aplicar_preset, args=("Prueba B · avalancha + diferencial",), width="stretch",
              help="Prioridad en efecto avalancha y resistencia diferencial (35 % c/u).")
    c1.button("Prueba C", on_click=aplicar_preset, args=("Prueba C · sólo avalancha",), width="stretch",
              help="Control: sólo efecto avalancha (100 %).")
    c2.button("Iguales", on_click=aplicar_preset, args=("Pesos iguales",), width="stretch")
    for c in S.CRITERIOS:
        st.slider(c["nombre"], 0.0, 1.0, step=0.025, key=f"w_{c['clave']}", help=c["ayuda"])
    pesos_crudos = {k: st.session_state[f"w_{k}"] for k in S.CLAVES}
    total = sum(pesos_crudos.values())
    if total == 0:
        st.error("Al menos un criterio debe tener peso mayor que cero.")
    else:
        st.caption("Pesos efectivos: " + " · ".join(
            f"{S.NOMBRES[k].split(' (')[0]} {100 * v / total:.0f}%" for k, v in pesos_crudos.items()))

    st.header("2 · Parámetros de PSO")
    n_part = st.number_input("Partículas", 5, 200, 30, step=5)
    n_iter = st.number_input("Iteraciones", 10, 1000, 100, step=10)
    semilla = st.number_input("Semilla aleatoria", 0, 10_000, 42, help="Misma semilla = mismo resultado.")
    st.text_input("Nombre de la ejecución", key="etiqueta")
    ejecutar = st.button("▶ Ejecutar selección", type="primary", width="stretch", disabled=total == 0)


# ------------------------------------------------------------------ ejecución
def correr(pesos, cfg: ConfigPSO, etiqueta: str) -> dict:
    sel = S.seleccion_exhaustiva(norm, pesos)
    rob = S.analisis_robustez(norm, pesos, cfg)
    directo = S.pso_seleccion_directa(norm, pesos, ConfigPSO(n_particulas=10, iteraciones=20, semilla=cfg.semilla))
    rival = S.rival_alcanzable(norm, sel)
    cm_segundo = S.pso_cambio_minimo(norm, pesos, rival, cfg) if rival else None
    reg = {
        "id": len(st.session_state["ejecuciones"]) + 1,
        "etiqueta": etiqueta or f"Ejecución {len(st.session_state['ejecuciones']) + 1}",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pesos": sel.pesos, "config_pso": cfg.a_dict(),
        "ganador": sel.ganador, "segundo": sel.segundo, "margen": sel.margen,
        "puntajes": sel.ranking["puntaje"].round(6).to_dict(),
        "robustez": rob.drop(columns="pesos_pso").replace({np.nan: None}).to_dict(orient="index"),
        "pso_directo": {"ganador": directo.ganador, "coincide": directo.coincide_con_exhaustiva,
                        "evaluaciones": directo.evaluaciones},
    }
    st.session_state["ejecuciones"].append(reg)
    try:  # registro persistente para reproducir experimentos
        RESULTADOS.mkdir(exist_ok=True)
        with open(RESULTADOS / "ejecuciones.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    except OSError:
        pass
    st.session_state["ultima"] = {"sel": sel, "rob": rob, "directo": directo, "cm": cm_segundo, "rival": rival,
                                  "reg": reg}
    return reg


if ejecutar:
    with st.spinner("Evaluando S-boxes y ejecutando PSO…"):
        correr(pesos_crudos, ConfigPSO(n_particulas=int(n_part), iteraciones=int(n_iter), semilla=int(semilla)),
               st.session_state.get("etiqueta", ""))

# ------------------------------------------------------------------ encabezado
st.title("🔐 Selección adaptativa de S-boxes")
st.markdown(
    "Una **S-box** (caja de sustitución) es la pieza no lineal de un cifrador por bloques: "
    "reemplaza cada byte por otro. Este prototipo compara las **7 S-boxes publicadas** en "
    "Khan (2025, *Scientific Reports*), cada una generada con una técnica de optimización distinta, "
    "y recomienda la mejor **según las prioridades que elijas**. "
    "Usa **PSO (optimización por enjambre de partículas)** para medir cuán estable es esa recomendación "
    "frente a cambios en las prioridades.")
st.caption("Nota: una de las siete S-boxes se llama **«PSO»** porque sus autores originales (ref. [9] del artículo) "
           "la generaron con PSO. Es independiente del PSO que usa este prototipo, que no genera S-boxes: "
           "analiza las prioridades (pesos).")
st.info("Las métricas describen la S-box aislada; **no** equivalen a la seguridad de un cifrador completo, "
        "que depende también de la difusión, el key schedule, el número de rondas y la implementación.", icon="ℹ️")

tab_exp, tab_cmp, tab_datos, tab_met = st.tabs(
    ["Experimento", "Comparar ejecuciones", "S-boxes y métricas", "Metodología y reproducción"])


def grafico_aportes(ranking: pd.DataFrame, titulo: str = ""):
    d = ranking[[f"aporte_{k}" for k in S.CLAVES]].copy()
    d.columns = [S.NOMBRES[k] for k in S.CLAVES]
    d = d.reset_index(names="S-box").melt("S-box", var_name="Criterio", value_name="Aporte")
    orden = list(ranking.index)
    base = alt.Chart(d).mark_bar(cornerRadiusEnd=3, stroke="white", strokeWidth=1).encode(
        y=alt.Y("S-box:N", sort=orden, title=None),
        x=alt.X("sum(Aporte):Q", title="Puntaje (0 = peor en todo, 1 = mejor en todo)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Criterio:N", scale=alt.Scale(domain=list(COLOR_CRIT), range=list(COLOR_CRIT.values())),
                        legend=alt.Legend(orient="bottom", columns=3, title=None)),
        order=alt.Order("Criterio:N"),
        tooltip=["S-box", "Criterio", alt.Tooltip("Aporte:Q", format=".3f")],
    )
    tot = alt.Chart(ranking.reset_index(names="S-box")).mark_text(align="left", dx=4, color="#52514e").encode(
        y=alt.Y("S-box:N", sort=orden), x="puntaje:Q", text=alt.Text("puntaje:Q", format=".3f"))
    return (base + tot).properties(height=260, title=titulo)


def grafico_perfil():
    d = norm.copy()
    d.columns = [S.NOMBRES[k] for k in S.CLAVES]
    d = d.reset_index(names="S-box").melt("S-box", var_name="Criterio", value_name="Valor normalizado")
    base = alt.Chart(d).encode(x=alt.X("Criterio:N", sort=list(COLOR_CRIT), title=None, axis=alt.Axis(labelAngle=-20)),
                               y=alt.Y("S-box:N", sort=list(norm.index), title=None))
    celdas = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("Valor normalizado:Q", scale=alt.Scale(range=["#cde2fb", "#104281"], domain=[0, 1]),
                        legend=alt.Legend(title="1 = mejor")),
        tooltip=["S-box", "Criterio", alt.Tooltip("Valor normalizado:Q", format=".3f")])
    texto = base.mark_text(fontSize=11).encode(
        text=alt.Text("Valor normalizado:Q", format=".2f"),
        color=alt.condition("datum['Valor normalizado'] > 0.55", alt.value("white"), alt.value("#0b0b0b")))
    return (celdas + texto).properties(height=260)


def explicar(sel: S.ResultadoSeleccion, cm: S.CambioMinimo | None, rival: str | None) -> list[str]:
    g, s2 = sel.ganador, sel.segundo
    r = sel.ranking
    aportes = r.loc[g, [f"aporte_{k}" for k in S.CLAVES]]
    aportes.index = S.CLAVES
    top = aportes.sort_values(ascending=False).index[:2]
    mejor_en = [S.NOMBRES[k] for k in S.CLAVES if norm.loc[g, k] == 1]
    lineas = [
        f"**{g}** ({TECNICA[g]}) obtiene el mayor puntaje: **{r.loc[g, 'puntaje']:.3f}**, contra "
        f"**{r.loc[s2, 'puntaje']:.3f}** de {s2} (diferencia {sel.margen:.3f}).",
        f"Lo que más suma a su puntaje: **{S.NOMBRES[top[0]]}** ({aportes[top[0]]:.3f}) y "
        f"**{S.NOMBRES[top[1]]}** ({aportes[top[1]]:.3f}).",
    ]
    if mejor_en:
        lineas.append(f"Es la mejor del conjunto en: {', '.join(mejor_en)}.")
    peores = [S.NOMBRES[k] for k in S.CLAVES if norm.loc[g, k] < 1 and sel.pesos[k] > 0]
    if peores:
        lineas.append("No es la mejor en: " + ", ".join(
            f"{S.NOMBRES[k]} (la mejor es {norm[k].idxmax()})" for k in S.CLAVES
            if norm.loc[g, k] < 1 and sel.pesos[k] > 0) + ".")
    dom = S.dominadas(norm)
    nd = [a for a, por in dom.items() if por]
    if nd:
        lineas.append(f"{len(nd)} de 7 S-boxes ({', '.join(nd)}) están **dominadas**: otra es igual o mejor en "
                      "todos los criterios, así que no pueden ganar con ningún conjunto de pesos.")
    if rival and rival != s2:
        lineas.append(f"La segunda ({s2}) está dominada, así que el rival que podría desplazar a {g} es **{rival}**.")
    if cm is not None and cm.factible:
        lineas.append(f"**Robustez (PSO):** para que {rival} pasara a ser la recomendada habría que reasignar al menos "
                      f"**{100 * cm.distancia:.1f} %** de la importancia total, por ejemplo con estos pesos: "
                      + ", ".join(f"{S.NOMBRES[k]} {100 * v:.0f}%" for k, v in cm.pesos.items() if v >= 0.005) + ".")
    elif rival:
        lineas.append(f"**Robustez (PSO):** PSO no encontró pesos con los que {rival} supere a {g}.")
    else:
        lineas.append(f"**Robustez:** las demás S-boxes están dominadas; {g} gana con cualquier conjunto de pesos.")
    return lineas


# ------------------------------------------------------------------ pestaña Experimento
with tab_exp:
    if "ultima" not in st.session_state:
        st.subheader("Cómo usar")
        st.markdown(
            "1. Elegí prioridades en la barra lateral (o usá los botones **Prueba A** / **Prueba B**).\n"
            "2. Presioná **▶ Ejecutar selección**.\n"
            "3. Repetí con otras prioridades y compará en la pestaña **Comparar ejecuciones**.")
        st.subheader("Perfil de cada S-box (valores normalizados)")
        st.altair_chart(grafico_perfil(), width="stretch")
    else:
        u = st.session_state["ultima"]
        sel, rob, directo, cm, rival, reg = u["sel"], u["rob"], u["directo"], u["cm"], u["rival"], u["reg"]
        st.subheader(f"Resultado: {reg['etiqueta']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("S-box recomendada", sel.ganador, TECNICA[sel.ganador], delta_color="off")
        m2.metric("Puntaje", f"{sel.ranking.loc[sel.ganador, 'puntaje']:.3f}")
        m3.metric("Ventaja sobre la 2.ª", f"{sel.margen:.3f}", sel.segundo, delta_color="off")
        m4.metric(f"Cambio de pesos para que gane {rival or 'otra'}",
                  f"{100 * cm.distancia:.1f} %" if cm is not None and cm.factible else "no alcanzable",
                  help="Calculado con PSO: fracción mínima de la importancia total que habría que reasignar.")
        if len(st.session_state["ejecuciones"]) >= 2:
            prev = st.session_state["ejecuciones"][-2]
            if prev["ganador"] == sel.ganador:
                st.success(f"La recomendación **no cambió** respecto de «{prev['etiqueta']}»: sigue siendo {sel.ganador}.")
            else:
                st.warning(f"La recomendación **cambió**: «{prev['etiqueta']}» elegía {prev['ganador']}, ahora {sel.ganador}.")

        izq, der = st.columns([3, 2])
        with izq:
            st.markdown("**Puntaje por S-box y aporte de cada criterio**")
            st.altair_chart(grafico_aportes(sel.ranking), width="stretch")
        with der:
            st.markdown("**¿Por qué fue seleccionada?**")
            for l in explicar(sel, cm, rival):
                st.markdown("- " + l)

        st.markdown("**Perfil normalizado** (1 = mejor del conjunto en ese criterio, 0 = peor)")
        st.altair_chart(grafico_perfil(), width="stretch")

        with st.expander("Análisis de robustez con PSO (detalle)", expanded=False):
            st.markdown(
                "Para cada S-box no ganadora, PSO busca los pesos **más parecidos a los elegidos** con los que esa "
                "S-box quedaría primera. La distancia es la fracción de la importancia que habría que reasignar "
                "(0 % = nada, 100 % = cambiar todo). Como referencia se muestra una búsqueda aleatoria (Monte Carlo) "
                "con muchas más evaluaciones. *Vacío* = no hay pesos que la hagan ganar.")
            t = rob.copy()
            t["dist_pso"] = (100 * t["dist_pso"]).round(1)
            t["dist_montecarlo"] = (100 * t["dist_montecarlo"]).round(1)
            t["pesos_pso"] = t["pesos_pso"].apply(
                lambda p: ", ".join(f"{k} {100 * v:.0f}%" for k, v in p.items() if v >= 0.005) if p else "-")
            st.dataframe(t.rename(columns={
                "dist_pso": "Cambio mínimo PSO (%)", "dist_montecarlo": "Cambio mínimo Monte Carlo (%)",
                "evals_pso": "Evaluaciones PSO", "evals_montecarlo": "Evaluaciones MC",
                "dominada_por": "Dominada por", "pesos_pso": "Pesos encontrados por PSO"}), width="stretch")
            if cm is not None and cm.factible:
                h = pd.DataFrame({"Iteración": range(len(cm.pso.historial)), "Mejor valor objetivo": cm.pso.historial})
                st.markdown(f"Convergencia de PSO al buscar pesos que hagan ganar a **{rival}**")
                st.altair_chart(alt.Chart(h).mark_line(strokeWidth=2, color=COLORES[0]).encode(
                    x="Iteración:Q", y=alt.Y("Mejor valor objetivo:Q", scale=alt.Scale(zero=False)),
                    tooltip=["Iteración", alt.Tooltip("Mejor valor objetivo:Q", format=".4f")]).properties(height=200),
                    width="stretch")
            st.caption(f"Parámetros PSO: {reg['config_pso']}")

        with st.expander("Método de referencia vs. PSO aplicado directamente a la selección"):
            st.markdown(
                f"- **Búsqueda exhaustiva** (referencia): evalúa las 7 S-boxes → **{sel.ganador}** con "
                f"{sel.evaluaciones} evaluaciones. Es exacta.\n"
                f"- **PSO directo** (10 partículas × 20 iteraciones, semilla {reg['config_pso']['semilla']}): "
                f"eligió **{directo.ganador}** con {directo.evaluaciones} evaluaciones "
                f"({'coincide' if directo.coincide_con_exhaustiva else 'NO coincide'} con la referencia).\n\n"
                "Con sólo 7 alternativas, usar PSO para elegir la S-box no tiene sentido práctico: gasta ~30 veces más "
                "evaluaciones y no garantiza el óptimo. Por eso aquí PSO se usa en el **espacio continuo de pesos**, "
                "donde no es posible enumerar todas las opciones.")

# ------------------------------------------------------------------ pestaña Comparar
with tab_cmp:
    ej = st.session_state["ejecuciones"]
    if len(ej) < 2:
        st.info("Ejecutá al menos dos configuraciones (por ejemplo, Prueba A y Prueba B) para compararlas.")
    else:
        nombres = [f"#{e['id']} · {e['etiqueta']}" for e in ej]
        a, b = st.columns(2)
        ia = a.selectbox("Ejecución 1", range(len(ej)), index=len(ej) - 2, format_func=lambda i: nombres[i])
        ib = b.selectbox("Ejecución 2", range(len(ej)), index=len(ej) - 1, format_func=lambda i: nombres[i])
        e1, e2 = ej[ia], ej[ib]
        if e1["ganador"] == e2["ganador"]:
            st.success(f"Ambas configuraciones recomiendan **{e1['ganador']}**. Cambiar las prioridades no cambió la "
                       "selección; sí cambiaron los puntajes y las distancias (ver abajo).")
        else:
            st.warning(f"La recomendación cambia: **{e1['ganador']}** → **{e2['ganador']}**.")
        pw = pd.DataFrame({nombres[ia]: e1["pesos"], nombres[ib]: e2["pesos"]})
        pw.index = [S.NOMBRES[k] for k in pw.index]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pesos (%)**")
            st.dataframe((100 * pw).round(1), width="stretch")
        with c2:
            st.markdown("**Puntajes**")
            ps = pd.DataFrame({nombres[ia]: e1["puntajes"], nombres[ib]: e2["puntajes"]})
            st.dataframe(ps.round(3), width="stretch")
        d = ps.reset_index(names="S-box").melt("S-box", var_name="Ejecución", value_name="Puntaje")
        st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("S-box:N", sort=list(ps[nombres[ib]].sort_values(ascending=False).index), title=None),
            yOffset="Ejecución:N", x=alt.X("Puntaje:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Ejecución:N", scale=alt.Scale(range=[COLORES[0], COLORES[1]]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["S-box", "Ejecución", alt.Tooltip("Puntaje:Q", format=".3f")]).properties(height=320),
            width="stretch")
        st.markdown("**Cambio mínimo de pesos (PSO) para que cada S-box pase a ganar**")
        rob_cmp = pd.DataFrame({
            nombres[ia]: {k: (None if v["dist_pso"] is None else round(100 * v["dist_pso"], 1)) for k, v in e1["robustez"].items()},
            nombres[ib]: {k: (None if v["dist_pso"] is None else round(100 * v["dist_pso"], 1)) for k, v in e2["robustez"].items()}})
        st.dataframe(rob_cmp, width="stretch")
        st.caption("Valores en % de importancia a reasignar; vacío = no alcanzable o es la ganadora.")
    if ej:
        st.download_button("Descargar todas las ejecuciones (JSON)", json.dumps(ej, ensure_ascii=False, indent=1),
                           "ejecuciones.json", "application/json")

# ------------------------------------------------------------------ pestaña Datos
with tab_datos:
    st.subheader("Valores calculados por este prototipo")
    vis = props[["nl_min", "nl_max", "nl_prom", "sac_prom", "sac_desvio", "bic_nl_min", "bic_nl_prom",
                 "bic_sac_prom", "bic_sac_desvio", "du", "dp", "lp"]].astype(float)
    vis.insert(0, "técnica", [TECNICA[i] for i in vis.index])
    st.dataframe(vis.round(4), width="stretch")
    st.caption("NL: no linealidad · SAC: avalancha · BIC: independencia de bits · DU/DP: uniformidad/probabilidad "
               "diferencial · LP: probabilidad lineal. Desvío = promedio de |valor − 0.5|.")
    st.subheader("Valores reportados en el artículo (Tablas 8 y 9)")
    st.dataframe(reportadas[["nl_min", "nl_max", "nl_prom", "sac_prom", "bic_nl_min", "bic_nl_prom",
                             "bic_sac_prom", "du", "lp"]], width="stretch")
    comp = pd.DataFrame({
        "NL prom (calc / art)": [f"{props.loc[i, 'nl_prom']:.2f} / {reportadas.loc[i, 'nl_prom']}" for i in props.index],
        "DU (calc / art)": [f"{props.loc[i, 'du']} / {reportadas.loc[i, 'du']}" for i in props.index],
        "LP (calc / art)": [f"{props.loc[i, 'lp']:.4f} / {reportadas.loc[i, 'lp']}" for i in props.index],
        "SAC prom (calc / art)": [f"{props.loc[i, 'sac_prom']:.4f} / {reportadas.loc[i, 'sac_prom']}" for i in props.index],
    }, index=props.index)
    st.markdown("**Reproducción** (calculado / artículo). Las diferencias están documentadas en "
                "`docs/referencia_cientifica.md`: TLBO (NL y LP), ABC (LP), FA (NL prom., SAC y BIC-SAC).")
    st.dataframe(comp, width="stretch")
    st.subheader("Ver una S-box")
    sid = st.selectbox("S-box", [s.id for s in sboxes], format_func=lambda i: f"{i} · {TECNICA[i]}")
    sb = next(s for s in sboxes if s.id == sid)
    st.caption(f"Tabla {sb.tabla_articulo} del artículo. Fila = nibble alto, columna = nibble bajo del byte de entrada.")
    st.dataframe(sb.como_matriz_hex(), width="stretch")
    f = fuente()
    st.caption(f"Fuente: {f['articulo']} doi:{f['doi']} · {f['licencia']} · {f['url_extraccion']}")

# ------------------------------------------------------------------ pestaña Metodología
with tab_met:
    st.subheader("Qué hace cada componente")
    st.markdown(
        "1. **Métricas** (`src/metricas.py`): calcula no linealidad (espectro de Walsh), SAC, BIC-NL, BIC-SAC, "
        "uniformidad diferencial (tabla DDT) y probabilidad lineal (tabla LAT) a partir de los 256 valores.\n"
        "2. **Normalización** min-max por criterio: 1 = mejor del conjunto, 0 = peor. Es relativa a estas 7 S-boxes.\n"
        "3. **Puntaje** = suma ponderada de los valores normalizados (SAW). La **búsqueda exhaustiva** sobre las 7 "
        "alternativas es el método de referencia y es exacta.\n"
        "4. **PSO** explora el espacio continuo de pesos: encuentra el cambio mínimo de prioridades que haría ganar "
        "a cada alternativa. Se valida contra una búsqueda Monte Carlo.\n"
        "5. **Dominancia de Pareto**: identifica alternativas que no pueden ganar con ningún peso.")
    st.subheader("Probabilidad de ganar con pesos al azar")
    fr = frecuencias(norm)
    st.dataframe((100 * fr).round(2).rename("% de 100 000 pesos aleatorios en que gana").to_frame(),
                 width="stretch")
    st.subheader("Reproducción del método EDAS del artículo")
    ed = S.edas(S.matriz_tabla10(reportadas), S.ART_TABLA11_PESOS, S.ART_TABLA10_BENEFICIO)
    t = pd.DataFrame({"AS reproducido": ed["AS"].round(4), "AS artículo (Tabla 16)": reportadas["edas_score"],
                      "Rank reproducido": pd.Series(ed["AS"]).rank(ascending=False).astype(int).values,
                      "Rank artículo": reportadas["edas_rank"]}, index=reportadas.index)
    st.dataframe(t, width="stretch")
    st.markdown(
        "Con la matriz de la Tabla 10 y los pesos de la Tabla 11 se reproducen exactamente el promedio (Tabla 12) y "
        "las matrices PDA/NDA (Tablas 13 y 14), y coinciden el primer y último puesto (PSO y FA). Los valores SP/SN "
        "de la Tabla 15 y los puntajes de la Tabla 16 **no** se obtienen con esos pesos: el artículo presenta una "
        "inconsistencia interna en esos pasos. Además, el último puesto de FA se debe a un valor BIC-SAC de 0.01276 "
        "que no es consistente con su mínimo (0.465) ni con el valor que calcula este prototipo (0.498).")
