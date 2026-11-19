"""
app.py — Segmentación de empresas del S&P 500 con ACP y K-Means.
Autora: Salome Reyes · Machine Learning aplicado a Economía y Finanzas · PUJ Cali

Ejecutar:  streamlit run app.py
Toda la lógica de procesamiento y modelado está en modelo.py; la app la ejecuta
en vivo con los parámetros que el usuario elige en la barra lateral.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import modelo as m

# ---------------------------------------------------------------------------
# Configuración de página e identidad visual
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="S&P 500 · ACP y K-Means",
    page_icon="📊",
    layout="wide",
)

PALETA = px.colors.qualitative.Safe
PLANTILLA = "plotly_white"
CONFIG_BASE = {"variables": list(m.VARIABLES_MODELO), "winsor": 0.01, "n_comp": 3, "k": 5}

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .nota {background:rgba(70,130,180,0.14);border-left:4px solid #4682b4;padding:0.8rem 1rem;
           border-radius:4px;margin:0.5rem 0 1rem 0;color:inherit;}
    </style>
    """,
    unsafe_allow_html=True,
)


# La traducción automática del navegador modifica la página por debajo y rompe
# Streamlit (error "removeChild"). Se marca la página como español y "no traducir".
components.html(
    """<script>
    const d = window.parent.document;
    d.documentElement.setAttribute("lang", "es");
    d.documentElement.setAttribute("translate", "no");
    d.documentElement.classList.add("notranslate");
    if (!d.querySelector('meta[name="google"]')) {
        const meta = d.createElement("meta");
        meta.name = "google"; meta.content = "notranslate";
        d.head.appendChild(meta);
    }
    </script>""",
    height=0,
)


def nota(texto: str):
    st.markdown(f'<div class="nota">{texto}</div>', unsafe_allow_html=True)


def pct(x: float) -> str:
    return f"{x * 100:.1f} %"


# ---------------------------------------------------------------------------
# Funciones con caché (envuelven a modelo.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Procesando datos y ajustando el ACP…")
def correr_pipeline(variables: tuple, winsor: float):
    return m.pipeline(variables=list(variables), winsor=winsor)


@st.cache_data(show_spinner="Evaluando valores de K…")
def correr_evaluar_k(datos: np.ndarray, k_max: int):
    return m.evaluar_k(datos, 2, k_max)


@st.cache_data(show_spinner="Ajustando K-Means…")
def correr_kmeans(datos: np.ndarray, k: int):
    _, etiquetas, sil = m.ajustar_kmeans(datos, k)
    return etiquetas.to_numpy(), sil


# ---------------------------------------------------------------------------
# Barra lateral: navegación y parámetros del modelo
# ---------------------------------------------------------------------------
st.sidebar.title("📊 S&P 500")
st.sidebar.caption(
    "**Salome Reyes**  \nMachine Learning aplicado a Economía y Finanzas · "
    "Taller ACP y K-Means · PUJ Cali"
)

seccion = st.sidebar.radio(
    "Secciones",
    [
        "1. Contexto y datos",
        "2. Exploración y preparación",
        "3. Componentes principales",
        "4. Segmentación K-Means",
        "5. Conclusiones",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Parámetros del modelo")
variables_sel = st.sidebar.multiselect(
    "Variables del modelo",
    options=list(m.VARIABLES_MODELO),
    default=CONFIG_BASE["variables"],
    format_func=lambda v: m.ETIQUETAS_CORTAS[v],
    help="Mínimo 4 variables. Se construyen a partir de las columnas originales.",
)
if len(variables_sel) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.stop()

winsor = st.sidebar.slider(
    "Recorte de atípicos (winsorización, % por cola)",
    0.0, 5.0, CONFIG_BASE["winsor"] * 100, 0.5,
    help="Los valores por debajo del percentil p y por encima del 100−p se llevan a esos límites.",
) / 100

R = correr_pipeline(tuple(variables_sel), winsor)
limpio, Xt, Z, varianza, cargas, puntajes = (
    R["limpio"], R["Xt"], R["Z"], R["varianza"], R["cargas"], R["puntajes"]
)
n_vars = Xt.shape[1]
sug = m.sugerir_componentes(varianza)

n_comp = st.sidebar.slider(
    "Componentes conservados para K-Means",
    2, n_vars, min(CONFIG_BASE["n_comp"], n_vars),
    help=f"Sugerencia → Kaiser (valor propio > 1): {sug['kaiser']} · "
    f"≥ 80 % de varianza: {sug['umbral']}",
)
base_km = st.sidebar.radio(
    "K-Means se aplica sobre",
    ["Componentes seleccionados", "Variables estandarizadas"],
    help="Por defecto se usan los componentes: eliminan redundancia y ruido.",
)
k_sel = st.sidebar.slider("Número de clústeres (K)", 2, 10, CONFIG_BASE["k"])

datos_km = puntajes[:, :n_comp] if base_km == "Componentes seleccionados" else Z
evaluacion = correr_evaluar_k(datos_km, 10)
etiquetas, sil = correr_kmeans(datos_km, k_sel)

df = limpio.copy()
df["Grupo"] = etiquetas
df["Silueta"] = sil
for i in range(min(3, puntajes.shape[1])):
    df[f"CP{i + 1}"] = puntajes[:, i]
perfil = m.perfil_clusters(df, Xt)
grupos = sorted(df["Grupo"].unique())
colores_grupo = {g: PALETA[i % len(PALETA)] for i, g in enumerate(grupos)}

st.sidebar.divider()
st.sidebar.caption(
    f"Empresas analizadas: **{len(df)}** · Varianza conservada: "
    f"**{pct(varianza['Varianza acumulada'].iloc[n_comp - 1])}**"
)


# ===========================================================================
# 1. CONTEXTO
# ===========================================================================
if seccion.startswith("1"):
    st.title("¿Qué perfiles financieros conviven dentro del S&P 500?")
    st.caption(
        "Salome Reyes · Taller individual ACP y K-Means · Machine Learning aplicado a "
        "Economía y Finanzas · Pontificia Universidad Javeriana Cali"
    )
    st.markdown(
        """
        El S&P 500 suele leerse por **sectores** (tecnología, salud, energía…). Pero dos
        empresas del mismo sector pueden tener perfiles de valoración muy distintos, y
        empresas de sectores diferentes pueden parecerse mucho ante los ojos de un
        inversionista. Este proyecto pregunta:

        > **¿Existen grupos de empresas con perfiles similares de tamaño, valoración,
        > rentabilidad, dividendos y comportamiento reciente del precio, y en qué se
        > parecen o no a la clasificación sectorial tradicional?**
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas en la fuente", len(R["crudo"]))
    c2.metric("Empresas analizadas", len(limpio))
    c3.metric("Variables del modelo", n_vars)
    c4.metric("Subindustrias GICS", limpio["Subindustria"].nunique())

    st.subheader("Fuente de los datos")
    st.markdown(
        """
        - **Archivo:** `constituents-financials.csv`, conjunto *S&P 500 Companies with
          Financial Information* publicado en DataHub (`datahub.io/core/s-and-p-500-companies-financials`).
          Reúne, para cada empresa del índice, su precio y razones financieras de mercado.
          Cada fila trae además el enlace a los reportes de la empresa en SEC EDGAR.
        - **Unidad de observación:** una empresa del S&P 500.
        - **Cobertura:** las ~500 empresas del índice en la fecha de descarga (corte transversal).
        - **Columnas originales:** símbolo, nombre, subindustria GICS, precio, P/E,
          rendimiento por dividendo, utilidad por acción (UPA), mínimo y máximo de 52
          semanas, capitalización bursátil, EBITDA, Precio/Ventas, Precio/Valor en libros
          y enlace a la SEC.
        """
    )

    st.subheader("Variables utilizadas en el modelo")
    st.markdown(
        "Las columnas de texto (símbolo, nombre, subindustria, enlace) **no entran** al "
        "modelo matemático; se conservan para describir los grupos. Las variables "
        "numéricas se transforman en razones comparables entre empresas:"
    )
    st.dataframe(
        pd.DataFrame(
            {"Variable": list(m.VARIABLES_MODELO), "Definición": list(m.VARIABLES_MODELO.values())}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "El precio y la UPA en niveles no se usan: dependen del número de acciones "
        "emitidas, no de la empresa. Se usan como cocientes (E/P) o para construir el "
        "rango de 52 semanas."
    )

    st.subheader("¿Por qué ACP y K-Means?")
    st.markdown(
        """
        - **ACP:** las razones de valoración están muy correlacionadas entre sí (una
          empresa "cara" suele serlo en P/E, P/Ventas y P/Libro a la vez). El ACP
          resume esa información redundante en pocas dimensiones interpretables
          (por ejemplo, un eje *valor vs. crecimiento*) y permite visualizar 480 empresas en un plano.
        - **K-Means:** sobre esas dimensiones busca grupos de empresas con perfiles
          parecidos sin usar la etiqueta sectorial. Así se puede contrastar si la
          segmentación por fundamentales coincide o no con los sectores, algo útil para
          construir portafolios diversificados o filtrar acciones por estilo.
        """
    )

    st.subheader("Limitaciones de la fuente")
    st.markdown(
        """
        - **Corte transversal:** es una fotografía en un solo momento. No hay evolución
          histórica, así que no se puede estudiar la estabilidad de los grupos en el tiempo
          ni separar efectos de ciclo económico.
        - **Razones de mercado:** P/E, P/Ventas, P/Libro y rendimiento por dividendo
          cambian con el precio diario; otro día de descarga puede mover empresas de grupo.
        - **Datos incompletos o no aplicables:** 17 empresas sin datos (salieron del
          índice), P/E indefinido con utilidades negativas, EBITDA no reportado para bancos
          y aseguradoras (no es una métrica relevante para ellos) y patrimonio negativo en
          empresas con grandes recompras de acciones.
        - **Pocas dimensiones contables:** no incluye endeudamiento, crecimiento de
          ventas, márgenes ni flujo de caja, que serían centrales en un análisis fundamental completo.
        - **Posibles errores de captura:** algunos valores lucen inconsistentes (por
          ejemplo, rendimientos por dividendo cercanos a cero en REIT que sí pagan
          dividendos). Se mitigan con la winsorización, pero no se corrigen uno a uno.
        - **Sesgo de supervivencia y tamaño:** solo grandes empresas estadounidenses; los
          resultados no se generalizan a empresas pequeñas ni a otros mercados.
        """
    )


# ===========================================================================
# 2. EXPLORACIÓN Y PREPARACIÓN
# ===========================================================================
elif seccion.startswith("2"):
    st.title("Exploración y preparación de los datos")

    st.subheader("Estructura de la base original")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", R["crudo"].shape[0])
    c2.metric("Columnas", R["crudo"].shape[1])
    c3.metric("Celdas faltantes", int(R["crudo"].isna().sum().sum()))
    st.dataframe(m.diagnostico(R["crudo"]), use_container_width=True)

    st.subheader("Decisiones de calidad")
    for linea in R["bitacora"]:
        st.markdown(f"- {linea}")
    st.markdown(
        """
        - **P/E faltante (23 empresas con pérdidas):** el P/E no está definido con
          utilidades negativas. En lugar de eliminarlas, se usa su inverso, el
          **rendimiento de utilidades (UPA/Precio)**, que es negativo para empresas con
          pérdidas y conserva esa información.
        - **P/B negativo (patrimonio negativo):** se usa el inverso, **Libro/Precio**, que
          es continuo y no se dispara cuando el patrimonio es cercano a cero.
        - **EBITDA faltante (26 bancos y financieras):** se imputa la **mediana** de
          EBITDA/Capitalización, para no excluir al sector financiero. Es una decisión
          neutral: ubica a estas empresas en el centro de esa variable.
        - **Asimetría:** capitalización y Precio/Ventas se transforman con log10.
        - **Atípicos:** se winsoriza cada variable en los percentiles elegidos en la barra
          lateral (por defecto 1 % y 99 %).
        """
    )
    st.dataframe(R["tratamiento"], use_container_width=True)

    with st.expander("Ver la tabla de empresas (datos limpios)"):
        st.dataframe(limpio.drop(columns="SEC Filings"), use_container_width=True, hide_index=True)

    st.subheader("Estadísticas descriptivas")
    t1, t2 = st.tabs(["Variables originales", "Variables del modelo (tratadas)"])
    with t1:
        st.dataframe(limpio[m.VARIABLES_ORIGINALES + ["Price"]].describe().T.round(3), use_container_width=True)
        st.caption(
            "Note la enorme dispersión (P/E máximo en miles, P/B negativo): por eso se "
            "transforman las variables antes del modelo."
        )
    with t2:
        st.dataframe(Xt.describe().T.round(3), use_container_width=True)

    st.subheader("Distribución de cada variable: antes y después del tratamiento")
    var_h = st.selectbox(
        "Variable", list(Xt.columns), format_func=lambda v: m.ETIQUETAS_CORTAS[v]
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.histogram(R["X"], x=var_h, nbins=50, template=PLANTILLA,
                     title="Construida, sin tratar", labels={var_h: m.ETIQUETAS_CORTAS[var_h]})
        .update_layout(yaxis_title="Número de empresas"),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.histogram(Xt, x=var_h, nbins=50, template=PLANTILLA, color_discrete_sequence=["#2c5f8a"],
                     title="Imputada y winsorizada (entra al modelo)",
                     labels={var_h: m.ETIQUETAS_CORTAS[var_h]})
        .update_layout(yaxis_title="Número de empresas"),
        use_container_width=True,
    )

    st.subheader("Matriz de correlaciones (variables del modelo)")
    corr = Xt.rename(columns=m.ETIQUETAS_CORTAS).corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template=PLANTILLA, title="Correlación de Pearson entre variables tratadas", aspect="auto",
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    pares = (
        corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
        .sort_values(key=np.abs, ascending=False).head(3)
    )
    nota(
        "<b>Lectura:</b> las correlaciones más fuertes son "
        + "; ".join(f"{a} – {b} ({v:+.2f})" for (a, b), v in pares.items())
        + ". Hay redundancia entre las razones de valoración, lo que justifica resumirlas con ACP."
    )

    st.subheader("Estandarización")
    st.markdown(
        "Las variables tienen escalas muy distintas (log de capitalización ≈ 10; "
        "rendimiento por dividendo ≈ 0,02). Como ACP y K-Means se basan en varianzas y "
        "distancias, se aplica `StandardScaler` (media 0, desviación 1) para que ninguna "
        "variable domine solo por su escala."
    )
    st.dataframe(
        pd.DataFrame(Z, columns=Xt.columns).describe().T[["mean", "std"]].round(3),
        use_container_width=True,
    )


# ===========================================================================
# 3. ACP
# ===========================================================================
elif seccion.startswith("3"):
    st.title("Análisis de Componentes Principales")

    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes con valor propio > 1 (Kaiser)", sug["kaiser"])
    c2.metric("Componentes para ≥ 80 % de varianza", sug["umbral"])
    c3.metric(f"Varianza con {n_comp} componentes", pct(varianza["Varianza acumulada"].iloc[n_comp - 1]))

    st.subheader("Varianza explicada")
    tabla_var = varianza.copy()
    for c in ["Varianza explicada", "Varianza acumulada"]:
        tabla_var[c] = (tabla_var[c] * 100).round(1).astype(str) + " %"
    tabla_var["Valor propio"] = tabla_var["Valor propio"].round(3)
    st.dataframe(tabla_var, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=varianza["Componente"], y=varianza["Varianza explicada"] * 100,
                name="Por componente", marker_color="#2c5f8a")
    fig.add_scatter(x=varianza["Componente"], y=varianza["Varianza acumulada"] * 100,
                    name="Acumulada", mode="lines+markers", line_color="#d1495b")
    fig.add_hline(y=80, line_dash="dot", annotation_text="80 %")
    fig.update_layout(template=PLANTILLA, title="Varianza explicada y acumulada",
                      yaxis_title="% de la varianza total", xaxis_title="Componente",
                      legend=dict(orientation="h", y=-0.2))
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.line(varianza, x="Componente", y="Valor propio", markers=True, template=PLANTILLA,
                  title="Gráfico de sedimentación (scree plot)")
    fig.add_hline(y=1, line_dash="dot", line_color="#d1495b", annotation_text="Criterio de Kaiser (λ = 1)")
    c2.plotly_chart(fig, use_container_width=True)

    nota(
        f"<b>¿Cuántos componentes conservar?</b> El criterio de Kaiser sugiere "
        f"<b>{sug['kaiser']}</b> y el umbral del 80 % sugiere <b>{sug['umbral']}</b>. "
        f"El codo del gráfico de sedimentación aparece después de CP{sug['kaiser']}: los "
        "componentes siguientes aportan poco cada uno y son más difíciles de interpretar. "
        f"Con la selección actual (<b>{n_comp}</b> componentes) se conserva el "
        f"<b>{pct(varianza['Varianza acumulada'].iloc[n_comp - 1])}</b> de la información. "
        "Puedes cambiar este número en la barra lateral."
    )

    st.subheader("Cargas: ¿qué significa cada componente?")
    n_mostrar = max(n_comp, 2)
    cargas_v = cargas.iloc[:, :n_mostrar].rename(index=m.ETIQUETAS_CORTAS)
    fig = px.imshow(cargas_v, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", template=PLANTILLA,
                    title="Cargas (correlación entre cada variable y cada componente)")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    lineas = []
    for cp in cargas_v.columns:
        col = cargas_v[cp]
        fuertes = col[col.abs() >= 0.45].sort_values(key=np.abs, ascending=False)
        pos = [i for i, v in fuertes.items() if v > 0]
        neg = [i for i, v in fuertes.items() if v < 0]
        lineas.append(
            f"<b>{cp}</b>: valores altos ↔ más {', '.join(pos) if pos else '—'}"
            f"; valores bajos ↔ más {', '.join(neg) if neg else '—'}."
        )
    nota("<br>".join(lineas))
    st.markdown(
        """
        **Interpretación con la configuración base (8 variables, 1 % de winsorización):**
        - **CP1 – Eje *valor vs. crecimiento*.** Cargas positivas altas en EBITDA/Cap.,
          Libro/Precio y dividendo; negativas en P/Ventas y tamaño. Un CP1 alto describe
          empresas "baratas" frente a sus fundamentales y que reparten utilidades; un CP1
          bajo describe empresas grandes por las que el mercado paga múltiplos altos
          (expectativa de crecimiento).
        - **CP2 – *Volatilidad / castigo reciente*.** Dominado por la amplitud del rango de
          52 semanas (+) frente al dividendo y la posición en el rango (−): empresas con
          precios que se movieron mucho y hoy están lejos de sus máximos.
        - **CP3 – *Rentabilidad contable*.** Casi exclusivamente el rendimiento de
          utilidades (E/P): separa a las empresas con pérdidas del resto.
        """
    )

    st.subheader("Empresas en el espacio de componentes")
    c1, c2, c3 = st.columns(3)
    opciones_cp = list(varianza["Componente"])
    eje_x = c1.selectbox("Eje X", opciones_cp, index=0)
    eje_y = c2.selectbox("Eje Y", opciones_cp, index=1)
    mostrar_cargas = c3.checkbox("Mostrar vectores de carga (biplot)", value=True)
    ix, iy = opciones_cp.index(eje_x), opciones_cp.index(eje_y)
    info = varianza["Varianza explicada"].iloc[[ix, iy]].sum()
    plano = pd.DataFrame({eje_x: puntajes[:, ix], eje_y: puntajes[:, iy]})
    plano[["Empresa", "Símbolo", "Sector GICS"]] = limpio[["Name", "Symbol", "Sector GICS"]].to_numpy()
    fig = px.scatter(
        plano, x=eje_x, y=eje_y, color="Sector GICS", hover_data=["Empresa", "Símbolo"],
        template=PLANTILLA, color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"{eje_x} vs {eje_y} — este plano resume el {pct(info)} de la variación total",
    )
    if mostrar_cargas:
        esc = np.abs(plano[[eje_x, eje_y]]).max().max() * 0.9
        for var, fila in cargas.iterrows():
            fig.add_annotation(
                x=fila[eje_x] * esc, y=fila[eje_y] * esc, ax=0, ay=0, xref="x", yref="y",
                axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#333",
                text=m.ETIQUETAS_CORTAS[var], font=dict(size=11),
            )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cada punto es una empresa, coloreada por su sector GICS (que no entra al modelo). "
        "Las flechas indican hacia dónde crece cada variable original."
    )


# ===========================================================================
# 4. K-MEANS
# ===========================================================================
elif seccion.startswith("4"):
    st.title("Segmentación mediante K-Means")
    st.markdown(
        f"K-Means se aplica sobre **{base_km.lower()}** "
        + (f"(CP1 a CP{n_comp}, {pct(varianza['Varianza acumulada'].iloc[n_comp - 1])} de la varianza)."
           if base_km == "Componentes seleccionados" else f"({n_vars} variables).")
        + " Se usan 20 inicializaciones aleatorias y semilla fija (42) para que el resultado sea reproducible."
    )

    st.subheader("Elección de K: método del codo y coeficiente de silueta")
    c1, c2 = st.columns(2)
    fig = px.line(evaluacion, x="K", y="Inercia", markers=True, template=PLANTILLA,
                  title="Método del codo (inercia intra-clúster)")
    fig.add_vline(x=k_sel, line_dash="dot", line_color="#d1495b", annotation_text=f"K = {k_sel}")
    fig.update_layout(yaxis_title="Suma de distancias² al centroide")
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.line(evaluacion, x="K", y="Silueta", markers=True, template=PLANTILLA,
                  title="Coeficiente de silueta promedio")
    fig.add_vline(x=k_sel, line_dash="dot", line_color="#d1495b", annotation_text=f"K = {k_sel}")
    fig.update_layout(yaxis_title="Silueta (−1 a 1; más alto = grupos más separados)")
    c2.plotly_chart(fig, use_container_width=True)

    k_mejor = int(evaluacion.loc[evaluacion["Silueta"].idxmax(), "K"])
    sil_k = evaluacion.set_index("K").loc[k_sel, "Silueta"]
    nota(
        f"La silueta más alta se obtiene con <b>K = {k_mejor}</b> "
        f"({evaluacion['Silueta'].max():.3f}); con el K elegido (<b>{k_sel}</b>) es "
        f"<b>{sil_k:.3f}</b>. Valores entre 0,25 y 0,35 indican una estructura de grupos "
        "<b>débil a moderada</b>: hay segmentos reconocibles, pero con fronteras difusas. "
        "<br><b>Justificación de K = 5 (configuración base):</b> K = 2 maximiza la silueta, "
        "pero solo separa empresas de valor y de crecimiento (algo que ya muestra CP1). "
        "La inercia cae con fuerza hasta K = 5 y luego se aplana (codo), y K = 5 tiene la "
        "segunda silueta más alta, mayor que K = 3 y K = 4. Además, cada grupo de K = 5 "
        "tiene una lectura económica clara, incluido un grupo pequeño de empresas con pérdidas."
    )

    st.subheader(f"Visualización de los {k_sel} grupos")
    t1, t2 = st.tabs(["Plano CP1–CP2", "Espacio 3D (CP1–CP3)"])
    hover = ["Name", "Symbol", "Sector GICS"]
    with t1:
        info = varianza["Varianza explicada"].iloc[:2].sum()
        fig = px.scatter(
            df, x="CP1", y="CP2", color="Grupo", hover_data=hover, template=PLANTILLA,
            color_discrete_map=colores_grupo, category_orders={"Grupo": grupos},
            title=f"Grupos en el plano CP1–CP2 (resume el {pct(info)} de la variación)",
            labels={"CP1": "CP1 · valor (+) vs crecimiento (−)", "CP2": "CP2 · volatilidad / castigo"},
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        if "CP3" in df:
            fig = px.scatter_3d(
                df, x="CP1", y="CP2", z="CP3", color="Grupo", hover_data=hover,
                template=PLANTILLA, color_discrete_map=colores_grupo,
                category_orders={"Grupo": grupos}, opacity=0.8,
                title=f"Resume el {pct(varianza['Varianza acumulada'].iloc[2])} de la variación",
            )
            fig.update_traces(marker_size=4)
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Las letras son solo nombres (A = grupo más numeroso). No indican orden, calidad ni jerarquía."
    )

    st.subheader("Silueta por empresa: ¿qué tan bien asignada está cada una?")
    ds = df.sort_values(["Grupo", "Silueta"], ascending=[True, False]).reset_index(drop=True)
    fig = px.bar(ds, x=ds.index, y="Silueta", color="Grupo", hover_data=["Name"],
                 color_discrete_map=colores_grupo, template=PLANTILLA,
                 title="Coeficiente de silueta de cada empresa, ordenado dentro de su grupo")
    fig.update_layout(bargap=0, xaxis_title="Empresas", xaxis_showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)
    n_neg = int((df["Silueta"] < 0).sum())
    nota(
        f"<b>{n_neg}</b> empresas ({pct(n_neg / len(df))}) tienen silueta negativa: están más "
        "cerca de otro grupo que del propio. Son casos de frontera, difíciles de clasificar."
    )

    st.subheader("Descripción de los grupos con las variables originales")
    st.markdown("**Medianas por grupo** (la mediana es robusta a los atípicos de estas razones):")
    st.dataframe(
        perfil["medianas"].style.format(precision=2).format({"Dividend Yield": "{:.2%}"}),
        use_container_width=True,
    )

    fig = px.imshow(
        perfil["perfil_z"].rename(columns=m.ETIQUETAS_CORTAS), text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto", template=PLANTILLA,
        title="Perfil de cada grupo: promedio en desviaciones estándar respecto al total del S&P 500",
    )
    st.plotly_chart(fig, use_container_width=True)

    resumen = pd.DataFrame(
        {
            "Empresas": perfil["medianas"]["Empresas"],
            "Silueta media": df.groupby("Grupo")["Silueta"].mean().round(2),
            "Rasgos más marcados": [m.describir_grupo(f) for _, f in perfil["perfil_z"].iterrows()],
            "Sectores principales": [
                ", ".join(f"{s} ({v:.0f} %)" for s, v in fila.sort_values(ascending=False).head(3).items())
                for _, fila in perfil["sectores"].iterrows()
            ],
            "Empresas representativas (mayor capitalización)": [
                ", ".join(df[df["Grupo"] == g].nlargest(5, "Market Cap")["Name"]) for g in grupos
            ],
        }
    )
    st.dataframe(resumen, use_container_width=True)

    sect = perfil["sectores"].reset_index().melt(id_vars="Grupo", var_name="Sector", value_name="%")
    fig = px.bar(sect, x="Grupo", y="%", color="Sector", template=PLANTILLA,
                 color_discrete_sequence=px.colors.qualitative.Bold,
                 title="Composición sectorial de cada grupo (el sector no entra al modelo)")
    fig.update_layout(yaxis_title="% de empresas del grupo")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparar una variable original entre grupos")
    var_box = st.selectbox("Variable", m.VARIABLES_ORIGINALES + ["Price"])
    log_y = st.checkbox("Escala logarítmica", value=var_box in ("Market Cap", "EBITDA", "Price"))
    datos_box = df if not log_y else df[df[var_box] > 0]
    fig = px.box(datos_box, x="Grupo", y=var_box, color="Grupo", points="outliers",
                 hover_data=["Name"], color_discrete_map=colores_grupo,
                 category_orders={"Grupo": grupos}, template=PLANTILLA, log_y=log_y,
                 title=f"Distribución de {var_box} por grupo")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Buscar empresas")
    g_sel = st.multiselect("Filtrar por grupo", grupos, default=grupos)
    tabla = df[df["Grupo"].isin(g_sel)][
        ["Symbol", "Name", "Sector GICS", "Subindustria", "Grupo", "Silueta"] + m.VARIABLES_ORIGINALES
    ]
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar empresas con su grupo (CSV)",
        df.drop(columns="SEC Filings").to_csv(index=False).encode("utf-8"),
        "sp500_grupos.csv", "text/csv",
    )


# ===========================================================================
# 5. CONCLUSIONES
# ===========================================================================
else:
    st.title("Conclusiones y discusión de resultados")

    # La discusión se escribe sobre la CONFIGURACIÓN BASE, que se recalcula aquí
    # para que el texto coincida siempre con los números, sin importar lo que el
    # usuario haya movido en la barra lateral.
    B = correr_pipeline(tuple(CONFIG_BASE["variables"]), CONFIG_BASE["winsor"])
    datos_b = B["puntajes"][:, : CONFIG_BASE["n_comp"]]
    ev_b = correr_evaluar_k(datos_b, 10)
    et_b, sil_b = correr_kmeans(datos_b, CONFIG_BASE["k"])
    db = B["limpio"].copy()
    db["Grupo"] = et_b
    db["Silueta"] = sil_b
    pb = m.perfil_clusters(db, B["Xt"])
    var_b = B["varianza"]

    def grupo_de(simbolo: str) -> str:
        return db.loc[db["Symbol"] == simbolo, "Grupo"].iloc[0]

    def n_de(simbolo: str) -> int:
        return int((db["Grupo"] == grupo_de(simbolo)).sum())

    def med(simbolo: str, col: str) -> float:
        return pb["medianas"].loc[grupo_de(simbolo), col]

    actual = (sorted(variables_sel) == sorted(CONFIG_BASE["variables"]) and winsor == CONFIG_BASE["winsor"]
              and n_comp == CONFIG_BASE["n_comp"] and k_sel == CONFIG_BASE["k"]
              and base_km == "Componentes seleccionados")
    if not actual:
        st.info(
            "Esta discusión corresponde a la **configuración base** (8 variables, 1 % de "
            "winsorización, 3 componentes, K = 5). Tu selección actual en la barra lateral es "
            "distinta, así que los grupos de la sección 4 pueden no coincidir con este texto."
        )

    st.subheader("1. ¿Qué variables definen los componentes y cuánta información se conserva?")
    st.markdown(
        f"""
        Los tres primeros componentes conservan el **{pct(var_b['Varianza acumulada'].iloc[2])}**
        de la variación de las 8 variables estandarizadas (CP1: {pct(var_b['Varianza explicada'].iloc[0])},
        CP2: {pct(var_b['Varianza explicada'].iloc[1])}, CP3: {pct(var_b['Varianza explicada'].iloc[2])}).
        Se eligieron tres porque son los únicos con valor propio mayor que 1 y porque
        tienen una lectura financiera clara:

        - **CP1 — valor frente a crecimiento.** Es la dimensión que más diferencia a las
          empresas del índice: de un lado, alto EBITDA/Capitalización, alto Libro/Precio
          y mayores dividendos; del otro, P/Ventas alto y gran tamaño. Es la conocida
          distinción entre acciones *value* y *growth*.
        - **CP2 — volatilidad y castigo reciente.** Amplitud del rango de 52 semanas frente
          a la cercanía del precio a su máximo y al dividendo.
        - **CP3 — rentabilidad contable.** Separa a las empresas con pérdidas.

        Se renunció a cerca de un tercio de la información, repartida en cinco componentes
        menores. El umbral del 80 % habría exigido cinco componentes, pero los dos adicionales
        tienen valores propios menores que 1 (aportan menos que una sola variable original).
        """
    )

    st.subheader("2. ¿Qué grupos aparecen y en qué se diferencian?")
    st.markdown(
        f"""
        Con **K = 5** aparecen cinco perfiles (letras asignadas por tamaño, sin orden implícito):

        - **{grupo_de('AAPL')} — Crecimiento de gran capitalización ({n_de('AAPL')} empresas).**
          Apple, Nvidia, Microsoft, Alphabet, Amazon. Mediana de P/E de
          {med('AAPL', 'Price/Earnings'):.1f} y de P/Ventas de {med('AAPL', 'Price/Sales'):.1f}, dividendos
          bajos ({med('AAPL', 'Dividend Yield'):.1%}) y precios cerca de sus máximos. El mercado paga
          por expectativas de crecimiento. Dominan tecnología, salud e industriales.
        - **{grupo_de('XOM')} — Valor estable con dividendo ({n_de('XOM')} empresas).**
          ExxonMobil, Chevron, los grandes bancos, servicios públicos. Múltiplos moderados
          (P/E {med('XOM', 'Price/Earnings'):.1f}, P/Libro {med('XOM', 'Price/Book'):.1f}), dividendo de
          {med('XOM', 'Dividend Yield'):.1%} y baja volatilidad: perfil defensivo.
        - **{grupo_de('MU')} — Alta volatilidad, lejos de máximos ({n_de('MU')} empresas).**
          Micron, Oracle, Intel, fabricantes de equipos para semiconductores, Caterpillar.
          Se distingue más por el comportamiento del precio que por la valoración:
          rangos de 52 semanas muy amplios y poco o ningún dividendo. Muchas son cíclicas.
        - **{grupo_de('T')} — Valor profundo ({n_de('T')} empresas).**
          AT&T, Comcast, CVS, refinadoras (Valero, Phillips 66). P/Ventas de apenas
          {med('T', 'Price/Sales'):.2f} y P/E de {med('T', 'Price/Earnings'):.1f}: generan mucho EBITDA por
          cada dólar de capitalización, pero el mercado las valora poco (negocios maduros o
          con dudas sobre su crecimiento).
        - **{grupo_de('F')} — Empresas con pérdidas ({n_de('F')} empresas).**
          Ford, Kraft Heinz, Dow, General Mills. Utilidad por acción negativa, pero dividendos
          altos ({med('F', 'Dividend Yield'):.1%}) y precios bajos frente a sus activos. Es el grupo
          mejor separado (silueta media {db.loc[db['Grupo'] == grupo_de('F'), 'Silueta'].mean():.2f}),
          aunque es pequeño y depende de una sola variable.
        """
    )

    st.subheader("3. ¿Son grupos claramente distinguibles?")
    sil_media = ev_b.set_index("K").loc[5, "Silueta"]
    n_neg_b = int((db["Silueta"] < 0).sum())
    st.markdown(
        f"""
        Solo en parte. La silueta promedio es **{sil_media:.2f}** y entre K = 2 y K = 6 se
        mueve en un rango estrecho (0,29–0,33): las empresas del S&P 500 forman más un
        **continuo** que islas separadas. {n_neg_b} empresas tienen silueta negativa y el grupo
        de valor profundo es el menos compacto (silueta media
        {db.loc[db['Grupo'] == grupo_de('T'), 'Silueta'].mean():.2f}): muchas de sus empresas están en
        la frontera con el grupo de valor estable. Los grupos son útiles como **tipologías**
        para describir estilos de inversión, no como categorías tajantes.
        """
    )

    st.subheader("4. ¿Coinciden los grupos con los sectores?")
    st.markdown(
        """
        No del todo, y ese es el hallazgo más interesante. Hay sectores que se concentran
        (casi toda la tecnología está en los grupos de crecimiento y de alta volatilidad; los
        servicios públicos y los bancos, en valor estable). Pero ningún grupo es un sector:
        por ejemplo, empresas de semiconductores aparecen tanto entre las de crecimiento
        (Nvidia, Broadcom) como entre las volátiles (Micron, Intel). Para un inversionista,
        esto sugiere que **diversificar solo por sector no garantiza diversificar por estilo
        de valoración**.
        """
    )

    st.subheader("5. Limitaciones para usar estos resultados")
    st.markdown(
        """
        - **Fotografía de un día:** los grupos dependen de precios de mercado que cambian a
          diario; con otra fecha de descarga, empresas de frontera cambiarían de grupo. Sin
          datos históricos no se puede comprobar si los grupos son estables.
        - **Imputación del EBITDA en financieras:** los bancos quedan en el centro de esa
          variable por construcción, lo que influye en que caigan en el grupo de valor estable.
        - **Variables limitadas:** sin deuda, márgenes ni crecimiento de ventas, la
          segmentación captura sobre todo cómo **valora el mercado** a la empresa, no la
          calidad de su negocio.
        - **K-Means asume grupos esféricos y de tamaño similar**, algo que no se cumple bien
          aquí (un grupo de 10 empresas frente a uno de más de 170).
        - Los resultados son **descriptivos**: no constituyen una recomendación de inversión.
        """
    )

    st.subheader("Datos clave de la configuración base")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas", len(db))
    c2.metric("Varianza conservada (3 CP)", pct(var_b["Varianza acumulada"].iloc[2]))
    c3.metric("Silueta (K = 5)", f"{sil_media:.3f}")
    c4.metric("Empresas con silueta < 0", n_neg_b)
