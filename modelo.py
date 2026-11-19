"""
modelo.py — Procesamiento y modelado (ACP + K-Means) para el S&P 500.
Autora: Salome Reyes · Machine Learning aplicado a Economía y Finanzas · PUJ Cali

Este módulo contiene TODA la lógica del análisis. Se usa de dos formas:
  1) Como script de validación:   python modelo.py
     (imprime el diagnóstico, la varianza del ACP, las cargas, el codo/silueta
      y el perfil de los clústeres, y guarda resultados en la carpeta "salidas").
  2) Como librería: app.py importa estas mismas funciones, de modo que la app
     y el script producen exactamente los mismos números.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
RUTA_DATOS = CARPETA / "constituents-financials.csv"
SEMILLA = 42

# Clases secundarias de acciones: misma empresa, mismos fundamentales.
# Se conserva solo una clase para no contar dos veces a la misma compañía.
CLASES_SECUNDARIAS = ["GOOG", "FOX", "NWS"]

# Variables cuantitativas construidas para el modelo (nombre -> descripción).
# Todas son razones o están en logaritmo, para que no dependan del número de
# acciones ni del tamaño absoluto de la empresa (salvo la de tamaño, a propósito).
VARIABLES_MODELO = {
    "log_cap": "Tamaño: log10 de la capitalización bursátil (USD)",
    "rend_utilidades": "Rendimiento de utilidades = UPA / Precio (inverso del P/E)",
    "rend_dividendo": "Rendimiento por dividendo",
    "log_p_ventas": "log10 de Precio / Ventas",
    "libro_precio": "Valor en libros / Precio (inverso del P/B)",
    "rend_ebitda": "EBITDA / Capitalización bursátil",
    "pos_52s": "Posición del precio en su rango de 52 semanas (0 = mínimo, 1 = máximo)",
    "amplitud_52s": "Amplitud del rango de 52 semanas: (Máx − Mín) / Máx",
}

ETIQUETAS_CORTAS = {
    "log_cap": "Tamaño (log cap.)",
    "rend_utilidades": "Rend. utilidades (E/P)",
    "rend_dividendo": "Rend. dividendo",
    "log_p_ventas": "log P/Ventas",
    "libro_precio": "Libro/Precio (B/P)",
    "rend_ebitda": "EBITDA/Cap.",
    "pos_52s": "Posición 52 sem.",
    "amplitud_52s": "Amplitud 52 sem.",
}

# Variables originales de la fuente con las que se describen los clústeres.
VARIABLES_ORIGINALES = [
    "Price/Earnings",
    "Dividend Yield",
    "Market Cap",
    "EBITDA",
    "Price/Sales",
    "Price/Book",
    "Earnings/Share",
]

# La columna "Sector" de la fuente es en realidad la SUBINDUSTRIA GICS (125
# categorías). Para describir los clústeres se agrupa en los 11 sectores GICS.
_SECTORES_GICS = {
    "Comunicaciones": [
        "Advertising", "Broadcasting", "Cable & Satellite",
        "Integrated Telecommunication Services", "Interactive Home Entertainment",
        "Interactive Media & Services", "Movies & Entertainment", "Publishing",
        "Wireless Telecommunication Services",
    ],
    "Consumo discrecional": [
        "Apparel Retail", "Apparel, Accessories & Luxury Goods",
        "Automobile Manufacturers", "Automotive Parts & Equipment", "Automotive Retail",
        "Broadline Retail", "Casinos & Gaming", "Computer & Electronics Retail",
        "Consumer Electronics", "Distributors", "Footwear", "Home Furnishings",
        "Home Improvement Retail", "Homebuilding", "Hotels, Resorts & Cruise Lines",
        "Leisure Products", "Other Specialty Retail", "Restaurants",
    ],
    "Consumo básico": [
        "Agricultural Products & Services", "Brewers",
        "Consumer Staples Merchandise Retail", "Distillers & Vintners",
        "Drug Retail", "Food Distributors", "Food Retail", "Household Products",
        "Packaged Foods & Meats", "Personal Care Products",
        "Soft Drinks & Non-alcoholic Beverages", "Tobacco",
    ],
    "Energía": [
        "Integrated Oil & Gas", "Oil & Gas Equipment & Services",
        "Oil & Gas Exploration & Production", "Oil & Gas Refining & Marketing",
        "Oil & Gas Storage & Transportation",
    ],
    "Financiero": [
        "Asset Management & Custody Banks", "Consumer Finance", "Diversified Banks",
        "Financial Exchanges & Data", "Insurance Brokers",
        "Investment Banking & Brokerage", "Life & Health Insurance",
        "Multi-line Insurance", "Multi-Sector Holdings",
        "Property & Casualty Insurance", "Regional Banks", "Reinsurance",
        "Transaction & Payment Processing Services",
    ],
    "Salud": [
        "Biotechnology", "Health Care Distributors", "Health Care Equipment",
        "Health Care Facilities", "Health Care Services", "Health Care Supplies",
        "Health Care Technology", "Life Sciences Tools & Services",
        "Managed Health Care", "Pharmaceuticals",
    ],
    "Industrial": [
        "Aerospace & Defense", "Agricultural & Farm Machinery",
        "Air Freight & Logistics", "Building Products", "Cargo Ground Transportation",
        "Construction & Engineering",
        "Construction Machinery & Heavy Transportation Equipment",
        "Data Processing & Outsourced Services", "Diversified Support Services",
        "Electrical Components & Equipment", "Environmental & Facilities Services",
        "Heavy Electrical Equipment", "Human Resource & Employment Services",
        "Industrial Conglomerates", "Industrial Machinery & Supplies & Components",
        "Passenger Airlines", "Passenger Ground Transportation", "Rail Transportation",
        "Research & Consulting Services", "Trading Companies & Distributors",
    ],
    "Tecnología": [
        "Application Software", "Communications Equipment", "Electronic Components",
        "Electronic Equipment & Instruments", "Electronic Manufacturing Services",
        "IT Consulting & Other Services", "Internet Services & Infrastructure",
        "Semiconductor Materials & Equipment", "Semiconductors", "Systems Software",
        "Technology Distributors", "Technology Hardware, Storage & Peripherals",
    ],
    "Materiales": [
        "Commodity Chemicals", "Construction Materials", "Copper",
        "Fertilizers & Agricultural Chemicals", "Gold", "Industrial Gases",
        "Metal, Glass & Plastic Containers",
        "Paper & Plastic Packaging Products & Materials", "Specialty Chemicals", "Steel",
    ],
    "Servicios públicos": [
        "Electric Utilities", "Gas Utilities",
        "Independent Power Producers & Energy Traders", "Multi-Utilities",
        "Water Utilities",
    ],
    "Bienes raíces": ["Real Estate Services"],
}
MAPA_SECTOR = {sub: sec for sec, subs in _SECTORES_GICS.items() for sub in subs}


def sector_gics(subindustria: str) -> str:
    """Devuelve el sector GICS (11 categorías) a partir de la subindustria."""
    if "REIT" in subindustria:
        return "Bienes raíces"
    return MAPA_SECTOR.get(subindustria, "Otro")


# ---------------------------------------------------------------------------
# 1. Carga y diagnóstico
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str | Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV original sin modificarlo."""
    return pd.read_csv(ruta)


def diagnostico(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con tipo de dato, faltantes y valores únicos de cada columna."""
    return pd.DataFrame(
        {
            "Tipo": df.dtypes.astype(str),
            "Faltantes": df.isna().sum(),
            "% faltantes": (df.isna().mean() * 100).round(1),
            "Valores únicos": df.nunique(),
        }
    )


# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aplica las decisiones de calidad y devuelve (datos limpios, bitácora)."""
    bitacora: list[str] = []
    d = df.copy()
    n0 = len(d)

    # a) Filas sin ningún dato financiero (empresas que salieron del índice).
    vacias = d["Price"].isna()
    d = d.loc[~vacias]
    bitacora.append(
        f"Se eliminaron {int(vacias.sum())} empresas sin ningún dato financiero "
        f"(p. ej. {', '.join(df.loc[vacias, 'Symbol'].head(5))})."
    )

    # b) Duplicados exactos y clases secundarias de acciones.
    dup = d.duplicated(subset="Symbol")
    d = d.loc[~dup]
    sec = d["Symbol"].isin(CLASES_SECUNDARIAS)
    d = d.loc[~sec]
    bitacora.append(
        f"Se eliminaron {int(dup.sum())} símbolos duplicados y {int(sec.sum())} clases "
        f"secundarias de acciones ({', '.join(CLASES_SECUNDARIAS)}), porque repiten los "
        "fundamentales de la misma empresa."
    )

    # c) Registros inconsistentes (precio o capitalización imposibles).
    anom = (d["Price"] < 1) | (d["Market Cap"] < 1e8)
    if anom.any():
        bitacora.append(
            "Se eliminaron registros con precio < 1 USD o capitalización < 100 millones "
            f"USD, incompatibles con una empresa del S&P 500: "
            f"{', '.join(d.loc[anom, 'Symbol'])}."
        )
    d = d.loc[~anom]

    d = d.reset_index(drop=True)
    d.insert(3, "Sector GICS", d["Sector"].map(sector_gics))
    d = d.rename(columns={"Sector": "Subindustria"})
    bitacora.append(f"Observaciones: {n0} en la fuente → {len(d)} empresas analizadas.")
    return d, bitacora


# ---------------------------------------------------------------------------
# 3. Construcción de variables para el modelo
# ---------------------------------------------------------------------------
def construir_variables(d: pd.DataFrame) -> pd.DataFrame:
    """Transforma las columnas originales en variables comparables entre empresas.

    - El P/E falta cuando la utilidad es negativa: se usa su inverso (E/P = UPA/Precio),
      que sí está definido y conserva la información de las pérdidas.
    - El P/B es negativo con patrimonio negativo (recompras): su inverso (B/P) es
      continuo y no explota cerca de cero.
    - Capitalización y P/Ventas son muy asimétricas: se usan en log10.
    """
    X = pd.DataFrame(index=d.index)
    X["log_cap"] = np.log10(d["Market Cap"])
    X["rend_utilidades"] = d["Earnings/Share"] / d["Price"]
    X["rend_dividendo"] = d["Dividend Yield"]
    X["log_p_ventas"] = np.log10(d["Price/Sales"])
    X["libro_precio"] = 1 / d["Price/Book"]
    X["rend_ebitda"] = d["EBITDA"] / d["Market Cap"]
    rango = d["52 Week High"] - d["52 Week Low"]
    X["pos_52s"] = ((d["Price"] - d["52 Week Low"]) / rango).clip(0, 1)
    X["amplitud_52s"] = rango / d["52 Week High"]
    return X


def tratar_faltantes_y_atipicos(
    X: pd.DataFrame, winsor: float = 0.01
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa faltantes con la mediana y winsoriza en los percentiles [winsor, 1-winsor].

    Devuelve (X tratada, resumen de lo que se hizo por variable).
    """
    Xt = X.copy()
    resumen = []
    for c in Xt.columns:
        n_na = int(Xt[c].isna().sum())
        med = Xt[c].median()
        Xt[c] = Xt[c].fillna(med)
        if winsor > 0:
            lo, hi = Xt[c].quantile([winsor, 1 - winsor])
            n_rec = int(((Xt[c] < lo) | (Xt[c] > hi)).sum())
            Xt[c] = Xt[c].clip(lo, hi)
        else:
            n_rec = 0
        resumen.append(
            {"Variable": c, "Faltantes imputados (mediana)": n_na, "Valores recortados": n_rec}
        )
    return Xt, pd.DataFrame(resumen).set_index("Variable")


def estandarizar(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Media 0 y desviación 1: necesario porque ACP y K-Means dependen de la escala."""
    escalador = StandardScaler()
    return escalador.fit_transform(X), escalador


# ---------------------------------------------------------------------------
# 4. ACP
# ---------------------------------------------------------------------------
def ajustar_acp(Z: np.ndarray, nombres: list[str]):
    """ACP completo. Devuelve (modelo, puntajes, tabla de varianza, cargas)."""
    pca = PCA(random_state=SEMILLA).fit(Z)
    puntajes = pca.transform(Z)
    comps = [f"CP{i + 1}" for i in range(pca.n_components_)]
    varianza = pd.DataFrame(
        {
            "Componente": comps,
            "Valor propio": pca.explained_variance_,
            "Varianza explicada": pca.explained_variance_ratio_,
            "Varianza acumulada": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    # Cargas = vectores propios × raíz del valor propio = correlación variable–componente.
    cargas = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=nombres,
        columns=comps,
    )
    return pca, puntajes, varianza, cargas


def sugerir_componentes(varianza: pd.DataFrame, umbral: float = 0.80) -> dict:
    """Dos criterios: Kaiser (valor propio > 1) y varianza acumulada ≥ umbral."""
    kaiser = int((varianza["Valor propio"] > 1).sum())
    por_umbral = int(np.argmax(varianza["Varianza acumulada"].to_numpy() >= umbral) + 1)
    return {"kaiser": max(kaiser, 2), "umbral": por_umbral}


# ---------------------------------------------------------------------------
# 5. K-Means
# ---------------------------------------------------------------------------
def evaluar_k(datos: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    """Inercia (método del codo) y coeficiente de silueta para cada K."""
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(datos)
        filas.append(
            {"K": k, "Inercia": km.inertia_, "Silueta": silhouette_score(datos, km.labels_)}
        )
    return pd.DataFrame(filas)


def ajustar_kmeans(datos: np.ndarray, k: int):
    """Ajusta K-Means y devuelve (modelo, etiquetas 'Grupo A, B, ...', silueta por empresa).

    Las etiquetas se renombran con letras para recalcar que el número del clúster
    no implica ningún orden ni jerarquía.
    """
    km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(datos)
    # Las letras se asignan por tamaño (A = grupo más numeroso). Es solo una
    # convención de nombre: hace que las etiquetas no cambien entre versiones
    # de scikit-learn, pero no significa que un grupo sea "mejor" que otro.
    orden = pd.Series(km.labels_).value_counts().index
    letras = {lab: f"Grupo {chr(65 + i)}" for i, lab in enumerate(orden)}
    etiquetas = pd.Series(km.labels_).map(letras)
    sil = silhouette_samples(datos, km.labels_)
    return km, etiquetas, sil


def perfil_clusters(d: pd.DataFrame, X: pd.DataFrame, col: str = "Grupo") -> dict:
    """Describe los clústeres con las variables ORIGINALES y las construidas."""
    g = d.groupby(col)
    tamano = g.size().rename("Empresas")
    medianas_orig = g[VARIABLES_ORIGINALES].median()
    medianas_orig["Market Cap"] = medianas_orig["Market Cap"] / 1e9
    medianas_orig["EBITDA"] = medianas_orig["EBITDA"] / 1e9
    medianas_orig = medianas_orig.rename(
        columns={"Market Cap": "Market Cap (miles de M USD)", "EBITDA": "EBITDA (miles de M USD)"}
    )
    medianas_orig.insert(0, "Empresas", tamano)
    # Perfil estandarizado (z) con las variables del modelo: útil para el mapa de calor.
    Z = (X - X.mean()) / X.std()
    perfil_z = Z.groupby(d[col].to_numpy()).mean()
    perfil_z.index.name = col
    # Composición sectorial (% de empresas de cada grupo en cada sector).
    sectores = pd.crosstab(d[col], d["Sector GICS"], normalize="index") * 100
    return {"medianas": medianas_orig, "perfil_z": perfil_z, "sectores": sectores}


def describir_grupo(fila_z: pd.Series, n: int = 3) -> str:
    """Texto automático con los rasgos más marcados de un grupo (en desviaciones estándar)."""
    top = fila_z.reindex(fila_z.abs().sort_values(ascending=False).index)[:n]
    partes = [
        f"{'↑' if v > 0 else '↓'} {ETIQUETAS_CORTAS.get(k, k)} ({v:+.2f} DE)" for k, v in top.items()
    ]
    return ", ".join(partes)


# ---------------------------------------------------------------------------
# Tubería completa
# ---------------------------------------------------------------------------
def pipeline(
    ruta: str | Path = RUTA_DATOS,
    variables: list[str] | None = None,
    winsor: float = 0.01,
):
    """Ejecuta carga → limpieza → variables → tratamiento → estandarización → ACP."""
    crudo = cargar_datos(ruta)
    limpio, bitacora = limpiar(crudo)
    X = construir_variables(limpio)
    if variables:
        X = X[variables]
    Xt, resumen_trat = tratar_faltantes_y_atipicos(X, winsor)
    Z, escalador = estandarizar(Xt)
    pca, puntajes, varianza, cargas = ajustar_acp(Z, list(Xt.columns))
    return {
        "crudo": crudo,
        "limpio": limpio,
        "bitacora": bitacora,
        "X": X,
        "Xt": Xt,
        "tratamiento": resumen_trat,
        "Z": Z,
        "pca": pca,
        "puntajes": puntajes,
        "varianza": varianza,
        "cargas": cargas,
    }


# ---------------------------------------------------------------------------
# Ejecución como script de validación
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    salidas = CARPETA / "salidas"
    salidas.mkdir(exist_ok=True)

    r = pipeline()
    print("=" * 70, "\n1. DIAGNÓSTICO DE LA BASE ORIGINAL\n", "=" * 70, sep="")
    print(f"Dimensiones: {r['crudo'].shape}")
    print(diagnostico(r["crudo"]))
    print("\nDecisiones de limpieza:")
    for linea in r["bitacora"]:
        print(" -", linea)

    print("\n", "=" * 70, "\n2. VARIABLES DEL MODELO\n", "=" * 70, sep="")
    print(r["tratamiento"])
    print(r["Xt"].describe().T.round(3))

    print("\n", "=" * 70, "\n3. ACP\n", "=" * 70, sep="")
    print(r["varianza"].round(3).to_string(index=False))
    sug = sugerir_componentes(r["varianza"])
    print(f"\nComponentes sugeridos → Kaiser: {sug['kaiser']} | ≥80 % varianza: {sug['umbral']}")
    print("\nCargas (correlación variable–componente):")
    print(r["cargas"].iloc[:, :4].round(2))

    n_comp = sug["kaiser"]
    datos_km = r["puntajes"][:, :n_comp]
    print("\n", "=" * 70, f"\n4. K-MEANS sobre {n_comp} componentes\n", "=" * 70, sep="")
    ev = evaluar_k(datos_km)
    print(ev.round(3).to_string(index=False))
    k = int(ev.loc[ev["Silueta"].idxmax(), "K"])
    print(f"\nK con mayor silueta: {k}")

    km, etiquetas, sil = ajustar_kmeans(datos_km, k)
    limpio = r["limpio"].copy()
    limpio["Grupo"] = etiquetas.to_numpy()
    limpio["Silueta"] = sil
    perfil = perfil_clusters(limpio, r["Xt"])
    print("\nMedianas de las variables originales por grupo:")
    print(perfil["medianas"].round(3))
    print("\nRasgos principales de cada grupo:")
    for g, fila in perfil["perfil_z"].iterrows():
        print(f" {g}: {describir_grupo(fila)}")
    print("\nComposición sectorial (%):")
    print(perfil["sectores"].round(0))

    limpio.to_csv(salidas / "empresas_con_grupo.csv", index=False)
    r["varianza"].to_csv(salidas / "varianza_acp.csv", index=False)
    r["cargas"].to_csv(salidas / "cargas_acp.csv")
    ev.to_csv(salidas / "codo_silueta.csv", index=False)
    print(f"\nResultados guardados en: {salidas}")
