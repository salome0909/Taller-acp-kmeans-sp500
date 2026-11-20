# Perfiles financieros en el S&P 500: ACP y K-Means

**Autora:** Salome Reyes

Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali

**Aplicación desplegada:** https://taller-acp-kmeans-salome.streamlit.app/
**Video de sustentación:** https://youtu.be/m0DWA6XIPI4

## Pregunta
¿Existen grupos de empresas del S&P 500 con perfiles similares de tamaño, valoración,
rentabilidad, dividendos y comportamiento reciente del precio? ¿Coinciden con la
clasificación sectorial tradicional?

## Datos
- **Fuente:** `constituents-financials.csv`, conjunto *S&P 500 Companies with Financial Information*
  de DataHub (https://datahub.io/core/s-and-p-500-companies-financials).
- **Unidad de observación:** una empresa del índice. 503 registros en la fuente, 482 analizados.
- **Corte transversal:** una sola fecha. No hay series históricas.
- **Limitaciones:** razones de mercado que cambian a diario, P/E indefinido con pérdidas, EBITDA
  no reportado para bancos, patrimonio negativo en algunas empresas, sin variables de deuda ni crecimiento.

## Metodología
1. **Limpieza:** se eliminan 17 empresas sin datos, 3 clases secundarias de acciones (GOOG, FOX, NWS)
   y 1 registro inconsistente (PARA).
2. **Variables (8):** log de capitalización, rendimiento de utilidades (UPA/Precio), rendimiento por
   dividendo, log de Precio/Ventas, Libro/Precio, EBITDA/Capitalización, posición y amplitud del rango
   de 52 semanas. Se usan los inversos del P/E y del P/B para no perder empresas con pérdidas o con patrimonio negativo.
3. **Tratamiento:** EBITDA faltante imputado con la mediana; winsorización al 1 % y 99 %; `StandardScaler`.
4. **ACP:** se conservan 3 componentes (criterio de Kaiser), con el 64,8 % de la varianza.
   CP1 = valor vs. crecimiento, CP2 = volatilidad/castigo reciente, CP3 = rentabilidad contable.
5. **K-Means** sobre los 3 componentes, comparando K = 2…10 con el codo y la silueta. Se elige K = 5.

## Resultados principales
Cinco perfiles: crecimiento de gran capitalización, valor estable con dividendo, alta volatilidad lejos
de máximos, valor profundo y empresas con pérdidas. La silueta (≈ 0,32) indica una estructura moderada:
los grupos son tipologías útiles, no categorías tajantes. Los grupos no reproducen los sectores. Por
ejemplo, las empresas de semiconductores se reparten entre crecimiento y alta volatilidad.

## Estructura del repositorio
```
├── app.py                       # Aplicación Streamlit (5 secciones)
├── modelo.py                    # Procesamiento, ACP y K-Means (lo usa app.py)
├── constituents-financials.csv  # Datos
├── requirements.txt
└── README.md
```

## Cómo ejecutar
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python modelo.py        # validación en consola; guarda tablas en /salidas
streamlit run app.py    # abre la app en http://localhost:8501
```
