"""
Aplicación Principal Streamlit: Dashboard de Logística, Rentabilidad e Integración IA.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Importación de funciones modularizadas de limpieza y carga de datos
from data_cleaner import load_raw_data, load_and_clean, build_joins, detect_iqr_outliers

# Intentar importar cliente de Groq e inicializarlo con el Secret
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Lectura de la API Key desde los Secrets de Streamlit
GROQ_API_KEY = st.secrets.get("groq_api_key", None)

# Intentar importar cliente de Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configuración de página
st.set_page_config(page_title="Dashboard Logístico & Rentabilidad", layout="wide", page_icon="📦")

DATA_DIR = Path(__file__).parent / "data"

# ----------------------------------------------------------------------------
# CARGA Y CACHÉ DE DATOS
# ----------------------------------------------------------------------------

@st.cache_data
def get_cleaned_data():
    return load_and_clean(DATA_DIR)

@st.cache_data
def get_joined_data(t, inv, f):
    return build_joins(t, inv, f)

@st.cache_data
def get_raw_data():
    return load_raw_data(DATA_DIR)

t, inv, f, clean_summary = get_cleaned_data()
tv, tvf = get_joined_data(t, inv, f)
t_raw, inv_raw, f_raw = get_raw_data()

# ----------------------------------------------------------------------------
# SIDEBAR (FILTROS)
# ----------------------------------------------------------------------------

st.sidebar.title("📦 Panel de Control")
st.sidebar.markdown("Filtra los datos para explorar el negocio.")

# 1. Selector de Rango de Fechas
min_date = t["Fecha_Venta"].min().date() if pd.notna(t["Fecha_Venta"].min()) else pd.Timestamp.today().date()
max_date = t["Fecha_Venta"].max().date() if pd.notna(t["Fecha_Venta"].max()) else pd.Timestamp.today().date()

date_range = st.sidebar.date_input(
    "Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 2. Filtro de Categoría
all_categories = sorted([c for c in inv["Categoria"].unique() if pd.notna(c)])
categorias = st.sidebar.multiselect("Categoría de Producto", all_categories, default=all_categories)

# 3. Filtro de Bodega
all_bodegas = sorted([b for b in inv["Bodega_Origen"].unique() if pd.notna(b)])
bodegas = st.sidebar.multiselect("Bodega de Origen", all_bodegas, default=all_bodegas)

# 4. Filtros de Canal y Ciudad
canales = st.sidebar.multiselect("Canal de Venta", sorted(t["Canal_Venta"].unique()), default=list(t["Canal_Venta"].unique()))
ciudades = st.sidebar.multiselect("Ciudad Destino", sorted(t["Ciudad_Destino"].unique()), default=list(t["Ciudad_Destino"].unique()))

st.sidebar.markdown("---")

# Botón de Refrescar Análisis
if st.sidebar.button("🔄 Refrescar Análisis", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.sidebar.expander("🧹 Notas de limpieza de datos", expanded=False):
    st.caption(f"Transacciones originales: {clean_summary['t_raw_n']:,} | Inventario: {clean_summary['i_raw_n']:,} | Feedback: {clean_summary['f_raw_n']:,}")
    for i_msg in clean_summary["issues"]:
        st.markdown(f"- {i_msg}")

# Generación del reporte de limpieza en CSV y botón de descarga en el Sidebar
report_data = [
    {"Tipo": "Métrica", "Detalle": f"Transacciones originales: {clean_summary.get('t_raw_n', 0)}"},
    {"Tipo": "Métrica", "Detalle": f"Inventario original: {clean_summary.get('i_raw_n', 0)}"},
    {"Tipo": "Métrica", "Detalle": f"Feedback original: {clean_summary.get('f_raw_n', 0)}"},
] + [{"Tipo": "Nota de Limpieza", "Detalle": msg} for msg in clean_summary.get("issues", [])]

df_reporte_limpieza = pd.DataFrame(report_data)
csv_reporte = df_reporte_limpieza.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="📥 Descargar Reporte de Limpieza (CSV)",
    data=csv_reporte,
    file_name="reporte_limpieza_datos.csv",
    mime="text/csv",
    use_container_width=True
)

st.sidebar.caption("Dashboard construido a partir de transacciones_logistica_v2, feedback_clientes_v2 e inventario_central_v2.")

# ----------------------------------------------------------------------------
# APLICACIÓN DE FILTROS A LOS DATASETS
# ----------------------------------------------------------------------------

start_d, end_d = (date_range[0], date_range[1]) if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else (min_date, max_date)

# Filtrado en dataset t
mask_t = (
    (t["Canal_Venta"].isin(canales)) &
    (t["Ciudad_Destino"].isin(ciudades)) &
    (t["Fecha_Venta"].dt.date >= start_d) &
    (t["Fecha_Venta"].dt.date <= end_d)
)
t_f = t[mask_t]

# Filtrado en dataset unificado tv y tvf
mask_tv = (
    (tv["Canal_Venta"].isin(canales)) &
    (tv["Ciudad_Destino"].isin(ciudades)) &
    (tv["Fecha_Venta"].dt.date >= start_d) &
    (tv["Fecha_Venta"].dt.date <= end_d) &
    ((tv["Categoria"].isin(categorias)) | (tv["Categoria"].isna())) &
    ((tv["Bodega_Origen"].isin(bodegas)) | (tv["Bodega_Origen"].isna()))
)
tv_f = tv[mask_tv]

mask_tvf = (
    (tvf["Canal_Venta"].isin(canales)) &
    (tvf["Ciudad_Destino"].isin(ciudades)) &
    (tvf["Fecha_Venta"].dt.date >= start_d) &
    (tvf["Fecha_Venta"].dt.date <= end_d) &
    ((tvf["Categoria"].isin(categorias)) | (tvf["Categoria"].isna())) &
    ((tvf["Bodega_Origen"].isin(bodegas)) | (tvf["Bodega_Origen"].isna()))
)
tvf_f = tvf[mask_tvf]

# ----------------------------------------------------------------------------
# HEADER & KPIs
# ----------------------------------------------------------------------------

st.title("📦 Dashboard de Rentabilidad, Logística y Fidelidad")
st.markdown("Análisis integrado de **transacciones**, **inventario** y **feedback de clientes** para diagnosticar fugas de capital, cuellos de botella logísticos y riesgos operativos.")

k1, k2, k3, k4 = st.columns(4)
ingreso_total = tv_f["Ingreso_Total"].sum()
margen_total = tv_f["Margen_Total"].sum()
n_trx = len(tv_f)
pct_sin_inv = (~tv_f["En_Inventario"]).mean() * 100 if n_trx > 0 else 0

k1.metric("Ingreso Total (USD)", f"${ingreso_total:,.0f}")
k2.metric("Margen Total (USD)", f"${margen_total:,.0f}", delta=f"{margen_total/ingreso_total*100:.1f}% del ingreso" if ingreso_total else None)
k3.metric("Transacciones", f"{n_trx:,}")
k4.metric("% Ventas sin match en inventario", f"{pct_sin_inv:.1f}%")

st.markdown("---")

tab0, tab_merge, tab1, tab2, tab3, tab4, tab5, tab_groq = st.tabs([
    "🔍 EDA",
    "🔗 Datos analizados",
    "1️⃣ Fuga de Capital",
    "2️⃣ Crisis Logística",
    "3️⃣ Venta Invisible",
    "4️⃣ Diagnóstico de Fidelidad",
    "5️⃣ Riesgo Operativo",
    "🤖 Diagnóstico IA (Groq)"
])

# ============================================================================
# TAB 0: EDA INTERACTIVO
# ============================================================================
with tab0:
    st.header("EDA Interactivo: Calidad de Datos")
    st.markdown(
        "Diagnóstico del estado **crudo** de los tres datasets antes de cualquier limpieza: nulidad, "
        "duplicados, outliers e integridad referencial (SKUs fantasma)."
    )

    ds_choice = st.selectbox(
        "Selecciona el dataset a explorar",
        ["transacciones_logistica_v2.csv", "inventario_central_v2.csv", "feedback_clientes_v2.csv"],
    )
    ds_map = {
        "transacciones_logistica_v2.csv": t_raw,
        "inventario_central_v2.csv": inv_raw,
        "feedback_clientes_v2.csv": f_raw,
    }
    df_sel = ds_map[ds_choice]

    st.subheader(f"1. Nulidad por columna — {ds_choice}")
    null_pct = (df_sel.isna().mean() * 100).round(2).sort_values(ascending=False)
    null_df = null_pct.reset_index()
    null_df.columns = ["Columna", "% Nulos"]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(null_df.style.format({"% Nulos": "{:.2f}%"}), use_container_width=True, height=350)
    with c2:
        fig = px.bar(null_df, x="% Nulos", y="Columna", orientation="h", color="% Nulos",
                     color_continuous_scale="Reds", title="% de valores nulos por columna")
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Total de registros:** {len(df_sel):,} | **Total de celdas nulas:** {int(df_sel.isna().sum().sum()):,}")

    st.subheader("2. Duplicados detectados")
    dup_rows = df_sel.duplicated().sum()
    id_col = {"transacciones_logistica_v2.csv": "Transaccion_ID", "inventario_central_v2.csv": "SKU_ID", "feedback_clientes_v2.csv": "Transaccion_ID"}[ds_choice]
    dup_id = df_sel[id_col].duplicated().sum() if id_col in df_sel.columns else None
    dc1, dc2 = st.columns(2)
    dc1.metric("Filas 100% duplicadas", f"{dup_rows:,}")
    dc2.metric(f"'{id_col}' duplicados", f"{dup_id:,}" if dup_id is not None else "N/A")

    st.subheader("3. Outliers detectados (método IQR: 1.5×RIC)")
    out_df = detect_iqr_outliers(df_sel)
    if len(out_df):
        st.dataframe(out_df.style.format({"Mínimo": "{:,.2f}", "Máximo": "{:,.2f}", "% Outliers": "{:.2f}%"}), use_container_width=True)
    else:
        st.info("No se detectaron outliers numéricos vía IQR en este dataset.")

# ============================================================================
# TAB MERGE: UNIÓN ESTRATÉGICA Y ANTES VS DESPUÉS
# ============================================================================
with tab_merge:
    st.header("📊 Antes vs. Después: Impacto de la Limpieza")
    st.markdown(
        "Comparación del estado **crudo** de cada dataset contra el estado **limpio** que alimenta el resto del dashboard: "
        "filas eliminadas, duplicados removidos y un puntaje compuesto de salud de datos."
    )

    def health_score(df_raw, df_clean, dup_removed):
        null_pct_raw = df_raw.isna().mean().mean() * 100
        null_pct_clean = df_clean.isna().mean().mean() * 100
        return null_pct_raw, null_pct_clean

    datasets_ba = [
        ("transacciones_logistica_v2.csv", t_raw, t, 0),
        ("inventario_central_v2.csv", inv_raw, inv, 0),
        ("feedback_clientes_v2.csv", f_raw, f, f_raw["Transaccion_ID"].duplicated().sum()),
    ]

    rows_ba = []
    for name, raw_df, clean_df, dups in datasets_ba:
        null_raw, null_clean = health_score(raw_df, clean_df, dups)
        rows_ba.append({
            "Dataset": name,
            "Filas (antes)": len(raw_df),
            "Filas (después)": len(clean_df),
            "Filas eliminadas": len(raw_df) - len(clean_df),
            "Duplicados eliminados": int(dups),
            "% Nulidad promedio (antes)": round(null_raw, 2),
            "% Nulidad promedio (después)": round(null_clean, 2),
            "Salud de datos (después)": round(100 - null_clean, 1),
        })
    ba_df = pd.DataFrame(rows_ba)

    st.dataframe(
        ba_df.style.format({
            "Filas (antes)": "{:,}", "Filas (después)": "{:,}", "Filas eliminadas": "{:,}",
            "Duplicados eliminados": "{:,}",
            "% Nulidad promedio (antes)": "{:.2f}%", "% Nulidad promedio (después)": "{:.2f}%",
            "Salud de datos (después)": "{:.1f}%",
        }),
        use_container_width=True,
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        fig_rows = go.Figure()
        fig_rows.add_bar(name="Antes", x=ba_df["Dataset"], y=ba_df["Filas (antes)"])
        fig_rows.add_bar(name="Después", x=ba_df["Dataset"], y=ba_df["Filas (después)"])
        fig_rows.update_layout(barmode="group", title="Filas: Antes vs. Después de la limpieza")
        st.plotly_chart(fig_rows, use_container_width=True)
    with bc2:
        fig_health = go.Figure()
        fig_health.add_bar(name="% Nulidad antes", x=ba_df["Dataset"], y=ba_df["% Nulidad promedio (antes)"], marker_color="#d62728")
        fig_health.add_bar(name="% Nulidad después", x=ba_df["Dataset"], y=ba_df["% Nulidad promedio (después)"], marker_color="#2ca02c")
        fig_health.update_layout(barmode="group", title="% Nulidad promedio: Antes vs. Después")
        st.plotly_chart(fig_health, use_container_width=True)

    st.info(
        f"**Lectura:** el único dataset donde se eliminan filas es `feedback_clientes_v2.csv`, donde se remueven "
        f"**{int(datasets_ba[2][3]):,} registros duplicados** (misma transacción encuestada más de una vez), pasando de "
        f"{len(f_raw):,} a {len(f):,} filas. En `transacciones` e `inventario` **no se elimina ninguna fila** — los valores "
        "problemáticos se corrigen, imputan o marcan (ver notas de limpieza en la barra lateral), preservando el 100% del "
        "volumen de negocio."
    )

    st.markdown("---")
    st.subheader("⚠️ Dilema del SKU Fantasma: ¿producto nuevo o error de digitación?")
    ghost_df = tv[~tv["En_Inventario"]]
    matched_df = tv[tv["En_Inventario"]]

    d1, d2 = st.columns(2)
    with d1:
        def sku_num(s):
            try:
                return int(str(s).split("-")[-1])
            except ValueError:
                return np.nan
        ghost_nums = ghost_df["SKU_ID"].drop_duplicates().apply(sku_num).dropna()
        matched_nums = inv_raw["SKU_ID"].apply(sku_num).dropna()
        st.metric("Rango numérico SKUs en inventario", f"{int(matched_nums.min())} – {int(matched_nums.max())}")
        st.metric("Rango numérico SKUs fantasma", f"{int(ghost_nums.min())} – {int(ghost_nums.max())}" if len(ghost_nums) else "N/A")
    with d2:
        fig_ghost = px.histogram(
            pd.concat([
                matched_nums.to_frame("SKU_Num").assign(Tipo="En inventario"),
                ghost_nums.to_frame("SKU_Num").assign(Tipo="Fantasma"),
            ]),
            x="SKU_Num", color="Tipo", barmode="overlay", nbins=60,
            title="Distribución del número de SKU: inventario vs. fantasma",
        )
        st.plotly_chart(fig_ghost, use_container_width=True)

    st.markdown(
        "**Decisión tomada:** los SKUs fantasma se tratan como **catálogo no registrado (productos nuevos / de terceros "
        "no dados de alta a tiempo)**, no como error de digitación aleatorio, por dos motivos observables en los datos: "
        "(1) sus códigos `PROD-XXXX` caen en un rango numérico **contiguo y superior** al de los SKUs sí catalogados, "
        "y (2) representan **480 SKUs distintos** con **1,751 transacciones**.\n\n"
        "**Impacto en el cálculo de margen:** la decisión es **excluirlos del margen total** (quedan como `NaN`), "
        "para no inventar rentabilidad donde no hay información de costos."
    )

    st.subheader("🧮 Variables Derivadas (Feature Engineering)")
    st.markdown("""
| # | Variable derivada | Fórmula | Para qué sirve |
|---|---|---|---|
| 1 | **Margen de Utilidad** (`Margen_Total`, `Margen_Pct`) | `(Precio_Venta_Final - Costo_Unitario_USD) * Cantidad_Vendida - Costo_Envio` ; `Margen_Total / Ingreso_Total * 100` | Detectar SKUs y canales con fuga de capital (pestaña 1) |
| 2 | **Brecha de Entrega** (`Dias_Desde_Revision`) | `Fecha_Hoy - Ultima_Revision` (días) | Medir qué tan "a ciegas" opera cada bodega respecto a su inventario (pestaña 5) |
| 3 | **Tasa/Ratio de Soporte** (`Tasa_Tickets`) | `mean(Ticket_Soporte_Abierto)` agrupado por Bodega/Categoría | Cuantificar el ratio de tickets de soporte por bodega (pestaña 5) |
| 4 | **Markup / Sobrecosto** (`Markup_Pct`) | `(Precio_Venta_Final - Costo_Unitario_USD) / Costo_Unitario_USD * 100` | Distinguir sobrecosto de mala calidad (pestaña 4) |
| 5 | **Flag de Venta Invisible** (`En_Inventario`) | `Categoria.notna()` tras el left join | Aislar el ingreso sin control de inventario (pestaña 3) |
""")

    prev_cols = ["Transaccion_ID", "SKU_ID", "En_Inventario", "Margen_Total", "Dias_Desde_Revision"]
    st.dataframe(tv[prev_cols].head(8), use_container_width=True)
    st.caption("Vista previa de la fuente única con variables derivadas ya calculadas (primeras 8 filas).")

# ============================================================================
# TAB 1: FUGA DE CAPITAL
# ============================================================================
with tab1:
    st.header("Fuga de Capital y Rentabilidad")
    st.markdown("Identificación de SKUs vendidos con **margen negativo** y evaluación de si la pérdida es por volumen o por una falla estructural de precios en un canal específico.")

    matched = tv_f[tv_f["En_Inventario"]].copy()

    sku_margin = matched.groupby("SKU_ID").agg(
        Categoria=("Categoria", "first"),
        Unidades_Vendidas=("Cantidad_Vendida", "sum"),
        Ingreso_Total=("Ingreso_Total", "sum"),
        Margen_Total=("Margen_Total", "sum"),
    ).reset_index()
    sku_margin["Margen_Pct"] = np.where(sku_margin["Ingreso_Total"] > 0, sku_margin["Margen_Total"] / sku_margin["Ingreso_Total"] * 100, 0)

    neg_skus = sku_margin[sku_margin["Margen_Total"] < 0].sort_values("Margen_Total")
    pos_skus = sku_margin[sku_margin["Margen_Total"] >= 0]
    
    margen_neto = sku_margin["Margen_Total"].sum()
    perdida_neg = neg_skus["Margen_Total"].sum()
    ganancia_pos = pos_skus["Margen_Total"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs en Pérdida (Margen < 0)", f"{len(neg_skus):,}", f"Acumulan: ${perdida_neg:,.0f} USD", delta_color="inverse")
    c2.metric("SKUs Rentables (Margen ≥ 0)", f"{len(pos_skus):,}", f"Acumulan: ${ganancia_pos:,.0f} USD", delta_color="normal")
    c3.metric("Margen Neto Real", f"${margen_neto:,.0f} USD", "Balance de SKUs controlados", delta_color="off")

    st.markdown("---")
    st.subheader("Desglose por Canal: ¿Es una falla crítica del canal Online?")
    
    channel_analysis = matched.groupby("Canal_Venta").agg(
        Ingreso=("Ingreso_Total", "sum"),
        Margen=("Margen_Total", "sum")
    ).reset_index()
    channel_analysis["Margen_Pct"] = np.where(channel_analysis["Ingreso"] > 0, channel_analysis["Margen"] / channel_analysis["Ingreso"] * 100, 0)
    channel_analysis = channel_analysis.sort_values("Margen")
    
    fig_channel = px.bar(
        channel_analysis, y="Canal_Venta", x="Margen", orientation="h", 
        title="Margen Total por Canal de Venta",
        text=channel_analysis["Margen_Pct"].apply(lambda x: f"{x:.1f}% del ingreso"),
        color="Margen", color_continuous_scale="RdYlGn"
    )
    fig_channel.update_layout(coloraxis_showscale=False, xaxis_title="Margen Total (USD)", yaxis_title="Canal de Venta")
    st.plotly_chart(fig_channel, use_container_width=True)

    st.markdown("---")
    invisible = tv_f[~tv_f["En_Inventario"]]
    st.warning(
        f"🚨 **Riesgo Adicional (SKUs Fantasma):** Existen **{invisible['SKU_ID'].nunique():,} SKUs** sin registro de costos en el inventario que generaron "
        f"**${invisible['Ingreso_Total'].sum():,.0f} USD** en ingresos."
    )

    st.subheader("Top SKUs con mayor pérdida individual")
    st.dataframe(
        neg_skus.head(20)[["SKU_ID", "Categoria", "Unidades_Vendidas", "Ingreso_Total", "Margen_Total", "Margen_Pct"]]
        .style.format({"Ingreso_Total": "${:,.0f}", "Margen_Total": "${:,.0f}", "Margen_Pct": "{:.1f}%"}),
        use_container_width=True,
    )

# ============================================================================
# TAB 2: CRISIS LOGÍSTICA
# ============================================================================
with tab2:
    st.header("Crisis Logística y Cuellos de Botella")
    st.markdown("Diagnóstico operativo de tiempos de entrega, cuellos de botella por bodega y tasas de falla en envíos.")

    log_df = tvf_f.dropna(subset=["Tiempo_Entrega_Valido", "Satisfaccion_NPS"]).copy()

    st.info(
        "💡 **Hallazgo Metodológico:** La correlación lineal de Pearson entre el Tiempo de Entrega y el NPS es nula "
        "($r \\approx 0.0036$, $p > 0.8$). Por ello, los cuellos de botella se diagnostican directamente mediante el **Tiempo Promedio de Entrega** y la **Tasa de Incidencias** (% retrasados / perdidos)."
    )

    fig_scatter = px.scatter(
        log_df, x="Tiempo_Entrega_Valido", y="Satisfaccion_NPS", trendline="ols", opacity=0.25,
        title="Dispersión: Tiempo de Entrega vs. Satisfacción NPS (Línea de tendencia plana: r ≈ 0)",
        labels={"Tiempo_Entrega_Valido": "Tiempo de Entrega (Días)", "Satisfaccion_NPS": "Puntaje NPS"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("Cuellos de Botella Operativos por Bodega y Ciudad")

    bodega_metrics = tvf_f.dropna(subset=["Bodega_Origen"]).groupby("Bodega_Origen").agg(
        Transacciones=("Transaccion_ID", "count"),
        Tiempo_Promedio=("Tiempo_Entrega_Valido", "mean"),
        NPS_Promedio=("Satisfaccion_NPS", "mean"),
        Pct_Retrasados=("Estado_Envio", lambda s: (s == "Retrasado").mean() * 100),
        Pct_Perdidos=("Estado_Envio", lambda s: (s == "Perdido").mean() * 100)
    ).reset_index()

    city_metrics = tvf_f.groupby("Ciudad_Destino").agg(
        Transacciones=("Transaccion_ID", "count"),
        Tiempo_Promedio=("Tiempo_Entrega_Valido", "mean"),
        NPS_Promedio=("Satisfaccion_NPS", "mean"),
        Pct_Retrasados=("Estado_Envio", lambda s: (s == "Retrasado").mean() * 100),
        Pct_Perdidos=("Estado_Envio", lambda s: (s == "Perdido").mean() * 100)
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_bodega = px.bar(
            bodega_metrics, x="Bodega_Origen", y="Pct_Retrasados", color="Pct_Retrasados",
            color_continuous_scale="Reds", title="% de Envíos Retrasados por Bodega de Origen",
            text=bodega_metrics["Pct_Retrasados"].apply(lambda x: f"{x:.1f}%")
        )
        fig_bodega.update_layout(coloraxis_showscale=False, yaxis_title="% Retrasados")
        st.plotly_chart(fig_bodega, use_container_width=True)

    with c2:
        fig_city = px.bar(
            city_metrics, x="Ciudad_Destino", y="Tiempo_Promedio", color="Tiempo_Promedio",
            color_continuous_scale="Oranges", title="Tiempo Promedio de Entrega (Días) por Ciudad",
            text=city_metrics["Tiempo_Promedio"].apply(lambda x: f"{x:.1f} días")
        )
        fig_city.update_layout(coloraxis_showscale=False, yaxis_title="Días Promedio")
        st.plotly_chart(fig_city, use_container_width=True)

    st.subheader("Resumen de Desempeño Logístico por Bodega")
    st.dataframe(
        bodega_metrics.style.format({
            "Transacciones": "{:,}", "Tiempo_Promedio": "{:.1f} días",
            "NPS_Promedio": "{:.1f}", "Pct_Retrasados": "{:.1f}%", "Pct_Perdidos": "{:.1f}%"
        }),
        use_container_width=True
    )

# ============================================================================
# TAB 3: VENTA INVISIBLE
# ============================================================================
with tab3:
    st.header("Análisis de la Venta Invisible")
    st.markdown("Cuantificación del impacto financiero de ventas cuyo SKU **no existe** en el maestro de inventario.")

    invisible = tv_f[~tv_f["En_Inventario"]]
    ingreso_invisible = invisible["Ingreso_Total"].sum()
    ingreso_total_f = tv_f["Ingreso_Total"].sum()
    pct_riesgo = ingreso_invisible / ingreso_total_f * 100 if ingreso_total_f else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Ingreso en riesgo (USD)", f"${ingreso_invisible:,.0f}")
    c2.metric("% del ingreso total en riesgo", f"{pct_riesgo:.1f}%")
    c3.metric("SKUs 'fantasma' distintos", f"{invisible['SKU_ID'].nunique():,}")

    fig = px.pie(
        values=[ingreso_invisible, max(0, ingreso_total_f - ingreso_invisible)],
        names=["Sin match en inventario", "Con match en inventario"],
        title="Ingreso Total: Ventas Controladas vs. Venta Invisible",
        color_discrete_sequence=["#d62728", "#2ca02c"], hole=0.45,
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: DIAGNÓSTICO DE FIDELIDAD
# ============================================================================
with tab4:
    st.header("Diagnóstico de Fidelidad")
    st.markdown("Categorías con **alta disponibilidad de stock** pero **sentimiento negativo del cliente**.")

    cat_df = tvf_f[tvf_f["En_Inventario"]].dropna(subset=["Categoria"])
    if not cat_df.empty:
        cat_summary = cat_df.groupby("Categoria").agg(
            Stock_Promedio=("Stock_Actual", "mean"),
            NPS_Promedio=("Satisfaccion_NPS", "mean"),
            Rating_Producto_Promedio=("Rating_Producto", "mean"),
            Precio_Promedio=("Precio_Venta_Final", "mean"),
            Costo_Promedio=("Costo_Unitario_USD", "mean"),
            Transacciones=("Transaccion_ID", "count"),
        ).reset_index()
        cat_summary["Markup_Pct"] = (cat_summary["Precio_Promedio"] - cat_summary["Costo_Promedio"]) / cat_summary["Costo_Promedio"] * 100

        fig = px.scatter(
            cat_summary, x="Stock_Promedio", y="NPS_Promedio", size="Transacciones", color="Markup_Pct",
            text="Categoria", color_continuous_scale="RdYlGn_r",
            title="Stock promedio vs NPS promedio por Categoría",
            labels={"Stock_Promedio": "Stock Promedio", "NPS_Promedio": "NPS Promedio"},
        )
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("""
        A partir del cruce entre el módulo de **Inventario**, **Transacciones** y **Feedback de Clientes**, se han detectado dos hallazgos críticos que comprometen tanto la retención del cliente como la salud financiera del catálogo en la categoría de **Smartphones**.
        """)
        
        # Métricas destacadas
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Riesgo de Percepción de Calidad",
                value="Rating Polarizado",
                delta="Atención en Smartphones",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Desviación Máxima de Margen",
                value="> -$1,000 USD",
                delta="Venta a Pérdida",
                delta_color="inverse"
            )
        
        st.divider()
        
        # Hallazgo 1: Calidad y Rating
        st.warning("""
        ### 1. Fricción en Fidelidad: Posible Baja Calidad en Smartphones
        * **Observación:** El análisis del feedback vinculado a las transacciones de *Smartphones* muestra valoraciones divididas e insatisfacción recurrente.
        * **Diagnóstico:** Los ratings bajos indican un desajuste severo entre las expectativas del cliente y el rendimiento real del dispositivo. 
        * **Impacto:** Esta insatisfacción directa debilita la fidelidad hacia la plataforma, incrementa la tasa de devoluciones/reclamos y disminuye el *Lifetime Value* (LTV) del usuario.
        """)
        
        # Hallazgo 2: Incongruencia Financiera
        st.error("""
        ### 2. Distorsión Financiera: Venta a Pérdida (> $1,000 USD de Brecha)
        * **Observación:** Al comparar los precios de venta percibidos por el cliente final, estos se mantienen dentro de los rangos habituales de mercado. Sin embargo, al cruzar el precio final con la estructura de costos/inventario, el producto se comercializa a precio de pérdida.
        * **Diagnóstico:** Existe una brecha negativa de **más de $1,000 USD por transacción** entre el costo real y el precio de venta final abonado por el cliente.
        * **Impacto en el Negocio:** Se está **subvencionando la insatisfacción del cliente**: la empresa asume un costo operativo y de margen masivo para colocar un producto que, además, está generando opiniones negativas en la plataforma.
        """)
        
        # Recomendaciones de Acción
        st.markdown("### 📋 Recomendaciones Operativas")
        st.markdown("""
        1. **Auditoría de Proveedores de Smartphones:** Suspender o revisar temporalmente la comercialización de los SKU de Smartphones con ratings bajos para frenar el detrimento en la fidelidad.
        2. **Corrección Sistemática de Precios / Costos:** Investigar la causa raíz de la brecha de $1,000 USD (¿error de catalogación, descuento no autorizado, o costo de adquisición inflado?).
        """)

# ============================================================================
# TAB 5: RIESGO OPERATIVO
# ============================================================================
with tab5:
    st.header("Storytelling de Riesgo Operativo")
    st.markdown("Relación entre la **antigüedad de la última revisión de stock** y la **tasa de tickets de soporte**.")

    op_df = tvf_f[tvf_f["En_Inventario"]].dropna(subset=["Dias_Desde_Revision", "Bodega_Origen"])
    if not op_df.empty:
        bodega_summary = op_df.groupby("Bodega_Origen").agg(
            Dias_Desde_Revision_Prom=("Dias_Desde_Revision", "mean"),
            Tasa_Tickets=("Ticket_Soporte_Abierto", "mean"),
            NPS_Promedio=("Satisfaccion_NPS", "mean"),
            Transacciones=("Transaccion_ID", "count"),
        ).reset_index()
        bodega_summary["Tasa_Tickets_Pct"] = bodega_summary["Tasa_Tickets"] * 100

        fig = px.scatter(
                    bodega_summary, x="Dias_Desde_Revision_Prom", y="Tasa_Tickets_Pct", size="Transacciones", color="NPS_Promedio",
                    text="Bodega_Origen", color_continuous_scale="RdYlGn",
                    title="Antigüedad de revisión vs. Tasa de Tickets por Bodega",
                )
        st.plotly_chart(fig, use_container_width=True)
    # =========================================================
    # SECCIÓN: STORYTELLING DEL RIESGO OPERATIVO
    # =========================================================
    st.header("📦 Storytelling del Riesgo Operativo: Operación a Ciegas y Satisfacción Final")
    
    st.markdown("""
    El **Riesgo Operativo** surge cuando el control físico de los inventarios se descuida.
    Al cruzar los **Días Sin Revisión**, la **Tasa de Tickets de Soporte** y la **Satisfacción del Cliente (NPS)**, 
    revelamos qué bodegas operan a ciegas y cómo esto impacta la experiencia final del usuario.
    """)
    
    # Tarjetas resumidas de KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Bodegas en Riesgo Crítico",
            value="3 Bodegas",
            delta="Occidente, Zona Franca, Norte",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Efecto en Satisfacción",
            value="Colapso de NPS",
            delta="Alta tasa de tickets",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Caso de Mitigación",
            value="BOD-EXT-99",
            delta="Soporte mitiga descontrol",
            delta_color="normal"
        )
    
    st.divider()
    
    # --- DIAGNÓSTICO DETALLADO ---
    
    st.subheader("📊 Análisis Causal: ¿Cómo afectan los días sin revisión al cliente?")
    
    # 1. El Trío Crítico
    st.error("""
    ### 🚨 1. El Trío en Red Zona: Occidente, Zona Franca y Norte
    * **Diagnóstico Operativo:** Estas 3 bodegas registran una combinación crítica de **acumulación extrema de días sin revisión** y las **tasas de tickets de soporte más altas** del sistema.
    * **Impacto en Satisfacción:** La falta de auditoría física genera errores recurrentes (productos defectuosos, stock fantasma o despachos erróneos) que llegan directamente al cliente, destruyendo la satisfacción y desplomando el NPS.
    * **Conclusión:** La ceguera operativa en esta zona no tiene filtro previo; cada falla en bodega se traduce en un reclamo y un cliente insatisfecho.
    """)
    
    # 2. Caso Particular BOD-EXT-99
    st.warning("""
    ### ⚠️ 2. El Caso Singular de BOD-EXT-99: La Red de Carga en Soporte
    * **Diagnóstico Operativo:** Muestra un volumen alto de días sin revisión y genera una cantidad considerable de tickets de soporte (aunque no tan masiva como el trío crítico).
    * **El Comportamiento de la Satisfacción:** A pesar del descontrol en revisiones, la satisfacción no colapsa al nivel catastrófico de las bodegas críticas.
    * **Causa Raíz:** Esta bodega evidencia que un **buen manejo y resolución de tickets de soporte** logra amortiguar el impacto. Las fallas físicas en bodega existen, pero el área de atención al cliente resuelve las incidencias a tiempo para evitar que la percepción del cliente final se destruya por completo.
    """)
    
    # 3. La Regla General del Riesgo
    st.info("""
    ### 💡 3. Conclusión: ¿Días sin revisión = Mala satisfacción?
    
    Los **días sin revisión** representan el **descontrol interno**, mientras que los **tickets** son los síntomas visibles ante el cliente:
    
    1. **Descontrol + Soporte Ineficiente (Occidente, Zona Franca, Norte):** La falla operativa destruye la fidelidad del cliente de forma inmediata.
    2. **Descontrol + Soporte Ágil (BOD-EXT-99):** El descontrol genera sobrecostos operativos por reclamos, pero la gestión rápida de tickets actúa como escudo para proteger el NPS.
    """)
    
    # --- RECOMENDACIONES DE ACCIÓN ---
    st.markdown("### 🛠️ Plan de Mitigación del Riesgo Operativo")
    st.markdown("""
    1. **Auditorías de Emergencia:** Intervenir de inmediato **Occidente, Zona Franca y Norte** con conteos cíclicos semanales para erradicar la causa raíz del volumen de tickets.
    2. **Control Preventivo en BOD-EXT-99:** Realizar auditoría de inventario en **BOD-EXT-99** antes de que el volumen de incidencias supere la capacidad del equipo de soporte para contener la insatisfacción.
    """)
# ============================================================================
# TAB GROQ: DIAGNÓSTICO IA
# ============================================================================
with tab_groq:
    st.header("🤖 Diagnóstico IA (Groq)")
    st.markdown("Genera diagnósticos automáticos y recomendaciones operativas usando Inteligencia Artificial.")

    if not GROQ_AVAILABLE:
        st.error("🚨 La librería `groq` no está instalada. Ejecuta `pip install groq` en tu entorno.")
    elif not GROQ_API_KEY:
        st.warning(
            "⚠️ No se encontró la API Key en los secretos de Streamlit. "
            "Asegúrate de haber creado el secret `groq_api_key` en tu archivo `.streamlit/secrets.toml` "
            "o en la configuración de Streamlit Cloud."
        )
    else:
        # Instanciar cliente con la clave del Secret
        client = Groq(api_key=GROQ_API_KEY)

        st.success("✅ Conexión con Groq configurada exitosamente desde Secrets.")

        # Botón para generar el informe con IA
        if st.button("🚀 Generar Diagnóstico Automático con IA", use_container_width=True):
            with st.spinner("Analizando KPIs y métricas con Groq..."):
                try:
                    # Resumen de contexto para el prompt del LLM
                    prompt_contexto = f"""
                    Eres un consultor experto en logística, inventarios y análisis financiero.
                    Analiza los siguientes KPIs del negocio y genera un diagnóstico estratégico conciso con recomendaciones:

                    - Ingreso Total: ${ingreso_total:,.0f} USD
                    - Margen Total: ${margen_total:,.0f} USD
                    - Transacciones Totales: {n_trx:,}
                    - % Ventas sin match en inventario: {pct_sin_inv:.1f}%
                    """

                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Eres un asistente analítico experto en cadena de suministro y finanzas."},
                            {"role": "user", "content": prompt_contexto}
                        ],
                        model="llama-3.3-70b-versatile",
                    )

                    st.subheader("💡 Resultado del Análisis")
                    st.write(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Error al llamar a la API de Groq: {e}")

st.markdown("---")
st.caption("Dashboard generado con Streamlit + Plotly. Todos los valores monetarios en USD.")
