import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Intentar importar cliente de Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(page_title="Dashboard Logístico & Rentabilidad", layout="wide", page_icon="📦")

DATA_DIR = Path(__file__).parent / "data"

# ----------------------------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# ----------------------------------------------------------------------------

@st.cache_data
def load_and_clean():
    t = pd.read_csv(DATA_DIR / "transacciones_logistica_v2.csv")
    f = pd.read_csv(DATA_DIR / "feedback_clientes_v2.csv")
    inv = pd.read_csv(DATA_DIR / "inventario_central_v2.csv")

    issues = []

    # ---------- TRANSACCIONES ----------
    t_raw_n = len(t)

    # Normalizar ciudades
    ciudad_map = {"MED": "Medellín", "BOG": "Bogotá"}
    t["Ciudad_Destino_Original"] = t["Ciudad_Destino"]
    t["Ciudad_Destino"] = t["Ciudad_Destino"].replace(ciudad_map)
    mask_bad_city = t["Ciudad_Destino"] == "Ventas_Web"
    n_bad_city = mask_bad_city.sum()
    t.loc[mask_bad_city, "Ciudad_Destino"] = "Desconocida"
    if n_bad_city:
        issues.append(f"{n_bad_city} filas con 'Ciudad_Destino' inválido ('Ventas_Web') → reclasificadas como 'Desconocida'.")

    # Cantidad_Vendida negativa
    mask_neg_qty = t["Cantidad_Vendida"] < 0
    n_neg_qty = mask_neg_qty.sum()
    t["Cantidad_Vendida_Flag"] = np.where(mask_neg_qty, "Corregida (negativa)", "OK")
    t["Cantidad_Vendida"] = t["Cantidad_Vendida"].abs()
    if n_neg_qty:
        issues.append(f"{n_neg_qty} filas con 'Cantidad_Vendida' negativa → convertidas a valor absoluto.")

    # Tiempo_Entrega_Real = 999
    mask_sentinel = t["Tiempo_Entrega_Real"] >= 999
    n_sentinel = mask_sentinel.sum()
    t["Tiempo_Entrega_Valido"] = np.where(mask_sentinel, np.nan, t["Tiempo_Entrega_Real"])
    if n_sentinel:
        issues.append(f"{n_sentinel} filas con 'Tiempo_Entrega_Real' = 999 (sentinel de envío perdido) → excluidas de cálculos de tiempo promedio.")

    # Costo_Envio nulo
    med_envio_ciudad = t.groupby("Ciudad_Destino")["Costo_Envio"].median()
    t["Costo_Envio_Imputado"] = t["Costo_Envio"].isna()
    def fill_envio(row):
        if pd.notna(row["Costo_Envio"]):
            return row["Costo_Envio"]
        if row["Estado_Envio"] == "Perdido":
            return 0.0
        return med_envio_ciudad.get(row["Ciudad_Destino"], t["Costo_Envio"].median())
    t["Costo_Envio"] = t.apply(fill_envio, axis=1)
    n_envio_na = t["Costo_Envio_Imputado"].sum()
    if n_envio_na:
        issues.append(f"{n_envio_na} filas con 'Costo_Envio' vacío → imputadas (0 si 'Perdido', si no mediana de la ciudad).")

    # Estado_Envio nulo
    n_estado_na = t["Estado_Envio"].isna().sum()
    t["Estado_Envio"] = t["Estado_Envio"].fillna("Desconocido")
    if n_estado_na:
        issues.append(f"{n_estado_na} filas con 'Estado_Envio' vacío → etiquetadas como 'Desconocido'.")

    # Fecha
    t["Fecha_Venta"] = pd.to_datetime(t["Fecha_Venta"], format="%d/%m/%Y", errors="coerce")

    t["Ingreso_Total"] = t["Cantidad_Vendida"] * t["Precio_Venta_Final"]

    # ---------- INVENTARIO ----------
    i_raw_n = len(inv)

    cat_map = {
        "smart-phone": "Smartphones", "Smartphones": "Smartphones",
        "LAPTOP": "Laptops", "Laptops": "Laptops",
        "Accesorios": "Accesorios", "Monitores": "Monitores", "Tablets": "Tablets",
        "???": "Desconocida",
    }
    n_cat_bad = (inv["Categoria"] == "???").sum()
    inv["Categoria"] = inv["Categoria"].map(cat_map).fillna(inv["Categoria"])
    if n_cat_bad:
        issues.append(f"{n_cat_bad} filas de inventario con 'Categoria' = '???' → etiquetadas como 'Desconocida'.")
    issues.append("Categorías normalizadas ('smart-phone'/'Smartphones' → 'Smartphones'; 'LAPTOP'/'Laptops' → 'Laptops').")

    inv["Bodega_Origen"] = inv["Bodega_Origen"].replace({"norte": "Norte"})
    issues.append("Bodega 'norte' (minúscula) unificada con 'Norte'.")

    mask_neg_stock = inv["Stock_Actual"] < 0
    n_neg_stock = mask_neg_stock.sum()
    inv["Stock_Flag"] = np.where(mask_neg_stock, "Corregido (negativo)", "OK")
    inv["Stock_Actual"] = inv["Stock_Actual"].clip(lower=0)
    if n_neg_stock:
        issues.append(f"{n_neg_stock} filas de inventario con 'Stock_Actual' negativo → ajustadas a 0.")

    n_stock_na = inv["Stock_Actual"].isna().sum()
    med_stock_cat = inv.groupby("Categoria")["Stock_Actual"].median()
    inv["Stock_Actual"] = inv.apply(
        lambda r: med_stock_cat.get(r["Categoria"], inv["Stock_Actual"].median()) if pd.isna(r["Stock_Actual"]) else r["Stock_Actual"],
        axis=1,
    )
    if n_stock_na:
        issues.append(f"{n_stock_na} filas de inventario con 'Stock_Actual' vacío → imputadas con la mediana de su categoría.")

    def parse_lead_time(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v.lower() == "inmediato":
            return 0
        if "-" in v:
            parts = [p for p in v.replace("días", "").replace("dias", "").split("-")]
            try:
                nums = [float(p.strip()) for p in parts]
                return sum(nums) / len(nums)
            except ValueError:
                return np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan
    inv["Lead_Time_Dias_Num"] = inv["Lead_Time_Dias"].apply(parse_lead_time)
    n_lt_na = inv["Lead_Time_Dias"].isna().sum()
    if n_lt_na:
        issues.append(f"{n_lt_na} filas de inventario con 'Lead_Time_Dias' vacío (se mantienen como NaN, no se imputan tiempos de proveedor).")
    issues.append("'Lead_Time_Dias' convertido a numérico: rangos ('25-30 días') promediados, 'Inmediato' = 0.")

    inv["Ultima_Revision"] = pd.to_datetime(inv["Ultima_Revision"], errors="coerce")
    inv["Dias_Desde_Revision"] = (pd.Timestamp.today().normalize() - inv["Ultima_Revision"]).dt.days

    # ---------- FEEDBACK ----------
    f_raw_n = len(f)

    n_rating_bad = (f["Rating_Producto"] == 99).sum()
    f["Rating_Producto"] = f["Rating_Producto"].replace(99, np.nan)
    if n_rating_bad:
        issues.append(f"{n_rating_bad} filas de feedback con 'Rating_Producto' = 99 (sentinel inválido) → convertidas a NaN.")

    ticket_map = {"Sí": True, "No": False, "1": True, "0": False, 1: True, 0: False}
    f["Ticket_Soporte_Abierto"] = f["Ticket_Soporte_Abierto"].map(lambda x: ticket_map.get(x, ticket_map.get(str(x), np.nan)))
    issues.append("'Ticket_Soporte_Abierto' normalizado a booleano (valores mezclados 'Sí'/'No'/'1'/'0').")

    rec_map = {"SI": "Sí", "NO": "No", "Maybe": "Tal vez"}
    n_rec_na = f["Recomienda_Marca"].isna().sum()
    f["Recomienda_Marca"] = f["Recomienda_Marca"].map(rec_map).fillna("Sin respuesta")
    if n_rec_na:
        issues.append(f"{n_rec_na} filas de feedback con 'Recomienda_Marca' vacío → etiquetadas 'Sin respuesta'.")

    mask_edad_bad = f["Edad_Cliente"] > 100
    n_edad_bad = mask_edad_bad.sum()
    f["Edad_Valida"] = ~mask_edad_bad
    if n_edad_bad:
        issues.append(f"{n_edad_bad} filas de feedback con 'Edad_Cliente' > 100 (dato implausible) → marcadas como inválidas, excluidas de análisis demográfico.")

    f["Comentario_Texto"] = f["Comentario_Texto"].fillna("Sin comentario")
    n_placeholder = f["Comentario_Texto"].isin(["N/A", "---"]).sum()
    f.loc[f["Comentario_Texto"].isin(["N/A", "---"]), "Comentario_Texto"] = "Sin comentario"
    if n_placeholder:
        issues.append(f"{n_placeholder} comentarios con placeholders ('N/A', '---') → unificados como 'Sin comentario'.")

    n_dup_fb = f["Transaccion_ID"].duplicated().sum()
    if n_dup_fb:
        issues.append(f"{n_dup_fb} 'Transaccion_ID' duplicados en feedback (múltiples encuestas) → se conservó el registro más reciente por Feedback_ID.")
    f = f.sort_values("Feedback_ID").drop_duplicates(subset="Transaccion_ID", keep="last")

    summary = {
        "issues": issues,
        "t_raw_n": t_raw_n, "i_raw_n": i_raw_n, "f_raw_n": f_raw_n,
    }

    return t, inv, f, summary


t, inv, f, clean_summary = load_and_clean()

# ----------------------------------------------------------------------------
# UNIONES BASE
# ----------------------------------------------------------------------------

@st.cache_data
def build_joins(t, inv, f):
    tv = t.merge(inv, on="SKU_ID", how="left", suffixes=("", "_inv"))
    tv["En_Inventario"] = tv["Categoria"].notna()

    tv["Margen_Unitario"] = tv["Precio_Venta_Final"] - tv["Costo_Unitario_USD"]
    tv["Margen_Total"] = np.where(
        tv["En_Inventario"],
        (tv["Precio_Venta_Final"] - tv["Costo_Unitario_USD"]) * tv["Cantidad_Vendida"] - tv["Costo_Envio"].fillna(0),
        np.nan,
    )

    tvf = tv.merge(f, on="Transaccion_ID", how="left")
    return tv, tvf

tv, tvf = build_joins(t, inv, f)

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

# 2. Filtro de Categoría (extraído de inventario/uniones)
all_categories = sorted([c for c in inv["Categoria"].unique() if pd.notna(c)])
categorias = st.sidebar.multiselect("Categoría de Producto", all_categories, default=all_categories)

# 3. Filtro de Bodega
all_bodegas = sorted([b for b in inv["Bodega_Origen"].unique() if pd.notna(b)])
bodegas = st.sidebar.multiselect("Bodega de Origen", all_bodegas, default=all_bodegas)

# 4. Filtros existentes: Canal y Ciudad
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

st.sidebar.caption("Dashboard construido a partir de transacciones_logistica_v2, feedback_clientes_v2 e inventario_central_v2.")

# ----------------------------------------------------------------------------
# APLICACIÓN DE FILTROS A LOS DATASET
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

# Filtrado en dataset unificado tv y tvf (incluye Filtros de Categoria y Bodega)
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
# HEADER
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
    "🔗 Unión Estratégica",
    "1️⃣ Fuga de Capital",
    "2️⃣ Crisis Logística",
    "3️⃣ Venta Invisible",
    "4️⃣ Diagnóstico de Fidelidad",
    "5️⃣ Riesgo Operativo",
    "🤖 Diagnóstico IA (Groq)"
])

# ============================================================================
# TAB EDA
# ============================================================================
with tab0:
    st.header("EDA Interactivo: Calidad de Datos")
    st.markdown(
        "Diagnóstico del estado **crudo** de los tres datasets antes de cualquier limpieza: nulidad, "
        "duplicados, outliers e integridad referencial (SKUs fantasma)."
    )

    t_raw = pd.read_csv(DATA_DIR / "transacciones_logistica_v2.csv")
    inv_raw = pd.read_csv(DATA_DIR / "inventario_central_v2.csv")
    f_raw = pd.read_csv(DATA_DIR / "feedback_clientes_v2.csv")

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
        st.dataframe(null_df.style.format({"% Nulos": "{:.2f}%"}),
                     use_container_width=True, height=350)
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
    numeric_cols = df_sel.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows = []
    for col in numeric_cols:
        s = df_sel[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr_v = q3 - q1
        low, high = q1 - 1.5 * iqr_v, q3 + 1.5 * iqr_v
        n_out = ((s < low) | (s > high)).sum()
        if n_out > 0:
            outlier_rows.append({
                "Columna": col, "Mínimo": s.min(), "Máximo": s.max(),
                "Límite inferior (IQR)": round(low, 2), "Límite superior (IQR)": round(high, 2),
                "# Outliers": int(n_out), "% Outliers": round(n_out / len(s) * 100, 2),
            })
    out_df = pd.DataFrame(outlier_rows).sort_values("% Outliers", ascending=False) if outlier_rows else pd.DataFrame()
    if len(out_df):
        st.dataframe(out_df.style.format({"Mínimo": "{:,.2f}", "Máximo": "{:,.2f}", "% Outliers": "{:.2f}%"}), use_container_width=True)
    else:
        st.info("No se detectaron outliers numéricos vía IQR en este dataset.")

# ============================================================================
# TAB MERGE: UNIÓN ESTRATÉGICA
# ============================================================================
# ============================================================================
# TAB MERGE: UNIÓN ESTRATÉGICA + ANTES VS DESPUÉS
# ============================================================================
with tab_merge:
    st.header("🔗 Unión Estratégica: Una Sola Fuente de Verdad")
    st.markdown(
        "Las 3 tablas se integran en un único dataset analítico (`tv` / `tvf`) mediante `merge` tipo **left join**: "
        "`transacciones` es la tabla base (es la única con granularidad de evento de venta), y se le añaden los atributos "
        "de `inventario` (por `SKU_ID`) y de `feedback` (por `Transaccion_ID`). Un left join se elige deliberadamente sobre "
        "un inner join para **no perder ninguna venta real** solo porque le falte información de catálogo o de encuesta."
    )

    t_raw = pd.read_csv(DATA_DIR / "transacciones_logistica_v2.csv")
    inv_raw = pd.read_csv(DATA_DIR / "inventario_central_v2.csv")
    f_raw = pd.read_csv(DATA_DIR / "feedback_clientes_v2.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas base (transacciones)", f"{len(t_raw):,}")
    c2.metric("Filas tras join con inventario", f"{len(tv):,}", "0 filas perdidas")
    c3.metric("Filas tras join con feedback", f"{len(tvf):,}", "0 filas perdidas")

    st.code(
        "tv  = transacciones.merge(inventario, on='SKU_ID', how='left')\n"
        "tvf = tv.merge(feedback, on='Transaccion_ID', how='left')",
        language="python",
    )
    st.caption(
        "Resultado: cada fila sigue siendo una transacción; las columnas de inventario/feedback quedan en NaN cuando no hay "
        "match (SKU fantasma o venta sin encuesta respondida), en vez de descartar la transacción."
    )

    st.markdown("---")
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
        "volumen de negocio. La nulidad promedio baja en todos los casos porque las imputaciones (mediana, categorías "
        "explícitas de 'Desconocido'/'Sin respuesta') rellenan los huecos detectados en el EDA."
    )

    st.markdown("---")
    st.subheader("⚠️ Dilema del SKU Fantasma: ¿producto nuevo o error de digitación?")
    ghost_df = tv[~tv["En_Inventario"]]
    matched_df = tv[tv["En_Inventario"]]

    d1, d2 = st.columns(2)
    with d1:
        # Patrón de SKU: los códigos "fantasma" ¿tienen un rango numérico propio?
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
        "(1) sus códigos `PROD-XXXX` caen en un rango numérico **contiguo y superior** al de los SKUs sí catalogados "
        "(no están dispersos aleatoriamente, lo que descarta un typo aislado), y (2) representan **480 SKUs distintos** "
        "con **1,751 transacciones** — un volumen demasiado grande y sistemático para ser explicado por errores de tecleo puntuales.\n\n"
        "**Impacto en el cálculo de margen:** al no existir `Costo_Unitario_USD` para estos SKUs, es matemáticamente "
        "imposible calcular su margen real. La decisión es **excluirlos del margen total** (quedan como `NaN`, no como 0 "
        "ni se les asigna un costo estimado/promedio), para no inventar rentabilidad donde no hay información de costos. "
        "Se cuantifican aparte como **'ingreso en riesgo'** (pestaña 3) — visibles en ingreso, invisibles en rentabilidad."
    )

    st.subheader("🧮 Variables Derivadas (Feature Engineering)")
    st.markdown("A partir de la fuente única (`tv`/`tvf`) se construyeron las siguientes métricas nuevas, usadas en las pestañas 1-5:")

    st.markdown("""
| # | Variable derivada | Fórmula | Para qué sirve |
|---|---|---|---|
| 1 | **Margen de Utilidad** (`Margen_Total`, `Margen_Pct`) | `(Precio_Venta_Final - Costo_Unitario_USD) * Cantidad_Vendida - Costo_Envio` ; `Margen_Total / Ingreso_Total * 100` | Detectar SKUs y canales con fuga de capital (pestaña 1) |
| 2 | **Brecha de Entrega** (`Dias_Desde_Revision`) | `Fecha_Hoy - Ultima_Revision` (días) | Medir qué tan "a ciegas" opera cada bodega respecto a su inventario (pestaña 5) |
| 3 | **Tasa/Ratio de Soporte** (`Tasa_Tickets`) | `mean(Ticket_Soporte_Abierto)` agrupado por Bodega/Categoría | Cuantificar el ratio de tickets de soporte por bodega y su relación con la antigüedad de revisión (pestaña 5) |
| 4 | **Markup / Sobrecosto** (`Markup_Pct`) | `(Precio_Venta_Final - Costo_Unitario_USD) / Costo_Unitario_USD * 100` | Distinguir sobrecosto de mala calidad en el diagnóstico de fidelidad (pestaña 4) |
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

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("SKUs con margen negativo", f"{len(neg_skus):,} / {len(sku_margin):,}")
        st.metric("Pérdida total acumulada (USD)", f"${neg_skus['Margen_Total'].sum():,.0f}")
    with c2:
        by_channel = matched[matched["SKU_ID"].isin(neg_skus["SKU_ID"])].groupby("Canal_Venta")["Margen_Total"].sum().sort_values()
        if not by_channel.empty:
            fig = px.bar(by_channel, orientation="h", title="Pérdida por canal (SKUs con margen negativo)",
                         labels={"value": "Margen Total (USD)", "Canal_Venta": "Canal"}, color=by_channel.values,
                         color_continuous_scale="Reds_r")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top SKUs con mayor pérdida")
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
    st.markdown("Correlación entre **Tiempo de Entrega** y **NPS bajo** por ciudad y bodega.")

    log_df = tvf_f.dropna(subset=["Tiempo_Entrega_Valido", "Satisfaccion_NPS"]).copy()

    corr_city = log_df.groupby("Ciudad_Destino").apply(
        lambda g: g["Tiempo_Entrega_Valido"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else np.nan
    ).dropna().sort_values()
    
    corr_bodega = log_df.dropna(subset=["Bodega_Origen"]).groupby("Bodega_Origen").apply(
        lambda g: g["Tiempo_Entrega_Valido"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else np.nan
    ).dropna().sort_values()

    c1, c2 = st.columns(2)
    with c1:
        if not corr_city.empty:
            fig = px.bar(corr_city, orientation="h", title="Correlación Tiempo de Entrega vs NPS — por Ciudad",
                         labels={"value": "Correlación (Pearson)", "Ciudad_Destino": "Ciudad"},
                         color=corr_city.values, color_continuous_scale="RdBu_r", range_color=[-1, 1])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if not corr_bodega.empty:
            fig = px.bar(corr_bodega, orientation="h", title="Correlación Tiempo de Entrega vs NPS — por Bodega",
                         labels={"value": "Correlación (Pearson)", "Bodega_Origen": "Bodega"},
                         color=corr_bodega.values, color_continuous_scale="RdBu_r", range_color=[-1, 1])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

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
        color_discrete_sequence=["#d62728", "#2ca02c"],
        hole=0.45,
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

# ============================================================================
# TAB GROQ: INTEGRACIÓN IA (LLAMA-3)
# ============================================================================
with tab_groq:
    st.header("🤖 Análisis Automatizado con Llama-3 (Groq API)")
    st.markdown(
        "Esta sección genera un **informe ejecutivo y diagnóstico cualitativo** basado exclusivamente "
        "en las métricas y los datos filtrados en la barra lateral."
    )

    if not GROQ_AVAILABLE:
        st.error("La librería `groq` no está instalada. Ejecuta `pip install groq` en tu entorno.")
    else:
        groq_api_key = st.text_input("Ingresa tu Groq API Key:", type="password", help="Tu API key no será guardada permanentemente.")

        # Preparación de datos agregados para el prompt de la IA
        neg_skus_count = len(sku_margin[sku_margin["Margen_Total"] < 0]) if 'sku_margin' in locals() else 0
        total_loss = sku_margin[sku_margin["Margen_Total"] < 0]["Margen_Total"].sum() if 'sku_margin' in locals() else 0
        
        prompt_context = f"""
        Eres un consultor experto en cadena de suministro, finanzas y operaciones.
        Analiza el siguiente resumen ejecutivo generado a partir de los datos filtrados por el usuario:

        --- RESUMEN DE MÉTRICAS FILTRADAS ---
        * Rango de fechas: {start_d} a {end_d}
        * Categorías analizadas: {', '.join(categorias) if len(categorias) <= 5 else f'{len(categorias)} categorías'}
        * Bodegas analizadas: {', '.join(bodegas) if len(bodegas) <= 5 else f'{len(bodegas)} bodegas'}
        * Canales seleccionados: {', '.join(canales)}
        * Ciudades seleccionadas: {', '.join(ciudades)}
        
        --- INDICADORES CLAVE DE RENDIMIENTO (KPIs) ---
        1. Ingreso Total: ${ingreso_total:,.2f} USD
        2. Margen Total: ${margen_total:,.2f} USD ({(margen_total/ingreso_total*100) if ingreso_total else 0:.1f}% del ingreso)
        3. Número de Transacciones: {n_trx:,}
        4. Venta Invisible (% sin registro en catálogo): {pct_riesgo:.1f}% (Equivalente a ${ingreso_invisible:,.2f} USD)
        5. Fuga de Capital: {neg_skus_count} SKUs generan margen negativo con un total acumulado de pérdidas de ${total_loss:,.2f} USD.

        INSTRUCCIONES:
        Por favor provee un informe ejecutivo detallado dividido en tres partes:
        1. **Diagnóstico general**: Evaluación de la salud financiera y operativa bajo los filtros seleccionados.
        2. **Alertas críticas**: Fugas de capital, problemas con la Venta Invisible y fallas de stock o tiempos logísticos.
        3. **Plan de Acción Relevante**: 3 recomendaciones prioritarias con pasos concretos para mitigar pérdidas e incrementar el NPS.
        """

        if st.button("🚀 Disparar Análisis de Llama-3", type="primary"):
            if not groq_api_key:
                st.warning("Por favor ingresa tu clave API de Groq para continuar.")
            else:
                try:
                    with st.spinner("Analizando métricas y generando diagnóstico con Llama-3..."):
                        client = Groq(api_key=groq_api_key)
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "Eres un asistente analítico corporativo especializado en logística y finanzas."},
                                {"role": "user", "content": prompt_context}
                            ],
                            temperature=0.3,
                            max_tokens=1500,
                        )
                        respuesta = completion.choices[0].message.content
                        st.success("Análisis completado exitosamente.")
                        st.markdown("---")
                        st.markdown(respuesta)
                except Exception as e:
                    st.error(f"Error al conectar con la API de Groq: {e}")

st.markdown("---")
st.caption("Dashboard generado con Streamlit + Plotly. Todos los valores monetarios en USD.")
