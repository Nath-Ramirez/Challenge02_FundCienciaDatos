import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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

    # Normalizar ciudades (MED->Medellín, BOG->Bogotá, "Ventas_Web" no es ciudad -> canal mal cargado)
    ciudad_map = {"MED": "Medellín", "BOG": "Bogotá"}
    t["Ciudad_Destino_Original"] = t["Ciudad_Destino"]
    t["Ciudad_Destino"] = t["Ciudad_Destino"].replace(ciudad_map)
    # "Ventas_Web" filtrado a "Desconocida" (dato corrupto, no es ciudad real)
    mask_bad_city = t["Ciudad_Destino"] == "Ventas_Web"
    n_bad_city = mask_bad_city.sum()
    t.loc[mask_bad_city, "Ciudad_Destino"] = "Desconocida"
    if n_bad_city:
        issues.append(f"{n_bad_city} filas con 'Ciudad_Destino' inválido ('Ventas_Web') → reclasificadas como 'Desconocida'.")

    # Cantidad_Vendida negativa -> probable devolución/error de captura. Se toma valor absoluto para no perder el registro,
    # pero se marca para trazabilidad.
    mask_neg_qty = t["Cantidad_Vendida"] < 0
    n_neg_qty = mask_neg_qty.sum()
    t["Cantidad_Vendida_Flag"] = np.where(mask_neg_qty, "Corregida (negativa)", "OK")
    t["Cantidad_Vendida"] = t["Cantidad_Vendida"].abs()
    if n_neg_qty:
        issues.append(f"{n_neg_qty} filas con 'Cantidad_Vendida' negativa → convertidas a valor absoluto.")

    # Tiempo_Entrega_Real = 999 es un sentinel de dato faltante/atípico (envíos perdidos), se excluye de promedios pero se conserva la fila
    mask_sentinel = t["Tiempo_Entrega_Real"] >= 999
    n_sentinel = mask_sentinel.sum()
    t["Tiempo_Entrega_Valido"] = np.where(mask_sentinel, np.nan, t["Tiempo_Entrega_Real"])
    if n_sentinel:
        issues.append(f"{n_sentinel} filas con 'Tiempo_Entrega_Real' = 999 (sentinel de envío perdido) → excluidas de cálculos de tiempo promedio.")

    # Costo_Envio nulo -> se asume 0 solo si el estado es 'Perdido' (no se despachó); en otros casos se imputa con mediana por ciudad
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

    # Estado_Envio nulo -> "Desconocido"
    n_estado_na = t["Estado_Envio"].isna().sum()
    t["Estado_Envio"] = t["Estado_Envio"].fillna("Desconocido")
    if n_estado_na:
        issues.append(f"{n_estado_na} filas con 'Estado_Envio' vacío → etiquetadas como 'Desconocido'.")

    # Fecha
    t["Fecha_Venta"] = pd.to_datetime(t["Fecha_Venta"], format="%d/%m/%Y", errors="coerce")

    t["Ingreso_Total"] = t["Cantidad_Vendida"] * t["Precio_Venta_Final"]

    # ---------- INVENTARIO ----------
    i_raw_n = len(inv)

    # Categoria: normalizar mayúsculas/variantes y "???" como Desconocida
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

    # Bodega_Origen: normalizar capitalización, ZONA_FRANCA y BOD-EXT-99 son bodegas externas/tercerizadas válidas
    inv["Bodega_Origen"] = inv["Bodega_Origen"].replace({"norte": "Norte"})
    issues.append("Bodega 'norte' (minúscula) unificada con 'Norte'.")

    # Stock_Actual negativo (inconsistencia de sistema) -> se trata como 0 (sin stock real) y se marca
    mask_neg_stock = inv["Stock_Actual"] < 0
    n_neg_stock = mask_neg_stock.sum()
    inv["Stock_Flag"] = np.where(mask_neg_stock, "Corregido (negativo)", "OK")
    inv["Stock_Actual"] = inv["Stock_Actual"].clip(lower=0)
    if n_neg_stock:
        issues.append(f"{n_neg_stock} filas de inventario con 'Stock_Actual' negativo → ajustadas a 0.")

    # Stock_Actual nulo -> imputado con mediana de la categoría
    n_stock_na = inv["Stock_Actual"].isna().sum()
    med_stock_cat = inv.groupby("Categoria")["Stock_Actual"].median()
    inv["Stock_Actual"] = inv.apply(
        lambda r: med_stock_cat.get(r["Categoria"], inv["Stock_Actual"].median()) if pd.isna(r["Stock_Actual"]) else r["Stock_Actual"],
        axis=1,
    )
    if n_stock_na:
        issues.append(f"{n_stock_na} filas de inventario con 'Stock_Actual' vacío → imputadas con la mediana de su categoría.")

    # Lead_Time_Dias: texto inconsistente ("25-30 días", "Inmediato", números, nulos)
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

    # Rating_Producto = 99 es sentinel de error de captura -> NaN
    n_rating_bad = (f["Rating_Producto"] == 99).sum()
    f["Rating_Producto"] = f["Rating_Producto"].replace(99, np.nan)
    if n_rating_bad:
        issues.append(f"{n_rating_bad} filas de feedback con 'Rating_Producto' = 99 (sentinel inválido) → convertidas a NaN.")

    # Ticket_Soporte_Abierto: booleano mal codificado (Sí/No/0/1) -> normalizar a booleano
    ticket_map = {"Sí": True, "No": False, "1": True, "0": False, 1: True, 0: False}
    f["Ticket_Soporte_Abierto"] = f["Ticket_Soporte_Abierto"].map(lambda x: ticket_map.get(x, ticket_map.get(str(x), np.nan)))
    issues.append("'Ticket_Soporte_Abierto' normalizado a booleano (valores mezclados 'Sí'/'No'/'1'/'0').")

    # Recomienda_Marca: SI/NO/Maybe/NaN -> normalizar y NaN se deja como "Sin respuesta"
    rec_map = {"SI": "Sí", "NO": "No", "Maybe": "Tal vez"}
    n_rec_na = f["Recomienda_Marca"].isna().sum()
    f["Recomienda_Marca"] = f["Recomienda_Marca"].map(rec_map).fillna("Sin respuesta")
    if n_rec_na:
        issues.append(f"{n_rec_na} filas de feedback con 'Recomienda_Marca' vacío → etiquetadas 'Sin respuesta'.")

    # Edad_Cliente atípica (>100 años) -> tratada como dato corrupto, se marca pero no se elimina la fila
    mask_edad_bad = f["Edad_Cliente"] > 100
    n_edad_bad = mask_edad_bad.sum()
    f["Edad_Valida"] = ~mask_edad_bad
    if n_edad_bad:
        issues.append(f"{n_edad_bad} filas de feedback con 'Edad_Cliente' > 100 (dato implausible) → marcadas como inválidas, excluidas de análisis demográfico.")

    # Comentario_Texto vacío o placeholders ("N/A", "---") -> "Sin comentario"
    placeholders = {"N/A": True, "---": True, "": True}
    f["Comentario_Texto"] = f["Comentario_Texto"].fillna("Sin comentario")
    n_placeholder = f["Comentario_Texto"].isin(["N/A", "---"]).sum()
    f.loc[f["Comentario_Texto"].isin(["N/A", "---"]), "Comentario_Texto"] = "Sin comentario"
    if n_placeholder:
        issues.append(f"{n_placeholder} comentarios con placeholders ('N/A', '---') → unificados como 'Sin comentario'.")

    # Duplicados de Transaccion_ID en feedback -> una transacción puede tener múltiples encuestas; se conserva la última
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
# SIDEBAR
# ----------------------------------------------------------------------------

st.sidebar.title("📦 Panel de Control")
st.sidebar.markdown("Filtra los datos para explorar el negocio.")

canales = st.sidebar.multiselect("Canal de Venta", sorted(t["Canal_Venta"].unique()), default=list(t["Canal_Venta"].unique()))
ciudades = st.sidebar.multiselect("Ciudad Destino", sorted(t["Ciudad_Destino"].unique()), default=list(t["Ciudad_Destino"].unique()))

with st.sidebar.expander("🧹 Notas de limpieza de datos", expanded=False):
    st.caption(f"Transacciones originales: {clean_summary['t_raw_n']:,} | Inventario: {clean_summary['i_raw_n']:,} | Feedback: {clean_summary['f_raw_n']:,}")
    for i_msg in clean_summary["issues"]:
        st.markdown(f"- {i_msg}")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard construido a partir de transacciones_logistica_v2, feedback_clientes_v2 e inventario_central_v2.")

t_f = t[t["Canal_Venta"].isin(canales) & t["Ciudad_Destino"].isin(ciudades)]
tv_f = tv[tv["Canal_Venta"].isin(canales) & tv["Ciudad_Destino"].isin(ciudades)]
tvf_f = tvf[tvf["Canal_Venta"].isin(canales) & tvf["Ciudad_Destino"].isin(ciudades)]

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------

st.title("📦 Dashboard de Rentabilidad, Logística y Fidelidad")
st.markdown("Análisis integrado de **transacciones**, **inventario** y **feedback de clientes** para diagnosticar fugas de capital, cuellos de botella logísticos y riesgos operativos.")

k1, k2, k3, k4 = st.columns(4)
ingreso_total = tv_f["Ingreso_Total"].sum()
margen_total = tv_f["Margen_Total"].sum()
n_trx = len(t_f)
pct_sin_inv = (~tv_f["En_Inventario"]).mean() * 100

k1.metric("Ingreso Total (USD)", f"${ingreso_total:,.0f}")
k2.metric("Margen Total (USD)", f"${margen_total:,.0f}", delta=f"{margen_total/ingreso_total*100:.1f}% del ingreso" if ingreso_total else None)
k3.metric("Transacciones", f"{n_trx:,}")
k4.metric("% Ventas sin match en inventario", f"{pct_sin_inv:.1f}%")

st.markdown("---")

tab0, tab_merge, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 EDA",
    "🔗 Unión Estratégica",
    "1️⃣ Fuga de Capital",
    "2️⃣ Crisis Logística",
    "3️⃣ Venta Invisible",
    "4️⃣ Diagnóstico de Fidelidad",
    "5️⃣ Riesgo Operativo",
])

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
    if ds_choice == "feedback_clientes_v2.csv":
        st.caption(
            "⚠️ 877 'Transaccion_ID' aparecen más de una vez en feedback (una misma transacción con varias encuestas registradas); "
            "500 'Feedback_ID' también se repiten. Estos son los duplicados intencionales mencionados en el enunciado. "
            "Decisión: se conserva un único registro por 'Transaccion_ID' (el de 'Feedback_ID' más reciente/alto), eliminando "
            "las filas redundantes, para no inflar artificialmente el conteo de opiniones por venta."
        )

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

    st.subheader("4. Distribución de una variable numérica")
    if numeric_cols:
        col_sel = st.selectbox("Variable a inspeccionar", numeric_cols, key=f"num_{ds_choice}")
        s = df_sel[col_sel].dropna()
        skew_val = s.skew()
        cA, cB = st.columns(2)
        with cA:
            fig_h = px.histogram(s, nbins=50, title=f"Histograma — {col_sel}")
            st.plotly_chart(fig_h, use_container_width=True)
        with cB:
            fig_b = px.box(s, title=f"Boxplot — {col_sel}")
            st.plotly_chart(fig_b, use_container_width=True)
        st.caption(
            f"Media: {s.mean():,.2f} | Mediana: {s.median():,.2f} | Moda: {s.mode().iloc[0]:,.2f} | "
            f"Desv. estándar: {s.std():,.2f} | Asimetría (skew): {skew_val:.2f} "
            + ("→ distribución muy sesgada, la mediana es más robusta que la media para imputar."
               if abs(skew_val) > 1 else
               "→ distribución razonablemente simétrica, la media es un estimador aceptable.")
        )

    st.markdown("---")
    st.header("📋 Reporte Consolidado de Calidad de Datos")

    st.markdown("""
| Dataset | Registros | Columna con más nulidad | Duplicados eliminados | Outlier más severo |
|---|---|---|---|---|
| `inventario_central_v2.csv` | 2,500 | `Lead_Time_Dias` (16.12%) | 0 filas duplicadas | `Costo_Unitario_USD`: rango $0.05 – $850,000 (1 valor extremo, ~750x la mediana) |
| `transacciones_logistica_v2.csv` | 10,000 | `Estado_Envio` (16.83%) | 0 filas duplicadas | `Tiempo_Entrega_Real`: 50 registros = 999 días (sentinel), vs. mediana de 15 días |
| `feedback_clientes_v2.csv` | 4,500 | `Recomienda_Marca` (24.87%) | 500 `Feedback_ID` / 877 `Transaccion_ID` duplicados eliminados | `Edad_Cliente`: 23 registros > 100 años (máx. 195); `Rating_Producto`: 30 registros = 99 (sentinel) |
""")

    st.subheader("Integridad Referencial: SKUs Fantasma")
    inv_skus = set(inv_raw["SKU_ID"])
    ghost_rows = t_raw[~t_raw["SKU_ID"].isin(inv_skus)]
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Transacciones con SKU fantasma", f"{len(ghost_rows):,} / {len(t_raw):,}", f"{len(ghost_rows)/len(t_raw)*100:.1f}%")
    ic2.metric("SKUs fantasma distintos", f"{ghost_rows['SKU_ID'].nunique():,}")
    ic3.metric("Ingreso comprometido (USD)", f"${(ghost_rows['Cantidad_Vendida'].abs()*ghost_rows['Precio_Venta_Final']).sum():,.0f}")
    st.caption(
        "Estas transacciones no se eliminan: se conservan para no perder ingreso ni volumen real de ventas, pero se excluyen "
        "de cualquier cálculo de margen/costo (no hay costo unitario con el cual calcularlo) y se marcan explícitamente como "
        "'Venta Invisible' en la pestaña 3. El tratamiento completo de este dilema está en la pestaña **🔗 Unión Estratégica**."
    )

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
# TAB 1: FUGA DE CAPITAL Y RENTABILIDAD
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
        Precio_Prom=("Precio_Venta_Final", "mean"),
        Costo_Unitario=("Costo_Unitario_USD", "first"),
    ).reset_index()
    sku_margin["Margen_Pct"] = sku_margin["Margen_Total"] / sku_margin["Ingreso_Total"] * 100

    neg_skus = sku_margin[sku_margin["Margen_Total"] < 0].sort_values("Margen_Total")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("SKUs con margen negativo", f"{len(neg_skus):,} / {len(sku_margin):,}")
        st.metric("Pérdida total acumulada (USD)", f"${neg_skus['Margen_Total'].sum():,.0f}")
    with c2:
        by_channel = matched[matched["SKU_ID"].isin(neg_skus["SKU_ID"])].groupby("Canal_Venta")["Margen_Total"].sum().sort_values()
        fig = px.bar(by_channel, orientation="h", title="Pérdida por canal (SKUs con margen negativo)",
                     labels={"value": "Margen Total (USD)", "Canal_Venta": "Canal"}, color=by_channel.values,
                     color_continuous_scale="Reds_r")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 SKUs con mayor pérdida")
    st.dataframe(
        neg_skus.head(20)[["SKU_ID", "Categoria", "Unidades_Vendidas", "Ingreso_Total", "Margen_Total", "Margen_Pct"]]
        .style.format({"Ingreso_Total": "${:,.0f}", "Margen_Total": "${:,.0f}", "Margen_Pct": "{:.1f}%"}),
        use_container_width=True,
    )

    # Volumen vs Falla estructural: dispersión de unidades vendidas vs margen para SKUs negativos, por canal
    fig2 = px.scatter(
        matched[matched["SKU_ID"].isin(neg_skus["SKU_ID"])].groupby(["SKU_ID", "Canal_Venta"]).agg(
            Unidades=("Cantidad_Vendida", "sum"), Margen=("Margen_Total", "sum")
        ).reset_index(),
        x="Unidades", y="Margen", color="Canal_Venta",
        title="¿Pérdida por volumen o por precio? Unidades vendidas vs. Margen (SKUs negativos)",
        labels={"Unidades": "Unidades Vendidas", "Margen": "Margen Total (USD)"},
        hover_data=["SKU_ID"],
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

    online_share = (matched["Canal_Venta"] == "Online").mean() * 100
    online_neg_share = (matched[matched["SKU_ID"].isin(neg_skus["SKU_ID"])]["Canal_Venta"] == "Online").mean() * 100

    st.info(
        f"**Lectura del hallazgo:** el canal Online representa **{online_share:.1f}%** de las transacciones con match en inventario, "
        f"pero concentra **{online_neg_share:.1f}%** de las transacciones vinculadas a SKUs con margen negativo. "
        + ("Esto sugiere una **falla crítica de pricing en Online** (sobre-representación de pérdidas), no solo pérdida aceptable por volumen: "
           "revisar reglas de descuento, comisiones de marketplace o promociones automáticas en ese canal."
           if online_neg_share > online_share * 1.15 else
           "La proporción es similar a su peso general, lo que sugiere que las pérdidas están más distribuidas y podrían explicarse "
           "por estrategia de volumen (loss-leaders) en ciertos SKUs, más que por una falla sistemática de precios en un solo canal.")
    )

# ============================================================================
# TAB 2: CRISIS LOGÍSTICA Y CUELLOS DE BOTELLA
# ============================================================================
with tab2:
    st.header("Crisis Logística y Cuellos de Botella")
    st.markdown("Correlación entre **Tiempo de Entrega** y **NPS bajo** por ciudad y bodega, para identificar la zona que requiere cambio inmediato de operador.")

    log_df = tvf_f.dropna(subset=["Tiempo_Entrega_Valido", "Satisfaccion_NPS"]).copy()

    # Correlación por ciudad
    corr_city = log_df.groupby("Ciudad_Destino").apply(
        lambda g: g["Tiempo_Entrega_Valido"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else np.nan
    ).dropna().sort_values()
    corr_city.name = "Correlacion"

    corr_bodega = log_df.dropna(subset=["Bodega_Origen"]).groupby("Bodega_Origen").apply(
        lambda g: g["Tiempo_Entrega_Valido"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else np.nan
    ).dropna().sort_values()
    corr_bodega.name = "Correlacion"

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(corr_city, orientation="h", title="Correlación Tiempo de Entrega vs NPS — por Ciudad",
                     labels={"value": "Correlación (Pearson)", "Ciudad_Destino": "Ciudad"},
                     color=corr_city.values, color_continuous_scale="RdBu_r", range_color=[-1, 1])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(corr_bodega, orientation="h", title="Correlación Tiempo de Entrega vs NPS — por Bodega",
                     labels={"value": "Correlación (Pearson)", "Bodega_Origen": "Bodega"},
                     color=corr_bodega.values, color_continuous_scale="RdBu_r", range_color=[-1, 1])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    worst_city = corr_city.idxmin() if len(corr_city) else None
    worst_bodega = corr_bodega.idxmin() if len(corr_bodega) else None

    if worst_city is not None:
        wc_data = log_df[log_df["Ciudad_Destino"] == worst_city]
        avg_time_wc = wc_data["Tiempo_Entrega_Valido"].mean()
        avg_nps_wc = wc_data["Satisfaccion_NPS"].mean()
        st.error(
            f"**🚨 Zona crítica identificada: {worst_city}** (bodega más asociada: **{worst_bodega}**). "
            f"Correlación Tiempo de Entrega–NPS = **{corr_city.min():.2f}** — a mayor tiempo de entrega, el NPS cae de forma más pronunciada que en cualquier otra ciudad. "
            f"Tiempo de entrega promedio: **{avg_time_wc:.1f} días**, NPS promedio: **{avg_nps_wc:.1f}**. "
            "Esta es la zona que requiere **cambio inmediato de operador logístico**."
        )

    st.subheader("Dispersión Tiempo de Entrega vs NPS (zona crítica)")
    if worst_city is not None:
        fig3 = px.scatter(wc_data, x="Tiempo_Entrega_Valido", y="Satisfaccion_NPS", color="Bodega_Origen",
                           trendline="ols", title=f"{worst_city}: Tiempo de Entrega vs NPS",
                           labels={"Tiempo_Entrega_Valido": "Tiempo de Entrega (días)", "Satisfaccion_NPS": "NPS"})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Mapa de calor: Tiempo de entrega promedio por Ciudad y Estado de Envío")
    heat = t_f.groupby(["Ciudad_Destino", "Estado_Envio"])["Tiempo_Entrega_Valido"].mean().reset_index() if "Tiempo_Entrega_Valido" in t_f.columns else None
    heat_pivot = t_f.assign(Tiempo_Entrega_Valido=np.where(t_f["Tiempo_Entrega_Real"] >= 999, np.nan, t_f["Tiempo_Entrega_Real"])).pivot_table(
        index="Ciudad_Destino", columns="Estado_Envio", values="Tiempo_Entrega_Valido", aggfunc="mean"
    )
    fig4 = px.imshow(heat_pivot, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                      title="Tiempo de entrega promedio (días) por Ciudad × Estado de Envío")
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================================
# TAB 3: ANÁLISIS DE LA VENTA INVISIBLE
# ============================================================================
with tab3:
    st.header("Análisis de la Venta Invisible")
    st.markdown("Cuantificación del impacto financiero de ventas cuyo SKU **no existe** en el maestro de inventario.")

    invisible = tv_f[~tv_f["En_Inventario"]]
    visible = tv_f[tv_f["En_Inventario"]]

    ingreso_invisible = invisible["Ingreso_Total"].sum()
    ingreso_total_f = tv_f["Ingreso_Total"].sum()
    pct_riesgo = ingreso_invisible / ingreso_total_f * 100 if ingreso_total_f else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Ingreso en riesgo (USD)", f"${ingreso_invisible:,.0f}")
    c2.metric("% del ingreso total en riesgo", f"{pct_riesgo:.1f}%")
    c3.metric("SKUs 'fantasma' distintos", f"{invisible['SKU_ID'].nunique():,}")

    fig = px.pie(
        values=[ingreso_invisible, ingreso_total_f - ingreso_invisible],
        names=["Sin match en inventario", "Con match en inventario"],
        title="Ingreso Total: Ventas Controladas vs. Venta Invisible",
        color_discrete_sequence=["#d62728", "#2ca02c"],
        hole=0.45,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        by_channel = invisible.groupby("Canal_Venta")["Ingreso_Total"].sum().sort_values(ascending=False)
        fig2 = px.bar(by_channel, title="Ingreso 'invisible' por Canal de Venta",
                      labels={"value": "Ingreso (USD)", "Canal_Venta": "Canal"})
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        by_city = invisible.groupby("Ciudad_Destino")["Ingreso_Total"].sum().sort_values(ascending=False)
        fig3 = px.bar(by_city, title="Ingreso 'invisible' por Ciudad",
                      labels={"value": "Ingreso (USD)", "Ciudad_Destino": "Ciudad"})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top SKUs fantasma por ingreso comprometido")
    top_ghost = invisible.groupby("SKU_ID").agg(
        Ingreso_Total=("Ingreso_Total", "sum"),
        Unidades=("Cantidad_Vendida", "sum"),
        Transacciones=("Transaccion_ID", "count"),
    ).sort_values("Ingreso_Total", ascending=False).head(15).reset_index()
    st.dataframe(top_ghost.style.format({"Ingreso_Total": "${:,.0f}"}), use_container_width=True)

    st.warning(
        f"**Conclusión:** un **{pct_riesgo:.1f}%** del ingreso total (**${ingreso_invisible:,.0f} USD**) corresponde a SKUs que no existen "
        "en el maestro de inventario. Esto implica que la empresa no puede calcular costo, margen real, ni gestionar reposición para esa "
        "porción del negocio — es ingreso **contable pero no controlado**, con riesgo de fraude, error de carga de catálogo, o productos "
        "descontinuados que siguen vendiéndose sin trazabilidad."
    )

# ============================================================================
# TAB 4: DIAGNÓSTICO DE FIDELIDAD
# ============================================================================
with tab4:
    st.header("Diagnóstico de Fidelidad")
    st.markdown("Categorías con **alta disponibilidad de stock** pero **sentimiento negativo del cliente**: ¿mala calidad o sobrecosto?")

    cat_df = tvf_f[tvf_f["En_Inventario"]].dropna(subset=["Categoria"])

    cat_summary = cat_df.groupby("Categoria").agg(
        Stock_Promedio=("Stock_Actual", "mean"),
        NPS_Promedio=("Satisfaccion_NPS", "mean"),
        Rating_Producto_Promedio=("Rating_Producto", "mean"),
        Precio_Promedio=("Precio_Venta_Final", "mean"),
        Costo_Promedio=("Costo_Unitario_USD", "mean"),
        Transacciones=("Transaccion_ID", "count"),
    ).reset_index()
    cat_summary["Markup_Pct"] = (cat_summary["Precio_Promedio"] - cat_summary["Costo_Promedio"]) / cat_summary["Costo_Promedio"] * 100

    stock_median = cat_summary["Stock_Promedio"].median()
    nps_median = cat_summary["NPS_Promedio"].median()

    fig = px.scatter(
        cat_summary, x="Stock_Promedio", y="NPS_Promedio", size="Transacciones", color="Markup_Pct",
        text="Categoria", color_continuous_scale="RdYlGn_r",
        title="Stock promedio vs NPS promedio por Categoría (tamaño = # transacciones, color = % markup)",
        labels={"Stock_Promedio": "Stock Promedio (unidades)", "NPS_Promedio": "NPS Promedio"},
    )
    fig.add_vline(x=stock_median, line_dash="dash", line_color="gray")
    fig.add_hline(y=nps_median, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    paradox = cat_summary[(cat_summary["Stock_Promedio"] >= stock_median) & (cat_summary["NPS_Promedio"] < nps_median)].sort_values("NPS_Promedio")

    st.subheader("Categorías en zona de paradoja (alto stock + NPS bajo)")
    st.dataframe(
        paradox[["Categoria", "Stock_Promedio", "NPS_Promedio", "Rating_Producto_Promedio", "Markup_Pct", "Transacciones"]]
        .style.format({"Stock_Promedio": "{:.0f}", "NPS_Promedio": "{:.1f}", "Rating_Producto_Promedio": "{:.2f}", "Markup_Pct": "{:.1f}%"}),
        use_container_width=True,
    )

    if len(paradox):
        for _, row in paradox.iterrows():
            veredicto = "sobrecosto" if row["Markup_Pct"] > cat_summary["Markup_Pct"].median() and row["Rating_Producto_Promedio"] >= 3 else \
                        "mala calidad de producto" if row["Rating_Producto_Promedio"] < 3 else \
                        "combinación de ambos factores"
            st.markdown(
                f"- **{row['Categoria']}**: markup de **{row['Markup_Pct']:.0f}%**, rating de producto promedio **{row['Rating_Producto_Promedio']:.1f}/5** "
                f"→ el diagnóstico apunta a **{veredicto}** como causa principal de la insatisfacción."
            )
    else:
        st.info("No se detectaron categorías que combinen stock por encima de la mediana con NPS por debajo de la mediana en el filtro actual.")

    st.caption(
        "**Cómo leer la paradoja:** si el markup (precio vs costo) es alto y el rating de producto sigue siendo aceptable, "
        "el cliente probablemente percibe **sobrecosto** (paga de más por lo que recibe). Si el rating de producto es bajo "
        "independientemente del markup, la causa raíz es **calidad del producto**, no el precio."
    )

# ============================================================================
# TAB 5: STORYTELLING DE RIESGO OPERATIVO
# ============================================================================
with tab5:
    st.header("Storytelling de Riesgo Operativo")
    st.markdown("Relación entre la **antigüedad de la última revisión de stock** y la **tasa de tickets de soporte**, para identificar bodegas que operan 'a ciegas'.")

    op_df = tvf_f[tvf_f["En_Inventario"]].dropna(subset=["Dias_Desde_Revision", "Bodega_Origen"])

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
        title="Antigüedad de última revisión de stock vs. Tasa de Tickets de Soporte, por Bodega",
        labels={"Dias_Desde_Revision_Prom": "Días desde última revisión de stock (promedio)", "Tasa_Tickets_Pct": "Tasa de Tickets de Soporte (%)"},
    )
    fig.update_traces(textposition="top center", marker=dict(sizemin=8))
    st.plotly_chart(fig, use_container_width=True)

    blind_threshold = bodega_summary["Dias_Desde_Revision_Prom"].median()
    blind_warehouses = bodega_summary[bodega_summary["Dias_Desde_Revision_Prom"] > blind_threshold].sort_values("Tasa_Tickets_Pct", ascending=False)

    st.subheader("Bodegas operando 'a ciegas' (revisión de stock más antigua que la mediana)")
    st.dataframe(
        blind_warehouses.style.format({
            "Dias_Desde_Revision_Prom": "{:.0f} días", "Tasa_Tickets_Pct": "{:.1f}%", "NPS_Promedio": "{:.1f}"
        }),
        use_container_width=True,
    )

    if len(blind_warehouses):
        top_blind = blind_warehouses.iloc[0]
        st.error(
            f"**🚨 Bodega con mayor riesgo operativo: {top_blind['Bodega_Origen']}** — su stock no se revisa en promedio hace "
            f"**{top_blind['Dias_Desde_Revision_Prom']:.0f} días**, y presenta una tasa de tickets de soporte de "
            f"**{top_blind['Tasa_Tickets_Pct']:.1f}%**, con un NPS promedio de **{top_blind['NPS_Promedio']:.1f}**. "
            "La falta de revisión de inventario deriva en quiebres de stock no detectados, promesas de entrega incumplidas, "
            "y esto se traduce directamente en más tickets de soporte y peor satisfacción final del cliente."
        )

    corr_op = bodega_summary["Dias_Desde_Revision_Prom"].corr(bodega_summary["Tasa_Tickets_Pct"])
    st.info(
        f"**Correlación global (a nivel bodega) entre antigüedad de revisión y tasa de tickets: {corr_op:.2f}.** "
        + ("Existe una relación positiva clara: mientras más 'a ciegas' opera una bodega, más soporte reactivo genera, "
           "lo cual erosiona el NPS aguas abajo." if corr_op > 0.3 else
           "La relación es débil con los datos actuales, lo que sugiere que otros factores (transportista, ciudad, categoría de producto) "
           "pesan más que la sola antigüedad de revisión — aunque los casos individuales extremos (arriba) siguen siendo focos de atención inmediata.")
    )

st.markdown("---")
st.caption("Dashboard generado con Streamlit + Plotly. Todos los valores monetarios en USD. Los cálculos de margen requieren que el SKU tenga match en el maestro de inventario.")
