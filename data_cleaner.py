"""
Módulo para la Carga, Limpieza, Preprocesamiento y Feature Engineering de Datos.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_data(data_dir: Path):
    """Carga y retorna los DataFrames en su estado crudo original."""
    t_raw = pd.read_csv(data_dir / "transacciones_logistica_v2.csv")
    f_raw = pd.read_csv(data_dir / "feedback_clientes_v2.csv")
    inv_raw = pd.read_csv(data_dir / "inventario_central_v2.csv")
    return t_raw, inv_raw, f_raw


def parse_lead_time(v):
    """Función auxiliar para normalizar Lead_Time_Dias a valores numéricos."""
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


def load_and_clean(data_dir: Path):
    """
    Realiza la lectura de los 3 datasets y aplica todas las reglas de limpieza y calidad de datos:
    - Normalización de ciudades, categorías y bodegas
    - Corrección de valores negativos u outliers sentinela (999, 99)
    - Imputaciones por mediana
    - Parsing de fechas y textos
    - Deduplicación de encuestas de feedback
    """
    t = pd.read_csv(data_dir / "transacciones_logistica_v2.csv")
    f = pd.read_csv(data_dir / "feedback_clientes_v2.csv")
    inv = pd.read_csv(data_dir / "inventario_central_v2.csv")

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

    # Fecha & Calculo de ingreso
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
        "t_raw_n": t_raw_n, 
        "i_raw_n": i_raw_n, 
        "f_raw_n": f_raw_n,
    }

    return t, inv, f, summary


def build_joins(t: pd.DataFrame, inv: pd.DataFrame, f: pd.DataFrame):
    """
    Combina transacciones, inventario y feedback creando uniones estratégicas (joins)
    y calculando variables financieras derivadas (Margen Unitario, Margen Total, etc.).
    """
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


def detect_iqr_outliers(df: pd.DataFrame):
    """Calcula detección de outliers numéricos vía IQR para el EDA."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr_v = q3 - q1
        low, high = q1 - 1.5 * iqr_v, q3 + 1.5 * iqr_v
        n_out = ((s < low) | (s > high)).sum()
        if n_out > 0:
            outlier_rows.append({
                "Columna": col, 
                "Mínimo": s.min(), 
                "Máximo": s.max(),
                "Límite inferior (IQR)": round(low, 2), 
                "Límite superior (IQR)": round(high, 2),
                "# Outliers": int(n_out), 
                "% Outliers": round(n_out / len(s) * 100, 2),
            })
    return pd.DataFrame(outlier_rows).sort_values("% Outliers", ascending=False) if outlier_rows else pd.DataFrame()
