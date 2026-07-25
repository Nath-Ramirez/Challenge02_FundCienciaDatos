"""TechLogistics S.A. — Consultoría de Datos (Challenge 02).

Aplicación de Streamlit que ejecuta el pipeline completo solicitado por el
consultor senior:

    Fase 1 - Auditoría de calidad (Health Score antes / después + reporte
              descargable de decisiones de limpieza).
    Fase 2 - Integración (merge estratégico) y Feature Engineering.
    Fase 3 - Inteligencia Artificial con Groq (Llama-3) para recomendaciones
              estratégicas en tiempo real, sobre los datos ya filtrados.

Autor: Consultor de Datos — EAFIT, Fundamentos en Ciencia de Datos, 2026-1
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:  # el módulo de IA se deshabilita si no está instalado
    GROQ_AVAILABLE = False

st.set_page_config(
    page_title="TechLogistics S.A. — Consultoría de Datos",
    layout="wide",
    page_icon="📦",
)

DATA_DIR = "data"
DEFAULT_FILES = {
    "inventario": os.path.join(DATA_DIR, "inventario_central_v2.csv"),
    "transacciones": os.path.join(DATA_DIR, "transacciones_logistica_v2.csv"),
    "feedback": os.path.join(DATA_DIR, "feedback_clientes_v2.csv"),
}

# =============================================================================
# 0. UTILIDADES GENÉRICAS DE CALIDAD DE DATOS
# =============================================================================


def null_report(df: pd.DataFrame) -> pd.Series:
    """Calcula el porcentaje de nulidad por columna."""
    return (df.isnull().mean() * 100).round(2).sort_values(ascending=False)


def count_duplicates(df: pd.DataFrame) -> int:
    """Cuenta filas exactamente duplicadas."""
    return int(df.duplicated().sum())


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Máscara booleana de outliers según el criterio de Tukey (rango IQR)."""
    numeric = pd.to_numeric(series, errors="coerce")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (numeric < lower) | (numeric > upper)


def health_score(df: pd.DataFrame, numeric_cols: Optional[list] = None) -> float:
    """Health Score (0-100) del dataset.

    Penaliza nulidad, duplicados y proporción de outliers en las columnas
    numéricas relevantes. Ninguno de los tres factores domina el resultado.
    """
    null_pct = df.isnull().mean().mean() * 100
    dup_pct = (df.duplicated().sum() / max(len(df), 1)) * 100

    outlier_pct = 0.0
    if numeric_cols:
        flags = [
            detect_outliers_iqr(df[col]).mean() * 100
            for col in numeric_cols
            if col in df.columns
        ]
        if flags:
            outlier_pct = float(np.mean(flags))

    score = 100 - (null_pct * 0.4) - (dup_pct * 0.4) - (outlier_pct * 0.2)
    return round(max(0.0, min(100.0, score)), 2)


# =============================================================================
# 1. CARGA DE DATOS (con manejo de excepciones)
# =============================================================================


@st.cache_data(show_spinner=False)
def read_csv_safe(path_or_buffer, nombre: str) -> Optional[pd.DataFrame]:
    """Lee un CSV controlando errores comunes de formato/codificación."""
    try:
        return pd.read_csv(path_or_buffer)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de **{nombre}**. Verifica la ruta o súbelo "
                 f"manualmente en la barra lateral.")
    except pd.errors.EmptyDataError:
        st.error(f"El archivo de **{nombre}** está vacío.")
    except pd.errors.ParserError as exc:
        st.error(f"El archivo de **{nombre}** no se pudo interpretar como CSV: {exc}")
    except UnicodeDecodeError:
        st.error(f"Problema de codificación al leer **{nombre}**. Intenta guardarlo en UTF-8.")
    except Exception as exc:  # último recurso: no tumbar la app por un error inesperado
        st.error(f"Error inesperado leyendo **{nombre}**: {exc}")
    return None


def cargar_datasets():
    """Resuelve la fuente de cada dataset: archivo subido > archivo en /data."""
    st.sidebar.header("📁 Fuente de datos")
    subir = st.sidebar.checkbox("Subir mis propios CSV", value=False)

    fuentes = {}
    if subir:
        fuentes["inventario"] = st.sidebar.file_uploader("Inventario", type="csv")
        fuentes["transacciones"] = st.sidebar.file_uploader("Transacciones", type="csv")
        fuentes["feedback"] = st.sidebar.file_uploader("Feedback", type="csv")
    else:
        fuentes = {k: v for k, v in DEFAULT_FILES.items()}

    inv = read_csv_safe(fuentes.get("inventario"), "Inventario") if fuentes.get("inventario") else None
    trans = read_csv_safe(fuentes.get("transacciones"), "Transacciones") if fuentes.get("transacciones") else None
    fb = read_csv_safe(fuentes.get("feedback"), "Feedback") if fuentes.get("feedback") else None
    return inv, trans, fb


# =============================================================================
# 2. LIMPIEZA — INVENTARIO CENTRAL
# =============================================================================

MAPA_CATEGORIAS = {
    "smart-phone": "Smartphones",
    "smartphones": "Smartphones",
    "laptop": "Laptops",
    "laptops": "Laptops",
    "monitores": "Monitores",
    "accesorios": "Accesorios",
    "tablets": "Tablets",
    "???": "Sin Categoría",
}

MAPA_BODEGAS = {
    "norte": "Norte",
    "sur": "Sur",
    "occidente": "Occidente",
    "zona_franca": "Zona Franca",
    "bod-ext-99": "Bodega Externa (Terceros)",
}


def parsear_lead_time(valor) -> float:
    """Convierte Lead_Time_Dias (texto heterogéneo) a un número de días.

    Reglas de negocio:
      - "Inmediato"        -> 0 días.
      - "25-30 días"       -> punto medio del rango (27.5).
      - cadena numérica     -> el número tal cual.
      - vacío / no parseable -> NaN (se imputa después).
    """
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip().lower()
    if texto in ("inmediato", "inmediata"):
        return 0.0
    if "-" in texto:
        partes = texto.replace("días", "").replace("dias", "").strip().split("-")
        try:
            numeros = [float(p.strip()) for p in partes]
            return float(np.mean(numeros))
        except ValueError:
            return np.nan
    texto_limpio = texto.replace("días", "").replace("dias", "").strip()
    try:
        return float(texto_limpio)
    except ValueError:
        return np.nan


def limpiar_inventario(df_raw: pd.DataFrame):
    """Limpia el dataset de inventario y documenta cada decisión tomada."""
    df = df_raw.copy()
    log = []

    # a) Categoria: unificar variantes de escritura ('smart-phone', 'LAPTOP', '???')
    df["Categoria"] = (
        df["Categoria"].astype(str).str.strip().str.lower().map(MAPA_CATEGORIAS)
        .fillna(df["Categoria"])
    )
    sin_cat = (df["Categoria"] == "Sin Categoría").sum()
    log.append(
        f"Categoria: variantes de escritura ('smart-phone', 'LAPTOP') unificadas a un "
        f"catálogo canónico; {sin_cat} registros con '???' se dejaron como categoría "
        f"explícita 'Sin Categoría' (no se adivina el segmento real)."
    )

    # b) Bodega_Origen: normalizar mayúsculas/minúsculas; conservar nodos externos
    df["Bodega_Origen"] = (
        df["Bodega_Origen"].astype(str).str.strip().str.lower().map(MAPA_BODEGAS)
        .fillna(df["Bodega_Origen"])
    )
    log.append(
        "Bodega_Origen: normalizado el casing ('norte' -> 'Norte'). 'Zona Franca' y "
        "'Bodega Externa (Terceros)' se conservan como nodos logísticos legítimos, "
        "no como errores, porque representan centros de distribución reales."
    )

    # c) Lead_Time_Dias: texto heterogéneo -> numérico, luego imputar por mediana
    df["Lead_Time_Dias"] = df["Lead_Time_Dias"].apply(parsear_lead_time)
    faltantes_lt = df["Lead_Time_Dias"].isnull().sum()
    df["Lead_Time_Dias"] = df.groupby("Bodega_Origen")["Lead_Time_Dias"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Lead_Time_Dias"] = df["Lead_Time_Dias"].fillna(df["Lead_Time_Dias"].median())
    log.append(
        f"Lead_Time_Dias: {faltantes_lt} valores (rangos de texto, 'Inmediato' o vacíos) "
        f"convertidos a numérico; los nulos remanentes se imputaron con la MEDIANA por "
        f"bodega — variable declarada como 'ruidosa' en el diccionario, por lo que la "
        f"mediana (robusta a colas largas) es preferible a la media."
    )

    # d) Ultima_Revision -> fecha
    df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")

    # e) Stock_Actual: nulos y negativos (contablemente imposibles)
    nulos_stock = df["Stock_Actual"].isnull().sum()
    negativos_stock = (df["Stock_Actual"] < 0).sum()
    df["Stock_Actual_Original"] = df["Stock_Actual"]
    df.loc[df["Stock_Actual"] < 0, "Stock_Actual"] = np.nan
    df["Stock_Actual"] = df.groupby("Categoria")["Stock_Actual"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Stock_Actual"] = df["Stock_Actual"].fillna(df["Stock_Actual"].median())
    log.append(
        f"Stock_Actual: {negativos_stock} registros negativos (error de captura del ERP, "
        f"no back-order real) y {nulos_stock} nulos se imputaron con la MEDIANA por "
        f"categoría (distribución del stock con cola derecha; la media se distorsiona)."
    )

    # f) Costo_Unitario_USD: rango extremo ($0.05 - $850,000) -> winsorize (capping)
    costo = pd.to_numeric(df["Costo_Unitario_USD"], errors="coerce")
    p1, p99 = costo.quantile(0.01), costo.quantile(0.99)
    outliers_costo = int(((costo < p1) | (costo > p99)).sum())
    df["Costo_Unitario_USD"] = costo.clip(lower=p1, upper=p99)
    df["Costo_Unitario_USD"] = df.groupby("Categoria")["Costo_Unitario_USD"].transform(
        lambda s: s.fillna(s.median())
    )
    log.append(
        f"Costo_Unitario_USD: {outliers_costo} outliers extremos capados al percentil "
        f"1-99 (winsorize) en vez de eliminarse, porque el costo es indispensable para "
        f"calcular el margen; los nulos (si existen) se imputan con la MEDIANA por "
        f"categoría por la fuerte asimetría de precios entre Accesorios y Laptops."
    )

    # g) Duplicados exactos
    dup = count_duplicates(df.drop(columns=["Stock_Actual_Original"], errors="ignore"))
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Stock_Actual_Original"])
    log.append(f"Registros duplicados exactos eliminados: {dup}.")

    return df, log


# =============================================================================
# 3. LIMPIEZA — TRANSACCIONES LOGÍSTICA
# =============================================================================

MAPA_CIUDADES = {
    "bog": "Bogotá",
    "bogotá": "Bogotá",
    "bogota": "Bogotá",
    "med": "Medellín",
    "medellín": "Medellín",
    "medellin": "Medellín",
    "cali": "Cali",
    "barranquilla": "Barranquilla",
    "bucaramanga": "Bucaramanga",
}


def limpiar_transacciones(df_raw: pd.DataFrame, inventario_skus: set):
    """Limpia el dataset de transacciones y documenta cada decisión tomada."""
    df = df_raw.copy()
    log = []

    # a) Fecha_Venta: formato consistente dd/mm/yyyy
    df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], dayfirst=True, errors="coerce")

    # b) Cantidad_Vendida negativa: error de signo (no hay campo de devolución
    #    separado), se corrige con valor absoluto en vez de descartar la venta.
    negativos_cant = int((df["Cantidad_Vendida"] < 0).sum())
    df["Cantidad_Vendida"] = df["Cantidad_Vendida"].abs()
    log.append(
        f"Cantidad_Vendida: {negativos_cant} registros con signo negativo (error de "
        f"digitación del ERP) corregidos con valor absoluto; se optó por esto y no por "
        f"eliminarlos porque el resto de la fila (precio, envío, fecha) es válido."
    )

    # c) Ciudad_Destino: abreviaturas y una fuga de 'Ventas_Web' (canal, no ciudad)
    ciudad_norm = df["Ciudad_Destino"].astype(str).str.strip().str.lower().map(MAPA_CIUDADES)
    fuga_canal = (df["Ciudad_Destino"] == "Ventas_Web").sum()
    df["Ciudad_Destino"] = ciudad_norm
    moda_ciudad = df["Ciudad_Destino"].mode(dropna=True)
    moda_ciudad = moda_ciudad.iloc[0] if not moda_ciudad.empty else "Desconocida"
    df["Ciudad_Destino"] = df["Ciudad_Destino"].fillna(moda_ciudad)
    log.append(
        f"Ciudad_Destino: abreviaturas ('BOG', 'MED') unificadas a nombre completo. "
        f"{fuga_canal} registros traían 'Ventas_Web' (una fuga del campo Canal_Venta "
        f"hacia Ciudad_Destino, error de exportación del sistema); al no poder recuperar "
        f"la ciudad real se imputaron con la MODA (ciudad más frecuente)."
    )

    # d) Costo_Envio: nulos -> mediana por Canal_Venta (el costo de envío depende
    #    fuertemente del canal: Físico vs. Online/App/WhatsApp).
    nulos_envio = df["Costo_Envio"].isnull().sum()
    df["Costo_Envio"] = df.groupby("Canal_Venta")["Costo_Envio"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Costo_Envio"] = df["Costo_Envio"].fillna(df["Costo_Envio"].median())
    log.append(
        f"Costo_Envio: {nulos_envio} nulos imputados con la MEDIANA por canal de venta "
        f"(el costo logístico varía sistemáticamente entre canales)."
    )

    # e) Tiempo_Entrega_Real: 999 es un valor centinela de "no entregado / sin
    #    registro", no un outlier orgánico -> se trata como dato faltante.
    tiempo = pd.to_numeric(df["Tiempo_Entrega_Real"], errors="coerce")
    centinela = int((tiempo >= 999).sum())
    df["Entrega_Sin_Registro"] = tiempo >= 999
    tiempo = tiempo.mask(tiempo >= 999, np.nan)
    df["Tiempo_Entrega_Real"] = tiempo
    df["Tiempo_Entrega_Real"] = df.groupby("Canal_Venta")["Tiempo_Entrega_Real"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Tiempo_Entrega_Real"] = df["Tiempo_Entrega_Real"].fillna(df["Tiempo_Entrega_Real"].median())
    log.append(
        f"Tiempo_Entrega_Real: {centinela} registros con el valor centinela 999 "
        f"(entrega nunca confirmada por el sistema) se trataron como NaN y se "
        f"imputaron con la MEDIANA por canal de venta; se conservó la bandera "
        f"'Entrega_Sin_Registro' para no perder la señal de falla operativa."
    )

    # f) Estado_Envio: nulos -> categoría explícita 'Desconocido' (17% de nulos,
    #    demasiado alto para imputar con la moda sin sesgar el KPI de servicio).
    nulos_estado = df["Estado_Envio"].isnull().sum()
    df["Estado_Envio"] = df["Estado_Envio"].fillna("Desconocido")
    log.append(
        f"Estado_Envio: {nulos_estado} nulos ({nulos_estado/len(df)*100:.1f}%) se "
        f"mantuvieron como categoría explícita 'Desconocido' en lugar de imputar con "
        f"la moda, para no inflar artificialmente ningún estado de envío."
    )

    # g) Venta Fantasma: SKU_ID que no existe en el inventario oficial
    df["Es_Venta_Fantasma"] = ~df["SKU_ID"].isin(inventario_skus)
    n_fantasma = int(df["Es_Venta_Fantasma"].sum())
    log.append(
        f"Venta Fantasma: {n_fantasma} transacciones ({n_fantasma/len(df)*100:.1f}%) con "
        f"SKU_ID fuera del catálogo. DECISIÓN: se conservan como ingresos reales "
        f"(producto nuevo no catalogado a tiempo), pero se excluyen del cálculo de "
        f"margen (no hay costo de referencia) y quedan marcadas para revisión de catálogo."
    )

    # h) Duplicados exactos
    dup = count_duplicates(df.drop(columns=["Entrega_Sin_Registro", "Es_Venta_Fantasma"], errors="ignore"))
    df = df.drop_duplicates(
        subset=[c for c in df.columns if c not in ("Entrega_Sin_Registro", "Es_Venta_Fantasma")]
    )
    log.append(f"Registros duplicados exactos eliminados: {dup}.")

    return df, log


# =============================================================================
# 4. LIMPIEZA — FEEDBACK CLIENTES
# =============================================================================


def limpiar_feedback(df_raw: pd.DataFrame):
    """Limpia el dataset de feedback y documenta cada decisión tomada."""
    df = df_raw.copy()
    log = []

    # a) Feedback_ID duplicado (colisión de llave, NO fila duplicada):
    #    el contenido de cada fila es distinto, así que no se elimina nada;
    #    se genera una llave sustituta única para trazabilidad interna.
    colisiones_id = int(df["Feedback_ID"].duplicated().sum())
    df["Feedback_UID"] = [f"FB-UID-{i:06d}" for i in range(len(df))]
    log.append(
        f"Feedback_ID: {colisiones_id} colisiones de identificador (mismo ID, "
        f"contenido distinto — bug de generación de IDs). No se eliminó ninguna fila; "
        f"se creó 'Feedback_UID' como llave sustituta única para el análisis."
    )

    # b) Duplicados de contenido exacto (fila 100% igual) -> sí se eliminan
    dup = count_duplicates(df.drop(columns=["Feedback_ID", "Feedback_UID"]))
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ("Feedback_ID", "Feedback_UID")])
    log.append(f"Registros 100% duplicados (mismo contenido) eliminados: {dup}.")

    # c) Rating_Producto fuera de escala (1-5) -> típicamente un dígito extra (9 -> 99)
    fuera_escala = int(((df["Rating_Producto"] < 1) | (df["Rating_Producto"] > 5)).sum())
    df["Rating_Producto"] = df["Rating_Producto"].mask(
        (df["Rating_Producto"] < 1) | (df["Rating_Producto"] > 5), np.nan
    )
    moda_rating = df["Rating_Producto"].mode(dropna=True)
    moda_rating = moda_rating.iloc[0] if not moda_rating.empty else 3
    df["Rating_Producto"] = df["Rating_Producto"].fillna(moda_rating)
    log.append(
        f"Rating_Producto: {fuera_escala} valores fuera del rango válido 1-5 (ej. 99, "
        f"probable error de digitación) se imputaron con la MODA — es una escala "
        f"ordinal tipo Likert, por lo que la moda es más apropiada que un promedio."
    )

    # d) Recomienda_Marca: normalizar valores y conservar el 'no responde' como señal
    df["Recomienda_Marca"] = (
        df["Recomienda_Marca"].astype(str).str.strip().str.upper()
        .replace({"SI": "Sí", "NO": "No", "MAYBE": "Tal vez", "NAN": np.nan})
    )
    nulos_recomienda = df["Recomienda_Marca"].isnull().sum()
    df["Recomienda_Marca"] = df["Recomienda_Marca"].fillna("No responde")
    log.append(
        f"Recomienda_Marca: valores normalizados ('SI'->'Sí', 'NO'->'No'); "
        f"{nulos_recomienda} nulos ({nulos_recomienda/len(df)*100:.1f}%) se dejaron como "
        f"categoría explícita 'No responde' en lugar de imputar con la moda, ya que es "
        f"casi la cuarta parte de los datos y forzar una respuesta sesgaría el NPS cualitativo."
    )

    # e) Ticket_Soporte_Abierto: normalizar a booleano ('1'/'Sí' -> True)
    df["Ticket_Soporte_Abierto"] = (
        df["Ticket_Soporte_Abierto"].astype(str).str.strip()
        .map({"Sí": True, "1": True, "No": False, "0": False})
        .fillna(False)
    )

    # f) Comentario_Texto: '---' es un placeholder de "sin comentario", no un dato real
    placeholders = int((df["Comentario_Texto"] == "---").sum())
    df["Comentario_Texto"] = df["Comentario_Texto"].replace("---", np.nan).fillna("Sin comentario")
    log.append(
        f"Comentario_Texto: {placeholders} registros con el placeholder '---' "
        f"unificados junto con los nulos bajo 'Sin comentario' (campo de texto libre, "
        f"no se imputa contenido inventado)."
    )

    # g) Edad_Cliente: edades imposibles (>100 años) -> mediana
    imposibles_edad = int((df["Edad_Cliente"] > 100).sum())
    df["Edad_Cliente"] = df["Edad_Cliente"].mask(df["Edad_Cliente"] > 100, np.nan)
    df["Edad_Cliente"] = df["Edad_Cliente"].fillna(df["Edad_Cliente"].median())
    log.append(
        f"Edad_Cliente: {imposibles_edad} edades imposibles (ej. 195 años) imputadas "
        f"con la MEDIANA (la media se dejaría arrastrar por esos valores extremos)."
    )

    # h) Satisfaccion_NPS: ya viene en una única escala continua (-100 a 100);
    #    se normaliza a 0-100 solo para facilitar la lectura del KPI.
    df["Satisfaccion_NPS_Normalizado"] = (df["Satisfaccion_NPS"] + 100) / 2
    log.append(
        "Satisfaccion_NPS: reescalada de [-100, 100] a [0, 100] "
        "(Satisfaccion_NPS_Normalizado) para una lectura más intuitiva del KPI, "
        "sin alterar el orden ni la distribución relativa de las respuestas."
    )

    return df, log


# =============================================================================
# 5. INTEGRACIÓN Y FEATURE ENGINEERING (Fase 2)
# =============================================================================


def integrar_y_enriquecer(inv: pd.DataFrame, trans: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    """Construye la 'fuente única de verdad' y calcula las variables derivadas."""
    df = trans.merge(
        inv[["SKU_ID", "Categoria", "Costo_Unitario_USD", "Lead_Time_Dias", "Bodega_Origen"]],
        on="SKU_ID",
        how="left",
        suffixes=("", "_inv"),
    )
    df["Categoria"] = df["Categoria"].fillna("Sin Catalogar")

    # Feature 1: Margen de Utilidad (NaN para ventas fantasma: no hay costo de referencia)
    df["Margen_Utilidad"] = np.where(
        df["Es_Venta_Fantasma"],
        np.nan,
        df["Precio_Venta_Final"] - df["Costo_Unitario_USD"] - df["Costo_Envio"],
    )

    # Feature 2: Brecha de Entrega vs. Prometido (Lead_Time_Dias = promesa teórica)
    df["Brecha_Entrega_Dias"] = df["Tiempo_Entrega_Real"] - df["Lead_Time_Dias"]

    # Feedback agregado a nivel de transacción (una transacción puede tener
    # varios registros de feedback: se promedian las métricas numéricas)
    fb_agg = fb.groupby("Transaccion_ID").agg(
        Rating_Producto=("Rating_Producto", "mean"),
        Rating_Logistica=("Rating_Logistica", "mean"),
        NPS=("Satisfaccion_NPS_Normalizado", "mean"),
        Ticket_Soporte=("Ticket_Soporte_Abierto", "max"),
    ).reset_index()
    df = df.merge(fb_agg, on="Transaccion_ID", how="left")

    # Feature 3: Ratio de Soporte por Categoría
    ratio_soporte = (
        df.groupby("Categoria")["Ticket_Soporte"].mean().rename("Ratio_Soporte_Categoria")
    )
    df = df.merge(ratio_soporte, on="Categoria", how="left")

    return df


# =============================================================================
# 6. FASE 3 — IA CON GROQ (Llama-3)
# =============================================================================


def construir_resumen_estadistico(df: pd.DataFrame) -> str:
    """Arma el resumen estadístico (texto) que se envía como prompt a la IA."""
    partes = [f"Registros analizados: {len(df)}"]
    partes.append(f"Ingresos totales: ${df['Precio_Venta_Final'].sum():,.2f}")
    if "Margen_Utilidad" in df.columns:
        partes.append(f"Margen de utilidad promedio: ${df['Margen_Utilidad'].mean():,.2f}")
        partes.append(
            f"% ventas fantasma (sin margen calculable): "
            f"{df['Es_Venta_Fantasma'].mean() * 100:.1f}%"
        )
    if "Brecha_Entrega_Dias" in df.columns:
        partes.append(
            f"Brecha promedio entrega vs. promesa: {df['Brecha_Entrega_Dias'].mean():.1f} días"
        )
    if "NPS" in df.columns:
        partes.append(f"NPS normalizado promedio: {df['NPS'].mean():.1f}/100")
    if "Ratio_Soporte_Categoria" in df.columns and not df.empty:
        top = df.groupby("Categoria")["Ratio_Soporte_Categoria"].mean().sort_values(ascending=False)
        if len(top):
            partes.append(
                f"Categoría con más tickets de soporte: {top.index[0]} "
                f"({top.iloc[0] * 100:.1f}% de sus ventas)"
            )
    return "\n".join(partes)


def generar_recomendaciones_ia(resumen_stats: str, api_key: str) -> str:
    """Llama a Llama-3 en Groq para generar 3 párrafos de recomendación."""
    client = Groq(api_key=api_key)
    prompt = f"""Eres un consultor senior de datos contratado por TechLogistics S.A.
A continuación tienes el resumen estadístico de los datos YA FILTRADOS por el analista:

{resumen_stats}

Genera EXACTAMENTE 3 párrafos de recomendación estratégica en español para la
gerencia general, cubriendo:
1) Diagnóstico de rentabilidad y riesgo operativo (margen, ventas fantasma, stock).
2) Diagnóstico de desempeño logístico y experiencia del cliente (entregas, NPS, soporte).
3) Recomendación accionable priorizada (qué hacer primero y por qué).
No uses viñetas ni encabezados, solo los 3 párrafos en prosa."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800,
    )
    return completion.choices[0].message.content


# =============================================================================
# 7. INTERFAZ STREAMLIT
# =============================================================================


def main() -> None:
    st.title("📦 TechLogistics S.A. — Consultoría de Datos")
    st.caption("Challenge 02 · Fundamentos en Ciencia de Datos · EAFIT 2026-1")

    inv_raw, trans_raw, fb_raw = cargar_datasets()
    if inv_raw is None or trans_raw is None or fb_raw is None:
        st.warning("Aún faltan uno o más datasets. Revisa la carpeta `data/` o sube los "
                   "archivos manualmente en la barra lateral.")
        st.stop()

    # ---- Limpieza (con manejo de errores para no tumbar la app) --------------
    try:
        inv_clean, log_inv = limpiar_inventario(inv_raw)
        trans_clean, log_trans = limpiar_transacciones(trans_raw, set(inv_clean["SKU_ID"]))
        fb_clean, log_fb = limpiar_feedback(fb_raw)
        df_master = integrar_y_enriquecer(inv_clean, trans_clean, fb_clean)
    except Exception as exc:  # noqa: BLE001 - queremos capturar cualquier fallo de pipeline
        st.error(f"Ocurrió un error durante el procesamiento de los datos: {exc}")
        st.stop()

    # ---- Filtros globales en la barra lateral ---------------------------------
    st.sidebar.header("🔎 Filtros de análisis")
    categorias_disp = ["(Todas)"] + sorted(df_master["Categoria"].dropna().unique().tolist())
    categoria_sel = st.sidebar.selectbox("Categoría", categorias_disp)

    ciudades_disp = ["(Todas)"] + sorted(df_master["Ciudad_Destino"].dropna().unique().tolist())
    ciudad_sel = st.sidebar.selectbox("Ciudad destino", ciudades_disp)

    excluir_fantasma = st.sidebar.checkbox("Excluir ventas fantasma del análisis", value=False)

    fecha_min = df_master["Fecha_Venta"].min()
    fecha_max = df_master["Fecha_Venta"].max()
    rango_fechas = st.sidebar.date_input(
        "Rango de fechas de venta",
        value=(fecha_min.date(), fecha_max.date()) if pd.notna(fecha_min) else None,
    )

    df_filtrado = df_master.copy()
    if categoria_sel != "(Todas)":
        df_filtrado = df_filtrado[df_filtrado["Categoria"] == categoria_sel]
    if ciudad_sel != "(Todas)":
        df_filtrado = df_filtrado[df_filtrado["Ciudad_Destino"] == ciudad_sel]
    if excluir_fantasma:
        df_filtrado = df_filtrado[~df_filtrado["Es_Venta_Fantasma"]]
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        inicio, fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
        df_filtrado = df_filtrado[
            (df_filtrado["Fecha_Venta"] >= inicio) & (df_filtrado["Fecha_Venta"] <= fin)
        ]

    st.sidebar.metric("Registros tras filtros", f"{len(df_filtrado):,}")

    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Auditoría de Calidad",
        "2️⃣ Integración y Features",
        "3️⃣ IA con Groq",
    ])

    # ---------------- TAB 1: AUDITORÍA -----------------------------------------
    with tab1:
        st.subheader("Health Score antes vs. después del procesamiento")

        numeric_map = {
            "Inventario": ["Stock_Actual", "Costo_Unitario_USD", "Lead_Time_Dias"],
            "Transacciones": ["Tiempo_Entrega_Real", "Precio_Venta_Final", "Cantidad_Vendida"],
            "Feedback": ["Satisfaccion_NPS", "Edad_Cliente", "Rating_Producto"],
        }
        score_before = {
            "Inventario": health_score(inv_raw, numeric_map["Inventario"]),
            "Transacciones": health_score(trans_raw, numeric_map["Transacciones"]),
            "Feedback": health_score(fb_raw, numeric_map["Feedback"]),
        }
        score_after = {
            "Inventario": health_score(inv_clean, numeric_map["Inventario"]),
            "Transacciones": health_score(trans_clean, numeric_map["Transacciones"]),
            "Feedback": health_score(fb_clean, numeric_map["Feedback"]),
        }

        col1, col2 = st.columns([1.3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Antes", x=list(score_before.keys()), y=list(score_before.values())))
            fig.add_trace(go.Bar(name="Después", x=list(score_after.keys()), y=list(score_after.values())))
            fig.update_layout(title="Health Score (0-100)", barmode="group", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            for nombre, df_ in [("Inventario", inv_raw), ("Transacciones", trans_raw), ("Feedback", fb_raw)]:
                with st.expander(f"% Nulidad por columna — {nombre}"):
                    st.dataframe(null_report(df_))

        st.markdown("### Decisiones de limpieza aplicadas")
        reporte_texto = [f"REPORTE DE LIMPIEZA — TechLogistics S.A. ({datetime.now():%Y-%m-%d %H:%M})", ""]
        for titulo, log, before, after in [
            ("Inventario", log_inv, score_before["Inventario"], score_after["Inventario"]),
            ("Transacciones", log_trans, score_before["Transacciones"], score_after["Transacciones"]),
            ("Feedback", log_fb, score_before["Feedback"], score_after["Feedback"]),
        ]:
            st.markdown(f"**{titulo}** — Health Score: {before} → {after}")
            reporte_texto.append(f"== {titulo} == (Health Score: {before} -> {after})")
            for linea in log:
                st.markdown(f"- {linea}")
                reporte_texto.append(f"- {linea}")
            reporte_texto.append("")

        st.download_button(
            label="⬇️ Descargar reporte de limpieza (.txt)",
            data="\n".join(reporte_texto).encode("utf-8"),
            file_name=f"reporte_limpieza_techlogistics_{datetime.now():%Y%m%d}.txt",
            mime="text/plain",
        )

    # ---------------- TAB 2: INTEGRACIÓN ---------------------------------------
    with tab2:
        st.subheader("Fuente única de verdad (merge estratégico)")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Ventas fantasma",
            f"{df_filtrado['Es_Venta_Fantasma'].sum():,}",
            f"{df_filtrado['Es_Venta_Fantasma'].mean() * 100:.1f}% del total filtrado",
        )
        c2.metric("Margen promedio", f"${df_filtrado['Margen_Utilidad'].mean():,.2f}")
        c3.metric("Brecha entrega promedio", f"{df_filtrado['Brecha_Entrega_Dias'].mean():.1f} días")

        st.dataframe(df_filtrado.head(200))

        fig2 = px.histogram(
            df_filtrado, x="Brecha_Entrega_Dias", nbins=40,
            title="Distribución: Brecha de Entrega (real - prometido)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
            df_filtrado.groupby("Categoria")["Ratio_Soporte_Categoria"].mean().reset_index(),
            x="Categoria", y="Ratio_Soporte_Categoria",
            title="Ratio de tickets de soporte por categoría",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.download_button(
            label="⬇️ Descargar dataset integrado (.csv)",
            data=df_filtrado.to_csv(index=False).encode("utf-8"),
            file_name="techlogistics_master_dataset.csv",
            mime="text/csv",
        )

    # ---------------- TAB 3: IA -------------------------------------------------
    with tab3:
        st.subheader("Recomendaciones estratégicas generadas con Llama-3 (Groq)")

        api_key_input = st.text_input(
            "GROQ_API_KEY", type="password", value=os.environ.get("GROQ_API_KEY", "")
        )
        generar = st.button("Generar recomendaciones para el segmento filtrado")

        if generar:
            resumen = construir_resumen_estadistico(df_filtrado)
            st.code(resumen, language="text")

            if not GROQ_AVAILABLE:
                st.error("El paquete `groq` no está instalado. Agrega `groq` a requirements.txt.")
            elif not api_key_input:
                st.warning("Ingresa tu GROQ_API_KEY para generar las recomendaciones.")
            elif df_filtrado.empty:
                st.warning("El filtro actual no tiene registros; ajusta los filtros de la barra lateral.")
            else:
                with st.spinner("Consultando a Llama-3 en Groq..."):
                    try:
                        texto = generar_recomendaciones_ia(resumen, api_key_input)
                        st.markdown(texto)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Error al llamar a la API de Groq: {exc}")


if __name__ == "__main__":
    main()
