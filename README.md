# Dashboard Logístico, Rentabilidad y Fidelidad de Clientes

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://challenge02fundcienciadatos-mubkenlfj3zzayqtetft2t.streamlit.app/)

Tablero de control interactivo desarrollado en Streamlit y Plotly para el diagnóstico operativo, financiero y de fidelidad de clientes. Integra análisis de calidad de datos, detección de fuga de capital, evaluación de cuellos de botella logísticos y un módulo de síntesis automatizada mediante Llama-3 (Groq API).

---

## Descripción del Problema

En operaciones de e-commerce y distribución logística, la falta de integración entre las ventas, el inventario central y el feedback post-venta genera puntos ciegos operativos que erosionan la rentabilidad y la reputación de la marca. 

Este dashboard aborda y resuelve cinco problemáticas de negocio clave:

1. **Fuga de Capital y Márgenes Negativos:**
   - Detección de SKUs comercializados por debajo del costo.
   - Identificación de la distorsión financiera en Smartphones: unidades vendidas a un precio promedio de ~$1,009 USD con costos unitarios de adquisición de ~$2,844 USD, generando una brecha de pérdida de más de $1,000 USD por transacción que subvenciona una operación con NPS negativo (-4.70).

2. **Venta Invisible (SKUs Fantasma):**
   - Cuantificación de ingresos generados por SKUs que no figuran en el maestro de inventario central (En_Inventario = False), identificando riesgos por falta de control de catalogación e imposibilidad de calcular el margen real.

3. **Crisis y Cuellos de Botella Logísticos:**
   - Evaluación de tiempos reales de entrega por ciudad y bodega de origen (ej. demoras promedio de ~23.4 días en Laptops).
   - Análisis de tasas de envíos retrasados o perdidos por punto de origen.

4. **Diagnóstico de Fidelidad y Satisfacción:**
   - Evaluación de la paradoja entre alta disponibilidad de stock y baja satisfacción percibida (NPS / Rating).
   - Análisis cualitativo de comentarios de clientes ("Lento", "Dañado", "No volvería").

5. **Riesgo Operativo y Auditoría:**
   - Relación entre la antigüedad de la última revisión física del stock en bodega y el incremento en la tasa de apertura de tickets de soporte técnico.

---

## Estructura del Repositorio

```text
.
├── data/
│   ├── transacciones_logistica_v2.csv   # Histórico de ventas, logística y precios
│   ├── feedback_clientes_v2.csv        # Encuestas de satisfacción, NPS, ratings y comentarios
│   └── inventario_central_v2.csv       # Catálogo maestro, stock, costos y tiempos de reorden
├── documents/
│   └── taller_practico_02.pdf           # Documento en formato PDF con el informe analítico extenso
├── app.py                              # Aplicación principal de Streamlit e interfaz gráfica
├── data_cleaner.py                     # Módulo backend de limpieza, imputaciones y ETL
├── requirements.txt                    # Lista de librerías y dependencias Python
└── README.md                           # Documentación general del proyecto
```

---

## Guía de Instalación y Ejecución Local

Sigue estos pasos para instalar y ejecutar la aplicación en tu entorno local:

### 1. Clonar el repositorio
```bash
git clone https://github.com/Nath-Ramirez/Challenge02_FundCienciaDatos.git
cd Challenge02_FundCienciaDatos
```

### 2. Crear y activar un entorno virtual
* **En Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **En Windows:**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Iniciar la aplicación
```bash
streamlit run app.py
```
El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## Integración del Módulo de IA (Groq API)

La pestaña Diagnóstico IA (Groq) utiliza el modelo de lenguaje Llama-3 (llama-3.3-70b-versatile) para generar un resumen ejecutivo con recomendaciones estratégicas basadas en los datos filtrados en tiempo real.

1. Consigue una API Key gratuita en [Groq Console](https://console.groq.com/).
2. Ingrésala en el campo desplegable de la pestaña correspondiente dentro de la aplicación.

---

## Enlace a la Aplicación en la Nube

Puedes acceder al dashboard desplegado en producción a través del siguiente enlace:

* [Ver Dashboard en Streamlit Community Cloud](https://challenge02fundcienciadatos-mubkenlfj3zzayqtetft2t.streamlit.app/)
