"""Plataforma educativa de Regresión Logística."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from model import predict_single, train_logistic_regression, validate_dataset

st.set_page_config(
    page_title="Plataforma de Regresión Logística",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        margin-bottom: 1.5rem;
    }
    .phase-box {
        background: #f0f7ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults = {
        "dataset": None,
        "target_column": None,
        "training_result": None,
        "file_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_dataset(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        st.error("Formato no soportado. Usa CSV o Excel.")
        return None
    except Exception as exc:
        st.error(f"No se pudo leer el archivo: {exc}")
        return None


def render_phase_1() -> None:
    st.markdown('<p class="main-header">Fase 1 · Presentación</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Qué entregamos y cuál es el objetivo del proyecto</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="phase-box">
        <strong>Entregable del proyecto:</strong> una plataforma web que permite cargar un dataset,
        entrenar un modelo de <em>regresión logística</em>, validarlo con métricas claras y usarlo
        para predecir nuevos casos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Objetivo")
        st.write(
            """
            Construir una herramienta sencilla que demuestre el ciclo completo de un modelo
            de clasificación binaria:

            1. **Carga de datos** — subir un archivo CSV o Excel.
            2. **Entrenamiento** — ajustar la regresión logística.
            3. **Validación** — medir qué tan bien generaliza el modelo.
            4. **Predicción** — clasificar nuevos registros.
            """
        )

    with col2:
        st.subheader("Requisitos del dataset")
        st.write(
            """
            - Formato: **CSV** o **Excel**
            - Una columna **objetivo** con **2 clases** (ej: 0/1, Sí/No)
            - Columnas predictoras **numéricas**
            - Sin valores nulos
            - Mínimo recomendado: **10 filas**
            """
        )

        example_path = "data/ejemplo.csv"
        try:
            example_df = pd.read_csv(example_path)
            st.download_button(
                label="Descargar dataset de ejemplo",
                data=example_df.to_csv(index=False).encode("utf-8"),
                file_name="ejemplo_regresion_logistica.csv",
                mime="text/csv",
            )
            with st.expander("Vista previa del ejemplo"):
                st.dataframe(example_df, use_container_width=True)
        except FileNotFoundError:
            st.info("El archivo de ejemplo no está disponible en esta instalación.")


def render_phase_2() -> None:
    st.markdown('<p class="main-header">Fase 2 · Fundamentos del modelo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Qué es la regresión logística y qué ocurre por detrás</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        La **regresión logística** es un modelo de clasificación binaria. En lugar de predecir
        un número continuo, estima la **probabilidad** de que una observación pertenezca a la
        clase positiva (por ejemplo, "aprobar" = 1).
        """
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("Ecuación del modelo")
        st.latex(r"P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots)}}")

        st.write(
            """
            - **β₀** → intercepto (sesgo del modelo)
            - **βᵢ** → coeficientes (peso de cada variable)
            - **Sigmoide** → convierte una combinación lineal en probabilidad entre 0 y 1
            - Si P ≥ 0.5 → clase 1; si P < 0.5 → clase 0
            """
        )

        st.subheader("Interpretación")
        st.write(
            """
            - Un coeficiente **positivo** aumenta la probabilidad de la clase 1.
            - Un coeficiente **negativo** la reduce.
            - **exp(βᵢ)** es el *odds ratio*: cuánto cambian las odds al aumentar xᵢ en 1 unidad.
            """
        )

    with col2:
        st.subheader("Función sigmoide")
        x = np.linspace(-6, 6, 200)
        y = 1 / (1 + np.exp(-x))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Sigmoide", line=dict(color="#1f77b4", width=3)))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Umbral 0.5")
        fig.update_layout(
            title="De score lineal a probabilidad",
            xaxis_title="Score lineal (β₀ + βX)",
            yaxis_title="Probabilidad P(Y=1)",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Detrás del modelo, scikit-learn optimiza los coeficientes β para minimizar "
            "la función de pérdida logística (equivalente a maximizar la verosimilitud)."
        )


def render_phase_3() -> None:
    st.markdown('<p class="main-header">Fase 3 · Entrenamiento del modelo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Sube tu dataset, elige la variable objetivo y entrena</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Sube tu dataset (CSV o Excel)",
        type=["csv", "xlsx", "xls"],
        help="Debe incluir una columna objetivo binaria y variables numéricas.",
    )

    if uploaded is not None:
        df = load_dataset(uploaded)
        if df is not None:
            st.session_state["dataset"] = df
            st.session_state["file_name"] = uploaded.name

    df = st.session_state.get("dataset")

    if df is None:
        st.warning("Sube un dataset para continuar con el entrenamiento.")
        return

    st.success(f"Dataset cargado: **{st.session_state.get('file_name', 'dataset')}** — {len(df)} filas, {len(df.columns)} columnas")

    st.subheader("Vista previa")
    st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No hay columnas numéricas en el dataset.")
        return

    target_column = st.selectbox(
        "Selecciona la columna objetivo (variable a predecir)",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
    )
    st.session_state["target_column"] = target_column

    feature_cols = [c for c in df.columns if c != target_column]
    st.write(f"**Variables predictoras:** {', '.join(feature_cols) if feature_cols else 'Ninguna'}")

    is_valid, message = validate_dataset(df, target_column)
    if not is_valid:
        st.error(message)
        return

    st.success(message)

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Porcentaje para prueba (validación)", 10, 40, 25, 5) / 100
    with col2:
        random_state = st.number_input("Semilla aleatoria (reproducibilidad)", min_value=0, value=42)

    st.markdown(
        """
        **Qué ocurre al entrenar:**
        1. Se divide el dataset en entrenamiento y prueba.
        2. Las variables se estandarizan (media 0, desviación 1).
        3. Se ajustan los coeficientes β mediante regresión logística.
        4. El modelo aprende a separar las dos clases con la mejor frontera probabilística.
        """
    )

    if st.button("Entrenar modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando regresión logística..."):
            try:
                result = train_logistic_regression(
                    df,
                    target_column=target_column,
                    test_size=test_size,
                    random_state=int(random_state),
                )
                st.session_state["training_result"] = result
                st.success("Modelo entrenado correctamente.")
            except Exception as exc:
                st.error(f"Error durante el entrenamiento: {exc}")
                return

    result = st.session_state.get("training_result")
    if result is None:
        return

    st.divider()
    st.subheader("Resultados del entrenamiento")

    m1, m2, m3 = st.columns(3)
    m1.metric("Filas entrenamiento", result.train_size)
    m2.metric("Filas prueba", result.test_size)
    m3.metric("Variables", len(result.feature_names))

    st.subheader("Coeficientes aprendidos")
    st.dataframe(result.coefficients, use_container_width=True)
    st.caption(f"Intercepto (β₀): **{result.intercept:.4f}**")

    fig, ax = plt.subplots(figsize=(8, max(3, len(result.feature_names) * 0.5)))
    sns.barplot(
        data=result.coefficients,
        y="Variable",
        x="Coeficiente",
        palette="coolwarm",
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Importancia direccional de cada variable")
    st.pyplot(fig)
    plt.close(fig)


def render_phase_4() -> None:
    st.markdown('<p class="main-header">Fase 4 · Validación y predicción</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Cómo evaluar el modelo y usarlo con datos nuevos</p>',
        unsafe_allow_html=True,
    )

    result = st.session_state.get("training_result")

    if result is None:
        st.warning("Primero entrena un modelo en la **Fase 3**.")
        return

    st.subheader("Validación del modelo")
    st.write(
        """
        El modelo se evalúa con datos que **no vio durante el entrenamiento** (conjunto de prueba).
        Esto permite saber si generaliza bien o solo memorizó los datos.
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{result.metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{result.metrics['precision']:.2%}")
    c3.metric("Recall", f"{result.metrics['recall']:.2%}")
    c4.metric("F1-Score", f"{result.metrics['f1']:.2%}")
    if not np.isnan(result.metrics.get("roc_auc", float("nan"))):
        c5.metric("ROC-AUC", f"{result.metrics['roc_auc']:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matriz de confusión")
        cm = result.confusion
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicho", y="Real", color="Conteo"),
            x=["Clase 0", "Clase 1"],
            y=["Clase 0", "Clase 1"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.write(
            """
            - **Verdaderos positivos / negativos** → aciertos
            - **Falsos positivos** → predijo 1 pero era 0
            - **Falsos negativos** → predijo 0 pero era 1
            """
        )

    with col2:
        st.subheader("Curva ROC")
        if len(result.roc_fpr) > 0:
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=result.roc_fpr,
                    y=result.roc_tpr,
                    mode="lines",
                    name="ROC",
                    line=dict(color="#2ca02c", width=3),
                )
            )
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash"))
            )
            fig_roc.update_layout(
                xaxis_title="Tasa de falsos positivos",
                yaxis_title="Tasa de verdaderos positivos",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("No se pudo calcular la curva ROC.")

    with st.expander("Reporte de clasificación detallado"):
        st.code(result.report)

    st.divider()
    st.subheader("Predicción con datos nuevos")

    st.write(
        "Introduce valores para cada variable y el modelo devolverá la clase predicha "
        "y la probabilidad estimada."
    )

    input_values: dict[str, float] = {}
    cols = st.columns(min(3, len(result.feature_names)))

    for i, feature in enumerate(result.feature_names):
        with cols[i % len(cols)]:
            default = float(st.session_state["dataset"][feature].median())
            input_values[feature] = st.number_input(
                feature,
                value=default,
                format="%.4f",
                key=f"pred_{feature}",
            )

    if st.button("Predecir", type="primary"):
        pred_class, positive_prob, confidence = predict_single(result, input_values)
        negative_prob = 1 - positive_prob

        st.markdown("### Resultado")
        p1, p2, p3 = st.columns(3)
        p1.metric("Clase predicha", int(pred_class))
        p2.metric("Probabilidad clase 1", f"{positive_prob:.2%}")
        p3.metric("Probabilidad clase 0", f"{negative_prob:.2%}")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Clase 0", "Clase 1"],
                    y=[negative_prob, positive_prob],
                    marker_color=["#ff7f0e", "#1f77b4"],
                    text=[f"{negative_prob:.1%}", f"{positive_prob:.1%}"],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title="Probabilidades predichas",
            yaxis_title="Probabilidad",
            yaxis_range=[0, 1],
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    init_session_state()

    with st.sidebar:
        st.title("Regresión Logística")
        st.caption("Plataforma educativa")
        st.divider()

        phase = st.radio(
            "Navegación",
            options=[
                "1. Presentación",
                "2. Fundamentos del modelo",
                "3. Entrenamiento",
                "4. Validación y predicción",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            """
            **Flujo recomendado**
            1. Lee qué entrega el proyecto
            2. Entiende la regresión logística
            3. Sube datos y entrena
            4. Valida y predice
            """
        )

        if st.session_state.get("training_result"):
            st.success("Modelo entrenado")
        elif st.session_state.get("dataset") is not None:
            st.info("Dataset cargado")
        else:
            st.warning("Sin dataset")

    if phase.startswith("1"):
        render_phase_1()
    elif phase.startswith("2"):
        render_phase_2()
    elif phase.startswith("3"):
        render_phase_3()
    elif phase.startswith("4"):
        render_phase_4()


if __name__ == "__main__":
    main()
