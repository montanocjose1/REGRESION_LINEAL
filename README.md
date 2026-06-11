<<<<<<< HEAD
# Plataforma de Regresión Logística

Aplicación web educativa para cargar un dataset, entrenar un modelo de **regresión logística binaria**, validarlo y usarlo para predecir.

## Requisitos

- Python 3.10 o superior

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar la página web

```bash
python web_app.py
```

Abre en el navegador: **http://localhost:5000**

## Publicar en internet (Render)

Guía completa paso a paso: **[DEPLOY.md](DEPLOY.md)**

Resumen rápido:

1. Sube el proyecto a GitHub
2. Crea cuenta en [Render](https://render.com)
3. **New → Blueprint** y conecta el repositorio
4. Render usa `render.yaml` y te da una URL pública

Ejemplo de URL final: `https://regresion-logistica.onrender.com`

## Estructura del proyecto

| Archivo | Descripción |
|---------|-------------|
| `web_app.py` | Servidor web Flask |
| `templates/index.html` | Página principal |
| `static/css/style.css` | Estilos |
| `static/js/app.js` | Lógica del frontend |
| `model.py` | Entrenamiento, validación y predicción |
| `data/ejemplo.csv` | Dataset de ejemplo |

## Fases de la plataforma

1. **Presentación** — Qué entrega el proyecto y requisitos del dataset
2. **Fundamentos del modelo** — Ecuación, sigmoide e interpretación
3. **Entrenamiento** — Subir CSV/Excel, elegir variable objetivo y entrenar
4. **Validación y predicción** — Métricas, matriz de confusión, curva ROC y predicción

## Formato del dataset

- Una columna **objetivo** con exactamente **2 clases** (ej: `0`/`1`)
- Columnas predictoras **numéricas**
- Sin valores nulos
- Mínimo recomendado: 10 filas

## Ejemplo de uso

1. Ejecuta `python web_app.py`
2. Abre `http://localhost:5000`
3. Ve a **Fase 3 · Entrenamiento**
4. Sube `data/ejemplo.csv` o descarga el ejemplo desde **Fase 1**
5. Selecciona la columna objetivo `aprobo`
6. Pulsa **Entrenar modelo**
7. Ve a **Fase 4** para validar y predecir
=======
# REGRESION_LINEAL
>>>>>>> c7fe7e5c4a200d07a27d2fae24598aadb7784b83
