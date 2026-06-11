# Publicar la plataforma en internet (Render)

Esta guía te lleva paso a paso para que cualquier persona entre a tu página web sin instalar Python.

## Resumen

1. Subir el proyecto a **GitHub**
2. Conectar **Render** con ese repositorio
3. Render publica la app y te da un enlace tipo `https://regresion-logistica.onrender.com`

> **Nota:** En el plan gratuito de Render, la app puede tardar **30–60 segundos** en arrancar si nadie la usa durante un rato.

---

## Paso 1 · Crear cuenta en GitHub

1. Entra a [https://github.com](https://github.com)
2. Crea una cuenta (si no tienes)
3. Inicia sesión

---

## Paso 2 · Subir el proyecto a GitHub

Abre PowerShell en la carpeta del proyecto:

```powershell
cd "C:\Users\Jose Montano\Desktop\REGRESION"
```

Inicializa git y sube los archivos:

```powershell
git init
git add .
git commit -m "Plataforma web de regresión logística"
```

En GitHub, crea un repositorio nuevo:

1. Clic en **New repository**
2. Nombre sugerido: `regresion-logistica`
3. Déjalo **público** o privado (Render funciona con ambos)
4. **No** marques “Add README” (ya tienes archivos locales)
5. Clic en **Create repository**

GitHub te mostrará comandos. Ejecuta (cambia `TU_USUARIO` por tu usuario de GitHub):

```powershell
git branch -M main
git remote add origin https://github.com/TU_USUARIO/regresion-logistica.git
git push -u origin main
```

Si te pide usuario y contraseña, usa tu cuenta de GitHub. Puede pedir un **Personal Access Token** en lugar de contraseña.

---

## Paso 3 · Crear cuenta en Render

1. Entra a [https://render.com](https://render.com)
2. Regístrate con **GitHub** (es lo más fácil)
3. Autoriza a Render para ver tus repositorios

---

## Paso 4 · Desplegar la aplicación

### Opción A — Automática (recomendada)

El proyecto ya incluye `render.yaml`.

1. En Render, clic en **New +** → **Blueprint**
2. Conecta el repositorio `regresion-logistica`
3. Render detectará `render.yaml` solo
4. Clic en **Apply**
5. Espera 3–5 minutos mientras instala dependencias y arranca

### Opción B — Manual

1. En Render, clic en **New +** → **Web Service**
2. Conecta tu repositorio de GitHub
3. Configura así:

| Campo | Valor |
|-------|-------|
| **Name** | `regresion-logistica` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn web_app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| **Plan** | Free |

4. En **Environment Variables**, agrega:

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.11.9` |
| `SECRET_KEY` | (genera una clave larga aleatoria) |
| `FLASK_DEBUG` | `0` |

5. Clic en **Create Web Service**

---

## Paso 5 · Probar la página publicada

Cuando el despliegue termine, Render te dará una URL como:

```text
https://regresion-logistica.onrender.com
```

Abre ese enlace y verifica:

1. **Fase 1** — se ve la presentación
2. **Fase 3** — sube `data/ejemplo.csv`
3. Entrena con columna objetivo `aprobo`
4. **Fase 4** — revisa métricas y predicción

---

## Compartir con tu equipo o profesor

Copia y envía la URL de Render. Cualquier persona podrá:

- Ver las 4 fases educativas
- Subir su propio CSV
- Entrenar regresión logística
- Validar y predecir

---

## Actualizar la app después de cambios

Cada vez que modifiques el código:

```powershell
cd "C:\Users\Jose Montano\Desktop\REGRESION"
git add .
git commit -m "Describe tu cambio"
git push
```

Render detectará el push y **volverá a desplegar** automáticamente.

---

## Problemas frecuentes

### La página tarda mucho en cargar
Normal en el plan gratis. La app “duerme” tras inactividad y tarda un poco en despertar.

### Error al subir CSV muy grande
El plan gratis tiene límites de memoria. Usa datasets pequeños o medianos (idealmente menos de 5 MB).

### Error 502 / Application failed to respond
Revisa en Render → **Logs** si falló la instalación. Suele deberse a:
- Falta `requirements.txt`
- Error en `gunicorn web_app:app`

### No encuentra el archivo de ejemplo
Asegúrate de que `data/ejemplo.csv` esté incluido en el repositorio de GitHub.

---

## Alternativa: Railway

Si prefieres Railway ([https://railway.app](https://railway.app)):

1. Conecta el mismo repositorio de GitHub
2. Railway detecta Python automáticamente
3. Start command: `gunicorn web_app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`

Railway también ofrece créditos gratis limitados.

---

## Ejecutar en local (desarrollo)

```powershell
pip install -r requirements.txt
python web_app.py
```

Abre: [http://localhost:5000](http://localhost:5000)
