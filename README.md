# 🛰️ Explorador de Filtros EuroSAT

Interfaz interactiva para explorar filtros espaciales y morfológicos sobre imágenes satelitales del dataset EuroSAT RGB.

## 🚀 Instalación y ejecución

### 1. Clona o descarga este proyecto
```bash
# Descarga app.py y requirements.txt en la misma carpeta
```

### 2. Crea un entorno virtual (recomendado)
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecuta la app
```bash
streamlit run app.py --  --server.headless true
```

La app se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 📁 Cómo obtener imágenes EuroSAT

1. Ve a [Kaggle — EuroSAT RGB](https://www.kaggle.com/datasets/salmaadell/eurosat-rgb)
2. Descarga el dataset (necesitas cuenta de Kaggle, es gratis)
3. Descomprime y elige imágenes de carpetas como:
   - `Highway/` → para detectar estructuras lineales
   - `River/` → para realzar bordes de cursos de agua
   - `Forest/` → para analizar texturas vegetales
   - `Residential/` → para distinguir estructuras urbanas

---

## 🗂️ Tabs disponibles

| Tab | Descripción |
|-----|-------------|
| 🧪 Pipeline | Encadena hasta 5 filtros en secuencia |
| ⚖️ Comparación | Aplica 4 filtros en paralelo y compáralos |
| 📊 Gráficas | Histogramas, perfil de intensidad, mapa de diferencia |
| 📋 Métricas | MSE, PSNR, SSIM por filtro |

---

## 🔧 Filtros disponibles

### Espaciales
- **Gaussiano** — suavizado con kernel configurable
- **Mediana** — reducción de ruido sal y pimienta
- **Bilateral** — suavizado preservando bordes
- **Sobel** — gradiente de bordes (X, Y o XY)
- **Laplaciano** — detección de bordes de segundo orden
- **Canny** — detección de bordes multiescala
- **Unsharp Mask** — realce de detalles

### Morfológicos
- Erosión, Dilatación, Apertura, Cierre
- Gradiente Morfológico (operación avanzada)
- Top-Hat Blanco (operación avanzada)
- Black-Hat (operación avanzada)
