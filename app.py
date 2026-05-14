"""
====================================================
 Explorador de Filtros Espaciales y Morfológicos
 Visión Artificial — EuroSAT RGB
====================================================
Ejecutar con:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import io, copy

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Filtros EuroSAT · Visión Artificial",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    code, .stCode { font-family: 'Space Mono', monospace; }

    .main-title {
        font-size: 2.2rem; font-weight: 800; letter-spacing: -1px;
        background: linear-gradient(135deg, #00d4ff, #0066ff, #7b2fff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #8899aa; font-size: 0.95rem; margin-top: 0; }

    .step-chip {
        display: inline-block; background: #1a2035; border: 1px solid #2a3555;
        color: #00d4ff; border-radius: 6px; padding: 3px 10px;
        font-family: 'Space Mono', monospace; font-size: 0.78rem; margin: 2px;
    }
    .metric-box {
        background: #0d1526; border: 1px solid #1e2d4a;
        border-radius: 10px; padding: 12px 16px; text-align: center;
    }
    .metric-label { color: #8899aa; font-size: 0.78rem; margin-bottom: 4px; }
    .metric-value { color: #00d4ff; font-size: 1.6rem; font-weight: 700; }

    .stButton>button {
        background: linear-gradient(135deg, #0066ff, #7b2fff);
        color: white; border: none; border-radius: 8px;
        font-family: 'Syne', sans-serif; font-weight: 700;
        padding: 0.5rem 1.5rem; width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }

    .sidebar-section {
        background: #0d1526; border-radius: 10px;
        padding: 12px; margin-bottom: 12px;
        border: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] { background: #080e1e; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE FILTROS
# ═══════════════════════════════════════════════════════════════════════════════

def asegurar_kernel_impar(k):
    return k if k % 2 == 1 else k + 1

# — Filtros Espaciales ————————————————————————————————————————————————————————

def filtro_gaussiano(img, ksize=5, sigma=1.0):
    k = asegurar_kernel_impar(ksize)
    return cv2.GaussianBlur(img, (k, k), sigma)

def filtro_mediana(img, ksize=5):
    k = asegurar_kernel_impar(ksize)
    return cv2.medianBlur(img, k)

def filtro_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def filtro_sobel(img, ksize=3, direction="XY"):
    k = asegurar_kernel_impar(ksize)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    if direction == "X":
        r = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k))
    elif direction == "Y":
        r = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k))
    else:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        r = cv2.magnitude(sx, sy)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

def filtro_laplaciano(img, ksize=3):
    k = asegurar_kernel_impar(ksize)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    r = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    r = cv2.normalize(np.abs(r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

def filtro_canny(img, t1=50, t2=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    r = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

def filtro_unsharp(img, ksize=5, sigma=1.0, fuerza=1.5):
    k = asegurar_kernel_impar(ksize)
    blur = cv2.GaussianBlur(img, (k, k), sigma)
    return cv2.addWeighted(img, 1 + fuerza, blur, -fuerza, 0)

# — Filtros Morfológicos ───────────────────────────────────────────────────────

FORMA_KERNEL = {
    "Rectángulo": cv2.MORPH_RECT,
    "Elipse":     cv2.MORPH_ELLIPSE,
    "Cruz":       cv2.MORPH_CROSS,
}

OP_MORFO = {
    "Erosión":              cv2.MORPH_ERODE,
    "Dilatación":           cv2.MORPH_DILATE,
    "Apertura":             cv2.MORPH_OPEN,
    "Cierre":               cv2.MORPH_CLOSE,
    "Gradiente Morfológico":cv2.MORPH_GRADIENT,
    "Top-Hat (Blanco)":     cv2.MORPH_TOPHAT,
    "Black-Hat":            cv2.MORPH_BLACKHAT,
}

def filtro_morfologico(img, operacion="Erosión", ksize=5, forma="Rectángulo", iteraciones=1):
    k = asegurar_kernel_impar(ksize)
    kernel = cv2.getStructuringElement(FORMA_KERNEL[forma], (k, k))
    return cv2.morphologyEx(img, OP_MORFO[operacion], kernel, iterations=iteraciones)

# — Diferencia e imagen de error —————————————————————————————————————————————

def imagen_diferencia(orig, proc):
    orig_f = orig.astype(np.float32)
    proc_f = proc.astype(np.float32) if proc.shape == orig.shape else \
             cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, axis=2).astype(np.float32)
    diff = np.abs(orig_f - proc_f)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return diff_norm

def calcular_metricas(orig, proc):
    orig_g = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    proc_g = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY) if proc.ndim == 3 else proc
    mse = float(np.mean((orig_g.astype(float) - proc_g.astype(float)) ** 2))
    psnr = float(cv2.PSNR(orig_g, proc_g)) if mse > 0 else 999.0
    ssim_val = float(ssim(orig_g, proc_g, data_range=255))
    mean_orig = float(np.mean(orig_g))
    mean_proc = float(np.mean(proc_g))
    std_orig  = float(np.std(orig_g))
    std_proc  = float(np.std(proc_g))
    return dict(MSE=mse, PSNR=psnr, SSIM=ssim_val,
                Media_orig=mean_orig, Media_proc=mean_proc,
                Std_orig=std_orig, Std_proc=std_proc)

# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG = "#080e1e"
GRID_C  = "#1a2535"

def fig_histograma(imgs, labels):
    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 3.5))
    if len(imgs) == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)
    colors_rgb = ("#ff4d6d", "#51cf66", "#339af0")
    for ax, img, label in zip(axes, imgs, labels):
        ax.set_facecolor(DARK_BG)
        for i, (col, name) in enumerate(zip(colors_rgb, ["R", "G", "B"])):
            if img.ndim == 3:
                h = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
            else:
                h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            ax.fill_between(range(256), h, alpha=0.35, color=col)
            ax.plot(h, color=col, linewidth=0.8, label=name)
        ax.set_title(label, color="white", fontsize=10)
        ax.set_xlim(0, 255); ax.set_xlabel("Intensidad", color="#8899aa", fontsize=8)
        ax.set_ylabel("Frecuencia", color="#8899aa", fontsize=8)
        ax.tick_params(colors="#8899aa", labelsize=7)
        ax.spines[:].set_color(GRID_C)
        ax.legend(fontsize=7, facecolor="#0d1526", labelcolor="white", edgecolor=GRID_C)
    fig.tight_layout()
    return fig

def fig_perfil(orig, proc, label_proc):
    row = orig.shape[0] // 2
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    colores = [("#ff4d6d","R"), ("#51cf66","G"), ("#339af0","B")]
    for ax, (img, title) in zip(axes, [(orig, "Original"), (proc, label_proc)]):
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, color="white", fontsize=9)
        if img.ndim == 3:
            for (col, name), ch in zip(colores, range(3)):
                ax.plot(img[row, :, ch], color=col, linewidth=0.9, label=name)
        else:
            ax.plot(img[row, :], color="#00d4ff", linewidth=0.9)
        ax.set_ylabel("Intensidad", color="#8899aa", fontsize=7)
        ax.tick_params(colors="#8899aa", labelsize=7)
        ax.spines[:].set_color(GRID_C)
        ax.legend(fontsize=7, facecolor="#0d1526", labelcolor="white", edgecolor=GRID_C)
    axes[-1].set_xlabel("Píxel (columna)", color="#8899aa", fontsize=8)
    fig.tight_layout()
    return fig

def fig_diferencia(orig, proc):
    diff = imagen_diferencia(orig, proc)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    titulos = ["Original", "Procesada", "Diferencia (amplificada)"]
    imgs_show = [orig, proc, diff]
    for ax, img, t in zip(axes, imgs_show, titulos):
        ax.imshow(img)
        ax.set_title(t, color="white", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig

def fig_comparacion_multiple(img_orig, resultados):
    n = len(resultados) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor(DARK_BG)
    if n == 1:
        axes = [axes]
    axes[0].imshow(img_orig); axes[0].set_title("Original", color="white", fontsize=9); axes[0].axis("off")
    for ax, (lbl, img) in zip(axes[1:], resultados):
        ax.imshow(img); ax.set_title(lbl, color="white", fontsize=8, wrap=True); ax.axis("off")
    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# APLICAR FILTRO POR NOMBRE
# ═══════════════════════════════════════════════════════════════════════════════

FILTROS_ESPACIALES = [
    "Gaussiano", "Mediana", "Bilateral",
    "Sobel", "Laplaciano", "Canny", "Unsharp Mask",
]
FILTROS_MORFOLOGICOS = list(OP_MORFO.keys())
TODOS_FILTROS = ["— ninguno —"] + FILTROS_ESPACIALES + FILTROS_MORFOLOGICOS

def aplicar_filtro(img, nombre, p):
    if nombre == "— ninguno —":
        return img
    if nombre == "Gaussiano":
        return filtro_gaussiano(img, p["gauss_k"], p["gauss_s"])
    if nombre == "Mediana":
        return filtro_mediana(img, p["med_k"])
    if nombre == "Bilateral":
        return filtro_bilateral(img, p["bil_d"], p["bil_sc"], p["bil_ss"])
    if nombre == "Sobel":
        return filtro_sobel(img, p["sob_k"], p["sob_dir"])
    if nombre == "Laplaciano":
        return filtro_laplaciano(img, p["lap_k"])
    if nombre == "Canny":
        return filtro_canny(img, p["can_t1"], p["can_t2"])
    if nombre == "Unsharp Mask":
        return filtro_unsharp(img, p["unsh_k"], p["unsh_s"], p["unsh_f"])
    if nombre in FILTROS_MORFOLOGICOS:
        return filtro_morfologico(img, nombre, p["morph_k"], p["morph_shape"], p["morph_iter"])
    return img

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">🛰️ Explorador de Filtros EuroSAT</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Filtros espaciales y morfológicos · Visión Artificial</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Imagen")
    uploaded = st.file_uploader("Cargar imagen EuroSAT (.jpg / .png)", type=["jpg","jpeg","png"])

    st.markdown("---")
    st.markdown("### 🔧 Parámetros Globales")

    with st.expander("Filtros Espaciales", expanded=True):
        gauss_k  = st.slider("Gaussiano — Kernel",    1, 31, 5, 2, key="gk")
        gauss_s  = st.slider("Gaussiano — Sigma",     0.1, 10.0, 1.0, 0.1, key="gs")
        med_k    = st.slider("Mediana — Kernel",      1, 31, 5, 2, key="mk")
        bil_d    = st.slider("Bilateral — Diámetro",  1, 25, 9, 2, key="bd")
        bil_sc   = st.slider("Bilateral — σ Color",   1, 200, 75, key="bsc")
        bil_ss   = st.slider("Bilateral — σ Espacio", 1, 200, 75, key="bss")
        sob_k    = st.slider("Sobel — Kernel",        1, 7, 3, 2, key="sk")
        sob_dir  = st.selectbox("Sobel — Dirección",  ["XY","X","Y"], key="sd")
        lap_k    = st.slider("Laplaciano — Kernel",   1, 31, 3, 2, key="lk")
        can_t1   = st.slider("Canny — Umbral 1",      0, 255, 50, key="ct1")
        can_t2   = st.slider("Canny — Umbral 2",      0, 255, 150, key="ct2")
        unsh_k   = st.slider("Unsharp — Kernel",      1, 31, 5, 2, key="uk")
        unsh_s   = st.slider("Unsharp — Sigma",       0.1, 10.0, 1.0, 0.1, key="us")
        unsh_f   = st.slider("Unsharp — Fuerza",      0.1, 5.0, 1.5, 0.1, key="uf")

    with st.expander("Filtros Morfológicos", expanded=True):
        morph_k     = st.slider("Kernel morfológico",   1, 31, 5, 2, key="mok")
        morph_shape = st.selectbox("Forma del elemento",["Rectángulo","Elipse","Cruz"], key="mosh")
        morph_iter  = st.slider("Iteraciones",          1, 10, 1, key="moit")

params = dict(
    gauss_k=gauss_k, gauss_s=gauss_s,
    med_k=med_k,
    bil_d=bil_d, bil_sc=bil_sc, bil_ss=bil_ss,
    sob_k=sob_k, sob_dir=sob_dir,
    lap_k=lap_k,
    can_t1=can_t1, can_t2=can_t2,
    unsh_k=unsh_k, unsh_s=unsh_s, unsh_f=unsh_f,
    morph_k=morph_k, morph_shape=morph_shape, morph_iter=morph_iter,
)

# ─── TABS ───────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 **Carga una imagen EuroSAT** desde la barra lateral para comenzar.")
    st.markdown("""
    #### ¿Cómo obtener imágenes?
    1. Descarga el dataset desde [Kaggle — EuroSAT RGB](https://www.kaggle.com/datasets/salmaadell/eurosat-rgb)
    2. Elige imágenes de clases como `Highway`, `River`, `Forest`, `Residential`, etc.
    3. Carga aquí cualquier imagen `.jpg` de 64×64 píxeles (o mayor)
    """)
    st.stop()

# Cargar imagen
pil_img  = Image.open(uploaded).convert("RGB")
img_orig = np.array(pil_img)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Pipeline de Filtros",
    "⚖️ Comparación Múltiple",
    "📊 Gráficas de Análisis",
    "📋 Métricas",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pipeline: encadena hasta 5 filtros en secuencia")
    cols_pip = st.columns(5)
    pipeline_names = []
    for i, col in enumerate(cols_pip):
        with col:
            sel = st.selectbox(f"Paso {i+1}", TODOS_FILTROS, key=f"pip_{i}")
            pipeline_names.append(sel)

    if st.button("▶ Aplicar Pipeline"):
        steps_imgs  = [img_orig]
        steps_names = ["Original"]
        current = img_orig.copy()

        for nombre in pipeline_names:
            if nombre == "— ninguno —":
                continue
            current = aplicar_filtro(current.copy(), nombre, params)
            steps_imgs.append(current.copy())
            steps_names.append(nombre)

        # Mostrar pasos
        n_steps = len(steps_imgs)
        cols_show = st.columns(n_steps)
        for col, img_s, name_s in zip(cols_show, steps_imgs, steps_names):
            with col:
                st.image(img_s, caption=name_s, use_container_width=True)

        # Chips del pipeline
        st.markdown("**Pipeline aplicado:** " + " → ".join(
            [f'<span class="step-chip">{n}</span>' for n in steps_names]
        ), unsafe_allow_html=True)

        # Diferencia
        if len(steps_imgs) > 1:
            st.markdown("---")
            st.markdown("#### Imagen de diferencia (original vs resultado final)")
            fig_diff = fig_diferencia(img_orig, steps_imgs[-1])
            st.pyplot(fig_diff, use_container_width=True)
            plt.close(fig_diff)

            # Histogramas orig vs final
            st.markdown("#### Histogramas")
            fig_h = fig_histograma(
                [img_orig, steps_imgs[-1]],
                ["Original", steps_names[-1]]
            )
            st.pyplot(fig_h, use_container_width=True)
            plt.close(fig_h)

            # Perfil de intensidad
            st.markdown("#### Perfil de intensidad — fila central")
            fig_p = fig_perfil(img_orig, steps_imgs[-1], steps_names[-1])
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)

    else:
        st.image(img_orig, caption="Imagen original", width=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARACIÓN MÚLTIPLE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Aplica varios filtros individuales y compáralos lado a lado")

    cols_cmp = st.columns(4)
    filtros_comp = []
    for i, col in enumerate(cols_cmp):
        with col:
            sel = st.selectbox(f"Filtro {chr(65+i)}", TODOS_FILTROS, key=f"cmp_{i}")
            filtros_comp.append(sel)

    if st.button("⚖️ Comparar Filtros"):
        resultados = []
        for nombre in filtros_comp:
            if nombre != "— ninguno —":
                res = aplicar_filtro(img_orig.copy(), nombre, params)
                resultados.append((nombre, res))

        if resultados:
            fig_cmp = fig_comparacion_multiple(img_orig, resultados)
            st.pyplot(fig_cmp, use_container_width=True)
            plt.close(fig_cmp)

            # Histogramas de cada uno
            st.markdown("#### Histogramas individuales")
            imgs_h  = [img_orig] + [r[1] for r in resultados]
            labels_h = ["Original"] + [r[0] for r in resultados]
            fig_hm = fig_histograma(imgs_h, labels_h)
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)
        else:
            st.warning("Selecciona al menos un filtro distinto de '— ninguno —'.")
    else:
        st.image(img_orig, caption="Imagen original", width=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Gráficas de análisis para un filtro individual")
    filtro_analisis = st.selectbox("Filtro a analizar", TODOS_FILTROS[1:], key="anal")

    if st.button("📊 Generar gráficas"):
        img_proc = aplicar_filtro(img_orig.copy(), filtro_analisis, params)

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img_orig,  caption="Original",         use_container_width=True)
        with col_b:
            st.image(img_proc, caption=filtro_analisis,    use_container_width=True)

        st.markdown("---")

        # Histogramas
        st.markdown("#### Histogramas — Distribución de intensidad")
        fig_h2 = fig_histograma([img_orig, img_proc], ["Original", filtro_analisis])
        st.pyplot(fig_h2, use_container_width=True); plt.close(fig_h2)

        # Perfil de intensidad
        st.markdown("#### Perfil de intensidad — fila central")
        fig_p2 = fig_perfil(img_orig, img_proc, filtro_analisis)
        st.pyplot(fig_p2, use_container_width=True); plt.close(fig_p2)

        # Imagen diferencia
        st.markdown("#### Mapa de diferencia")
        fig_d2 = fig_diferencia(img_orig, img_proc)
        st.pyplot(fig_d2, use_container_width=True); plt.close(fig_d2)

        # Histograma de diferencia
        diff = imagen_diferencia(img_orig, img_proc)
        diff_g = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        fig_dh, ax = plt.subplots(figsize=(8, 2.5))
        fig_dh.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.fill_between(range(256), cv2.calcHist([diff_g],[0],None,[256],[0,256]).flatten(),
                        alpha=0.5, color="#ff6b6b")
        ax.plot(cv2.calcHist([diff_g],[0],None,[256],[0,256]).flatten(), color="#ff6b6b", lw=0.9)
        ax.set_title("Histograma de diferencia", color="white", fontsize=9)
        ax.set_facecolor(DARK_BG); ax.tick_params(colors="#8899aa", labelsize=7)
        ax.spines[:].set_color(GRID_C); ax.set_xlim(0,255)
        st.pyplot(fig_dh, use_container_width=True); plt.close(fig_dh)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Métricas de calidad de imagen")
    st.markdown("Compara cuantitativamente el efecto de cada filtro sobre la imagen original.")

    filtros_metrica = st.multiselect(
        "Selecciona filtros a medir",
        TODOS_FILTROS[1:],
        default=["Gaussiano", "Mediana", "Erosión", "Canny"],
        key="met_sel"
    )

    if st.button("📋 Calcular métricas"):
        if not filtros_metrica:
            st.warning("Selecciona al menos un filtro.")
        else:
            tabla = []
            for nombre in filtros_metrica:
                img_proc = aplicar_filtro(img_orig.copy(), nombre, params)
                # Convertir a RGB si es necesario para métricas
                if img_proc.shape != img_orig.shape:
                    img_proc = cv2.cvtColor(
                        cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB
                    )
                m = calcular_metricas(img_orig, img_proc)
                m["Filtro"] = nombre
                tabla.append(m)

            import pandas as pd
            df = pd.DataFrame(tabla).set_index("Filtro")
            df.columns = ["MSE ↓", "PSNR ↑ (dB)", "SSIM ↑",
                          "Media orig", "Media proc", "Std orig", "Std proc"]
            df = df.round(3)
            st.dataframe(df.style.background_gradient(cmap="Blues", subset=["SSIM ↑"])
                                  .background_gradient(cmap="Reds_r", subset=["MSE ↓"]),
                         use_container_width=True)

            st.markdown("---")
            # Gráfico de barras de SSIM
            fig_bar, ax = plt.subplots(figsize=(10, 3))
            fig_bar.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(DARK_BG)
            bars = ax.barh(df.index, df["SSIM ↑"], color="#339af0", alpha=0.85)
            ax.set_xlim(0, 1)
            ax.set_xlabel("SSIM (↑ mejor)", color="#8899aa", fontsize=8)
            ax.set_title("Similitud estructural (SSIM) por filtro", color="white", fontsize=10)
            ax.tick_params(colors="#8899aa", labelsize=8)
            ax.spines[:].set_color(GRID_C)
            for bar in bars:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():.3f}", va="center", color="white", fontsize=7)
            st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

            st.caption("**MSE** = Error cuadrático medio (↓ mejor) · **PSNR** = Relación señal-ruido (↑ mejor) · **SSIM** = Similitud estructural (↑ mejor, 1.0 = idéntica)")
