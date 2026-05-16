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
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import io, copy
import pandas as pd

st.set_page_config(
    page_title="Filtros EuroSAT · Visión Artificial",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .stButton>button {
        background: linear-gradient(135deg, #0066ff, #7b2fff);
        color: white; border: none; border-radius: 8px;
        font-family: 'Syne', sans-serif; font-weight: 700;
        padding: 0.5rem 1.5rem; width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }
    [data-testid="stSidebar"] { background: #080e1e; }
</style>
""", unsafe_allow_html=True)

# ── FILTROS ───────────────────────────────────────────────────────────────────

def asegurar_kernel_impar(k):
    return k if k % 2 == 1 else k + 1

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

def a_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def filtro_clahe(img, clip_limit=2.0, tile_size=8):
    img = a_uint8(img)
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    else:
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(img)

FORMA_KERNEL = {"Rectángulo": cv2.MORPH_RECT, "Elipse": cv2.MORPH_ELLIPSE, "Cruz": cv2.MORPH_CROSS}
OP_MORFO = {
    "Erosión": cv2.MORPH_ERODE, "Dilatación": cv2.MORPH_DILATE,
    "Apertura": cv2.MORPH_OPEN, "Cierre": cv2.MORPH_CLOSE,
    "Gradiente Morfológico": cv2.MORPH_GRADIENT,
    "Top-Hat (Blanco)": cv2.MORPH_TOPHAT, "Black-Hat": cv2.MORPH_BLACKHAT,
}

def filtro_morfologico(img, operacion="Erosión", ksize=5, forma="Rectángulo", iteraciones=1):
    img = a_uint8(img)
    k = asegurar_kernel_impar(ksize)
    kernel = cv2.getStructuringElement(FORMA_KERNEL[forma], (k, k))
    return cv2.morphologyEx(img, OP_MORFO[operacion], kernel, iterations=iteraciones)

def imagen_diferencia(orig, proc):
    orig_f = orig.astype(np.float32)
    proc_f = proc.astype(np.float32) if proc.shape == orig.shape else \
             cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, axis=2).astype(np.float32)
    diff = np.abs(orig_f - proc_f)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def calcular_metricas(orig, proc):
    orig_g = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    proc_g = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY) if proc.ndim == 3 else proc
    mse = float(np.mean((orig_g.astype(float) - proc_g.astype(float)) ** 2))
    psnr = float(cv2.PSNR(orig_g, proc_g)) if mse > 0 else 999.0
    ssim_val = float(ssim(orig_g, proc_g, data_range=255))
    return dict(MSE=mse, PSNR=psnr, SSIM=ssim_val,
                Media_orig=float(np.mean(orig_g)), Media_proc=float(np.mean(proc_g)),
                Std_orig=float(np.std(orig_g)), Std_proc=float(np.std(proc_g)))

# ── GRÁFICAS ──────────────────────────────────────────────────────────────────

DARK_BG = "#080e1e"
GRID_C  = "#1a2535"

def fig_histograma(imgs, labels):
    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 3.5))
    if len(imgs) == 1: axes = [axes]
    fig.patch.set_facecolor(DARK_BG)
    colors_rgb = ("#ff4d6d", "#51cf66", "#339af0")
    for ax, img, label in zip(axes, imgs, labels):
        ax.set_facecolor(DARK_BG)
        for i, (col, name) in enumerate(zip(colors_rgb, ["R", "G", "B"])):
            h = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten() if img.ndim == 3 else \
                cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            ax.fill_between(range(256), h, alpha=0.35, color=col)
            ax.plot(h, color=col, linewidth=0.8, label=name)
        ax.set_title(label, color="white", fontsize=10)
        ax.set_xlim(0, 255); ax.tick_params(colors="#8899aa", labelsize=7)
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
        ax.set_facecolor(DARK_BG); ax.set_title(title, color="white", fontsize=9)
        if img.ndim == 3:
            for (col, name), ch in zip(colores, range(3)):
                ax.plot(img[row, :, ch], color=col, linewidth=0.9, label=name)
        else:
            ax.plot(img[row, :], color="#00d4ff", linewidth=0.9)
        ax.tick_params(colors="#8899aa", labelsize=7); ax.spines[:].set_color(GRID_C)
        ax.legend(fontsize=7, facecolor="#0d1526", labelcolor="white", edgecolor=GRID_C)
    axes[-1].set_xlabel("Píxel (columna)", color="#8899aa", fontsize=8)
    fig.tight_layout()
    return fig

def fig_diferencia(orig, proc):
    diff = imagen_diferencia(orig, proc)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    for ax, img, t in zip(axes, [orig, proc, diff], ["Original", "Procesada", "Diferencia"]):
        ax.imshow(img); ax.set_title(t, color="white", fontsize=9); ax.axis("off")
    fig.tight_layout()
    return fig

def fig_comparacion_multiple(img_orig, resultados):
    n = len(resultados) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor(DARK_BG)
    if n == 1: axes = [axes]
    axes[0].imshow(img_orig); axes[0].set_title("Original", color="white", fontsize=9); axes[0].axis("off")
    for ax, (lbl, img) in zip(axes[1:], resultados):
        ax.imshow(img); ax.set_title(lbl, color="white", fontsize=8); ax.axis("off")
    fig.tight_layout()
    return fig

# ── CATÁLOGO DE FILTROS ───────────────────────────────────────────────────────

FILTROS_ESPACIALES   = ["Gaussiano", "CLAHE", "Mediana", "Bilateral", "Sobel", "Laplaciano", "Canny", "Unsharp Mask"]
FILTROS_MORFOLOGICOS = list(OP_MORFO.keys())
TODOS_FILTROS        = ["— ninguno —"] + FILTROS_ESPACIALES + FILTROS_MORFOLOGICOS

# Parámetros por defecto para el pipeline dinámico
DEFAULTS = {
    "Gaussiano":             {"ksize": 5,  "sigma": 1.0},
    "CLAHE":                 {"clip_limit": 2.0, "tile_size": 8},
    "Mediana":               {"ksize": 5},
    "Bilateral":             {"d": 9, "sigma_color": 75, "sigma_space": 75},
    "Sobel":                 {"ksize": 3, "direction": "XY"},
    "Laplaciano":            {"ksize": 3},
    "Canny":                 {"t1": 50, "t2": 150},
    "Unsharp Mask":          {"ksize": 5, "sigma": 1.0, "fuerza": 1.5},
    "Erosión":               {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
    "Dilatación":            {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
    "Apertura":              {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
    "Cierre":                {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
    "Gradiente Morfológico": {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
    "Top-Hat (Blanco)":      {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
    "Black-Hat":             {"ksize": 5, "forma": "Rectángulo", "iteraciones": 1},
}

def params_por_defecto(filtro):
    return copy.deepcopy(DEFAULTS.get(filtro, {}))

# ── APLICAR FILTRO (global params, usado por Tabs 2-4) ───────────────────────

def aplicar_filtro(img, nombre, p):
    if nombre == "— ninguno —": return img
    if nombre == "Gaussiano":    return filtro_gaussiano(img, p["gauss_k"], p["gauss_s"])
    if nombre == "CLAHE":        return filtro_clahe(img, p["clahe_clip"], p["clahe_tile"])
    if nombre == "Mediana":      return filtro_mediana(img, p["med_k"])
    if nombre == "Bilateral":    return filtro_bilateral(img, p["bil_d"], p["bil_sc"], p["bil_ss"])
    if nombre == "Sobel":        return filtro_sobel(img, p["sob_k"], p["sob_dir"])
    if nombre == "Laplaciano":   return filtro_laplaciano(img, p["lap_k"])
    if nombre == "Canny":        return filtro_canny(img, p["can_t1"], p["can_t2"])
    if nombre == "Unsharp Mask": return filtro_unsharp(img, p["unsh_k"], p["unsh_s"], p["unsh_f"])
    if nombre in FILTROS_MORFOLOGICOS:
        resultado = filtro_morfologico(img, nombre, p["morph_k"], p["morph_shape"], p["morph_iter"])
        if nombre in ("Top-Hat (Blanco)", "Black-Hat", "Gradiente Morfológico"):
            resultado = cv2.add(a_uint8(img), resultado)
        return resultado
    return img

# ── APLICAR FILTRO (params por paso, usado por Tab 1) ────────────────────────

def aplicar_filtro_paso(img, filtro, p):
    """Aplica un filtro usando su propio dict de parámetros independiente."""
    if filtro == "Gaussiano":    return filtro_gaussiano(img, p["ksize"], p["sigma"])
    if filtro == "CLAHE":        return filtro_clahe(img, p["clip_limit"], p["tile_size"])
    if filtro == "Mediana":      return filtro_mediana(img, p["ksize"])
    if filtro == "Bilateral":    return filtro_bilateral(img, p["d"], p["sigma_color"], p["sigma_space"])
    if filtro == "Sobel":        return filtro_sobel(img, p["ksize"], p["direction"])
    if filtro == "Laplaciano":   return filtro_laplaciano(img, p["ksize"])
    if filtro == "Canny":        return filtro_canny(img, p["t1"], p["t2"])
    if filtro == "Unsharp Mask": return filtro_unsharp(img, p["ksize"], p["sigma"], p["fuerza"])
    if filtro in FILTROS_MORFOLOGICOS:
        resultado = filtro_morfologico(img, filtro, p["ksize"], p["forma"], p["iteraciones"])
        if filtro in ("Top-Hat (Blanco)", "Black-Hat", "Gradiente Morfológico"):
            resultado = cv2.add(a_uint8(img), resultado)
        return resultado
    return img

# ── WIDGETS DE PARÁMETROS POR PASO ───────────────────────────────────────────

def render_params_paso(filtro, p, prefix):
    """Renderiza los sliders del filtro dado y actualiza p en-place."""
    if filtro == "Gaussiano":
        p["ksize"] = st.slider("Kernel", 1, 31, p.get("ksize", 5), 2, key=f"{prefix}_ksize")
        p["sigma"] = st.slider("Sigma",  0.1, 10.0, float(p.get("sigma", 1.0)), 0.1, key=f"{prefix}_sigma")

    elif filtro == "CLAHE":
        p["clip_limit"] = st.slider("Clip Limit", 0.5, 10.0, float(p.get("clip_limit", 2.0)), 0.5, key=f"{prefix}_clip")
        p["tile_size"]  = st.select_slider("Tile Size", [4, 8, 16, 32], p.get("tile_size", 8), key=f"{prefix}_tile")

    elif filtro == "Mediana":
        p["ksize"] = st.slider("Kernel", 1, 31, p.get("ksize", 5), 2, key=f"{prefix}_ksize")

    elif filtro == "Bilateral":
        p["d"]           = st.slider("Diámetro",  1, 25,  p.get("d", 9), 2,   key=f"{prefix}_d")
        p["sigma_color"] = st.slider("σ Color",   1, 200, p.get("sigma_color", 75), key=f"{prefix}_sc")
        p["sigma_space"] = st.slider("σ Espacio", 1, 200, p.get("sigma_space", 75), key=f"{prefix}_ss")

    elif filtro == "Sobel":
        p["ksize"]     = st.slider("Kernel", 1, 7, p.get("ksize", 3), 2, key=f"{prefix}_ksize")
        dirs = ["XY", "X", "Y"]
        p["direction"] = st.selectbox("Dirección", dirs,
                                       index=dirs.index(p.get("direction", "XY")),
                                       key=f"{prefix}_dir")

    elif filtro == "Laplaciano":
        p["ksize"] = st.slider("Kernel", 1, 31, p.get("ksize", 3), 2, key=f"{prefix}_ksize")

    elif filtro == "Canny":
        p["t1"] = st.slider("Umbral bajo",  0, 255, p.get("t1", 50),  key=f"{prefix}_t1")
        p["t2"] = st.slider("Umbral alto", 0, 255, p.get("t2", 150), key=f"{prefix}_t2")

    elif filtro == "Unsharp Mask":
        p["ksize"] = st.slider("Kernel", 1, 31, p.get("ksize", 5), 2, key=f"{prefix}_ksize")
        p["sigma"] = st.slider("Sigma",  0.1, 10.0, float(p.get("sigma", 1.0)), 0.1, key=f"{prefix}_sigma")
        p["fuerza"] = st.slider("Fuerza", 0.1, 5.0, float(p.get("fuerza", 1.5)), 0.1, key=f"{prefix}_fuerza")

    elif filtro in FILTROS_MORFOLOGICOS:
        p["ksize"] = st.slider("Kernel", 1, 15, p.get("ksize", 5), 2, key=f"{prefix}_ksize")
        formas = ["Rectángulo", "Elipse", "Cruz"]
        p["forma"] = st.selectbox("Forma", formas,
                                   index=formas.index(p.get("forma", "Rectángulo")),
                                   key=f"{prefix}_forma")
        p["iteraciones"] = st.slider("Iteraciones", 1, 10, p.get("iteraciones", 1), key=f"{prefix}_iter")

# ── EXPORTAR COMO CÓDIGO OPENCV ───────────────────────────────────────────────

def codigo_opencv(pipeline):
    lines = [
        "import cv2", "import numpy as np", "",
        "# img: numpy array uint8 RGB", "",
    ]
    for i, step in enumerate(pipeline):
        f, p = step["filtro"], step["p"]
        src = "img" if i == 0 else f"paso{i}"
        dst = f"paso{i+1}"

        if f == "Gaussiano":
            k = asegurar_kernel_impar(p["ksize"])
            lines.append(f"{dst} = cv2.GaussianBlur({src}, ({k},{k}), {p['sigma']})")

        elif f == "CLAHE":
            lines += [
                f"_lab = cv2.cvtColor({src}, cv2.COLOR_RGB2LAB)",
                f"_l, _a, _b = cv2.split(_lab)",
                f"_clahe = cv2.createCLAHE(clipLimit={p['clip_limit']}, tileGridSize=({p['tile_size']},{p['tile_size']}))",
                f"{dst} = cv2.cvtColor(cv2.merge([_clahe.apply(_l), _a, _b]), cv2.COLOR_LAB2RGB)",
            ]

        elif f == "Mediana":
            k = asegurar_kernel_impar(p["ksize"])
            lines.append(f"{dst} = cv2.medianBlur({src}, {k})")

        elif f == "Bilateral":
            lines.append(f"{dst} = cv2.bilateralFilter({src}, {p['d']}, {p['sigma_color']}, {p['sigma_space']})")

        elif f == "Sobel":
            k = asegurar_kernel_impar(p["ksize"])
            lines += [
                f"_gray = cv2.cvtColor({src}, cv2.COLOR_RGB2GRAY)",
                f"_sx = cv2.Sobel(_gray, cv2.CV_64F, 1, 0, ksize={k})",
                f"_sy = cv2.Sobel(_gray, cv2.CV_64F, 0, 1, ksize={k})  # dirección: {p['direction']}",
                f"_r = cv2.magnitude(_sx, _sy)",
                f"{dst} = cv2.cvtColor(cv2.normalize(_r, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), cv2.COLOR_GRAY2RGB)",
            ]

        elif f == "Laplaciano":
            k = asegurar_kernel_impar(p["ksize"])
            lines += [
                f"_gray = cv2.cvtColor({src}, cv2.COLOR_RGB2GRAY)",
                f"_r = cv2.Laplacian(_gray, cv2.CV_64F, ksize={k})",
                f"{dst} = cv2.cvtColor(cv2.normalize(abs(_r), None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), cv2.COLOR_GRAY2RGB)",
            ]

        elif f == "Canny":
            lines += [
                f"_gray = cv2.cvtColor({src}, cv2.COLOR_RGB2GRAY)",
                f"{dst} = cv2.cvtColor(cv2.Canny(_gray, {p['t1']}, {p['t2']}), cv2.COLOR_GRAY2RGB)",
            ]

        elif f == "Unsharp Mask":
            k = asegurar_kernel_impar(p["ksize"])
            w1, w2 = round(1 + p["fuerza"], 2), round(-p["fuerza"], 2)
            lines += [
                f"_blur = cv2.GaussianBlur({src}, ({k},{k}), {p['sigma']})",
                f"{dst} = cv2.addWeighted({src}, {w1}, _blur, {w2}, 0)",
            ]

        elif f in FILTROS_MORFOLOGICOS:
            op_map = {
                "Erosión": "cv2.MORPH_ERODE", "Dilatación": "cv2.MORPH_DILATE",
                "Apertura": "cv2.MORPH_OPEN", "Cierre": "cv2.MORPH_CLOSE",
                "Gradiente Morfológico": "cv2.MORPH_GRADIENT",
                "Top-Hat (Blanco)": "cv2.MORPH_TOPHAT", "Black-Hat": "cv2.MORPH_BLACKHAT",
            }
            shape_map = {"Rectángulo": "cv2.MORPH_RECT", "Elipse": "cv2.MORPH_ELLIPSE", "Cruz": "cv2.MORPH_CROSS"}
            k = asegurar_kernel_impar(p["ksize"])
            lines += [
                f"_kernel = cv2.getStructuringElement({shape_map[p['forma']]}, ({k},{k}))",
                f"{dst} = cv2.morphologyEx({src}, {op_map[f]}, _kernel, iterations={p['iteraciones']})",
            ]

        lines.append("")

    if pipeline:
        lines.append(f"resultado = paso{len(pipeline)}")
    return "\n".join(lines)

# ── SESSION STATE — PIPELINE ──────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = []

def _pip_agregar():
    st.session_state.pipeline.append({"filtro": "Gaussiano", "p": params_por_defecto("Gaussiano")})

def _pip_limpiar():
    st.session_state.pipeline = []

def _pip_eliminar(i):
    st.session_state.pipeline.pop(i)

def _pip_subir(i):
    p = st.session_state.pipeline
    p[i], p[i - 1] = p[i - 1], p[i]

def _pip_bajar(i):
    p = st.session_state.pipeline
    p[i], p[i + 1] = p[i + 1], p[i]

# ── UI — ENCABEZADO ───────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🛰️ Explorador de Filtros EuroSAT</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Filtros espaciales y morfológicos · Visión Artificial</p>', unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Imagen")
    uploaded = st.file_uploader("Cargar imagen EuroSAT (.jpg / .png)", type=["jpg","jpeg","png"])

    st.markdown("---")
    st.markdown("### 🔧 Parámetros Globales")
    st.caption("Usados en Comparación, Análisis y Métricas")

    with st.expander("Filtros Espaciales", expanded=False):
        gauss_k    = st.slider("Gaussiano — Kernel",    1, 31, 5, 2, key="gk")
        gauss_s    = st.slider("Gaussiano — Sigma",     0.1, 10.0, 1.0, 0.1, key="gs")
        clahe_clip = st.slider("CLAHE — Clip Limit",    0.5, 10.0, 1.5, 0.5, key="cc")
        clahe_tile = st.select_slider("CLAHE — Tile Size", options=[4,8,16,32], value=32, key="ct")
        med_k      = st.slider("Mediana — Kernel",      1, 31, 5, 2, key="mk")
        bil_d      = st.slider("Bilateral — Diámetro",  1, 25, 9, 2, key="bd")
        bil_sc     = st.slider("Bilateral — σ Color",   1, 200, 75, key="bsc")
        bil_ss     = st.slider("Bilateral — σ Espacio", 1, 200, 75, key="bss")
        sob_k      = st.slider("Sobel — Kernel",        1, 7, 3, 2, key="sk")
        sob_dir    = st.selectbox("Sobel — Dirección",  ["XY","X","Y"], key="sd")
        lap_k      = st.slider("Laplaciano — Kernel",   1, 31, 3, 2, key="lk")
        can_t1     = st.slider("Canny — Umbral 1",      0, 255, 50, key="ct1")
        can_t2     = st.slider("Canny — Umbral 2",      0, 255, 150, key="ct2")
        unsh_k     = st.slider("Unsharp — Kernel",      1, 31, 5, 2, key="uk")
        unsh_s     = st.slider("Unsharp — Sigma",       0.1, 10.0, 1.0, 0.1, key="us")
        unsh_f     = st.slider("Unsharp — Fuerza",      0.1, 5.0, 1.5, 0.1, key="uf")

    with st.expander("Filtros Morfológicos", expanded=False):
        morph_k     = st.slider("Kernel morfológico", 1, 15, 3, 2, key="mok")
        morph_shape = st.selectbox("Forma del elemento", ["Rectángulo","Elipse","Cruz"], key="mosh")
        morph_iter  = st.slider("Iteraciones", 1, 10, 1, key="moit")

params_global = dict(
    gauss_k=gauss_k, gauss_s=gauss_s,
    clahe_clip=clahe_clip, clahe_tile=clahe_tile,
    med_k=med_k,
    bil_d=bil_d, bil_sc=bil_sc, bil_ss=bil_ss,
    sob_k=sob_k, sob_dir=sob_dir, lap_k=lap_k,
    can_t1=can_t1, can_t2=can_t2,
    unsh_k=unsh_k, unsh_s=unsh_s, unsh_f=unsh_f,
    morph_k=morph_k, morph_shape=morph_shape, morph_iter=morph_iter,
)

# ── GUARD: sin imagen ─────────────────────────────────────────────────────────

if not uploaded:
    st.info("👈 **Carga una imagen EuroSAT** desde la barra lateral para comenzar.")
    st.markdown("""
    #### ¿Cómo obtener imágenes?
    1. Descarga el dataset desde [Kaggle — EuroSAT RGB](https://www.kaggle.com/datasets/salmaadell/eurosat-rgb)
    2. Elige imágenes de clases como `Highway`, `River`, `Forest`, `Residential`, etc.
    3. Carga aquí cualquier imagen `.jpg` de 64×64 píxeles (o mayor)
    """)
    st.stop()

pil_img  = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
img_orig = np.array(pil_img)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Pipeline de Filtros",
    "⚖️ Comparación Múltiple",
    "📊 Gráficas de Análisis",
    "📋 Métricas",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE DINÁMICO
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Pipeline dinámico de filtros")
    st.caption("Encadena filtros en secuencia con parámetros independientes por paso. El resultado se actualiza en tiempo real.")

    # ── Controles del pipeline ──
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 6])
    ctrl1.button("＋ Agregar paso", on_click=_pip_agregar, use_container_width=True, key="btn_agregar")
    if st.session_state.pipeline:
        ctrl2.button("🗑 Limpiar todo", on_click=_pip_limpiar, use_container_width=True, key="btn_limpiar")

    st.markdown("---")

    # ── Sin pasos ──
    if not st.session_state.pipeline:
        st.info("Agrega pasos con **＋ Agregar paso** para construir tu pipeline.")
        st.image(img_orig, caption="Imagen original", width=300)

    else:
        # ── Lista de pasos ──
        n_pasos = len(st.session_state.pipeline)
        filtros_disp = FILTROS_ESPACIALES + FILTROS_MORFOLOGICOS

        for i in range(n_pasos):
            step = st.session_state.pipeline[i]

            with st.container():
                col_num, col_sel, col_par, col_up, col_dn, col_del = st.columns([0.35, 2.2, 3.5, 0.45, 0.45, 0.45])

                col_num.markdown(f"<br><b>#{i+1}</b>", unsafe_allow_html=True)

                # Selector de filtro
                with col_sel:
                    cur_idx = filtros_disp.index(step["filtro"]) if step["filtro"] in filtros_disp else 0
                    new_f = st.selectbox(
                        "Filtro", filtros_disp, index=cur_idx,
                        key=f"pip_f_{i}", label_visibility="collapsed"
                    )
                    if new_f != step["filtro"]:
                        st.session_state.pipeline[i] = {"filtro": new_f, "p": params_por_defecto(new_f)}
                        step = st.session_state.pipeline[i]

                # Parámetros del paso
                with col_par:
                    with st.expander(f"⚙ Parámetros — {step['filtro']}", expanded=False):
                        render_params_paso(step["filtro"], step["p"], f"pip_{i}")

                # Botones de control
                with col_up:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.button("⬆", key=f"pip_up_{i}",
                              on_click=_pip_subir, args=(i,),
                              disabled=(i == 0), use_container_width=True)
                with col_dn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.button("⬇", key=f"pip_dn_{i}",
                              on_click=_pip_bajar, args=(i,),
                              disabled=(i == n_pasos - 1), use_container_width=True)
                with col_del:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.button("✕", key=f"pip_del_{i}",
                              on_click=_pip_eliminar, args=(i,), use_container_width=True)

        st.markdown("---")

        # ── Ejecución del pipeline en tiempo real ──
        steps_imgs  = [img_orig]
        steps_names = ["Original"]
        current     = img_orig.copy()
        error_paso  = None

        for j, step in enumerate(st.session_state.pipeline):
            try:
                result = aplicar_filtro_paso(current, step["filtro"], step["p"])
                result = a_uint8(result)
                steps_imgs.append(result)
                steps_names.append(f"{j+1}. {step['filtro']}")
                current = result
            except Exception as e:
                error_paso = (j + 1, step["filtro"], str(e))
                break

        if error_paso:
            st.error(f"Error en paso {error_paso[0]} ({error_paso[1]}): {error_paso[2]}")

        # Diagrama de flujo
        st.markdown(
            "**Flujo:** " + " → ".join(
                f'<span class="step-chip">{n}</span>' for n in steps_names
            ),
            unsafe_allow_html=True,
        )

        # Thumbnails por paso
        cols_th = st.columns(len(steps_imgs))
        for col, img_s, name_s in zip(cols_th, steps_imgs, steps_names):
            with col:
                st.image(img_s, caption=name_s, use_container_width=True)

        # ── Análisis del resultado final ──
        if len(steps_imgs) > 1:
            st.markdown("---")

            col_an1, col_an2 = st.columns(2)
            with col_an1:
                st.markdown("#### Diferencia acumulada")
                st.pyplot(fig_diferencia(img_orig, steps_imgs[-1]), use_container_width=True)
            with col_an2:
                st.markdown("#### Histogramas")
                st.pyplot(
                    fig_histograma([img_orig, steps_imgs[-1]], ["Original", steps_names[-1]]),
                    use_container_width=True,
                )

            st.markdown("#### Perfil de intensidad — fila central")
            st.pyplot(fig_perfil(img_orig, steps_imgs[-1], steps_names[-1]), use_container_width=True)

            # Métricas rápidas
            st.markdown("#### Métricas del resultado final")
            final = steps_imgs[-1]
            if final.shape != img_orig.shape:
                final = cv2.cvtColor(cv2.cvtColor(final, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            m = calcular_metricas(img_orig, final)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MSE ↓", f"{m['MSE']:.2f}")
            mc2.metric("PSNR ↑", f"{m['PSNR']:.2f} dB")
            mc3.metric("SSIM ↑", f"{m['SSIM']:.4f}")

            # Exportar como código
            st.markdown("---")
            st.markdown("#### Código OpenCV equivalente")
            st.code(codigo_opencv(st.session_state.pipeline), language="python")

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
        resultados = [(n, aplicar_filtro(img_orig.copy(), n, params_global))
                      for n in filtros_comp if n != "— ninguno —"]
        if resultados:
            st.pyplot(fig_comparacion_multiple(img_orig, resultados), use_container_width=True)
            st.markdown("#### Histogramas individuales")
            st.pyplot(fig_histograma([img_orig]+[r[1] for r in resultados],
                                     ["Original"]+[r[0] for r in resultados]), use_container_width=True)
        else:
            st.warning("Selecciona al menos un filtro.")
    else:
        st.image(img_orig, caption="Imagen original", width=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRÁFICAS DE ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Gráficas de análisis para un filtro individual")
    filtro_analisis = st.selectbox("Filtro a analizar", TODOS_FILTROS[1:], key="anal")
    if st.button("📊 Generar gráficas"):
        img_proc = aplicar_filtro(img_orig.copy(), filtro_analisis, params_global)
        col_a, col_b = st.columns(2)
        with col_a: st.image(img_orig,  caption="Original", use_container_width=True)
        with col_b: st.image(img_proc, caption=filtro_analisis, use_container_width=True)
        st.markdown("---")
        st.markdown("#### Histogramas")
        st.pyplot(fig_histograma([img_orig, img_proc], ["Original", filtro_analisis]), use_container_width=True)
        st.markdown("#### Perfil de intensidad")
        st.pyplot(fig_perfil(img_orig, img_proc, filtro_analisis), use_container_width=True)
        st.markdown("#### Mapa de diferencia")
        st.pyplot(fig_diferencia(img_orig, img_proc), use_container_width=True)
    else:
        st.image(img_orig, caption="Imagen original", width=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Métricas de calidad de imagen")
    filtros_metrica = st.multiselect(
        "Selecciona filtros a medir", TODOS_FILTROS[1:],
        default=["Gaussiano","Mediana","Erosión","Canny"], key="met_sel"
    )
    if st.button("📋 Calcular métricas"):
        if not filtros_metrica:
            st.warning("Selecciona al menos un filtro.")
        else:
            tabla = []
            for nombre in filtros_metrica:
                img_proc = aplicar_filtro(img_orig.copy(), nombre, params_global)
                if img_proc.shape != img_orig.shape:
                    img_proc = cv2.cvtColor(cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                m = calcular_metricas(img_orig, img_proc); m["Filtro"] = nombre
                tabla.append(m)
            df = pd.DataFrame(tabla).set_index("Filtro")
            df.columns = ["MSE ↓","PSNR ↑ (dB)","SSIM ↑","Media orig","Media proc","Std orig","Std proc"]
            st.dataframe(df.round(3).style
                         .background_gradient(cmap="Blues", subset=["SSIM ↑"])
                         .background_gradient(cmap="Reds_r", subset=["MSE ↓"]),
                         use_container_width=True)
            fig_bar, ax = plt.subplots(figsize=(10, 3))
            fig_bar.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
            bars = ax.barh(df.index, df["SSIM ↑"], color="#339af0", alpha=0.85)
            ax.set_xlim(0, 1); ax.set_title("SSIM por filtro", color="white", fontsize=10)
            ax.tick_params(colors="#8899aa", labelsize=8); ax.spines[:].set_color(GRID_C)
            for bar in bars:
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f"{bar.get_width():.3f}", va="center", color="white", fontsize=7)
            st.pyplot(fig_bar, use_container_width=True)
            st.caption("MSE ↓ mejor · PSNR ↑ mejor · SSIM ↑ mejor (1.0 = idéntica)")
    else:
        st.image(img_orig, caption="Imagen original", width=300)
