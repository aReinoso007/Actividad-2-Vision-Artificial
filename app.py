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
import io
import pandas as pd
import json
import os
from pathlib import Path

# ── Directorio de presets ────────────────────────────────────────────────────
PRESETS_DIR = Path("presets")
PRESETS_DIR.mkdir(exist_ok=True)

def guardar_preset(nombre, pre_ops, filt_names, P):
    data = {"nombre": nombre, "pre_ops": pre_ops, "filt_names": filt_names, "params": P}
    ruta = PRESETS_DIR / f"{nombre}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return ruta

def cargar_preset(nombre):
    ruta = PRESETS_DIR / f"{nombre}.json"
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def listar_presets():
    return sorted([p.stem for p in PRESETS_DIR.glob("*.json")])

def eliminar_preset(nombre):
    ruta = PRESETS_DIR / f"{nombre}.json"
    if ruta.exists():
        os.remove(ruta)

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

    .main-title {
        font-size: 2.0rem; font-weight: 800; letter-spacing: -1px;
        background: linear-gradient(135deg, #00d4ff, #0066ff, #7b2fff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #8899aa; font-size: 0.9rem; margin-top: 0; }

    /* Stage headers */
    .stage-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; padding: 6px 14px;
        border-radius: 4px; display: inline-block; margin-bottom: 8px;
    }
    .stage-pre  { background: #1a2d1a; color: #51cf66; border: 1px solid #2a552a; }
    .stage-filt { background: #1a2035; color: #00d4ff; border: 1px solid #2a3555; }
    .stage-res  { background: #2d1a2d; color: #cc88ff; border: 1px solid #55205a; }

    .chip-pre  { display:inline-block; background:#1a2d1a; border:1px solid #2a552a;
                 color:#51cf66; border-radius:5px; padding:2px 9px;
                 font-family:'Space Mono',monospace; font-size:0.72rem; margin:2px; }
    .chip-filt { display:inline-block; background:#1a2035; border:1px solid #2a3555;
                 color:#00d4ff; border-radius:5px; padding:2px 9px;
                 font-family:'Space Mono',monospace; font-size:0.72rem; margin:2px; }

    .img-label { color:#8899aa; font-size:0.72rem; font-family:'Space Mono',monospace;
                 text-transform:uppercase; letter-spacing:1px; text-align:center;
                 margin-bottom:3px; }

    .stButton>button {
        background: linear-gradient(135deg,#0066ff,#7b2fff);
        color:white; border:none; border-radius:8px; font-weight:700;
        padding:0.4rem 1.2rem; width:100%;
    }
    [data-testid="stSidebar"] { background: #080e1e; }
    hr.stage-sep { border:none; border-top:1px solid #1e2d4a; margin:18px 0; }
</style>
""", unsafe_allow_html=True)

DARK_BG = "#080e1e"
GRID_C  = "#1a2535"

# ═══════════════════════════════════════════════════════════════════════════════
# FILTROS ESPACIALES
# ═══════════════════════════════════════════════════════════════════════════════

def k_odd(k): return k if k % 2 == 1 else k + 1

def f_gaussiano(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (k_odd(ksize),)*2, sigma)

def f_mediana(img, ksize=3):
    return cv2.medianBlur(img, k_odd(ksize))

def f_bilateral(img, d=7, sc=50, ss=50):
    return cv2.bilateralFilter(img, d, sc, ss)

def f_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:,:,0] = c.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def f_sobel(img, ksize=3, direction="XY"):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim==3 else img
    k = k_odd(ksize)
    if direction=="X":   r = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k))
    elif direction=="Y": r = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k))
    else:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        r = cv2.magnitude(sx, sy)
    return cv2.cvtColor(cv2.normalize(r,None,0,255,cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)

def f_laplaciano(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim==3 else img
    r = cv2.Laplacian(gray, cv2.CV_64F, ksize=k_odd(ksize))
    return cv2.cvtColor(cv2.normalize(np.abs(r),None,0,255,cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)

def f_canny(img, t1=30, t2=80):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim==3 else img
    return cv2.cvtColor(cv2.Canny(gray, t1, t2), cv2.COLOR_GRAY2RGB)

def f_unsharp(img, ksize=3, sigma=1.0, strength=1.5):
    blur = cv2.GaussianBlur(img, (k_odd(ksize),)*2, sigma)
    return np.clip(cv2.addWeighted(img, 1+strength, blur, -strength, 0), 0, 255).astype(np.uint8)

def f_anisotropico(img, alpha=0.075, K=50, niters=10):
    """Difusión anisotrópica de Perona-Malik — suaviza texturas preservando bordes."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    try:
        result = cv2.ximgproc.anisotropicDiffusion(
            gray.astype(np.float32), alpha=float(alpha), K=float(K), niters=int(niters)
        )
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except AttributeError:
        # Fallback si no está opencv-contrib: usar bilateral como aproximación
        result = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

FORMAS = {"Rectángulo": cv2.MORPH_RECT, "Elipse": cv2.MORPH_ELLIPSE, "Cruz": cv2.MORPH_CROSS}
OPS_MORFO = {
    "Erosión": cv2.MORPH_ERODE, "Dilatación": cv2.MORPH_DILATE,
    "Apertura": cv2.MORPH_OPEN, "Cierre": cv2.MORPH_CLOSE,
    "Gradiente Morfológico": cv2.MORPH_GRADIENT,
    "Top-Hat (Blanco)": cv2.MORPH_TOPHAT, "Black-Hat": cv2.MORPH_BLACKHAT,
}

def f_morfo(img, op="Erosión", ksize=3, forma="Rectángulo", iters=1):
    kern = cv2.getStructuringElement(FORMAS[forma], (k_odd(ksize),)*2)
    return cv2.morphologyEx(img, OPS_MORFO[op], kern, iterations=iters)


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD (UMBRALIZACIÓN)
# ═══════════════════════════════════════════════════════════════════════════════

TIPOS_THRESH = ["Binario fijo", "Otsu (automático)", "Adaptativo Gaussiano",
                "Adaptativo Media", "Inverso fijo", "Inverso Otsu"]

def f_threshold(img, tipo="Otsu (automático)", valor=127, block=11, C=2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    bs   = block if block % 2 == 1 else block + 1
    if tipo == "Binario fijo":
        _, mask = cv2.threshold(gray, valor, 255, cv2.THRESH_BINARY)
    elif tipo == "Otsu (automático)":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tipo == "Adaptativo Gaussiano":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, bs, C)
    elif tipo == "Adaptativo Media":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, bs, C)
    elif tipo == "Inverso fijo":
        _, mask = cv2.threshold(gray, valor, 255, cv2.THRESH_BINARY_INV)
    elif tipo == "Inverso Otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# ═══════════════════════════════════════════════════════════════════════════════
# OPERACIONES ELEMENTALES
# ═══════════════════════════════════════════════════════════════════════════════

def op_suma(img, v=50):
    return np.clip(img.astype(np.int32)+int(v), 0, 255).astype(np.uint8)

def op_resta(img, v=50):
    return np.clip(img.astype(np.int32)-int(v), 0, 255).astype(np.uint8)

def op_gamma(img, g=1.0):
    t = np.array([(i/255.0)**(1/max(g,0.01))*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, t)

def op_stretch(img, pmin=0, pmax=100):
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, pmin), np.percentile(f, pmax)
    if hi==lo: return img
    return np.clip((f-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)

def op_log(img, c=40.0):
    return np.clip(c*np.log(img.astype(np.float32)+1), 0, 255).astype(np.uint8)

def op_he(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[:,:,0] = cv2.equalizeHist(ycc[:,:,0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

def op_negativa(img):
    return (255 - img.astype(np.int32)).clip(0,255).astype(np.uint8)

def op_escala_grises(img):
    """Convierte a escala de grises y devuelve 3 canales iguales (compatible con el pipeline)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

NINGUNA_PRE  = "— ninguna —"
NINGUNO_FILT = "— ninguno —"

OPS_PRE = [NINGUNA_PRE, "Suma", "Resta", "Corrección Gamma",
           "Linear Stretching", "Transformada Log", "Ecualización HE", "Negativa",
           "Escala de Grises"]

F_ESPACIALES = ["Gaussiano","CLAHE","Mediana","Bilateral","Sobel","Laplaciano","Canny","Unsharp Mask","Threshold","Anisotrópico"]
F_MORFO      = list(OPS_MORFO.keys())
FILTROS      = [NINGUNO_FILT] + F_ESPACIALES + F_MORFO

# ═══════════════════════════════════════════════════════════════════════════════
# DESPACHO
# ═══════════════════════════════════════════════════════════════════════════════

def aplicar_pre(img, nombre, P):
    if nombre==NINGUNA_PRE:           return img
    if nombre=="Suma":                return op_suma(img, P["suma_v"])
    if nombre=="Resta":               return op_resta(img, P["resta_v"])
    if nombre=="Corrección Gamma":    return op_gamma(img, P["gamma"])
    if nombre=="Linear Stretching":   return op_stretch(img, P["ls_min"], P["ls_max"])
    if nombre=="Transformada Log":    return op_log(img, P["log_c"])
    if nombre=="Ecualización HE":     return op_he(img)
    if nombre=="Negativa":            return op_negativa(img)
    if nombre=="Escala de Grises":    return op_escala_grises(img)
    return img

def aplicar_filtro(img, nombre, P, grupo="A"):
    if nombre==NINGUNO_FILT:  return img
    if nombre=="Gaussiano":   return f_gaussiano(img, P["gk"], P["gs"])
    if nombre=="CLAHE":       return f_clahe(img, P["cc"], P["ct"])
    if nombre=="Mediana":     return f_mediana(img, P["mk"])
    if nombre=="Bilateral":   return f_bilateral(img, P["bd"], P["bsc"], P["bss"])
    if nombre=="Sobel":       return f_sobel(img, P["sk"], P["sd"])
    if nombre=="Laplaciano":  return f_laplaciano(img, P["lk"])
    if nombre=="Canny":       return f_canny(img, P["ct1"], P["ct2"])
    if nombre=="Unsharp Mask":return f_unsharp(img, P["uk"], P["us"], P["uf"])
    if nombre=="Threshold":    return f_threshold(img, P["th_tipo"], P["th_val"], P["th_block"], P["th_c"])
    if nombre=="Anisotrópico": return f_anisotropico(img, P["an_alpha"], P["an_K"], P["an_niters"])
    if nombre in F_MORFO:
        g = grupo.lower()
        return f_morfo(img, nombre, P[f"mok_{g}"], P[f"mosh_{g}"], P[f"moit_{g}"])
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS DE DETECCIÓN DE BORDES
# ═══════════════════════════════════════════════════════════════════════════════

def edge_density(img):
    """% de píxeles clasificados como borde por Canny."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    return round(float(np.sum(edges > 0) / edges.size * 100), 4)

def mean_edge_strength(img):
    """Magnitud media y std del gradiente Sobel."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return round(float(np.mean(mag)), 4), round(float(np.std(mag)), 4)

def local_contrast(img):
    """Contraste de alta frecuencia (img - blur gaussiano)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return round(float(np.std(gray - blur)), 4)

def imagen_entropy(img):
    """Entropía de Shannon sobre el histograma de luminancia."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return round(float(-np.sum(hist * np.log2(hist))), 4)

def mask_coverage(img):
    """% de píxeles de borde en imagen binarizada."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return round(float(np.sum(gray > 127) / gray.size * 100), 4)

def calcular_metricas_borde(orig, proc):
    """Calcula todas las métricas de borde para original y procesada."""
    dens_o,  dens_p  = edge_density(orig),      edge_density(proc)
    mes_o,   mes_p   = mean_edge_strength(orig), mean_edge_strength(proc)
    lc_o,    lc_p    = local_contrast(orig),     local_contrast(proc)
    ent_o,   ent_p   = imagen_entropy(orig),     imagen_entropy(proc)
    cov_o,   cov_p   = mask_coverage(orig),      mask_coverage(proc)
    return {
        "Edge Density (%)":          (dens_o, dens_p),
        "Edge Strength — media":     (mes_o[0], mes_p[0]),
        "Edge Strength — std":       (mes_o[1], mes_p[1]),
        "Contraste local":           (lc_o,   lc_p),
        "Entropía (bits)":           (ent_o,  ent_p),
        "Mask Coverage (%)":         (cov_o,  cov_p),
    }

def plot_edge_comparison(orig, proc, label_proc):
    """Figura: imagen de bordes Canny original vs procesada + gradiente."""
    gray_o = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    gray_p = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY) if proc.ndim == 3 else proc
    edges_o = cv2.Canny(gray_o, 50, 150)
    edges_p = cv2.Canny(gray_p, 50, 150)
    sx = cv2.Sobel(gray_p.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_p.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.normalize(np.sqrt(sx**2+sy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    # white background for consistency
    fig.patch.set_facecolor("white")
    datos = [(orig, "Original"), (proc, label_proc),
             (edges_o, "Bordes Canny — Original"), (edges_p, f"Bordes Canny — {label_proc}")]
    for ax, (img, t) in zip(axes, datos):
        ax.set_facecolor("white")
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(t, color="black", fontsize=8); ax.axis("off")
    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES Y GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def diferencia(a, b):
    af, bf = a.astype(np.float32), b.astype(np.float32)
    if af.shape != bf.shape:
        bf = cv2.cvtColor(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB).astype(np.float32)
    return cv2.normalize(np.abs(af-bf), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def metricas(orig, proc):
    og = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    pg = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY) if proc.ndim==3 else proc
    if og.shape!=pg.shape: pg=cv2.resize(pg,(og.shape[1],og.shape[0]))
    mse = float(np.mean((og.astype(float)-pg.astype(float))**2))
    return dict(
        MSE=round(mse,2),
        PSNR=round(float(cv2.PSNR(og,pg)) if mse>0 else 999,2),
        SSIM=round(float(ssim(og,pg,data_range=255)),4),
        Media=round(float(np.mean(pg)),2),
        Std=round(float(np.std(pg)),2),
    )

def plot_hist(imgs, labels):
    fig, axes = plt.subplots(1, len(imgs), figsize=(4.5*len(imgs), 3.2))
    if len(imgs)==1: axes=[axes]
    # White background for histogram figures
    fig.patch.set_facecolor("white")
    for ax, img, lbl in zip(axes, imgs, labels):
        ax.set_facecolor("white")
        # draw grid with a light gray color
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        for i,(col,name) in enumerate(zip(("#ff4d6d","#51cf66","#339af0"),"RGB")):
            ch = min(i, img.shape[2]-1) if img.ndim==3 else 0
            h = cv2.calcHist([img],[ch],None,[256],[0,256]).flatten()
            ax.fill_between(range(256),h,alpha=0.3,color=col)
            ax.plot(h,color=col,lw=0.8,label=name)
        ax.set_title(lbl,color="black",fontsize=9)
        ax.set_xlim(0,255)
        # darker ticks/labels for white background
        ax.tick_params(colors="#222222",labelsize=7)
        # light colored spines to match grid
        ax.spines[:].set_color("#cccccc")
        ax.legend(fontsize=7,facecolor="white",labelcolor="black",edgecolor="#dddddd")
    fig.tight_layout(); return fig

def plot_perfil(orig, proc, lbl):
    row = orig.shape[0]//2
    fig, axes = plt.subplots(2,1,figsize=(10,4),sharex=True)
    fig.patch.set_facecolor("white")
    for ax,(img,title) in zip(axes,[(orig,"Original"),(proc,lbl)]):
        ax.set_facecolor("white")
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_title(title,color="black",fontsize=9)
        if img.ndim==3:
            for col,name,ch in zip(("#ff4d6d","#51cf66","#339af0"),"RGB",range(3)):
                ax.plot(img[row,:,ch],color=col,lw=0.9,label=name)
        else: ax.plot(img[row,:],color="#00d4ff",lw=0.9)
        ax.tick_params(colors="#222222",labelsize=7); ax.spines[:].set_color("#cccccc")
        ax.legend(fontsize=7,facecolor="white",labelcolor="black",edgecolor="#dddddd")
    axes[-1].set_xlabel("Píxel (columna)",color="#222222",fontsize=8)
    fig.tight_layout(); return fig

def plot_curva(nombre_op, P):
    r = np.arange(256,dtype=np.float32)
    if   nombre_op=="Suma":              s=np.clip(r+P["suma_v"],0,255)
    elif nombre_op=="Resta":             s=np.clip(r-P["resta_v"],0,255)
    elif nombre_op=="Corrección Gamma":  s=255*(r/255)**(1/max(P["gamma"],0.01))
    elif nombre_op=="Linear Stretching": s=np.clip((r-np.percentile(r,P["ls_min"]))/(max(np.percentile(r,P["ls_max"])-np.percentile(r,P["ls_min"]),1))*255,0,255)
    elif nombre_op=="Transformada Log":  s=np.clip(P["log_c"]*np.log(r+1),0,255)
    elif nombre_op=="Negativa":          s=255-r
    else:                                s=r
    fig,ax=plt.subplots(figsize=(3.5,3))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.plot(r,s,color="#51cf66",lw=2)
    ax.plot([0,255],[0,255],color="#cccccc",lw=0.8,linestyle="--")
    ax.set_title("Curva T(r)",color="black",fontsize=9)
    ax.set_xlabel("r",color="#222222",fontsize=8); ax.set_ylabel("s",color="#222222",fontsize=8)
    ax.tick_params(colors="#222222",labelsize=7); ax.spines[:].set_color("#cccccc")
    ax.set_xlim(0,255); ax.set_ylim(0,255)
    fig.tight_layout(); return fig

def mostrar_tira(imgs, labels, highlight_last=False):
    """Muestra una tira horizontal de imágenes con etiquetas."""
    cols = st.columns(len(imgs))
    for i,(col,img,lbl) in enumerate(zip(cols,imgs,labels)):
        with col:
            st.markdown(f'<p class="img-label">{lbl}</p>', unsafe_allow_html=True)
            border = "2px solid #cc88ff" if (highlight_last and i==len(imgs)-1) else "none"
            st.image(img, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UI — CABECERA
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">🛰️ Explorador de Filtros EuroSAT</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Pre-procesamiento · Filtros espaciales y morfológicos · Análisis en tiempo real</p>', unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📂 Imagen")
    uploaded = st.file_uploader("Cargar imagen EuroSAT", type=["jpg","jpeg","png"])

    # ── GESTOR DE PRESETS ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Presets de pipeline")
    presets_disponibles = listar_presets()

    with st.expander("📂 Cargar preset", expanded=len(presets_disponibles) > 0):
        if presets_disponibles:
            preset_elegido = st.selectbox("Seleccionar preset", presets_disponibles, key="preset_sel")
            col_load, col_del = st.columns(2)
            with col_load:
                if st.button("⬆️ Cargar", key="btn_load", use_container_width=True):
                    datos = cargar_preset(preset_elegido)
                    if datos:
                        for i, op in enumerate(datos["pre_ops"][:3]):
                            st.session_state[f"saved_pre_{i}"] = op
                            st.session_state[f"pre_{i}"] = op
                        for i, fn in enumerate(datos["filt_names"][:5]):
                            st.session_state[f"saved_f{i}"] = fn
                            st.session_state[f"f{i}"] = fn
                        params_g = datos.get("params", {})
                        for k in ["suma_v","resta_v","gamma","ls_min","ls_max","log_c",
                                   "gk","gs","cc","mk","bd","bsc","bss","sk","lk",
                                   "ct1","ct2","uk","us","uf","mok_a","moit_a","mok_b","moit_b",
                                   "sd","mosh_a","mosh_b","ct"]:
                            if k in params_g:
                                st.session_state[k] = params_g[k]
                        st.success(f"✅ Preset '{preset_elegido}' cargado")
                        st.rerun()
            with col_del:
                if st.button("🗑️ Eliminar", key="btn_del", use_container_width=True):
                    eliminar_preset(preset_elegido)
                    st.warning(f"Preset '{preset_elegido}' eliminado")
                    st.rerun()
        else:
            st.caption("No hay presets guardados aún.")

    with st.expander("💾 Guardar preset actual", expanded=False):
        nombre_preset = st.text_input("Nombre del preset",
            placeholder="ej: Pipeline_A_Highway, Arido_v2...", key="preset_nombre")
        if st.button("💾 Guardar", key="btn_save", use_container_width=True):
            if nombre_preset.strip():
                ops_act  = [st.session_state.get(f"saved_pre_{i}", "— ninguna —") for i in range(3)]
                filts_act = [st.session_state.get(f"saved_f{i}", "— ninguno —") for i in range(5)]
                defaults = dict(gk=3,gs=0.8,cc=3.0,ct=8,mk=3,bd=7,bsc=50,bss=50,
                                sk=3,sd="XY",lk=3,ct1=30,ct2=80,uk=3,us=1.0,uf=1.5,
                                mok_a=5,mosh_a="Rectángulo",moit_a=1,
                                mok_b=3,mosh_b="Rectángulo",moit_b=1,
                                suma_v=50,resta_v=50,gamma=0.4,ls_min=0,ls_max=100,log_c=40.0,
                                th_tipo="Otsu (automático)",th_val=127,th_block=11,th_c=2,
                                an_alpha=0.075,an_K=50,an_niters=10)
                P_act = {k: st.session_state.get(k, v) for k, v in defaults.items()}
                nom = nombre_preset.strip().replace(" ","_").replace("/","_")
                guardar_preset(nom, ops_act, filts_act, P_act)
                st.success(f"✅ Preset '{nom}' guardado")
                st.rerun()
            else:
                st.warning("Escribe un nombre para el preset.")

    with st.expander("📤 Importar preset desde archivo .json", expanded=False):
        arch_json = st.file_uploader("Cargar .json", type=["json"], key="preset_upload")
        if arch_json is not None:
            try:
                datos_imp = json.load(arch_json)
                nom_imp = datos_imp.get("nombre", arch_json.name.replace(".json","")).replace(" ","_")
                with open(PRESETS_DIR / f"{nom_imp}.json", "w", encoding="utf-8") as ff:
                    json.dump(datos_imp, ff, ensure_ascii=False, indent=2)
                st.success(f"✅ Preset '{nom_imp}' importado")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # También mostrar botón para descargar preset activo como JSON
    with st.expander("📥 Exportar preset activo como archivo", expanded=False):
        nom_exp = st.text_input("Nombre para exportar", key="preset_exp_nombre",
                                placeholder="Pipeline_A_Highway...")
        if st.button("📥 Descargar JSON", key="btn_exp_dl", use_container_width=True):
            if nom_exp.strip():
                ops_exp   = [st.session_state.get(f"saved_pre_{i}", "— ninguna —") for i in range(3)]
                filts_exp = [st.session_state.get(f"saved_f{i}", "— ninguno —") for i in range(5)]
                defaults2 = dict(gk=3,gs=0.8,cc=3.0,ct=8,mk=3,bd=7,bsc=50,bss=50,
                                 sk=3,sd="XY",lk=3,ct1=30,ct2=80,uk=3,us=1.0,uf=1.5,
                                 mok_a=5,mosh_a="Rectángulo",moit_a=1,
                                 mok_b=3,mosh_b="Rectángulo",moit_b=1,
                                 suma_v=50,resta_v=50,gamma=0.4,ls_min=0,ls_max=100,log_c=40.0)
                P_exp = {k: st.session_state.get(k, v) for k, v in defaults2.items()}
                data_exp = {"nombre": nom_exp.strip(), "pre_ops": ops_exp,
                            "filt_names": filts_exp, "params": P_exp}
                st.download_button("⬇️ Descargar", data=json.dumps(data_exp, ensure_ascii=False, indent=2).encode(),
                                   file_name=f"{nom_exp.strip().replace(chr(32),chr(95))}.json",
                                   mime="application/json", key="dl_json_preset")

    st.markdown("---")
    st.markdown("### 🔧 Parámetros")

    with st.expander("⚡ Pre-procesamiento (Ops. Elementales)", expanded=True):
        suma_v  = st.slider("Suma — valor",         0,200,50,      key="suma_v")
        resta_v = st.slider("Resta — valor",        0,200,50,      key="resta_v")
        gamma   = st.slider("Gamma — γ",            0.1,4.0,0.4,0.05, key="gamma")
        ls_min  = st.slider("Stretch — % mín",      0,10,0,        key="ls_min")
        ls_max  = st.slider("Stretch — % máx",      90,100,100,    key="ls_max")
        log_c   = st.slider("Log — constante c",    1.0,80.0,40.0,1.0, key="log_c")

    with st.expander("🔵 Filtros Espaciales", expanded=False):
        gk  = st.slider("Gaussiano — Kernel",    1,15,3,2,   key="gk")
        gs  = st.slider("Gaussiano — Sigma",     0.1,5.0,0.8,0.1, key="gs")
        cc  = st.slider("CLAHE — Clip Limit",    0.5,8.0,3.0,0.5, key="cc")
        ct  = st.select_slider("CLAHE — Tile",   options=[4,8,16,32],value=8,key="ct")
        mk  = st.slider("Mediana — Kernel",      1,11,3,2,   key="mk")
        bd  = st.slider("Bilateral — Diám.",     1,15,7,2,   key="bd")
        bsc = st.slider("Bilateral — σ Color",   1,150,50,   key="bsc")
        bss = st.slider("Bilateral — σ Espacio", 1,150,50,   key="bss")
        sk  = st.slider("Sobel — Kernel",        1,7,3,2,    key="sk")
        sd  = st.selectbox("Sobel — Dir.",       ["XY","X","Y"], key="sd")
        lk  = st.slider("Laplaciano — Kernel",   1,7,3,2,    key="lk")
        ct1 = st.slider("Canny — Umbral 1",      0,128,30,   key="ct1")
        ct2 = st.slider("Canny — Umbral 2",      0,255,80,   key="ct2")
        uk  = st.slider("Unsharp — Kernel",      1,11,3,2,   key="uk")
        us  = st.slider("Unsharp — Sigma",       0.1,5.0,1.0,0.1, key="us")
        uf  = st.slider("Unsharp — Fuerza",      0.1,4.0,1.5,0.1, key="uf")

    with st.expander("🎯 Threshold (Umbralización)", expanded=False):
        th_tipo  = st.selectbox("Tipo de threshold", TIPOS_THRESH, key="th_tipo")
        th_val   = st.slider("Umbral fijo",     0, 255, 127,    key="th_val",
                             help="Solo para 'Binario fijo' e 'Inverso fijo'")
        th_block = st.slider("Block size (adapt.)", 3, 51, 11, 2, key="th_block",
                             help="Solo para métodos Adaptativos")
        th_c     = st.slider("Constante C (adapt.)", -20, 20, 2, key="th_c",
                             help="Solo para métodos Adaptativos")

    with st.expander("🌊 Anisotrópico (Perona-Malik)", expanded=False):
        an_alpha  = st.slider("Alpha — tasa difusión", 0.01, 0.25, 0.075, 0.005, key="an_alpha",
                              help="Menor = más suavizado. Rango recomendado: 0.05–0.15")
        an_K      = st.slider("K — sensibilidad borde",  5, 150, 50, 5,    key="an_K",
                              help="Mayor = preserva más bordes. Rango: 30–80 para satelital")
        an_niters = st.slider("Iteraciones",              1,  50, 10, 1,    key="an_niters",
                              help="Más iteraciones = más suavizado")

    with st.expander("🔴 Morfológico A  (Pasos 1–3)", expanded=True):
        mok_a  = st.slider("Kernel A",   1,15,5,2,key="mok_a")
        mosh_a = st.selectbox("Forma A",["Rectángulo","Elipse","Cruz"],key="mosh_a")
        moit_a = st.slider("Iters A",   1,5,1,key="moit_a")

    with st.expander("🟠 Morfológico B  (Pasos 4–5)", expanded=True):
        mok_b  = st.slider("Kernel B",   1,15,3,2,key="mok_b")
        mosh_b = st.selectbox("Forma B",["Rectángulo","Elipse","Cruz"],key="mosh_b")
        moit_b = st.slider("Iters B",   1,5,1,key="moit_b")

P = dict(
    gk=gk, gs=gs, cc=cc, ct=ct, mk=mk, bd=bd, bsc=bsc, bss=bss,
    sk=sk, sd=sd, lk=lk, ct1=ct1, ct2=ct2, uk=uk, us=us, uf=uf,
    mok_a=mok_a, mosh_a=mosh_a, moit_a=moit_a,
    mok_b=mok_b, mosh_b=mosh_b, moit_b=moit_b,
    suma_v=suma_v, resta_v=resta_v, gamma=gamma,
    ls_min=ls_min, ls_max=ls_max, log_c=log_c,
    th_tipo=th_tipo, th_val=th_val, th_block=th_block, th_c=th_c,
    an_alpha=an_alpha, an_K=an_K, an_niters=an_niters,
)

# ─── IMAGEN ──────────────────────────────────────────────────────────────────
if not uploaded:
    st.info("👈 Carga una imagen EuroSAT desde la barra lateral.")
    st.stop()

img_orig = np.array(Image.open(io.BytesIO(uploaded.read())).convert("RGB"))

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Pipeline completo",
    "⚖️ Comparación",
    "📊 Gráficas",
    "📋 Métricas",
])

# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCIA DEL PIPELINE EN SESSION STATE
# Guarda las selecciones independientemente del file uploader
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS_PRE  = [NINGUNA_PRE]  * 3
_DEFAULTS_FILT = [NINGUNO_FILT] * 5

for i, v in enumerate(_DEFAULTS_PRE):
    if f"saved_pre_{i}" not in st.session_state:
        st.session_state[f"saved_pre_{i}"] = v
for i, v in enumerate(_DEFAULTS_FILT):
    if f"saved_f{i}" not in st.session_state:
        st.session_state[f"saved_f{i}"] = v

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE COMPLETO EN TIEMPO REAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Botón de reset (esquina derecha del título)
    col_title, col_reset = st.columns([6, 1])
    with col_title:
        st.markdown('<span class="stage-header stage-pre">① Pre-procesamiento</span>', unsafe_allow_html=True)
        st.caption("Operaciones elementales de intensidad — se aplican sobre la imagen original.")
    with col_reset:
        if st.button("🔄 Reset", key="reset_pipeline", help="Limpia todos los filtros del pipeline"):
            for i in range(3):
                st.session_state[f"saved_pre_{i}"] = NINGUNA_PRE
                st.session_state[f"pre_{i}"]       = NINGUNA_PRE
            for i in range(5):
                st.session_state[f"saved_f{i}"] = NINGUNO_FILT
                st.session_state[f"f{i}"]       = NINGUNO_FILT
            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 1: PRE-PROCESAMIENTO
    # ──────────────────────────────────────────────────────────────────────────
    pre_cols = st.columns(3)
    pre_ops  = []
    for i, col in enumerate(pre_cols):
        with col:
            # Inicializar solo si no existe — no sobreescribir cambios del usuario
            if f"pre_{i}" not in st.session_state:
                st.session_state[f"pre_{i}"] = st.session_state.get(f"saved_pre_{i}", NINGUNA_PRE)
            sel = st.selectbox(f"Op. {i+1}", OPS_PRE, key=f"pre_{i}")
            st.session_state[f"saved_pre_{i}"] = sel  # mantener saved en sync
            pre_ops.append(sel)

    # Cómputo reactivo del pre-procesamiento
    pre_imgs  = [img_orig]
    pre_names = ["Original"]
    cur_pre   = img_orig.copy()
    for nombre in pre_ops:
        if nombre == NINGUNA_PRE:
            continue
        cur_pre = aplicar_pre(cur_pre.copy(), nombre, P)
        pre_imgs.append(cur_pre.copy())
        pre_names.append(nombre)

    img_preprocesada = cur_pre.copy()

    # Mostrar tira de pre-procesamiento
    mostrar_tira(pre_imgs, pre_names)

    # Chips del flujo
    if len(pre_names) > 1:
        st.markdown(
            " → ".join([f'<span class="chip-pre">{n}</span>' for n in pre_names]),
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 2: FILTROS ESPACIALES Y MORFOLÓGICOS
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("<hr class='stage-sep'>", unsafe_allow_html=True)
    st.markdown('<span class="stage-header stage-filt">② Filtros espaciales y morfológicos</span>', unsafe_allow_html=True)
    st.caption("Se aplican sobre el resultado del pre-procesamiento. Morfológico A = pasos 1–3 · B = pasos 4–5.")

    filt_cols  = st.columns(5)
    filt_names = []
    for i, col in enumerate(filt_cols):
        with col:
            grupo_label = " 🔴" if i < 3 else " 🟠"
            # Inicializar solo si no existe — no sobreescribir cambios del usuario
            if f"f{i}" not in st.session_state:
                st.session_state[f"f{i}"] = st.session_state.get(f"saved_f{i}", NINGUNO_FILT)
            sel = st.selectbox(f"Paso {i+1}{grupo_label}", FILTROS, key=f"f{i}")
            st.session_state[f"saved_f{i}"] = sel  # mantener saved en sync
            filt_names.append(sel)

    # Cómputo reactivo de filtros
    filt_imgs  = [img_preprocesada]
    filt_labels = ["Pre-proc."]
    cur_filt = img_preprocesada.copy()
    for i, nombre in enumerate(filt_names):
        if nombre == NINGUNO_FILT:
            continue
        grupo = "A" if i < 3 else "B"
        cur_filt = aplicar_filtro(cur_filt.copy(), nombre, P, grupo)
        filt_imgs.append(cur_filt.copy())
        filt_labels.append(nombre)

    img_final = cur_filt.copy()

    # Mostrar tira de filtros
    mostrar_tira(filt_imgs, filt_labels)

    if len(filt_labels) > 1:
        st.markdown(
            " → ".join([f'<span class="chip-filt">{n}</span>' for n in filt_labels]),
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ETAPA 3: RESULTADO Y ANÁLISIS
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("<hr class='stage-sep'>", unsafe_allow_html=True)
    st.markdown('<span class="stage-header stage-res">③ Resultado final</span>', unsafe_allow_html=True)

    # Comparación original → pre-proc → final → diferencia
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<p class="img-label">Original</p>', unsafe_allow_html=True)
        st.image(img_orig, use_container_width=True)
    with col2:
        st.markdown('<p class="img-label">Pre-procesada</p>', unsafe_allow_html=True)
        st.image(img_preprocesada, use_container_width=True)
    with col3:
        st.markdown('<p class="img-label">Final (pipeline completo)</p>', unsafe_allow_html=True)
        st.image(img_final, use_container_width=True)
    with col4:
        st.markdown('<p class="img-label">Diferencia (orig → final)</p>', unsafe_allow_html=True)
        st.image(diferencia(img_orig, img_final), use_container_width=True)

    # ── Análisis opcional ─────────────────────────────────────────────────────
    st.markdown("<hr class='stage-sep'>", unsafe_allow_html=True)

    col_tog1, col_tog2, col_tog3 = st.columns(3)
    with col_tog1:
        show_hist = st.toggle("📊 Histogramas",        value=False, key="sh")
    with col_tog2:
        show_metr = st.toggle("📋 Métricas generales", value=False, key="sm")
    with col_tog3:
        show_edge = st.toggle("🔍 Métricas de bordes", value=False, key="se")

    if show_hist:
        st.markdown("#### Histogramas: Original → Pre-proc → Final")
        fig_h = plot_hist(
            [img_orig, img_preprocesada, img_final],
            ["Original", "Pre-procesada", "Final"]
        )
        st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

        st.markdown("#### Perfil de intensidad — fila central")
        fig_p = plot_perfil(img_orig, img_final, "Final")
        st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

        # Curvas de transformación del pre-procesamiento
        ops_activas = [op for op in pre_ops if op != NINGUNA_PRE]
        if ops_activas:
            st.markdown("#### Curvas de transformación (pre-procesamiento)")
            curva_cols = st.columns(len(ops_activas))
            for col, op in zip(curva_cols, ops_activas):
                with col:
                    fig_c = plot_curva(op, P)
                    st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

    if show_metr:
        st.markdown("#### Métricas por etapa (referencia: imagen original)")

        img_diff = diferencia(img_orig, img_final)

        etapas = [
            ("Pre-procesada",        img_preprocesada),
            ("Final (pipeline)",     img_final),
            ("Diferencia orig→final",img_diff),
        ]
        tabla = []
        for lbl, img in etapas:
            m = metricas(img_orig, img); m["Etapa"] = lbl
            tabla.append(m)
        df = pd.DataFrame(tabla).set_index("Etapa")
        st.dataframe(
            df.style.background_gradient(cmap="Blues",  subset=["SSIM"])
                    .background_gradient(cmap="Reds_r", subset=["MSE"]),
            use_container_width=True
        )

        # Tarjetas de métricas clave para Final y Diferencia
        st.markdown("#### Resumen rápido")
        m_final = metricas(img_orig, img_final)
        m_diff  = metricas(img_orig, img_diff)

        col_f1,col_f2,col_f3,col_sep,col_d1,col_d2,col_d3 = st.columns([1,1,1,0.2,1,1,1])
        st.markdown("""
        <style>
        .mcard { background:#0d1526; border:1px solid #1e2d4a; border-radius:8px;
                 padding:10px; text-align:center; }
        .mcard .mlabel { color:#8899aa; font-size:0.7rem; font-family:'Space Mono',monospace;
                         text-transform:uppercase; letter-spacing:1px; }
        .mcard .mval   { font-size:1.3rem; font-weight:700; font-family:'Space Mono',monospace; }
        .mcard .mval.blue  { color:#00d4ff; }
        .mcard .mval.green { color:#51cf66; }
        .mcard .mval.red   { color:#ff6b6b; }
        .mcard .msub   { color:#8899aa; font-size:0.65rem; margin-top:2px; }
        </style>
        """, unsafe_allow_html=True)

        with col_f1:
            st.markdown(f'<div class="mcard"><div class="mlabel">MSE · Final</div>'
                        f'<div class="mval red">{m_final["MSE"]}</div>'
                        f'<div class="msub">↓ mejor</div></div>', unsafe_allow_html=True)
        with col_f2:
            st.markdown(f'<div class="mcard"><div class="mlabel">PSNR · Final</div>'
                        f'<div class="mval blue">{m_final["PSNR"]} dB</div>'
                        f'<div class="msub">↑ mejor</div></div>', unsafe_allow_html=True)
        with col_f3:
            st.markdown(f'<div class="mcard"><div class="mlabel">SSIM · Final</div>'
                        f'<div class="mval green">{m_final["SSIM"]}</div>'
                        f'<div class="msub">↑ mejor · máx 1.0</div></div>', unsafe_allow_html=True)
        with col_sep:
            st.markdown('<div style="border-left:1px solid #1e2d4a;height:80px;margin:auto;width:1px"></div>',
                        unsafe_allow_html=True)
        with col_d1:
            st.markdown(f'<div class="mcard"><div class="mlabel">MSE · Diferencia</div>'
                        f'<div class="mval red">{m_diff["MSE"]}</div>'
                        f'<div class="msub">↓ mejor</div></div>', unsafe_allow_html=True)
        with col_d2:
            st.markdown(f'<div class="mcard"><div class="mlabel">PSNR · Diferencia</div>'
                        f'<div class="mval blue">{m_diff["PSNR"]} dB</div>'
                        f'<div class="msub">↑ mejor</div></div>', unsafe_allow_html=True)
        with col_d3:
            st.markdown(f'<div class="mcard"><div class="mlabel">SSIM · Diferencia</div>'
                        f'<div class="mval green">{m_diff["SSIM"]}</div>'
                        f'<div class="msub">↑ mejor · máx 1.0</div></div>', unsafe_allow_html=True)

        st.caption("**MSE** = Error cuadrático medio · **PSNR** = Relación señal/ruido · **SSIM** = Similitud estructural — todos calculados respecto a la imagen original")


    if show_edge:
        st.markdown("<hr class='stage-sep'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Métricas de detección de bordes")
        st.caption("Calculadas sobre la imagen original y el resultado final del pipeline.")

        resultados_borde = calcular_metricas_borde(img_orig, img_final)

        # Tabla comparativa
        filas = []
        for metrica, (val_orig, val_proc) in resultados_borde.items():
            cambio = val_proc - val_orig
            signo  = "▲" if cambio > 0 else "▼"
            filas.append({
                "Métrica":    metrica,
                "Original":   val_orig,
                "Final":      val_proc,
                "Cambio":     f"{signo} {abs(round(cambio, 4))}",
            })
        df_edge = pd.DataFrame(filas).set_index("Métrica")
        st.dataframe(df_edge, use_container_width=True)

        # Tarjetas de las métricas clave
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        dens_o, dens_p = resultados_borde["Edge Density (%)"]
        mes_o,  mes_p  = resultados_borde["Edge Strength — media"]
        ent_o,  ent_p  = resultados_borde["Entropía (bits)"]
        cov_o,  cov_p  = resultados_borde["Mask Coverage (%)"]

        st.markdown("""
        <style>
        .ecard { background:#0d1526; border:1px solid #1e2d4a; border-radius:8px;
                 padding:10px; text-align:center; margin-bottom:6px; }
        .ecard .elabel { color:#8899aa; font-size:0.68rem; font-family:"Space Mono",monospace;
                         text-transform:uppercase; letter-spacing:1px; }
        .ecard .eval   { font-size:1.1rem; font-weight:700; font-family:"Space Mono",monospace; color:#00d4ff; }
        .ecard .ediff  { font-size:0.72rem; margin-top:2px; }
        .ecard .up     { color:#51cf66; }
        .ecard .down   { color:#ff6b6b; }
        </style>""", unsafe_allow_html=True)

        def delta_html(orig, proc, higher_better=True):
            diff = proc - orig
            cls  = "up" if (diff > 0) == higher_better else "down"
            sign = "▲" if diff > 0 else "▼"
            return f'<span class="{cls}">{sign} {abs(round(diff,4))}</span>'

        with col_e1:
            st.markdown(f"""<div class="ecard">
                <div class="elabel">Edge Density</div>
                <div class="eval">{dens_p}%</div>
                <div class="ediff">orig: {dens_o}% &nbsp; {delta_html(dens_o, dens_p)}</div>
            </div>""", unsafe_allow_html=True)
        with col_e2:
            st.markdown(f"""<div class="ecard">
                <div class="elabel">Edge Strength</div>
                <div class="eval">{mes_p}</div>
                <div class="ediff">orig: {mes_o} &nbsp; {delta_html(mes_o, mes_p)}</div>
            </div>""", unsafe_allow_html=True)
        with col_e3:
            st.markdown(f"""<div class="ecard">
                <div class="elabel">Entropía (bits)</div>
                <div class="eval">{ent_p}</div>
                <div class="ediff">orig: {ent_o} &nbsp; {delta_html(ent_o, ent_p)}</div>
            </div>""", unsafe_allow_html=True)
        with col_e4:
            st.markdown(f"""<div class="ecard">
                <div class="elabel">Mask Coverage</div>
                <div class="eval">{cov_p}%</div>
                <div class="ediff">orig: {cov_o}% &nbsp; {delta_html(cov_o, cov_p)}</div>
            </div>""", unsafe_allow_html=True)

        # Comparación visual de bordes Canny
        st.markdown("#### Comparación visual de bordes Canny")
        fig_edge = plot_edge_comparison(img_orig, img_final, "Final")
        st.pyplot(fig_edge, use_container_width=True); plt.close(fig_edge)

        st.caption("**Edge Density** = % píxeles borde (Canny) · **Edge Strength** = magnitud gradiente Sobel · "
                   "**Entropía** = información Shannon · **Mask Coverage** = % píxeles blancos en máscara binarizada")


    # ── Exportar configuración y métricas ─────────────────────────────────────
    st.markdown("<hr class='stage-sep'>", unsafe_allow_html=True)
    st.markdown('<span class="stage-header stage-res">📋 Exportar configuración y métricas</span>',
                unsafe_allow_html=True)
    st.caption("Copia este bloque directamente en tu informe o cuaderno de notas.")

    # Construir nombre de archivo si hay uno cargado
    nombre_img = uploaded.name if uploaded else "imagen_sin_nombre"

    # Construir texto de pipeline
    # Leer SIEMPRE desde session_state para evitar lag de render
    pre_activos  = [st.session_state.get(f"saved_pre_{i}", NINGUNA_PRE)
                    for i in range(3)
                    if st.session_state.get(f"saved_pre_{i}", NINGUNA_PRE) != NINGUNA_PRE]
    filt_activos = [st.session_state.get(f"saved_f{i}", NINGUNO_FILT)
                    for i in range(5)
                    if st.session_state.get(f"saved_f{i}", NINGUNO_FILT) != NINGUNO_FILT]
    pipeline_str = " → ".join(pre_activos + filt_activos) if (pre_activos or filt_activos) else "Sin filtros aplicados"

    # Recopilar parámetros relevantes según filtros usados
    params_usados = {}
    for op in pre_activos:
        if op == "Corrección Gamma":   params_usados["Gamma γ"]            = P["gamma"]
        if op == "Linear Stretching":  params_usados["Stretch % mín/máx"]  = f"{P['ls_min']} / {P['ls_max']}"
        if op == "Suma":               params_usados["Suma valor"]          = P["suma_v"]
        if op == "Resta":              params_usados["Resta valor"]         = P["resta_v"]
        if op == "Transformada Log":   params_usados["Log constante c"]     = P["log_c"]
    for fn in filt_activos:
        if fn == "Gaussiano":          params_usados["Gaussiano k / σ"]     = f"{P['gk']} / {P['gs']}"
        if fn == "CLAHE":              params_usados["CLAHE clip / tile"]   = f"{P['cc']} / {P['ct']}"
        if fn == "Mediana":            params_usados["Mediana k"]           = P["mk"]
        if fn == "Bilateral":         params_usados["Bilateral d/σC/σS"]   = f"{P['bd']}/{P['bsc']}/{P['bss']}"
        if fn == "Sobel":              params_usados["Sobel k / dir"]       = f"{P['sk']} / {P['sd']}"
        if fn == "Laplaciano":         params_usados["Laplaciano k"]        = P["lk"]
        if fn == "Canny":              params_usados["Canny T1 / T2"]       = f"{P['ct1']} / {P['ct2']}"
        if fn == "Unsharp Mask":       params_usados["Unsharp k/σ/fuerza"] = f"{P['uk']}/{P['us']}/{P['uf']}"
        if fn == "Threshold":          params_usados[f"Threshold tipo"]    = P["th_tipo"]
        if fn == "Anisotrópico":       params_usados["Anisotr. α/K/iters"] = f"{P['an_alpha']}/{P['an_K']}/{P['an_niters']}"
        if fn == "Threshold" and P["th_tipo"] in ["Binario fijo","Inverso fijo"]:
                                       params_usados["Threshold umbral"]   = P["th_val"]
        if fn == "Threshold" and "Adaptativo" in P["th_tipo"]:
                                       params_usados["Threshold block/C"]  = f"{P['th_block']} / {P['th_c']}"
        if fn in F_MORFO:
            pos = next((i for i,x in enumerate(filt_activos) if x == fn), 0)
            grupo = "A" if pos < 3 else "B"
            g = grupo.lower()
            params_usados[f"{fn} kernel/forma/iters ({grupo})"] = \
                f"{P[f'mok_{g}']} / {P[f'mosh_{g}']} / {P[f'moit_{g}']}"

    # Calcular métricas para el bloque
    img_diff_exp  = diferencia(img_orig, img_final)
    m_pre_exp     = metricas(img_orig, img_preprocesada)
    m_fin_exp     = metricas(img_orig, img_final)
    m_dif_exp     = metricas(img_orig, img_diff_exp)

    # Métricas de bordes
    rb = calcular_metricas_borde(img_orig, img_final)

    # Construir el texto exportable
    separador = "─" * 56
    params_lines = "\n".join(f"  {k:<32} {v}" for k, v in params_usados.items())

    # Líneas de métricas de bordes
    edge_lines = "\n".join(
        f"  {metrica:<30} orig={vals[0]:>8}  final={vals[1]:>8}  Δ={round(vals[1]-vals[0],4):>+9}"
        for metrica, vals in rb.items()
    )

    texto_export = f"""
{'═'*56}
  CONFIGURACIÓN DEL PIPELINE — EuroSAT Visión Artificial
{'═'*56}
  Imagen        : {nombre_img}

  PIPELINE
  {pipeline_str}

  PARÁMETROS
{params_lines if params_lines else '  (sin parámetros adicionales)'}

  MORFOLÓGICO A (Pasos 1–3)
  Kernel: {P['mok_a']}  |  Forma: {P['mosh_a']}  |  Iters: {P['moit_a']}

  MORFOLÓGICO B (Pasos 4–5)
  Kernel: {P['mok_b']}  |  Forma: {P['mosh_b']}  |  Iters: {P['moit_b']}

{separador}
  MÉTRICAS DE CALIDAD  (referencia: imagen original)
{separador}
  {'Etapa':<26} {'MSE':>10} {'PSNR (dB)':>10} {'SSIM':>8} {'Media':>8} {'Std':>8}
  {'─'*26} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}
  {'Pre-procesada':<26} {m_pre_exp['MSE']:>10} {m_pre_exp['PSNR']:>10} {m_pre_exp['SSIM']:>8} {m_pre_exp['Media']:>8} {m_pre_exp['Std']:>8}
  {'Final (pipeline)':<26} {m_fin_exp['MSE']:>10} {m_fin_exp['PSNR']:>10} {m_fin_exp['SSIM']:>8} {m_fin_exp['Media']:>8} {m_fin_exp['Std']:>8}
  {'Diferencia orig→final':<26} {m_dif_exp['MSE']:>10} {m_dif_exp['PSNR']:>10} {m_dif_exp['SSIM']:>8} {m_dif_exp['Media']:>8} {m_dif_exp['Std']:>8}

  MSE ↓ mejor · PSNR ↑ mejor · SSIM ↑ mejor (máx 1.0)

{separador}
  MÉTRICAS DE DETECCIÓN DE BORDES  (original → final)
{separador}
{edge_lines}

  Edge Density ▲ mejor · Edge Strength ▲ mejor · Entropía ▲ mejor
{'═'*56}
""".strip()

    # Sin key= para que Streamlit use siempre value=texto_export actualizado
    st.text_area(
        label="📋 Configuración y métricas — listo para copiar",
        value=texto_export,
        height=370,
        help="Selecciona todo (Ctrl+A dentro del cuadro) y copia (Ctrl+C)"
    )

    # Botón de descarga como .txt
    st.download_button(
        label="⬇️ Descargar como .txt",
        data=texto_export.encode("utf-8"),
        file_name=f"pipeline_{nombre_img.rsplit('.',1)[0]}.txt",
        mime="text/plain",
        key="dl_config"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARACIÓN MÚLTIPLE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Comparación de filtros individuales — reactivo")

    st.markdown("**Fuente de comparación:**")
    fuente_cmp = st.radio("", ["Original", "Pre-procesada (del pipeline)"],
                          horizontal=True, key="fuente_cmp")
    img_cmp_src = img_orig if fuente_cmp=="Original" else img_preprocesada

    filtros_comp = []
    for i, col in enumerate(st.columns(4)):
        with col:
            filtros_comp.append(st.selectbox(f"Filtro {chr(65+i)}", FILTROS, key=f"cmp_{i}"))

    resultados = [(n, aplicar_filtro(img_cmp_src.copy(), n, P, "A"))
                  for n in filtros_comp if n != NINGUNO_FILT]

    if resultados:
        n_tot = len(resultados)+1
        fig, axes = plt.subplots(1, n_tot, figsize=(4*n_tot, 4))
        fig.patch.set_facecolor("white")
        if n_tot==1: axes=[axes]
        lbl_src = "Original" if fuente_cmp=="Original" else "Pre-proc."
        axes[0].set_facecolor("white"); axes[0].imshow(img_cmp_src); axes[0].set_title(lbl_src,color="black",fontsize=9); axes[0].axis("off")
        for ax,(lbl,img) in zip(axes[1:],resultados):
            ax.set_facecolor("white"); ax.imshow(img); ax.set_title(lbl,color="black",fontsize=8); ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown("#### Histogramas")
        fig_h = plot_hist([img_cmp_src]+[r[1] for r in resultados],
                          [lbl_src]+[r[0] for r in resultados])
        st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)
    else:
        st.image(img_cmp_src, caption="Imagen fuente", width=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRÁFICAS INDIVIDUALES
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Gráficas detalladas")

    fuente_graf = st.radio("Imagen de entrada", ["Original","Pre-procesada"],
                           horizontal=True, key="fuente_gr")
    img_gr_src = img_orig if fuente_graf=="Original" else img_preprocesada

    fa = st.selectbox("Filtro a analizar", FILTROS[1:], key="anal")
    img_proc = aplicar_filtro(img_gr_src.copy(), fa, P, "A")

    col_a, col_b = st.columns(2)
    with col_a: st.image(img_gr_src, caption=fuente_graf,  use_container_width=True)
    with col_b: st.image(img_proc,   caption=fa,            use_container_width=True)
    st.markdown("---")

    fig_h = plot_hist([img_gr_src, img_proc], [fuente_graf, fa])
    st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

    fig_p = plot_perfil(img_gr_src, img_proc, fa)
    st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

    diff_img = diferencia(img_gr_src, img_proc)
    fig_d, axes = plt.subplots(1,3,figsize=(12,3.5))
    fig_d.patch.set_facecolor("white")
    for ax,img,t in zip(axes,[img_gr_src,img_proc,diff_img],[fuente_graf,fa,"Diferencia"]):
        ax.set_facecolor("white"); ax.imshow(img); ax.set_title(t,color="black",fontsize=9); ax.axis("off")
    fig_d.tight_layout()
    st.pyplot(fig_d, use_container_width=True); plt.close(fig_d)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Métricas de calidad")

    todas_op = FILTROS[1:] + OPS_PRE[1:]
    sel = st.multiselect(
        "Selecciona filtros/operaciones a comparar", todas_op,
        default=["Gaussiano","CLAHE","Black-Hat","Corrección Gamma","Ecualización HE"],
        key="met_sel"
    )

    fuente_met = st.radio("Imagen de referencia para métricas",
                          ["Original","Pre-procesada"], horizontal=True, key="fuente_met")
    img_met_src = img_orig if fuente_met=="Original" else img_preprocesada

    if sel:
        tabla = []
        for nombre in sel:
            if nombre in FILTROS[1:]:
                ip = aplicar_filtro(img_met_src.copy(), nombre, P, "A")
            else:
                ip = aplicar_pre(img_met_src.copy(), nombre, P)
            if ip.shape != img_met_src.shape:
                ip = cv2.cvtColor(cv2.cvtColor(ip,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
            m = metricas(img_met_src, ip); m["Filtro/Op"] = nombre
            tabla.append(m)

        df = pd.DataFrame(tabla).set_index("Filtro/Op")
        st.dataframe(
            df.style.background_gradient(cmap="Blues",  subset=["SSIM"])
                    .background_gradient(cmap="Reds_r", subset=["MSE"]),
            use_container_width=True
        )

        fig_b, ax = plt.subplots(figsize=(10, max(3, len(df)*0.55)))
        fig_b.patch.set_facecolor("white"); ax.set_facecolor("white")
        bars = ax.barh(df.index, df["SSIM"], color="#339af0", alpha=0.85)
        ax.set_xlim(0,1); ax.set_title("SSIM por filtro/op",color="black",fontsize=10)
        ax.grid(True, axis='x', color="#e6e6e6", linewidth=0.8)
        ax.tick_params(colors="#222222",labelsize=8); ax.spines[:].set_color("#cccccc")
        for bar in bars:
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.4f}", va="center", color="black", fontsize=7)
        st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
        st.caption("MSE ↓ mejor · PSNR ↑ mejor · SSIM ↑ mejor")
    else:
        st.info("Selecciona al menos un filtro u operación arriba.")