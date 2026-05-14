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
    .step-chip {
        display: inline-block; background: #1a2035; border: 1px solid #2a3555;
        color: #00d4ff; border-radius: 6px; padding: 3px 10px;
        font-family: 'Space Mono', monospace; font-size: 0.75rem; margin: 2px;
    }
    .op-chip {
        display: inline-block; background: #1a2d1a; border: 1px solid #2a552a;
        color: #51cf66; border-radius: 6px; padding: 3px 10px;
        font-family: 'Space Mono', monospace; font-size: 0.75rem; margin: 2px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0066ff, #7b2fff);
        color: white; border: none; border-radius: 8px;
        font-weight: 700; padding: 0.4rem 1.2rem; width: 100%;
    }
    .result-header {
        color: #00d4ff; font-size: 0.8rem; font-family: 'Space Mono', monospace;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;
    }
    [data-testid="stSidebar"] { background: #080e1e; }
    .section-divider {
        border: none; border-top: 1px solid #1e2d4a; margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

DARK_BG = "#080e1e"
GRID_C  = "#1a2535"

# ═══════════════════════════════════════════════════════════════════════════════
# FILTROS
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

OPS_ELEMENTALES = ["— ninguna —","Suma","Resta","Corrección Gamma",
                   "Linear Stretching","Transformada Log","Ecualización HE","Negativa"]

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def diferencia(orig, proc):
    o = orig.astype(np.float32)
    p = proc.astype(np.float32)
    if o.shape != p.shape:
        p = cv2.cvtColor(cv2.cvtColor(proc,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB).astype(np.float32)
    return cv2.normalize(np.abs(o-p), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def metricas(orig, proc):
    og = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    pg = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY) if proc.ndim==3 else proc
    if og.shape != pg.shape: pg = cv2.resize(pg,(og.shape[1],og.shape[0]))
    mse = float(np.mean((og.astype(float)-pg.astype(float))**2))
    return dict(
        MSE=round(mse,2),
        PSNR=round(float(cv2.PSNR(og,pg)) if mse>0 else 999,2),
        SSIM=round(float(ssim(og,pg,data_range=255)),4),
        Media_proc=round(float(np.mean(pg)),2),
        Std_proc=round(float(np.std(pg)),2),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_histogramas(imgs, labels):
    fig, axes = plt.subplots(1, len(imgs), figsize=(5*len(imgs), 3.2))
    if len(imgs)==1: axes=[axes]
    fig.patch.set_facecolor(DARK_BG)
    for ax, img, lbl in zip(axes, imgs, labels):
        ax.set_facecolor(DARK_BG)
        for i,(col,name) in enumerate(zip(("#ff4d6d","#51cf66","#339af0"),"RGB")):
            ch = min(i, img.shape[2]-1) if img.ndim==3 else 0
            h = cv2.calcHist([img],[ch],None,[256],[0,256]).flatten()
            ax.fill_between(range(256), h, alpha=0.3, color=col)
            ax.plot(h, color=col, lw=0.8, label=name)
        ax.set_title(lbl, color="white", fontsize=9)
        ax.set_xlim(0,255); ax.tick_params(colors="#8899aa",labelsize=7)
        ax.spines[:].set_color(GRID_C)
        ax.legend(fontsize=7,facecolor="#0d1526",labelcolor="white",edgecolor=GRID_C)
    fig.tight_layout(); return fig

def plot_perfil(orig, proc, lbl):
    row = orig.shape[0]//2
    fig, axes = plt.subplots(2,1,figsize=(10,4),sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    for ax,(img,title) in zip(axes,[(orig,"Original"),(proc,lbl)]):
        ax.set_facecolor(DARK_BG); ax.set_title(title,color="white",fontsize=9)
        if img.ndim==3:
            for col,name,ch in zip(("#ff4d6d","#51cf66","#339af0"),"RGB",range(3)):
                ax.plot(img[row,:,ch],color=col,lw=0.9,label=name)
        else: ax.plot(img[row,:],color="#00d4ff",lw=0.9)
        ax.tick_params(colors="#8899aa",labelsize=7); ax.spines[:].set_color(GRID_C)
        ax.legend(fontsize=7,facecolor="#0d1526",labelcolor="white",edgecolor=GRID_C)
    axes[-1].set_xlabel("Píxel (columna)",color="#8899aa",fontsize=8)
    fig.tight_layout(); return fig

def plot_diferencia(orig, proc, lbl_proc):
    diff = diferencia(orig, proc)
    fig, axes = plt.subplots(1,3,figsize=(12,3.5))
    fig.patch.set_facecolor(DARK_BG)
    for ax,img,t in zip(axes,[orig,proc,diff],["Original",lbl_proc,"Diferencia"]):
        ax.imshow(img); ax.set_title(t,color="white",fontsize=9); ax.axis("off")
    fig.tight_layout(); return fig

def plot_curva(nombre_op, p):
    r = np.arange(256,dtype=np.float32)
    if   nombre_op=="Suma":              s = np.clip(r+p["suma_v"],0,255)
    elif nombre_op=="Resta":             s = np.clip(r-p["resta_v"],0,255)
    elif nombre_op=="Corrección Gamma":  s = 255*(r/255)**(1/max(p["gamma"],0.01))
    elif nombre_op=="Linear Stretching": s = np.clip((r-np.percentile(r,p["ls_min"]))/(max(np.percentile(r,p["ls_max"])-np.percentile(r,p["ls_min"]),1))*255,0,255)
    elif nombre_op=="Transformada Log":  s = np.clip(p["log_c"]*np.log(r+1),0,255)
    elif nombre_op=="Negativa":          s = 255-r
    else:                                s = r
    fig,ax = plt.subplots(figsize=(3.5,3))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.plot(r,s,color="#00d4ff",lw=1.8)
    ax.plot([0,255],[0,255],color="#2a3555",lw=0.8,linestyle="--")
    ax.set_title("Curva T(r)",color="white",fontsize=9)
    ax.set_xlabel("r",color="#8899aa",fontsize=8); ax.set_ylabel("s",color="#8899aa",fontsize=8)
    ax.tick_params(colors="#8899aa",labelsize=7); ax.spines[:].set_color(GRID_C)
    ax.set_xlim(0,255); ax.set_ylim(0,255)
    fig.tight_layout(); return fig

# ═══════════════════════════════════════════════════════════════════════════════
# DESPACHO
# ═══════════════════════════════════════════════════════════════════════════════

F_ESPACIALES = ["Gaussiano","CLAHE","Mediana","Bilateral","Sobel","Laplaciano","Canny","Unsharp Mask"]
F_MORFO      = list(OPS_MORFO.keys())
TODOS        = ["— ninguno —"] + F_ESPACIALES + F_MORFO

def aplicar(img, nombre, p):
    if nombre=="— ninguno —":    return img
    if nombre=="Gaussiano":      return f_gaussiano(img, p["gk"], p["gs"])
    if nombre=="CLAHE":          return f_clahe(img, p["cc"], p["ct"])
    if nombre=="Mediana":        return f_mediana(img, p["mk"])
    if nombre=="Bilateral":      return f_bilateral(img, p["bd"], p["bsc"], p["bss"])
    if nombre=="Sobel":          return f_sobel(img, p["sk"], p["sd"])
    if nombre=="Laplaciano":     return f_laplaciano(img, p["lk"])
    if nombre=="Canny":          return f_canny(img, p["ct1"], p["ct2"])
    if nombre=="Unsharp Mask":   return f_unsharp(img, p["uk"], p["us"], p["uf"])
    if nombre in F_MORFO:
        # Usa el conjunto A o B según el paso
        return f_morfo(img, nombre, p["mok_a"], p["mosh_a"], p["moit_a"])
    return img

def aplicar_con_grupo(img, nombre, p, grupo="A"):
    """Igual que aplicar() pero elige el grupo de params morfológicos."""
    if nombre not in F_MORFO:
        return aplicar(img, nombre, p)
    k  = p[f"mok_{grupo.lower()}"]
    sh = p[f"mosh_{grupo.lower()}"]
    it = p[f"moit_{grupo.lower()}"]
    return f_morfo(img, nombre, k, sh, it)

def aplicar_op(img, nombre, p):
    if nombre=="— ninguna —":         return img
    if nombre=="Suma":                return op_suma(img, p["suma_v"])
    if nombre=="Resta":               return op_resta(img, p["resta_v"])
    if nombre=="Corrección Gamma":    return op_gamma(img, p["gamma"])
    if nombre=="Linear Stretching":   return op_stretch(img, p["ls_min"], p["ls_max"])
    if nombre=="Transformada Log":    return op_log(img, p["log_c"])
    if nombre=="Ecualización HE":     return op_he(img)
    if nombre=="Negativa":            return op_negativa(img)
    return img

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">🛰️ Explorador de Filtros EuroSAT</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Filtros espaciales · morfológicos · operaciones elementales en tiempo real</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Imagen")
    uploaded = st.file_uploader("Cargar imagen EuroSAT", type=["jpg","jpeg","png"])
    st.markdown("---")
    st.markdown("### 🔧 Parámetros")

    with st.expander("Espaciales", expanded=False):
        gk  = st.slider("Gaussiano — Kernel",    1,15, 5,2,key="gk")
        gs  = st.slider("Gaussiano — Sigma",     0.1,5.0,1.0,0.1,key="gs")
        cc  = st.slider("CLAHE — Clip Limit",    0.5,8.0,2.0,0.5,key="cc")
        ct  = st.select_slider("CLAHE — Tile",   options=[4,8,16,32],value=8,key="ct")
        mk  = st.slider("Mediana — Kernel",      1,11,3,2,key="mk")
        bd  = st.slider("Bilateral — Diám.",     1,15,7,2,key="bd")
        bsc = st.slider("Bilateral — σ Color",   1,150,50,key="bsc")
        bss = st.slider("Bilateral — σ Espacio", 1,150,50,key="bss")
        sk  = st.slider("Sobel — Kernel",        1,7,3,2,key="sk")
        sd  = st.selectbox("Sobel — Dir.",       ["XY","X","Y"],key="sd")
        lk  = st.slider("Laplaciano — Kernel",   1,7,3,2,key="lk")
        ct1 = st.slider("Canny — Umbral 1",      0,128,30,key="ct1")
        ct2 = st.slider("Canny — Umbral 2",      0,255,80,key="ct2")
        uk  = st.slider("Unsharp — Kernel",      1,11,3,2,key="uk")
        us  = st.slider("Unsharp — Sigma",       0.1,5.0,1.0,0.1,key="us")
        uf  = st.slider("Unsharp — Fuerza",      0.1,4.0,1.5,0.1,key="uf")

    with st.expander("Morfológico A  (Pasos 1–3)", expanded=True):
        mok_a  = st.slider("Kernel A",    1,15,3,2,key="mok_a")
        mosh_a = st.selectbox("Forma A", ["Rectángulo","Elipse","Cruz"],key="mosh_a")
        moit_a = st.slider("Iters A",    1,5,1,key="moit_a")

    with st.expander("Morfológico B  (Pasos 4–5)", expanded=True):
        mok_b  = st.slider("Kernel B",    1,15,3,2,key="mok_b")
        mosh_b = st.selectbox("Forma B", ["Rectángulo","Elipse","Cruz"],key="mosh_b")
        moit_b = st.slider("Iters B",    1,5,1,key="moit_b")

    with st.expander("⚡ Ops. Elementales", expanded=True):
        suma_v  = st.slider("Suma — valor",         0,200,50,     key="suma_v")
        resta_v = st.slider("Resta — valor",        0,200,50,     key="resta_v")
        gamma   = st.slider("Gamma — γ",            0.1,4.0,1.0,0.1,key="gamma")
        ls_min  = st.slider("Stretch — % mín",      0,10,0,       key="ls_min")
        ls_max  = st.slider("Stretch — % máx",      90,100,100,   key="ls_max")
        log_c   = st.slider("Log — constante c",    1.0,80.0,40.0,1.0,key="log_c")

P = dict(
    gk=gk, gs=gs, cc=cc, ct=ct, mk=mk, bd=bd, bsc=bsc, bss=bss,
    sk=sk, sd=sd, lk=lk, ct1=ct1, ct2=ct2, uk=uk, us=us, uf=uf,
    mok_a=mok_a, mosh_a=mosh_a, moit_a=moit_a,
    mok_b=mok_b, mosh_b=mosh_b, moit_b=moit_b,
    suma_v=suma_v, resta_v=resta_v, gamma=gamma,
    ls_min=ls_min, ls_max=ls_max, log_c=log_c,
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
    "🧪 Pipeline + Ops. en tiempo real",
    "⚖️ Comparación",
    "📊 Gráficas",
    "📋 Métricas",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE REACTIVO + OPS. ELEMENTALES
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Selección de pasos ────────────────────────────────────────────────────
    st.markdown("#### Pasos del pipeline")
    cols_sel = st.columns(5)
    pasos = []
    for i, col in enumerate(cols_sel):
        with col:
            grupo_label = " (A)" if i < 3 else " (B)"
            pasos.append(st.selectbox(f"Paso {i+1}{grupo_label}", TODOS, key=f"p{i}"))

    # ── Cómputo reactivo del pipeline ─────────────────────────────────────────
    steps_imgs  = [img_orig]
    steps_names = ["Original"]
    current = img_orig.copy()
    for i, nombre in enumerate(pasos):
        if nombre == "— ninguno —":
            continue
        grupo = "A" if i < 3 else "B"
        if nombre in F_MORFO:
            current = f_morfo(current.copy(), nombre,
                              P[f"mok_{grupo.lower()}"],
                              P[f"mosh_{grupo.lower()}"],
                              P[f"moit_{grupo.lower()}"])
        else:
            current = aplicar(current.copy(), nombre, P)
        steps_imgs.append(current.copy())
        steps_names.append(nombre)

    img_pipeline = current.copy()  # resultado del pipeline

    # ── Visualización de pasos ────────────────────────────────────────────────
    st.markdown("#### Pasos intermedios")
    n_cols = len(steps_imgs)
    cols_show = st.columns(n_cols)
    for col, img_s, name_s in zip(cols_show, steps_imgs, steps_names):
        with col:
            st.image(img_s, caption=name_s, use_container_width=True)

    st.markdown("**Pipeline:** " + " → ".join(
        [f'<span class="step-chip">{n}</span>' for n in steps_names]
    ), unsafe_allow_html=True)

    # ── OPERACIÓN ELEMENTAL sobre resultado del pipeline ─────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Operación elemental sobre el resultado del pipeline")
    st.caption("Mueve los sliders de la barra lateral y el resultado se actualiza en tiempo real.")

    op_sel = st.selectbox("Operación elemental", OPS_ELEMENTALES, key="op_rt")

    img_final = aplicar_op(img_pipeline.copy(), op_sel, P)

    # Vista comparativa en tiempo real
    col_pipe, col_op, col_diff = st.columns(3)
    with col_pipe:
        st.markdown('<p class="result-header">Resultado pipeline</p>', unsafe_allow_html=True)
        st.image(img_pipeline, use_container_width=True)
    with col_op:
        lbl_op = op_sel if op_sel != "— ninguna —" else "Sin operación"
        st.markdown(f'<p class="result-header">+ {lbl_op}</p>', unsafe_allow_html=True)
        st.image(img_final, use_container_width=True)
    with col_diff:
        st.markdown('<p class="result-header">Diferencia (pipeline → final)</p>', unsafe_allow_html=True)
        st.image(diferencia(img_pipeline, img_final), use_container_width=True)

    # ── Gráficas reactivas ────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    show_graphs = st.toggle("📊 Mostrar gráficas en tiempo real", value=False, key="show_gr")
    if show_graphs:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**Histograma: Original → Pipeline → Final**")
            st.pyplot(plot_histogramas(
                [img_orig, img_pipeline, img_final],
                ["Original", "Pipeline", lbl_op]
            ), use_container_width=True)
        with col_g2:
            if op_sel != "— ninguna —":
                st.markdown("**Curva de transformación**")
                st.pyplot(plot_curva(op_sel, P), use_container_width=True)
            else:
                st.markdown("**Diferencia: Original vs Final**")
                st.pyplot(plot_diferencia(img_orig, img_final, "Final"), use_container_width=True)

        st.markdown("**Perfil de intensidad — fila central**")
        st.pyplot(plot_perfil(img_orig, img_final, "Final"), use_container_width=True)

    # ── Métricas en tiempo real ───────────────────────────────────────────────
    show_metrics = st.toggle("📋 Mostrar métricas en tiempo real", value=False, key="show_mt")
    if show_metrics:
        m_pipe  = metricas(img_orig, img_pipeline)
        m_final = metricas(img_orig, img_final)
        df_m = pd.DataFrame([
            {"Imagen": "Pipeline",         **m_pipe},
            {"Imagen": f"Pipeline + {lbl_op}", **m_final},
        ]).set_index("Imagen")
        st.dataframe(df_m.style.background_gradient(cmap="Blues", subset=["SSIM"])
                              .background_gradient(cmap="Reds_r", subset=["MSE"]),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARACIÓN MÚLTIPLE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Hasta 4 filtros lado a lado — reactivo")
    filtros_comp = []
    for i, col in enumerate(st.columns(4)):
        with col:
            filtros_comp.append(st.selectbox(f"Filtro {chr(65+i)}", TODOS, key=f"cmp_{i}"))

    resultados = [(n, aplicar(img_orig.copy(), n, P))
                  for n in filtros_comp if n != "— ninguno —"]

    if resultados:
        n = len(resultados)+1
        fig, axes = plt.subplots(1, n, figsize=(4*n,4))
        fig.patch.set_facecolor(DARK_BG)
        if n==1: axes=[axes]
        axes[0].imshow(img_orig); axes[0].set_title("Original",color="white",fontsize=9); axes[0].axis("off")
        for ax,(lbl,img) in zip(axes[1:], resultados):
            ax.imshow(img); ax.set_title(lbl,color="white",fontsize=8); ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Histogramas")
        st.pyplot(plot_histogramas([img_orig]+[r[1] for r in resultados],
                                   ["Original"]+[r[0] for r in resultados]),
                  use_container_width=True)
    else:
        st.image(img_orig, caption="Original", width=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRÁFICAS INDIVIDUALES
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Gráficas detalladas — filtro individual")
    fa = st.selectbox("Filtro a analizar", TODOS[1:], key="anal")
    img_proc = aplicar(img_orig.copy(), fa, P)

    col_a, col_b = st.columns(2)
    with col_a: st.image(img_orig, caption="Original",  use_container_width=True)
    with col_b: st.image(img_proc, caption=fa,           use_container_width=True)
    st.markdown("---")
    st.pyplot(plot_histogramas([img_orig,img_proc],["Original",fa]), use_container_width=True)
    st.pyplot(plot_perfil(img_orig, img_proc, fa),                   use_container_width=True)
    st.pyplot(plot_diferencia(img_orig, img_proc, fa),               use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Métricas de calidad")
    todas_op = TODOS[1:] + OPS_ELEMENTALES[1:]
    sel = st.multiselect("Selecciona filtros/operaciones",
                         todas_op,
                         default=["Gaussiano","CLAHE","Black-Hat","Corrección Gamma","Ecualización HE"],
                         key="met_sel")
    if sel:
        tabla = []
        for nombre in sel:
            if nombre in TODOS[1:]:
                ip = aplicar(img_orig.copy(), nombre, P)
            else:
                ip = aplicar_op(img_orig.copy(), nombre, P)
            if ip.shape != img_orig.shape:
                ip = cv2.cvtColor(cv2.cvtColor(ip,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
            m = metricas(img_orig, ip); m["Filtro/Op"] = nombre
            tabla.append(m)

        df = pd.DataFrame(tabla).set_index("Filtro/Op")
        st.dataframe(df.style.background_gradient(cmap="Blues",subset=["SSIM"])
                             .background_gradient(cmap="Reds_r",subset=["MSE"]),
                     use_container_width=True)

        fig_b, ax = plt.subplots(figsize=(10, max(3,len(df)*0.55)))
        fig_b.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
        bars = ax.barh(df.index, df["SSIM"], color="#339af0", alpha=0.85)
        ax.set_xlim(0,1); ax.set_title("SSIM por filtro/op",color="white",fontsize=10)
        ax.tick_params(colors="#8899aa",labelsize=8); ax.spines[:].set_color(GRID_C)
        for bar in bars:
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.4f}", va="center", color="white", fontsize=7)
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)
        st.caption("MSE ↓ mejor · PSNR ↑ mejor · SSIM ↑ mejor")
    else:
        st.info("Selecciona al menos un filtro u operación arriba.")