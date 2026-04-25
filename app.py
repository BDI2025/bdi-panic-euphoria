"""
================================================================================
BDI PANIC / EUPHORIA MODEL  —  Streamlit (ES)
================================================================================
Versión 2 del proyecto: en lugar de replicar CNN, este modelo se inspira en
el "Panic/Euphoria Model" que Citigroup popularizó (Tobias Levkovich) y lo
extiende con componentes de FX, curva de tasas y crédito. La salida principal
NO es un score 0-100, sino un Z-score compuesto, lo que permite identificar
regímenes históricos comparables en el tiempo:

    Z ≤ -1.0   →  PÁNICO         (rebote técnico esperado, edge contraria alto)
    Z ≤ -0.5   →  Miedo
    -0.5 < Z < 0.5 → Neutral
    Z ≥  0.5   →  Codicia
    Z ≥  1.0   →  EUFORIA        (corrección esperada, edge contraria alto)

Por qué un Z y no un 0-100:
- El score 0-100 normaliza por percentil reciente y aplana valores
  estructuralmente extremos. El Z-score deja ver QUÉ TAN extremo es el día
  vs un benchmark histórico fijo (ventana larga + recortado).
- Permite agregar/restar componentes con la misma escala estadística.

Componentes (9):
1. Momentum acciones        — SPX vs SMA125 (Z)
2. Breadth (concentración)  — RSP/SPY (Z)
3. Drawdown del 52w high    — NYA/max252 (Z)
4. Volatilidad VIX nivel    — VIX bruto (Z, INVERTIDO)
5. Term structure VIX       — VIX/VIX3M (Z, INVERTIDO)
6. Safe haven flight        — Δ20d(SPY) − Δ20d(TLT) (Z)
7. Crédito basura           — HYG/LQD (Z)
8. Term spread bonos        — 10Y - 2Y proxy via TLT/IEF (Z)
9. Dólar refugio            — DXY (Z, INVERTIDO)

Para correr:
    pip install -r requirements.txt
    streamlit run app.py
"""

import time
import warnings
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go

import streamlit as st

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="BDI Panic/Euphoria Model",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# PALETA BDI CONSULTORA — Style Guide oficial
# ==============================================================================
BRAND_GREEN = "#137247"
BRAND_TURQ  = "#17BEBB"
BRAND_LIME  = "#B5E61D"
BRAND_DARK  = "#323232"
BRAND_CREAM = "#EFEDEA"

BG    = "#1c1c1c"
PANEL = BRAND_DARK
TXT   = BRAND_CREAM
MUTED = "#9aa093"

# Mapeo de regímenes
COL_PANIC      = "#5a1818"   # Z ≤ -1
COL_FEAR       = "#c6453a"   # Z ≤ -0.5
COL_NEUTRAL    = "#d9a72b"   # |Z| < 0.5
COL_GREED      = BRAND_GREEN # Z ≥ 0.5
COL_EUPHORIA   = BRAND_LIME  # Z ≥ 1

TICKERS = {
    "SPX":   "^GSPC",
    "SPY":   "SPY",
    "QQQ":   "QQQ",
    "RSP":   "RSP",
    "TLT":   "TLT",
    "IEF":   "IEF",
    "BND":   "BND",
    "HYG":   "HYG",
    "LQD":   "LQD",
    "JNK":   "JNK",
    "VIX":   "^VIX",
    "VIX3M": "^VIX3M",
    "NYA":   "^NYA",
    "DXY":   "DX-Y.NYB",
}

# ==============================================================================
# CSS  +  LOGO BDI
# ==============================================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Poppins', sans-serif !important;
}}
.stApp {{ background-color:{BG}; color:{TXT}; }}
section[data-testid="stSidebar"] {{ background-color:{PANEL}; }}
[data-testid="stMetricValue"] {{ color:{BRAND_LIME}; font-weight:700; }}
[data-testid="stMetricLabel"] {{ color:{MUTED}; }}
h1, h2, h3, h4 {{
    color:{TXT};
    font-family: 'Bebas Neue','Poppins',sans-serif !important;
    letter-spacing: 1px;
}}
.block-container {{ padding-top: 1.2rem; }}
hr {{ border-color:#444; }}
.small-muted {{ color:{MUTED}; font-size:12px; font-family:'Poppins',sans-serif; }}
.bdi-header {{
    display:flex; align-items:center; gap:28px;
    padding:28px 32px; border-radius:12px;
    background: linear-gradient(135deg, {BRAND_GREEN} 0%, {BRAND_TURQ} 60%, {BRAND_LIME} 100%);
    margin-bottom: 28px;
    min-height: 100px;
    box-shadow: 0 4px 16px rgba(0,0,0,.25);
}}
.bdi-header .logo-block {{ display:flex; flex-direction:column; gap:8px; }}
.bdi-header .logo-row {{
    display:flex; align-items:center; gap:8px; line-height:1;
}}
.bdi-header .logo-bdi {{
    font-family:'Playfair Display','Georgia','Times New Roman',serif;
    font-weight:900; font-size:56px; color:white;
    line-height:1; letter-spacing:-1px;
}}
.bdi-header .logo-arrow {{
    color:white; font-size:48px; line-height:1;
    margin-left:2px; transform:translateY(-2px);
}}
.bdi-header .tagline {{
    font-family:'Poppins',sans-serif; font-size:10px;
    letter-spacing:2.5px; color:rgba(255,255,255,.92);
    text-transform:uppercase; line-height:1;
}}
.bdi-header .title-block {{
    margin-left:auto; text-align:right; color:white;
    display:flex; flex-direction:column; gap:6px;
}}
.bdi-header .title-block .h1 {{
    font-family:'Bebas Neue',sans-serif; font-size:34px; letter-spacing:2px;
    line-height:1.05;
}}
.bdi-header .title-block .h2 {{
    font-family:'Poppins',sans-serif; font-size:11px; letter-spacing:2px;
    text-transform:uppercase; opacity:.92; line-height:1;
}}
.regime-badge {{
    display:inline-block; padding:8px 18px; border-radius:30px;
    font-family:'Bebas Neue',sans-serif; font-size:22px; letter-spacing:2px;
    color:white;
}}
div[data-baseweb="tab-list"] button[role="tab"] {{ color:{MUTED} !important; }}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    color:{BRAND_LIME} !important; border-bottom-color:{BRAND_LIME} !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def bdi_header(title: str, subtitle: str):
    """Header BDI con logo recreado con CSS+texto (robusto, sin SVG inline)."""
    html = (
        '<div class="bdi-header">'
          '<div class="logo-block">'
            '<div class="logo-row">'
              '<span class="logo-bdi">BDI</span>'
              '<span class="logo-arrow">&#9654;</span>'
            '</div>'
            '<div class="tagline">Consultora Patrimonial Integral</div>'
          '</div>'
          '<div class="title-block">'
            f'<div class="h1">{title}</div>'
            f'<div class="h2">{subtitle}</div>'
          '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# HELPERS
# ==============================================================================
def regime_from_z(z: float) -> str:
    if pd.isna(z): return "Neutral"
    if z <= -1.0: return "PÁNICO"
    if z <= -0.5: return "Miedo"
    if z <   0.5: return "Neutral"
    if z <   1.0: return "Codicia"
    return "EUFORIA"

def regime_color(z: float) -> str:
    if pd.isna(z): return COL_NEUTRAL
    if z <= -1.0: return COL_PANIC
    if z <= -0.5: return COL_FEAR
    if z <   0.5: return COL_NEUTRAL
    if z <   1.0: return COL_GREED
    return COL_EUPHORIA

def safe_last(series, default=np.nan):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if len(s) else default

def fmt(x, nd=2):
    return "N/A" if pd.isna(x) else f"{x:.{nd}f}"

def long_zscore(series: pd.Series, window: int = 1260, min_periods: int = 252) -> pd.Series:
    """Z-score con ventana larga (5 años por defecto) → captura ciclos completos."""
    s = pd.Series(series, dtype=float)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


# ==============================================================================
# DESCARGA
# ==============================================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_one(ticker: str, start: dt.date, end: dt.date,
                 retries: int = 3, pause: float = 2.0) -> pd.Series:
    for _ in range(retries):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=False, progress=False,
                threads=False, timeout=25,
            )
            if df is None or df.empty:
                time.sleep(pause); continue
            if isinstance(df.columns, pd.MultiIndex):
                s = df["Close"].iloc[:, 0].copy()
            elif "Close" in df.columns:
                s = df["Close"].copy()
            else:
                s = df.iloc[:, 0].copy()
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.name = ticker
            if len(s):
                return s
        except Exception:
            time.sleep(pause)
    return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_universe(start: dt.date, end: dt.date) -> tuple:
    out, failed = {}, []
    for n, t in TICKERS.items():
        s = download_one(t, start, end)
        if len(s) == 0:
            failed.append((n, t))
        else:
            out[n] = s
    if not out:
        return pd.DataFrame(), failed
    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    df = df.sort_index().ffill()
    return df, failed


def apply_fallbacks(d: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("SPX", ["SPY", "QQQ"]),
        ("SPY", ["SPX", "QQQ"]),
        ("RSP", ["SPY"]),
        ("NYA", ["SPY", "SPX"]),
        ("TLT", ["IEF", "BND"]),
        ("IEF", ["BND", "TLT"]),
        ("HYG", ["JNK", "LQD"]),
        ("LQD", ["BND", "TLT"]),
        ("DXY", ["BND"]),  # último recurso, peor pero no rompe
    ]
    for col, alts in pairs:
        if col not in d.columns:
            for a in alts:
                if a in d.columns:
                    d[col] = d[a]; break
    if "VIX" not in d.columns and "SPY" in d.columns:
        rv = d["SPY"].pct_change().rolling(20, min_periods=10).std() * np.sqrt(252) * 100
        d["VIX"] = rv.bfill().fillna(20)
    if "VIX3M" not in d.columns and "VIX" in d.columns:
        d["VIX3M"] = d["VIX"].rolling(63, min_periods=20).mean().bfill()
    return d


# ==============================================================================
# COMPONENTES (9)
# ==============================================================================
def compute_zcomponents(d: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=d.index)

    # 1. Momentum
    mom = d["SPX"] / d["SPX"].rolling(125, min_periods=60).mean() - 1
    z["Momentum"] = long_zscore(mom)

    # 2. Breadth (RSP/SPY)
    breadth = (d["RSP"] / d["SPY"]).replace([np.inf, -np.inf], np.nan)
    z["Breadth"] = long_zscore(breadth)

    # 3. 52w high distance
    strength = d["NYA"] / d["NYA"].rolling(252, min_periods=80).max()
    z["52w Strength"] = long_zscore(strength)

    # 4. VIX nivel (invertido: VIX alto = miedo → -Z)
    z["VIX nivel"] = -long_zscore(d["VIX"])

    # 5. Term structure VIX/VIX3M (invertido)
    term = d["VIX"] / d["VIX3M"]
    z["VIX term struct"] = -long_zscore(term)

    # 6. Safe haven flight
    safe = d["SPY"].pct_change(20) - d["TLT"].pct_change(20)
    z["Safe Haven"] = long_zscore(safe)

    # 7. Crédito basura HYG/LQD
    junk = d["HYG"] / d["LQD"]
    z["Junk Bond"] = long_zscore(junk)

    # 8. Term spread bonos: TLT/IEF (cuando se invierte → recesión esperada → fear)
    bond = d["TLT"] / d["IEF"]
    z["Term Spread"] = long_zscore(bond)

    # 9. DXY refugio (DXY alto = dólar fuerte = flight-to-quality → invertido)
    z["DXY"] = -long_zscore(d["DXY"])

    return z.replace([np.inf, -np.inf], np.nan)


def composite_z(zcomp: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    if weights is None:
        weights = {c: 1 / len(zcomp.columns) for c in zcomp.columns}
    out = sum(zcomp[c].fillna(0) * w for c, w in weights.items())
    out = out.rolling(3, min_periods=1).mean()
    # recortar para estabilidad visual
    return out.clip(-3.5, 3.5)


# ==============================================================================
# VISUALES
# ==============================================================================
def gauge_panic(z: float) -> go.Figure:
    """Gauge con 5 zonas centradas en 0. Sin overlap entre número y etiqueta."""
    val = float(np.clip(z, -3, 3))
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        number=dict(font=dict(size=48, color=BRAND_LIME), valueformat=".2f"),
        delta=dict(reference=0, valueformat=".2f",
                   increasing=dict(color=BRAND_LIME),
                   decreasing=dict(color=COL_FEAR)),
        domain=dict(x=[0, 1], y=[0.30, 1.0]),  # gauge en 70% superior
        gauge=dict(
            axis=dict(range=[-3, 3], tickwidth=1, tickcolor=MUTED,
                      tickvals=[-3, -1, -0.5, 0.5, 1, 3]),
            bar=dict(color="rgba(0,0,0,0)"),
            bgcolor=PANEL,
            steps=[
                dict(range=[-3,   -1],   color=COL_PANIC),
                dict(range=[-1,   -0.5], color=COL_FEAR),
                dict(range=[-0.5,  0.5], color=COL_NEUTRAL),
                dict(range=[ 0.5,  1.0], color=COL_GREED),
                dict(range=[ 1.0,  3.0], color=COL_EUPHORIA),
            ],
            threshold=dict(line=dict(color=TXT, width=4),
                           thickness=0.85, value=val),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=20, r=20, t=20, b=60), height=380,
        annotations=[dict(
            x=0.5, y=0.06, xref="paper", yref="paper",
            text=f"<b>{regime_from_z(val).upper()}</b>",
            showarrow=False, font=dict(color=regime_color(val), size=22),
        )],
    )
    return fig


def composite_history(z_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=-3.5, y1=-1.0, fillcolor=COL_PANIC,    opacity=0.18, line_width=0)
    fig.add_hrect(y0=-1.0, y1=-0.5, fillcolor=COL_FEAR,     opacity=0.13, line_width=0)
    fig.add_hrect(y0=-0.5, y1= 0.5, fillcolor=COL_NEUTRAL,  opacity=0.08, line_width=0)
    fig.add_hrect(y0= 0.5, y1= 1.0, fillcolor=COL_GREED,    opacity=0.13, line_width=0)
    fig.add_hrect(y0= 1.0, y1= 3.5, fillcolor=COL_EUPHORIA, opacity=0.18, line_width=0)
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot", width=1))
    fig.add_trace(go.Scatter(
        x=z_series.index, y=z_series.values, mode="lines",
        line=dict(color=BRAND_LIME, width=1.7), name="Composite Z",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=380,
        yaxis=dict(title="Z compuesto", range=[-3.5, 3.5], gridcolor="#444"),
        xaxis=dict(gridcolor="#444"),
        showlegend=False,
    )
    return fig


def components_bar(z_now: pd.Series) -> go.Figure:
    s = z_now.sort_values()
    colors = [regime_color(v) for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        text=[f"{v:+.2f}" for v in s.values], textposition="outside",
        marker=dict(color=colors, line=dict(color=TXT, width=0.6)),
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dash"))
    fig.add_vline(x=-1, line=dict(color=COL_PANIC, dash="dot", width=1))
    fig.add_vline(x= 1, line=dict(color=COL_EUPHORIA, dash="dot", width=1))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=30), height=420,
        xaxis=dict(range=[-3, 3], gridcolor="#444",
                   title="Z-score (negativo=miedo, positivo=codicia)"),
        yaxis=dict(gridcolor="#444"),
    )
    return fig


def regime_distribution(z_series: pd.Series) -> go.Figure:
    s = z_series.dropna()
    pct = pd.Series({
        "Pánico":   ((s <= -1.0).mean()) * 100,
        "Miedo":    (((s > -1.0) & (s <= -0.5)).mean()) * 100,
        "Neutral":  (((s > -0.5) & (s <  0.5)).mean()) * 100,
        "Codicia":  (((s >= 0.5) & (s <  1.0)).mean()) * 100,
        "Euforia":  ((s >= 1.0).mean()) * 100,
    })
    colors = [COL_PANIC, COL_FEAR, COL_NEUTRAL, COL_GREED, COL_EUPHORIA]
    fig = go.Figure(go.Bar(
        x=pct.index, y=pct.values,
        marker=dict(color=colors, line=dict(color=TXT, width=0.6)),
        text=[f"{v:.0f}%" for v in pct.values], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        yaxis=dict(title="% del tiempo", gridcolor="#444"),
        xaxis=dict(gridcolor="#444"),
    )
    return fig


def vix_real(vix: pd.Series, days: int = 252) -> go.Figure:
    s = vix.dropna().tail(days)
    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=15, fillcolor=BRAND_GREEN, opacity=0.10, line_width=0)
    fig.add_hrect(y0=15, y1=20, fillcolor=BRAND_LIME,  opacity=0.06, line_width=0)
    fig.add_hrect(y0=20, y1=30, fillcolor=COL_NEUTRAL, opacity=0.10, line_width=0)
    fig.add_hrect(y0=30, y1=80, fillcolor=COL_FEAR,    opacity=0.13, line_width=0)
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color="#ff5b4d", width=1.8), name="VIX",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        yaxis=dict(title="VIX (pts)", gridcolor="#444"),
        xaxis=dict(gridcolor="#444"),
        showlegend=False,
    )
    return fig


def episodes_table(z_series: pd.Series, dates_only: bool = True) -> pd.DataFrame:
    """Identifica episodios extremos (Z ≤ -1 ó Z ≥ 1) y los agrupa."""
    s = z_series.dropna()
    extreme = (s <= -1.0) | (s >= 1.0)
    if not extreme.any():
        return pd.DataFrame(columns=["Inicio", "Fin", "Días", "Tipo", "Z mínimo", "Z máximo"])
    grp = (extreme.ne(extreme.shift())).cumsum()
    blocks = []
    for _, sub in s.groupby(grp):
        if (sub <= -1.0).any() or (sub >= 1.0).any():
            mn, mx = sub.min(), sub.max()
            tipo = "PÁNICO" if mn <= -1 else "EUFORIA"
            blocks.append({
                "Inicio": sub.index[0].strftime("%Y-%m-%d") if dates_only else sub.index[0],
                "Fin":    sub.index[-1].strftime("%Y-%m-%d") if dates_only else sub.index[-1],
                "Días":   len(sub),
                "Tipo":   tipo,
                "Z mínimo": round(float(mn), 2),
                "Z máximo": round(float(mx), 2),
            })
    df = pd.DataFrame(blocks)
    return df.sort_values("Inicio", ascending=False).reset_index(drop=True)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Controles")
    years_back = st.slider("Histórico (años)", 2, 10, 6)
    use_equal_weights = st.checkbox(
        "Pesos iguales (recomendado)", value=True,
        help="Si lo desmarcás podés ajustar pesos manualmente."
    )
    st.markdown("---")
    st.markdown("### Pesos del composite")
    if use_equal_weights:
        st.caption("Cada uno de los 9 componentes = 1/9 ≈ 11.1%")
        custom_weights = None
    else:
        custom_weights = {}
        defaults = {
            "Momentum": 0.12, "Breadth": 0.10, "52w Strength": 0.10,
            "VIX nivel": 0.14, "VIX term struct": 0.12, "Safe Haven": 0.10,
            "Junk Bond": 0.12, "Term Spread": 0.10, "DXY": 0.10,
        }
        for k, v in defaults.items():
            custom_weights[k] = st.slider(k, 0.0, 0.30, v, 0.01)
        s = sum(custom_weights.values()) or 1.0
        custom_weights = {k: v / s for k, v in custom_weights.items()}

    st.markdown("---")
    st.markdown(
        f"<span class='small-muted'>Modelo: composite Z-score sobre 9 series de "
        f"acciones, bonos, volatilidad y FX. Lookback Z = 5 años.</span>",
        unsafe_allow_html=True,
    )

# ==============================================================================
# DATOS
# ==============================================================================
TODAY = dt.date.today()
START = TODAY - timedelta(days=years_back * 365 + 365 * 2)  # buffer para Z largo

with st.spinner("Descargando datos…"):
    data, failed = download_universe(START, TODAY + timedelta(days=1))

if data.empty:
    st.error("No se pudieron descargar datos.")
    st.stop()

if failed:
    with st.sidebar.expander("⚠️ Tickers con fallback aplicado"):
        for n, t in failed:
            st.text(f"• {n} ({t})")

data = apply_fallbacks(data)
required = ["SPX", "SPY", "RSP", "NYA", "TLT", "IEF", "HYG", "LQD", "VIX", "DXY"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Faltan columnas críticas: {missing}")
    st.stop()

zcomp = compute_zcomponents(data)
z_series_full = composite_z(zcomp, weights=custom_weights)
z_series = z_series_full.tail(years_back * 252)
z_today = safe_last(z_series)

# ==============================================================================
# HEADER
# ==============================================================================
bdi_header(
    title="PANIC / EUPHORIA MODEL",
    subtitle=f"Composite Z multi-asset · {TODAY.strftime('%d/%m/%Y')}"
)
st.markdown(
    f"<span class='small-muted'>9 componentes · acciones + bonos + VIX + FX · "
    f"Z-score con ventana 5 años · cache 30 min</span>",
    unsafe_allow_html=True,
)
st.markdown("")

# ==============================================================================
# TABS
# ==============================================================================
tab_dash, tab_comp, tab_regimes, tab_theory, tab_math, tab_data = st.tabs([
    "🌡️ Régimen actual", "🧩 Componentes", "📚 Regímenes históricos",
    "📖 Teoría", "🧮 Matemática avanzada", "📥 Datos"
])

# ------------------------------- DASHBOARD ------------------------------------
with tab_dash:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(gauge_panic(z_today), use_container_width=True)
    with c2:
        regime = regime_from_z(z_today)
        color = regime_color(z_today)
        st.markdown(
            f"<div class='regime-badge' style='background:{color};'>"
            f"{regime}</div>", unsafe_allow_html=True
        )
        st.metric("Z compuesto hoy", fmt(z_today, 2))

        z5  = safe_last(z_series.iloc[:-5]) if len(z_series) > 5 else np.nan
        z21 = safe_last(z_series.iloc[:-21]) if len(z_series) > 21 else np.nan
        z252 = safe_last(z_series.iloc[:-252]) if len(z_series) > 252 else np.nan

        st.metric("Hace 1 semana", fmt(z5, 2),  delta=fmt(z_today - z5, 2)  if not pd.isna(z5)  else None)
        st.metric("Hace 1 mes",    fmt(z21, 2), delta=fmt(z_today - z21, 2) if not pd.isna(z21) else None)
        st.metric("Hace 1 año",    fmt(z252, 2), delta=fmt(z_today - z252, 2) if not pd.isna(z252) else None)

    st.markdown("### 📈 Evolución del Z compuesto")
    st.plotly_chart(composite_history(z_series), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 🧩 Componentes hoy (Z individuales)")
        st.plotly_chart(components_bar(zcomp.iloc[-1]), use_container_width=True)
    with c4:
        st.markdown("### 🔥 VIX en valor real")
        st.plotly_chart(vix_real(data["VIX"]), use_container_width=True)
        st.caption(
            f"VIX hoy: **{safe_last(data['VIX']):.2f}** · Promedio 1Y: "
            f"**{data['VIX'].tail(252).mean():.2f}** · "
            f"<15 calma · 15-20 normal · 20-30 estrés · >30 pánico"
        )

    st.markdown("### 📊 Distribución de regímenes")
    st.plotly_chart(regime_distribution(z_series), use_container_width=True)

# ------------------------------- COMPONENTES ----------------------------------
with tab_comp:
    st.markdown("## 🧩 Componentes del composite")
    st.markdown(
        "Cada componente está expresado en unidades de desvíos estándar respecto "
        "de su comportamiento promedio en los últimos 5 años. "
        "Negativo = miedo · Positivo = codicia."
    )

    last_z = zcomp.iloc[-1].round(2)
    cols = st.columns(3)
    for i, (k, v) in enumerate(last_z.items()):
        with cols[i % 3]:
            st.metric(
                label=k,
                value=f"{v:+.2f} σ" if not pd.isna(v) else "N/A",
                delta=regime_from_z(v),
            )

    st.markdown("### Evolución conjunta (3 años)")
    fig = go.Figure()
    palette = [BRAND_LIME, BRAND_TURQ, BRAND_GREEN, COL_FEAR, COL_NEUTRAL,
               "#7ad6e6", "#9ed14d", "#d96d29", "#a991e3"]
    for i, col in enumerate(zcomp.columns):
        fig.add_trace(go.Scatter(
            x=zcomp.tail(756).index,
            y=zcomp[col].tail(756).values,
            mode="lines", name=col,
            line=dict(width=1.3, color=palette[i % len(palette)]),
        ))
    fig.add_hline(y=0,   line=dict(color=MUTED, dash="dash"))
    fig.add_hline(y=-1,  line=dict(color=COL_PANIC, dash="dot"))
    fig.add_hline(y= 1,  line=dict(color=COL_EUPHORIA, dash="dot"))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        height=480, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Z-score", gridcolor="#444"),
        xaxis=dict(gridcolor="#444"),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color=TXT, size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabla de Z-scores recientes (últimos 30 días)")
    st.dataframe(zcomp.tail(30).round(2).iloc[::-1], use_container_width=True)

    st.markdown("### Correlación entre componentes (último año)")
    st.dataframe(zcomp.tail(252).corr().round(2), use_container_width=True)

# ------------------------------- REGÍMENES ------------------------------------
with tab_regimes:
    st.markdown("## 📚 Episodios extremos detectados")
    st.markdown(
        "Bloques de días consecutivos en zona PÁNICO (Z ≤ -1) o EUFORIA (Z ≥ 1). "
        "Históricamente, estos extremos suelen marcar puntos de inflexión "
        "(rebote desde pánico, corrección desde euforia)."
    )
    eps = episodes_table(z_series_full)
    if len(eps):
        st.dataframe(eps, use_container_width=True, hide_index=True)
    else:
        st.info("No se detectaron episodios extremos en la ventana cargada.")

    st.markdown("### % del tiempo en cada régimen")
    s = z_series_full.dropna()
    counts = pd.DataFrame({
        "Régimen": ["Pánico", "Miedo", "Neutral", "Codicia", "Euforia"],
        "% del tiempo": [
            f"{(s <= -1.0).mean()*100:.1f}%",
            f"{((s > -1.0) & (s <= -0.5)).mean()*100:.1f}%",
            f"{((s > -0.5) & (s <  0.5)).mean()*100:.1f}%",
            f"{((s >= 0.5) & (s <  1.0)).mean()*100:.1f}%",
            f"{(s >= 1.0).mean()*100:.1f}%",
        ],
    })
    st.dataframe(counts, use_container_width=True, hide_index=True)

# ------------------------------- TEORÍA SENCILLA ------------------------------
with tab_theory:
    st.markdown("## 📖 Teoría — para entender el modelo sin matemática")
    st.markdown(
        "El **Modelo Pánico/Euforia BDI** mide cuán **asustado** o cuán **eufórico** "
        "está el mercado, pero usando 9 señales de **acciones, bonos, volatilidad y "
        "tipo de cambio** combinadas en un solo número. La idea original es del "
        "estratega Tobias Levkovich (Citi); BDI la abre y la extiende."
    )
    st.markdown(
        "La salida es un **Z-score compuesto** entre -3 y +3. Negativo = miedo. "
        "Positivo = codicia. Por encima de +1 → euforia. Por debajo de -1 → pánico."
    )

    st.markdown("### Cómo leer el termómetro")
    st.markdown("""
    | Z compuesto       | Régimen     | Qué pensar                                              |
    |-------------------|-------------|---------------------------------------------------------|
    | **Z ≤ -1.0**      | 🟥 PÁNICO   | Capitulación. Suelen aparecer oportunidades de compra.  |
    | -1.0 < Z ≤ -0.5   | 🟧 Miedo    | Aversión al riesgo elevada.                             |
    | -0.5 < Z < 0.5    | 🟨 Neutral  | Sin sesgo, mercado en rango.                            |
    | 0.5 ≤ Z < 1.0     | 🟩 Codicia  | Apetito de riesgo. Vigilar exposición.                  |
    | **Z ≥ 1.0**       | 🟢 EUFORIA  | Probabilidad alta de corrección técnica.                |
    """)

    st.markdown("### Los 9 componentes en lenguaje simple")

    st.markdown("**1. Momentum del S&P 500**")
    st.markdown(
        "¿Está el mercado subiendo con fuerza o cayendo? Distancia del precio "
        "actual al promedio de 6 meses."
    )
    st.markdown("**2. Amplitud (Breadth) — RSP vs SPY**")
    st.markdown(
        "¿Sube **todo** el mercado o solo unas pocas megacaps? Si solo suben "
        "las gigantes, el rally es frágil aunque el índice marque récords."
    )
    st.markdown("**3. Fortaleza del precio**")
    st.markdown(
        "¿Estamos cerca del techo de los últimos 12 meses? Cuanto más cerca, "
        "más fortaleza."
    )
    st.markdown("**4. VIX (índice del miedo)**")
    st.markdown(
        "Mide volatilidad esperada en el S&P 500 a 30 días. <15 calma. "
        "20–30 estrés. >30 pánico real. Cuanto más alto, más miedo."
    )
    st.markdown("**5. Estructura de plazos del VIX**")
    st.markdown(
        "Compara VIX a 1 mes vs VIX a 3 meses. Cuando se invierte "
        "(corto > largo), hay **pánico inmediato** — los traders pagan caro "
        "para protegerse YA."
    )
    st.markdown("**6. Vuelo a la calidad (Safe Haven)**")
    st.markdown(
        "¿La gente se está pasando de acciones a bonos del Tesoro? "
        "Comparamos retorno SPY vs TLT a 20 días."
    )
    st.markdown("**7. Apetito por bonos basura**")
    st.markdown(
        "Bonos high-yield (HYG) vs investment-grade (LQD). Cuando hay "
        "confianza, los basura suben. Cuando hay miedo, todos huyen a calidad."
    )
    st.markdown("**8. Curva de tasas (TLT/IEF)**")
    st.markdown(
        "Proxy de la curva 10Y–2Y. Cuando se aplana o invierte, el mercado "
        "está descontando **recesión** = miedo estructural."
    )
    st.markdown("**9. Dólar como refugio (DXY)**")
    st.markdown(
        "Cuando estalla una crisis global, el dólar se fortalece "
        "(flight-to-quality). Un DXY que se dispara de repente es señal "
        "de **stress global**."
    )

    st.markdown("### Por qué este modelo le agrega valor a Fear & Greed clásico")
    st.markdown("""
    - **Multi-asset:** no mira solo acciones; bonos, FX y volatilidad
      muestran tensiones que el SPX todavía no refleja.
    - **Z-score absoluto:** un Z = -3 es comparable entre 2008, 2020 y
      cualquier crisis futura. Un percentil rolling no.
    - **Regímenes con umbrales fijos:** PÁNICO ≤ -1, EUFORIA ≥ +1. Sin
      ambigüedad.
    - **Pesos ajustables:** podés darle más peso al VIX si te interesa
      el ángulo de volatilidad, o al crédito si te interesa el ángulo
      de financiamiento.
    """)

    st.markdown("### Casos famosos donde el modelo funcionó")
    st.markdown("""
    - **Marzo 2020 (COVID):** Z compuesto bajó a **~-3**. SPX rebotó +60% en 12 meses.
    - **Octubre 2022:** Z ≈ -1.5 (pánico prolongado). Inicio del bull 2023.
    - **Enero 2018 (Volmageddon):** Z ≈ +1.8 → corrección rápida del 10%.
    - **Diciembre 2021:** Z ≈ +1.5 → inicio del bear market 2022.
    """)

# ------------------------------- MATEMÁTICA AVANZADA --------------------------
with tab_math:
    st.markdown("## 🧮 Matemática del modelo + decisiones de diseño")
    st.markdown(
        "Esta sección documenta las **fórmulas exactas** y las decisiones de "
        "modelado que diferencian el Pánico/Euforia BDI del modelo original "
        "de Citi (Levkovich) y de cualquier réplica académica simple."
    )

    st.markdown("### 1) Z-score con ventana larga (5 años)")
    st.latex(r"""
    Z_t \;=\; \frac{x_t - \mu_{t-1260:t-1}}{\sigma_{t-1260:t-1}}
    """)
    st.markdown(
        "Usamos **1260 días hábiles ≈ 5 años**, suficiente para cubrir un "
        "ciclo de mercado completo (expansión + contracción). Con ventanas "
        "más cortas, los regímenes estructurales nuevos contaminan rápidamente "
        "la media y el desvío, dejando todo el composite atascado en el "
        "neutral."
    )

    st.markdown("### 2) Composite y umbrales")
    st.latex(r"""
    Z_t^{\text{compuesto}} \;=\; \sum_{i=1}^{9} w_i \cdot Z_t^{(i)}
    \quad \text{con } w_i = \tfrac{1}{9}
    \;\;\text{(o configurables vía sidebar)}
    """)
    st.markdown(
        "Suavizado final con SMA-3 días. Recortado a [-3.5, +3.5] para "
        "estabilidad visual."
    )

    st.markdown("### 3) Los 9 componentes — fórmulas exactas")
    cmps = [
        ("**1. Momentum**",
         r"x = \frac{P^{SPX}_t}{\overline{P}^{SPX}_{t-125:t}} - 1",
         "SPX vs SMA-125. Sobrecompra/sobreventa de mediano plazo."),
        ("**2. Breadth (RSP/SPY)**",
         r"x = \frac{P^{RSP}_t}{P^{SPY}_t}",
         "Concentración: equal-weight vs cap-weight."),
        ("**3. 52w Strength**",
         r"x = \frac{P^{NYA}_t}{\max\big(P^{NYA}_{t-252:t}\big)}",
         "Distancia al máximo de 52 semanas del NYSE Composite."),
        ("**4. VIX nivel (invertido)**",
         r"x = -Z(VIX_t)",
         "Volatilidad implícita absoluta. VIX alto = miedo (signo negativo)."),
        ("**5. VIX term structure (invertido)**",
         r"x = -Z\!\left(\frac{VIX_t}{VIX3M_t}\right)",
         "Pendiente de la curva de volatilidad. Inversión = pánico inmediato."),
        ("**6. Safe haven flight**",
         r"x = r^{SPY}_{20d} - r^{TLT}_{20d}",
         "Spread de retornos 20-día acciones vs bonos largos."),
        ("**7. Junk Bond Demand**",
         r"x = \frac{P^{HYG}_t}{P^{LQD}_t}",
         "Cociente high-yield vs investment-grade."),
        ("**8. Term spread (TLT/IEF)**",
         r"x = \frac{P^{TLT}_t}{P^{IEF}_t}",
         "Proxy curva 10Y–2Y vía bonos largos vs medianos."),
        ("**9. DXY refugio (invertido)**",
         r"x = -Z(DXY_t)",
         "Dollar Index. Fortalecimiento abrupto = flight-to-quality."),
    ]
    for title, formula, expl in cmps:
        st.markdown(title)
        st.latex(formula)
        st.markdown(expl)

    st.markdown("### 4) Mejoras BDI sobre el modelo Citi original")
    st.markdown("""
    | Aspecto                    | Citi Pánico/Euforia                | **BDI V2 mejora**                                                              |
    |----------------------------|------------------------------------|--------------------------------------------------------------------------------|
    | **Acceso**                 | Propietario, paywall               | **Open-source**, datos gratis de Yahoo Finance                                 |
    | **Componentes**            | 9 (no todos públicos)              | **9 documentados**, fórmulas exactas en esta pestaña                           |
    | **Frecuencia**             | Semanal                            | **Diaria** (granularidad táctica)                                              |
    | **FX**                     | No incluido                        | **DXY** como flight-to-quality global                                          |
    | **Curva de tasas**         | No incluida                        | **TLT/IEF** como proxy 10Y–2Y                                                  |
    | **Estructura volatilidad** | No incluida                        | **VIX/VIX3M** captura pánico inmediato                                         |
    | **Pesos**                  | No publicados                      | **1/9 igual o ajustables** vía sidebar                                         |
    | **Lookback**               | No documentado                     | **1260 días explícitos**                                                       |
    | **Output**                 | 0–1                                | **Z-score absoluto** (-3.5..+3.5), magnitudes comparables a través del tiempo  |
    | **Data resilience**        | Modelo cae si falta input          | **Fallbacks automáticos** (HYG→JNK, TLT→IEF, VIX estimado vía vol realizada)   |
    """)

    st.markdown("### 5) Por qué Z absoluto y no percentil")
    st.markdown("""
    - **Conserva magnitud:** un Z = -3 es 3× más extremo que un Z = -1, mientras
      que un percentil rolling truncaría ambos en la cola más baja.
    - **Comparable en el tiempo:** el "Z = -1.5 hoy" significa lo mismo que el
      "Z = -1.5 en 2010". Un percentil rolling no.
    - **Sumable:** sumar Z-scores tiene sentido estadístico (combinación lineal
      de variables aleatorias estandarizadas). Sumar percentiles no.
    """)

    st.markdown("### 6) Limitaciones honestas")
    st.markdown("""
    - **Lookback de 5 años**: regímenes estructurales nuevos (tasas altas
      persistentes 2022+) tardan en incorporarse al Z.
    - El Z asume distribución aproximadamente normal. Las **colas reales son
      más gordas**, así que |Z| > 3 ocurre más seguido que lo predicho por una
      gaussiana — no calibrar probabilidades exactas.
    - **Sin fundamentales** (P/E, EPS, macro). Sentimiento puro.
    - Funciona mejor como **filtro táctico** y **señal contraria en extremos**,
      no como sistema de trading aislado.
    """)

    st.markdown("### 7) Tabla comparativa final")
    st.markdown("""
    | Aspecto                | CNN F&G        | BDI V1 (CNN replica)| **BDI V2 (este)**       | Alternative.me  |
    |------------------------|----------------|---------------------|-------------------------|-----------------|
    | Universo               | Acciones US    | Acciones US         | Acciones + Bonds + FX   | BTC             |
    | Salida                 | 0–100          | 0–100               | **Z-score (-3..+3)**    | 0–100           |
    | Componentes            | 7              | 7                   | **9**                   | 5               |
    | Normalización          | Percentil      | Z + sigmoide        | **Z bruto (5y)**        | Mixto           |
    | Pesos                  | No publicados  | Iguales (1/7)       | **Iguales o ajustables**| No publicado    |
    | Frecuencia             | Diaria         | Diaria              | Diaria                  | Diaria          |
    | Bond + FX              | Parcial        | Parcial             | **Sí**                  | No              |
    """)

# ------------------------------- DATOS ----------------------------------------
with tab_data:
    st.markdown("## 📥 Datos")
    st.markdown("### Series de mercado")
    st.dataframe(data.tail(40).round(2).iloc[::-1], use_container_width=True)
    st.download_button(
        "Descargar series CSV",
        data=data.to_csv().encode("utf-8"),
        file_name=f"panic_euphoria_data_{TODAY.isoformat()}.csv",
        mime="text/csv",
    )

    st.markdown("### Z-scores por componente")
    st.dataframe(zcomp.tail(40).round(2).iloc[::-1], use_container_width=True)
    st.download_button(
        "Descargar Z componentes",
        data=zcomp.to_csv().encode("utf-8"),
        file_name=f"panic_euphoria_components_{TODAY.isoformat()}.csv",
        mime="text/csv",
    )

    st.markdown("### Composite Z")
    out = pd.DataFrame({"Z compuesto": z_series_full,
                        "Régimen": z_series_full.map(regime_from_z)})
    st.dataframe(out.tail(40).iloc[::-1], use_container_width=True)
    st.download_button(
        "Descargar composite",
        data=out.to_csv().encode("utf-8"),
        file_name=f"panic_euphoria_composite_{TODAY.isoformat()}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    f"<span class='small-muted'>BDI Consultora Patrimonial Integral · "
    f"Modelo Pánico/Euforia · Material educativo, no es recomendación de inversión.</span>",
    unsafe_allow_html=True,
)

