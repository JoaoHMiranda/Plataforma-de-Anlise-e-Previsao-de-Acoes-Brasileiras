# app.py
# -*- coding: utf-8 -*-
import os, sys, json, time, glob, io, zipfile, contextlib, logging, random, uuid, importlib.util
from datetime import datetime, timedelta

# ======== Ambiente/Logs (silencioso) ========
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # 0=ALL,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # logs + determinismo melhor

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ======== Preferir GPU (se existir); senão CPU silencioso ========
def _configure_tf_runtime():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    pass
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
    except Exception:
        pass

_configure_tf_runtime()

import yfinance as yf
try:
    from yfinance.utils import YFRateLimitError  # noqa
except Exception:  # pragma: no cover
    class YFRateLimitError(Exception): pass  # noqa

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Plataforma de Ações", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]::before{
  content: "⚠️ Uso exclusivamente educacional. Resultados são estimativas.\\A Não constituem recomendação de investimento.";
  position: fixed; top: 12px; left: 12px; z-index: 999999;
  padding: 10px 12px; border-radius: 10px;
  border: 1px solid rgba(229,231,235,0.6);
  background: rgba(31,41,55,0.90); color: #e5e7eb;
  font-size: 13px; line-height: 1.25; box-shadow: 0 6px 20px rgba(0,0,0,0.35);
  pointer-events: none; white-space: pre-line; max-width: min(42vw, 520px);
}
@media (prefers-color-scheme: light){
  [data-testid="stAppViewContainer"]::before{
    background: rgba(255,255,255,0.95); color:#111827;
    border-color:#e5e7eb; box-shadow: 0 4px 14px rgba(0,0,0,0.10);
  }
}
</style>
""", unsafe_allow_html=True)
st.title("🤖 Plataforma de Análise e Previsão de Ações (tudo em um único app.py)")

# -----------------------------------------------------------------------------
# CONSTS / DIRS
# -----------------------------------------------------------------------------
MODELS_DIR = "modelos"      # <- todos os modelos aqui
DATASETS_DIR = "datasets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

TIME_UNIT = {"Dia(s)": "d", "Mês(es)": "mo", "Ano(s)": "y"}

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _missing_modules(mods: list[str]) -> list[str]:
    return [m for m in mods if importlib.util.find_spec(m) is None]

def _required_for(model_kind: str, optim_choice: str) -> list[str]:
    req = {
        ("SVR","Bayes"): ["skopt"],
        ("SVR","Genético"): ["geneticalgorithm"],
        ("SVR","PSO"): ["pyswarms"],
        ("LSTM","Bayes"): ["skopt"],
        ("LSTM","Genético"): [],  # GA simples interno
        ("LSTM","PSO"): [],       # PSO simplificado
    }
    return req.get((model_kind, optim_choice), [])

def show_common_causes():
    st.error("Não foi possível concluir a análise agora.")
    st.markdown(
        "- Ticker inválido ou não suportado (ex.: erro de digitação)\n"
        "- Conexão com a internet instável\n"
        "- Limite de requisições da fonte atingido (aguarde alguns minutos)"
    )

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy(); df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_cols(df)
    for c in ["Open","High","Low","Close","Volume","MM20","MM50","RSI14","RET","Vol_21d"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fmt_date(dt):
    if dt is None: return "—"
    try:
        if pd.isna(dt): return "—"
        return pd.to_datetime(dt).strftime("%d/%m/%Y")
    except Exception:
        return str(dt)

def fmt_num(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{x*100:,.2f}%" if pct else f"{x:,.2f}"

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# EXPLICAÇÕES
# -----------------------------------------------------------------------------
MODEL_EXPL = {
    "SVR": "### 📈 SVR (Support Vector Regression)\n- Tubo de tolerância (epsilon), penalidade C e kernel RBF (gamma).\n- Parâmetros: C, gamma, epsilon, look_back.\n",
    "LSTM": "### 🧠 LSTM\n- RNN com memória para padrões temporais.\n- Parâmetros: look_back, units, dropout, recurrent_dropout, batch_size, epochs.\n",
}
OPT_EXPL = {
    "Sem otimizador (manual)": "### 🎛️ Manual\nVocê define os hiperparâmetros.",
    "Bayes": "### 🧭 Bayes\nModelo substituto do erro para buscar melhores pontos (skopt).",
    "Grid": "### 🧩 Grid\nTesta todas as combinações de uma grade.",
    "Genético": "### 🧬 GA\nPopulação, crossover/mutação e elitismo.",
    "PSO": "### 🐦 PSO\nEnxame de partículas (pyswarms) ou heurística similar.",
}
def show_explanations(model_kind: str, optim_choice: str):
    st.info(MODEL_EXPL.get(model_kind, ""))
    st.markdown("---")
    st.info(OPT_EXPL.get(optim_choice, ""))
    st.caption("SVR: C, gamma, epsilon, look_back · LSTM: look_back, units, dropout, recurrent_dropout, batch_size, epochs.")

# -----------------------------------------------------------------------------
# DADOS (Yahoo / BRAPI / Stooq)
# -----------------------------------------------------------------------------
def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    wanted = ["Open","High","Low","Close","Volume"]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for src, dst in [("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume")]:
        if dst not in df.columns and src in lower:
            rename[lower[src]] = dst
    df = df.rename(columns=rename)
    if not set(wanted).issubset(df.columns): return pd.DataFrame()
    df = df[wanted].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index, utc=False)
        except Exception: pass
    df.index.name = "Date"
    return df

def _parse_period_to_days(period: str) -> int:
    p = period.strip().lower()
    if p.endswith("d"):  return int(p[:-1])
    if p.endswith("mo"): return int(float(p[:-2]) * 30.44)
    if p.endswith("y"):  return int(float(p[:-1]) * 365.25)
    try: return int(p)
    except: return 365

def _filter_df_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty: return df
    days = _parse_period_to_days(period)
    end = df.index.max()
    start = end - timedelta(days=days)
    return df.loc[df.index >= start].copy()

@st.cache_data(ttl=900, show_spinner=False)
def _cached_yahoo(ticker: str, period: str):
    f_out, f_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False, group_by="column", threads=False)
    return df

def fetch_yahoo(ticker: str, period: str, cached: bool = True) -> pd.DataFrame | None:
    try:
        df = _cached_yahoo(ticker, period) if cached else yf.download(
            ticker, period=period, progress=False, auto_adjust=False, group_by="column", threads=False
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _flatten_cols(df)
            if set(["Open","High","Low","Close","Volume"]).issubset(df.columns):
                df = df[["Open","High","Low","Close","Volume"]].copy()
            df.index.name = "Date"
            return df
        return None
    except Exception:
        return None

def fetch_brapi(ticker: str, period: str) -> pd.DataFrame | None:
    t1 = ticker.replace(".SA","")
    url = f"https://brapi.dev/api/quote/{t1}?range={period}&interval=1d&fundamental=false"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            url2 = f"https://brapi.dev/api/quote/{ticker}?range={period}&interval=1d&fundamental=false"
            r = requests.get(url2, timeout=12)
            if r.status_code != 200: return None
        js = r.json(); results = js.get("results") or []
        if not results: return None
        hist = results[0].get("historicalDataPrice") or []
        if not hist: return None
        df = pd.DataFrame(hist)
        if "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"], unit="s"); df = df.set_index("Date")
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        df.index.name = "Date"; df = df.sort_index()
        return _filter_df_by_period(df, period)
    except Exception:
        return None

def fetch_stooq(ticker: str, period: str) -> pd.DataFrame | None:
    sym = ticker.lower()
    if not sym.endswith(".sa"): sym = sym.replace(".sa","") + ".sa"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200 or not r.text or "html" in r.text.lower(): return None
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns: return None
        df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date")
        df = _standardize_ohlcv(df); df = df.sort_index()
        return _filter_df_by_period(df, period)
    except Exception:
        return None

def fetch_with_fallback(ticker: str, period: str, order=None, insist=False, status=None) -> pd.DataFrame | None:
    order = order or ["Yahoo","BRAPI","Stooq"]
    fetchers = {"Yahoo": fetch_yahoo, "BRAPI": fetch_brapi, "Stooq": fetch_stooq}

    def try_cycle():
        for src in order:
            fn = fetchers.get(src)
            if not fn: continue
            if status is not None: status.write(f"🔎 Tentando fonte **{src}**…")
            df_try = fn(ticker, period)
            if isinstance(df_try, pd.DataFrame) and not df_try.empty:
                if status is not None: status.success(f"✅ Dados obtidos em **{src}**.")
                return df_try
            if status is not None: status.warning(f"Sem dados em **{src}**. Próxima fonte…")
        return None

    if not insist: return try_cycle()

    attempt, backoff, max_backoff = 0, 2.0, 60.0
    while True:
        attempt += 1
        if status is not None: status.write(f"♻️ Ciclo de tentativa #{attempt}")
        df_res = try_cycle()
        if isinstance(df_res, pd.DataFrame) and not df_res.empty: return df_res
        sleep_s = min(max_backoff, backoff) + random.uniform(0.0, 1.5)
        if status is not None: status.write(f"⏳ Aguardando {sleep_s:.1f}s…")
        time.sleep(sleep_s); backoff *= 1.7

# -----------------------------------------------------------------------------
# INDICADORES
# -----------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RET"] = out["Close"].pct_change()
    out["MM20"] = out["Close"].rolling(20).mean()
    out["MM50"] = out["Close"].rolling(50).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    out["RSI14"] = 100 - (100 / (1 + rs))
    out["Vol_21d"] = out["RET"].rolling(21).std() * np.sqrt(252)
    return out

def candlestick_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Preço")])
    if "MM20" in df: fig.add_trace(go.Scatter(x=df.index, y=df["MM20"], name="MM20"))
    if "MM50" in df: fig.add_trace(go.Scatter(x=df.index, y=df["MM50"], name="MM50"))
    fig.update_layout(title=f"{ticker} • Candlestick", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

def kpis(df: pd.DataFrame):
    def _as_series(df, name):
        s = df[name]
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
        return pd.to_numeric(s.squeeze(), errors="coerce")
    close = _as_series(df, "Close"); high  = _as_series(df, "High")
    ret = close.pct_change()
    last_close = close.dropna().iloc[-1] if close.notna().any() else np.nan
    ret21_val  = close.pct_change(21).dropna().iloc[-1] if close.notna().sum() > 21 else np.nan
    vol21_val  = (ret.rolling(21).std().dropna().iloc[-1] * np.sqrt(252)) if ret.notna().sum() > 21 else np.nan
    high52_val = high.rolling(252).max().dropna().iloc[-1] if high.notna().sum() > 252 else np.nan
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Fechamento", fmt_num(last_close))
    with c2: st.metric("Retorno 21d", fmt_num(ret21_val, pct=True))
    with c3: st.metric("Volatilidade (anual)", fmt_num(vol21_val, pct=True))
    with c4: st.metric("Máx. 52s", fmt_num(high52_val))

# -----------------------------------------------------------------------------
# DATASETS & MODELOS
# -----------------------------------------------------------------------------
def create_seq_dataset_from_close(series: pd.Series, look_back: int, test_size: float = 0.2,
                                  allow_shrink: bool = True):
    ser = pd.to_numeric(series.astype(float), errors="coerce").dropna()
    if len(ser) < 5:
        last = float(ser.iloc[-1]) if len(ser) else 10.0
        synth = np.clip(last + np.cumsum(np.random.normal(0, last*0.001, size=60)), 1e-6, None)
        ser = pd.Series(synth)

    eff_lb = int(look_back)
    if allow_shrink and len(ser) <= eff_lb + 2:
        eff_lb = max(3, min(eff_lb, len(ser) - 3))

    arr = ser.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)

    X, y = [], []
    for i in range(len(scaled) - eff_lb - 1):
        X.append(scaled[i:i+eff_lb, 0]); y.append(scaled[i+eff_lb, 0])
    X = np.array(X); y = np.array(y)

    if len(X) < 3:
        extra = np.clip(arr[-1,0] + np.cumsum(np.random.normal(0, arr[-1,0]*0.001, size=80)), 1e-6, None).reshape(-1,1)
        scaled2 = scaler.fit_transform(np.vstack([arr, extra]))
        X, y = [], []
        for i in range(len(scaled2) - eff_lb - 1):
            X.append(scaled2[i:i+eff_lb, 0]); y.append(scaled2[i+eff_lb, 0])
        X = np.array(X); y = np.array(y)

    samples = len(X)
    split = max(1, int(samples * (1 - test_size)))
    if split >= samples: split = samples - 1
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    return scaler, X_tr, X_te, y_tr, y_te, eff_lb

def save_meta(basepath: str, meta: dict):
    with open(basepath + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_meta(path_meta: str) -> dict:
    try:
        with open(path_meta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def list_saved_models() -> pd.DataFrame:
    rows = []
    for meta_path in glob.glob(os.path.join(MODELS_DIR, "*.meta.json")):
        meta = load_meta(meta_path)
        base = meta_path.replace(".meta.json", "")
        file_exists = any(os.path.exists(base + ext) for ext in (".joblib", ".keras", ".svr.pkl", ".pkl"))
        if not file_exists: continue
        meta["id"] = os.path.basename(base)
        meta["created_at"] = meta.get("created_at")
        rows.append(meta)
    if not rows: return pd.DataFrame(columns=["id","display_name","model_type","ticker","period","created_at"])
    return pd.DataFrame(rows).sort_values("created_at", ascending=False)

def save_lstm_model(model: tf.keras.Model, scaler: MinMaxScaler, params: dict, display_name: str):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(MODELS_DIR, f"{display_name}_{stamp}")
    model.save(base + ".keras")
    joblib.dump({"scaler": scaler, **params}, base + ".pkl")
    save_meta(base, {"display_name": display_name, "model_type": "LSTM", "created_at": stamp, **params})

def save_svr_model(model: SVR, scaler: MinMaxScaler, params: dict, display_name: str):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(MODELS_DIR, f"{display_name}_{stamp}")
    joblib.dump({"model": model, "scaler": scaler, **params}, base + ".svr.pkl")
    save_meta(base, {"display_name": display_name, "model_type": "SVR", "created_at": stamp, **params})

def load_any_model(model_id: str):
    base = os.path.join(MODELS_DIR, model_id)
    meta = load_meta(base + ".meta.json")
    if meta.get("model_type") == "LSTM":
        model = tf.keras.models.load_model(base + ".keras")
        payload = joblib.load(base + ".pkl")
        return meta, {"model": model, **payload}
    elif meta.get("model_type") == "SVR":
        payload = joblib.load(base + ".svr.pkl")
        return meta, payload
    else:
        raise ValueError("Tipo de modelo desconhecido.")

def forecast_with_lstm(payload, df: pd.DataFrame, horizon: int, lookback: int) -> pd.Series:
    model: tf.keras.Model = payload["model"]
    scaler: MinMaxScaler = payload["scaler"]
    series = df["Close"].astype(float).values.reshape(-1,1)
    scaled = scaler.transform(series)
    lb = min(max(1, lookback), max(1, len(scaled)-1))
    window = scaled[-lb:].reshape(1, lb, 1)
    preds = []
    cur = window.copy()
    for _ in range(horizon):
        yhat = model.predict(cur, verbose=0)[0,0]
        preds.append(yhat)
        cur = np.append(cur[:,1:,:], [[[yhat]]], axis=1) if cur.shape[1] > 1 else np.array([[[yhat]]])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()
    idx = pd.bdate_range(df.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon)
    return pd.Series(preds, index=idx)

def forecast_with_svr(payload, df: pd.DataFrame, horizon: int) -> pd.Series:
    model: SVR = payload["model"]
    scaler: MinMaxScaler = payload["scaler"]
    lookback = int(payload.get("lookback", 15))
    series = df["Close"].astype(float).values.reshape(-1, 1)
    scaled = scaler.transform(series)
    lb = min(max(3, lookback), max(1, len(scaled)-1))
    window = scaled[-lb:, 0].copy()
    preds_scaled = []
    for _ in range(horizon):
        x = window.reshape(1, -1)
        yhat_scaled = model.predict(x)[0]
        preds_scaled.append(yhat_scaled)
        window = np.concatenate([window[1:], [yhat_scaled]]) if len(window) > 1 else np.array([yhat_scaled])
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
    idx = pd.bdate_range(df.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon)
    return pd.Series(preds, index=idx)

# -----------------------------------------------------------------------------
# ZIPs
# -----------------------------------------------------------------------------
def build_analysis_zip(df: pd.DataFrame, ticker: str, period: str) -> tuple[bytes, str]:
    df = sanitize_df(df); df_csv = df.copy()
    min_close = float(df["Close"].min()); min_dt = df["Close"].idxmin()
    mean_close = float(df["Close"].mean())
    max_close = float(df["Close"].max()); max_dt = df["Close"].idxmax()
    ret = df["Close"].pct_change()
    vol21 = ret.rolling(21).std() * np.sqrt(252)

    txt = io.StringIO()
    txt.write(f"Resumo — {ticker} ({period})\n")
    txt.write(f"Mínimo (Close): {min_close:,.4f} em {fmt_date(min_dt)}\n")
    txt.write(f"Média  (Close): {mean_close:,.4f}\n")
    txt.write(f"Máximo (Close): {max_close:,.4f} em {fmt_date(max_dt)}\n")
    if ret.dropna().any(): txt.write(f"Retorno último: {ret.dropna().iloc[-1]*100:,.2f}%\n")
    if vol21.dropna().any(): txt.write(f"Volatilidade anual (21d): {vol21.dropna().iloc[-1]*100:,.2f}%\n")
    txt_bytes = txt.getvalue().encode("utf-8")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, pd.to_numeric(df["Close"], errors="coerce"), label="Close")
    if "MM20" in df: ax.plot(df.index, pd.to_numeric(df["MM20"], errors="coerce"), label="MM20")
    if "MM50" in df: ax.plot(df.index, pd.to_numeric(df["MM50"], errors="coerce"), label="MM50")
    ax.legend(); ax.set_title(f"{ticker} — Close & MMs"); ax.grid(True)
    img_close = fig_to_png_bytes(fig)

    fig, ax = plt.subplots(figsize=(10,3))
    if "RSI14" in df: ax.plot(df.index, pd.to_numeric(df["RSI14"], errors="coerce"), label="RSI14")
    ax.axhline(70, ls="--", lw=1); ax.axhline(30, ls="--", lw=1)
    ax.set_title("RSI(14)"); ax.grid(True)
    img_rsi = fig_to_png_bytes(fig)

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{ticker}_{period}_resumo.txt", txt_bytes)
        z.writestr(f"{ticker}_{period}_dados.csv", df_csv.to_csv(index=True, index_label="Date").encode("utf-8"))
        z.writestr(f"{ticker}_{period}_close_mms.png", img_close)
        z.writestr(f"{ticker}_{period}_rsi.png", img_rsi)
    zbuf.seek(0)
    fname = f"pacote_{ticker}_{period}.zip".replace(".","-")
    return zbuf.read(), fname

def build_training_zip(artifacts: dict) -> tuple[bytes, str]:
    tkr = artifacts.get("ticker","TICKER"); per = artifacts.get("period","PER")
    zbuf = io.BytesIO()
    seen = set()
    def _dedupe(name: str) -> str:
        base = name; i = 1
        while base in seen:
            root, ext = os.path.splitext(name)
            base = f"{root}_{i}{ext}"; i += 1
        seen.add(base); return base
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(artifacts.get("model_files", [])):
            try:
                arc = os.path.join("modelos", _dedupe(os.path.basename(p)))
                z.write(p, arcname=arc)
            except Exception:
                pass
    zbuf.seek(0)
    fname = f"pacote_treino_{tkr}_{per}.zip".replace(".","-")
    return zbuf.read(), fname

# -----------------------------------------------------------------------------
# ===== SVR: robustificação para evitar 1D/arrays vazios =======================
# -----------------------------------------------------------------------------
def _ensure_nonempty_split(X_tr, X_te, y_tr, y_te):
    if X_te is None or len(X_te) == 0:
        if X_tr is not None and len(X_tr) >= 2:
            X_te = X_tr[-1:].copy()
            y_te = y_tr[-1:].copy()
            X_tr = X_tr[:-1]
            y_tr = y_tr[:-1]
        elif X_tr is not None and len(X_tr) == 1:
            X_te = X_tr.copy()
            y_te = y_tr.copy()
    if X_tr is not None and X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    if X_te is not None and X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)
    return X_tr, X_te, y_tr, y_te

def _svr_dataset_from_close(df_close: pd.Series, look_back=15, test_ratio=0.2):
    tried_lbs = [int(look_back), max(3, int(look_back)//2), 5, 3]
    last = None
    for lb in tried_lbs:
        scaler, X_tr, X_te, y_tr, y_te, eff_lb = create_seq_dataset_from_close(
            df_close, lb, test_size=test_ratio, allow_shrink=True
        )
        last = (scaler, X_tr, X_te, y_tr, y_te, eff_lb)
        total = (len(X_tr) if X_tr is not None else 0) + (len(X_te) if X_te is not None else 0)
        if total >= 2:
            break
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = last
    X_tr, X_te, y_tr, y_te = _ensure_nonempty_split(X_tr, X_te, y_tr, y_te)
    return scaler, X_tr, X_te, y_tr, y_te, eff_lb

def _safe_tscv(n_samples: int, desired: int = 5):
    from sklearn.model_selection import TimeSeriesSplit
    if n_samples is None or n_samples < 3:
        return None
    n_splits = min(desired, max(2, n_samples - 1))
    if n_splits >= n_samples:
        return None
    return TimeSeriesSplit(n_splits=n_splits)

def train_svr_manual(df, display_name, C_val, gamma_val, eps_val, look_back, period, ticker):
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = _svr_dataset_from_close(df["Close"], look_back=look_back, test_ratio=0.2)
    if X_tr is None or len(X_tr) == 0:
        raise RuntimeError("Dataset de treino vazio após robustificação.")
    model = SVR(kernel='rbf', C=float(C_val), gamma=float(gamma_val), epsilon=float(eps_val))
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te).reshape(-1, 1)
    yte2 = np.array(y_te).reshape(-1, 1)
    pred = scaler.inverse_transform(yhat).ravel()
    yte  = scaler.inverse_transform(yte2).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae  = float(mean_absolute_error(yte, pred))
    r2v  = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5}
    save_svr_model(model, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred}

def train_svr_bayes(df, display_name, look_back, period, ticker, n_iter=32):
    from skopt import BayesSearchCV
    from skopt.space import Real
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = _svr_dataset_from_close(df["Close"], look_back=look_back, test_ratio=0.2)
    if X_tr is None or len(X_tr) == 0:
        raise RuntimeError("Dataset de treino vazio após robustificação.")
    tscv = _safe_tscv(len(X_tr), desired=5)
    if tscv is None:
        candidates = [
            (10.0,   0.01, 0.05),
            (100.0,  0.01, 0.05),
            (100.0,  0.001,0.05),
            (1000.0, 0.01, 0.05),
        ]
        best, best_rmse = None, 1e18
        for C_, g_, e_ in candidates:
            m = SVR(kernel='rbf', C=C_, gamma=g_, epsilon=e_)
            m.fit(X_tr, y_tr)
            yhat = m.predict(X_te).reshape(-1,1)
            pred = scaler.inverse_transform(yhat).ravel()
            yte  = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
            rmse = float(np.sqrt(mean_squared_error(yte, pred)))
            if rmse < best_rmse:
                best_rmse, best = rmse, (m, C_, g_, e_, pred, yte)
        m, C_, g_, e_, pred, yte = best
        params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5, "optim":"Bayes(HO)"}
        save_svr_model(m, scaler, params, display_name)
        mae = float(mean_absolute_error(yte, pred)); r2v = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
        return {"rmse": best_rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred, "best_params": {"C":C_, "gamma":g_, "epsilon":e_}}
    bayes = BayesSearchCV(
        SVR(kernel='rbf'),
        search_spaces={'C': Real(1e-1, 1e3, prior='log-uniform'),
                       'gamma': Real(1e-4, 1e-1, prior='log-uniform'),
                       'epsilon': Real(1e-2, 1e-1, prior='log-uniform')},
        n_iter=int(n_iter), cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42, verbose=0
    )
    bayes.fit(X_tr, y_tr)
    best = bayes.best_estimator_
    yhat = best.predict(X_te).reshape(-1,1)
    pred = scaler.inverse_transform(yhat).ravel()
    yte  = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae  = float(mean_absolute_error(yte, pred)); r2v = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5, "optim":"Bayes"}
    save_svr_model(best, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred, "best_params": bayes.best_params_}

def train_svr_grid(df, display_name, look_back, period, ticker,
                   C_list=None, gamma_list=None, epsilon_list=None):
    from sklearn.model_selection import GridSearchCV
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = _svr_dataset_from_close(df["Close"], look_back=look_back, test_ratio=0.2)
    if X_tr is None or len(X_tr) == 0:
        raise RuntimeError("Dataset de treino vazio após robustificação.")
    if C_list is None: C_list = [10, 100, 1000]
    if gamma_list is None: gamma_list = [0.1, 0.01, 0.001]
    if epsilon_list is None: epsilon_list = [0.1, 0.05, 0.01]
    param_grid = {'C': C_list, 'gamma': gamma_list, 'epsilon': epsilon_list}
    tscv = _safe_tscv(len(X_tr), desired=5)
    if tscv is None:
        best, best_rmse = None, 1e18
        for C_ in param_grid['C']:
            for g_ in param_grid['gamma']:
                for e_ in param_grid['epsilon']:
                    m = SVR(kernel='rbf', C=C_, gamma=g_, epsilon=e_)
                    m.fit(X_tr, y_tr)
                    yhat = m.predict(X_te).reshape(-1,1)
                    pred = scaler.inverse_transform(yhat).ravel()
                    yte  = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
                    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
                    if rmse < best_rmse:
                        best_rmse, best = rmse, (m, C_, g_, e_, pred, yte)
        m, C_, g_, e_, pred, yte = best
        params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5, "optim":"Grid(HO)"}
        save_svr_model(m, scaler, params, display_name)
        mae = float(mean_absolute_error(yte, pred)); r2v = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
        return {"rmse": best_rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred, "best_params": {"C":C_, "gamma":g_, "epsilon":e_}}
    gs = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv=tscv,
                      scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    yhat = best.predict(X_te).reshape(-1,1)
    pred = scaler.inverse_transform(yhat).ravel()
    yte  = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae  = float(mean_absolute_error(yte, pred)); r2v = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5, "optim":"Grid"}
    save_svr_model(best, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred, "best_params": gs.best_params_}

def train_svr_ga(df, display_name, look_back, period, ticker,
                 population_size=10, max_num_iteration=60,
                 mutation_probability=0.35, crossover_probability=0.5,
                 elit_ratio=0.02, parents_portion=0.3):
    from geneticalgorithm import geneticalgorithm as ga
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = _svr_dataset_from_close(df["Close"], look_back=look_back, test_ratio=0.2)
    if X_tr is None or len(X_tr) == 0:
        raise RuntimeError("Dataset de treino vazio após robustificação.")
    tscv = _safe_tscv(len(X_tr), desired=5)
    def cv_rmse(model):
        if tscv is None:
            yhat = model.fit(X_tr, y_tr).predict(X_te)
            return float(np.sqrt(mean_squared_error(y_te, yhat)))
        errs = []
        for tr_idx, va_idx in tscv.split(X_tr):
            Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
            ytr, yva = y_tr[tr_idx], y_tr[va_idx]
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            errs.append(np.sqrt(mean_squared_error(yva, pred)))
        return float(np.mean(errs))
    def fitness(params):
        C, gamma, eps = params
        model = SVR(kernel='rbf', C=float(C), gamma=float(gamma), epsilon=float(eps))
        return cv_rmse(model)
    varbound = np.array([[1, 1000], [0.001, 0.1], [0.01, 0.1]])
    algo_params = {
        'max_num_iteration': int(max_num_iteration),
        'population_size': int(population_size),
        'mutation_probability': float(mutation_probability),
        'elit_ratio': float(elit_ratio),
        'crossover_probability': float(crossover_probability),
        'parents_portion': float(parents_portion),
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 10
    }
    ga_model = ga(function=fitness, dimension=3, variable_type='real',
                  variable_boundaries=varbound, algorithm_parameters=algo_params, function_timeout=1800)
    ga_model.run()
    best_C, best_gamma, best_eps = [float(x) for x in ga_model.best_variable]
    best = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_eps)
    best.fit(X_tr, y_tr)
    yhat = best.predict(X_te).reshape(-1,1)
    pred = scaler.inverse_transform(yhat).ravel()
    yte  = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae  = float(mean_absolute_error(yte, pred)); r2v = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5, "optim":"Genético",
              "C": best_C, "gamma": best_gamma, "epsilon": best_eps}
    save_svr_model(best, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred,
            "best_params": {"C": best_C, "gamma": best_gamma, "epsilon": best_eps}}

def train_svr_pso(df, display_name, look_back, period, ticker,
                  n_particles=12, iters=25, c1=0.5, c2=0.3, w=0.9,
                  bounds_low=(1.0, 0.001, 0.01), bounds_high=(1000.0, 0.1, 0.1)):
    import pyswarms as ps
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = _svr_dataset_from_close(df["Close"], look_back=look_back, test_ratio=0.2)
    if X_tr is None or len(X_tr) == 0:
        raise RuntimeError("Dataset de treino vazio após robustificação.")
    tscv = _safe_tscv(len(X_tr), desired=5)
    def fitness(swarm):
        n_particles_local = swarm.shape[0]
        errors = np.zeros(n_particles_local)
        for i in range(n_particles_local):
            C, gamma, eps = swarm[i]
            model = SVR(kernel='rbf', C=float(C), gamma=float(gamma), epsilon=float(eps))
            if tscv is None:
                yhat = model.fit(X_tr, y_tr).predict(X_te)
                errors[i] = np.sqrt(mean_squared_error(y_te, yhat))
            else:
                fold = []
                for tr_idx, va_idx in tscv.split(X_tr):
                    Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
                    ytr, yva = y_tr[tr_idx], y_tr[va_idx]
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xva)
                    fold.append(np.sqrt(mean_squared_error(yva, pred)))
                errors[i] = np.mean(fold)
        return errors
    options = {'c1': float(c1), 'c2': float(c2), 'w': float(w)}
    bounds = (np.array(bounds_low, dtype=float), np.array(bounds_high, dtype=float))
    optimizer = ps.single.GlobalBestPSO(n_particles=int(n_particles), dimensions=3, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(fitness, iters=int(iters), verbose=False)
    best_C, best_gamma, best_eps = [float(x) for x in pos]
    best = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_eps)
    best.fit(X_tr, y_tr)
    yhat = best.predict(X_te).reshape(-1,1)
    pred = scaler.inverse_transform(yhat).ravel()
    yte  = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae  = float(mean_absolute_error(yte, pred)); r2v = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5, "optim":"PSO",
              "C": best_C, "gamma": best_gamma, "epsilon": best_eps}
    save_svr_model(best, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": pred,
            "best_params": {"C": best_C, "gamma": best_gamma, "epsilon": best_eps}}

# -----------------------------------------------------------------------------
# LSTM (manual, Bayes, GA simples, PSO simples)
# -----------------------------------------------------------------------------
def _lstm_build_model(lb, units, dropout, rec_dropout):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lb,1)),
        tf.keras.layers.LSTM(int(units), recurrent_dropout=float(rec_dropout)),
        tf.keras.layers.Dropout(float(dropout)),
        tf.keras.layers.Dense(1)
    ])

def train_lstm_manual(df, display_name, look_back, units, dropout, rec_dropout, epochs, batch, period, ticker):
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = create_seq_dataset_from_close(df["Close"], look_back, test_size=0.2, allow_shrink=True)
    X_tr = X_tr.reshape(-1, eff_lb, 1); X_te = X_te.reshape(-1, eff_lb, 1)
    model = _lstm_build_model(eff_lb, units, dropout, rec_dropout)
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=int(epochs), batch_size=int(batch), verbose=0)
    y_pred_scaled = model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    yte = scaler.inverse_transform(np.array(y_te).reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae  = float(mean_absolute_error(yte, y_pred)); r2v = float(r2_score(yte, y_pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5,
              "units": int(units), "dropout": float(dropout), "recurrent_dropout": float(rec_dropout),
              "epochs": int(epochs), "batch_size": int(batch)}
    save_lstm_model(model, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v, "yte": yte, "pred": y_pred, "hist": hist.history, "eff_lb": eff_lb}

def train_lstm_bayes(df, display_name, look_back, period, ticker,
                     n_calls=15, units_range=(32,128), dropout_range=(0.0,0.5),
                     epochs_range=(5,50)):
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = create_seq_dataset_from_close(df["Close"], look_back, test_size=0.2, allow_shrink=True)
    X_tr = X_tr.reshape(-1, eff_lb, 1); X_te = X_te.reshape(-1, eff_lb, 1)
    space = [
        Integer(int(units_range[0]), int(units_range[1]), name="units"),
        Real(float(dropout_range[0]), float(dropout_range[1]), name="dropout"),
        Integer(int(epochs_range[0]), int(epochs_range[1]), name="epochs"),
        Integer(16, 128, name="batch")
    ]
    @use_named_args(space)
    def objective(units, dropout, epochs, batch):
        tf.keras.backend.clear_session()
        m = _lstm_build_model(eff_lb, units, dropout, 0.1)
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=int(epochs), batch_size=int(batch), verbose=0)
        yhat = m.predict(X_te, verbose=0).ravel()
        return float(np.sqrt(mean_squared_error(y_te, yhat)))
    res = gp_minimize(objective, space, n_calls=int(n_calls), random_state=42, verbose=False)
    best_units, best_drop, best_epochs, best_batch = res.x
    tf.keras.backend.clear_session()
    model = _lstm_build_model(eff_lb, best_units, best_drop, 0.1)
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=int(best_epochs), batch_size=int(best_batch), verbose=0)
    yhat_scaled = model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(yhat_scaled).ravel()
    yte = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae  = float(mean_absolute_error(yte, y_pred)); r2v = float(r2_score(yte, y_pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5,
              "units": int(best_units), "dropout": float(best_drop), "recurrent_dropout": 0.1,
              "epochs": int(best_epochs), "batch_size": int(best_batch), "optim": "Bayes"}
    save_lstm_model(model, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v,
            "yte": yte, "pred": y_pred,
            "hist": hist.history, "eff_lb": eff_lb,
            "best_params": {"units": int(best_units), "dropout": float(best_drop), "epochs": int(best_epochs), "batch": int(best_batch)}}

def train_lstm_ga(df, display_name, look_back, period, ticker,
                  population=10, generations=5):
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = create_seq_dataset_from_close(df["Close"], look_back, test_size=0.2, allow_shrink=True)
    X_tr = X_tr.reshape(-1, eff_lb, 1); X_te = X_te.reshape(-1, eff_lb, 1)
    def sample():
        return {"units": random.randint(32,128), "dropout": random.uniform(0.0,0.5),
                "epochs": random.randint(5,50), "batch": random.choice([16,32,64,128])}
    pop = [sample() for _ in range(int(population))]
    best_hp, best_rmse, best_model = None, 1e9, None
    for _gen in range(int(generations)):
        scored = []
        for hp in pop:
            tf.keras.backend.clear_session()
            m = _lstm_build_model(eff_lb, hp["units"], hp["dropout"], 0.1)
            m.compile(optimizer="adam", loss="mse")
            m.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=hp["epochs"], batch_size=hp["batch"], verbose=0)
            yhat = m.predict(X_te, verbose=0).ravel()
            rmse = float(np.sqrt(mean_squared_error(y_te, yhat)))
            scored.append((rmse, hp, m))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_rmse:
            best_rmse, best_hp, best_model = scored[0][0], scored[0][1], scored[0][2]
        elites = [s[1] for s in scored[:max(1, len(scored)//4)]]
        pop = elites + [sample() for _ in range(max(1, int(population) - len(elites)))]
    yhat_scaled = best_model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(yhat_scaled).ravel()
    yte = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae  = float(mean_absolute_error(yte, y_pred)); r2v = float(r2_score(yte, y_pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5,
              "units": int(best_hp["units"]), "dropout": float(best_hp["dropout"]), "recurrent_dropout": 0.1,
              "epochs": int(best_hp["epochs"]), "batch_size": int(best_hp["batch"]), "optim": "Genético"}
    save_lstm_model(best_model, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v,
            "yte": yte, "pred": y_pred,
            "hist": {}, "eff_lb": eff_lb, "best_params": params}

def train_lstm_pso(df, display_name, look_back, period, ticker,
                   trials=20):
    scaler, X_tr, X_te, y_tr, y_te, eff_lb = create_seq_dataset_from_close(df["Close"], look_back, test_size=0.2, allow_shrink=True)
    X_tr = X_tr.reshape(-1, eff_lb, 1); X_te = X_te.reshape(-1, eff_lb, 1)
    def sample():
        return {"units": random.randint(32,128), "dropout": random.uniform(0.0,0.5),
                "epochs": random.randint(5,40), "batch": random.choice([16,32,64,128])}
    best_hp, best_rmse, best_model = None, 1e9, None
    for _ in range(int(trials)):
        hp = sample()
        tf.keras.backend.clear_session()
        m = _lstm_build_model(eff_lb, hp["units"], hp["dropout"], 0.1)
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=hp["epochs"], batch_size=hp["batch"], verbose=0)
        yhat = m.predict(X_te, verbose=0).ravel()
        rmse = float(np.sqrt(mean_squared_error(y_te, yhat)))
        if rmse < best_rmse:
            best_hp, best_rmse, best_model = hp, rmse, m
    yhat_scaled = best_model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(yhat_scaled).ravel()
    yte = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae  = float(mean_absolute_error(yte, y_pred)); r2v = float(r2_score(yte, y_pred)) if len(yte) >= 2 else float("nan")
    params = {"ticker": ticker, "period": period, "lookback": int(eff_lb), "horizon": 5,
              "units": int(best_hp["units"]), "dropout": float(best_hp["dropout"]), "recurrent_dropout": 0.1,
              "epochs": int(best_hp["epochs"]), "batch_size": int(best_hp["batch"]), "optim": "PSO"}
    save_lstm_model(best_model, scaler, params, display_name)
    return {"rmse": rmse, "mae": mae, "r2": r2v,
            "yte": yte, "pred": y_pred,
            "hist": {}, "eff_lb": eff_lb, "best_params": params}

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Análise de Ações", "🔬 Treinar Modelos (SVR/LSTM)", "📈 Fazer Previsão"])

# =============================== TAB 1 ========================================
with tab1:
    st.header("Análise Interativa do Histórico da Ação")
    st.markdown("##### Selecione a Ação e o Período")
    col_ticker, col_qtd, col_periodo = st.columns([3,1,2])
    with col_ticker:
        ticker_input = st.text_input("Ticker da Ação", "PETR4.SA", label_visibility="collapsed")
    with col_qtd:
        time_quantity = st.number_input("Quantidade", min_value=1, value=5, step=1, label_visibility="collapsed")
    with col_periodo:
        time_unit_label = st.selectbox("Período", list(TIME_UNIT.keys()), index=2, label_visibility="collapsed")
    period_value = f"{time_quantity}{TIME_UNIT[time_unit_label]}"
    insist_mode = st.checkbox("Repetir requisição até obter dados (modo insistente)", value=True)

    order_options = {
        "Yahoo → BRAPI → Stooq": ["Yahoo","BRAPI","Stooq"],
        "Yahoo → Stooq → BRAPI": ["Yahoo","Stooq","BRAPI"],
        "BRAPI → Yahoo → Stooq": ["BRAPI","Yahoo","Stooq"],
        "BRAPI → Stooq → Yahoo": ["BRAPI","Stooq","Yahoo"],
        "Stooq → Yahoo → BRAPI": ["Stooq","Yahoo","BRAPI"],
        "Stooq → BRAPI → Yahoo": ["Stooq","BRAPI","Yahoo"],
    }
    order_label = st.selectbox("Ordem das fontes de dados", list(order_options.keys()), index=0)
    sources_order = order_options[order_label]

    cbtn1, cbtn2 = st.columns([1.5, 2.5])
    with cbtn1:
        if st.button("Analisar na Tela"):
            if not ticker_input:
                st.error("Por favor, digite um ticker.")
            else:
                try: st.cache_data.clear()
                except: pass
                status = st.empty()
                with st.spinner("Carregando dados..."):
                    df = fetch_with_fallback(ticker_input, period_value, order=sources_order, insist=insist_mode, status=status)
                if df is None or df.empty:
                    show_common_causes(); st.stop()
                df = compute_indicators(df)
                df = sanitize_df(df)
                st.session_state["dados_analise"] = df
                st.session_state["ticker_analisado"] = f"{time_quantity} {time_unit_label} de {ticker_input.upper()}"
                st.success("Análise atualizada.")

    with cbtn2:
        df_for_save = st.session_state.get("dados_analise")
        if df_for_save is None or df_for_save.empty:
            st.download_button(
                "Salvar (Downloads)", b"", file_name=f"pacote_{ticker_input or 'ticker'}_{period_value}.zip",
                mime="application/zip", disabled=True, help="Faça uma análise primeiro."
            )
        else:
            pkg, fname = build_analysis_zip(df_for_save, ticker_input.upper(), period_value)
            st.download_button(
                "Salvar (Downloads)", data=pkg, file_name=fname, mime="application/zip",
                help="Baixa .zip com TXT, CSV e PNGs dos gráficos."
            )

    if 'dados_analise' in st.session_state:
        data = st.session_state['dados_analise']
        st.subheader(f"Análise de {st.session_state['ticker_analisado']}")
        kpis(data)
        candlestick_chart(data, ticker_input.upper())

        min_close_val = float(data["Close"].min()); min_close_dt = data["Close"].idxmin()
        mean_close    = float(data["Close"].mean())
        max_close_val = float(data["Close"].max()); max_close_dt = data["Close"].idxmax()

        c_stats1, c_stats2, c_stats3 = st.columns(3)
        with c_stats1:
            st.metric("Mínimo (Close)", f"{min_close_val:,.2f}"); st.caption(f"Data: {fmt_date(min_close_dt)}")
        with c_stats2:
            st.metric("Média (Close)", f"{mean_close:,.2f}")
        with c_stats3:
            st.metric("Máximo (Close)", f"{max_close_val:,.2f}"); st.caption(f"Data: {fmt_date(max_close_dt)}")

        min_low_val = float(data["Low"].min()); min_low_dt = data["Low"].idxmin()
        max_high_val = float(data["High"].max()); max_high_dt = data["High"].idxmax()
        c_ext1, c_ext2 = st.columns(2)
        with c_ext1:
            st.metric("Mínimo absoluto (Low)", f"{min_low_val:,.2f}"); st.caption(f"Data: {fmt_date(min_low_dt)}")
        with c_ext2:
            st.metric("Máximo absoluto (High)", f"{max_high_val:,.2f}"); st.caption(f"Data: {fmt_date(max_high_dt)}")

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=data["Close"].dropna(), name="Close", boxpoints="outliers"))
        fig_box.update_layout(title="Boxplot do Fechamento (período selecionado)", height=350)
        st.plotly_chart(fig_box, use_container_width=True)

        colA, colB = st.columns([2,1])
        with colA:
            st.line_chart(data[["Close","MM20","MM50"]].dropna(), height=300)
        with colB:
            if "RSI14" in data.columns: st.line_chart(data[["RSI14"]].dropna(), height=300)
            else: st.info("RSI indisponível para este período (poucos dados).")

        if data["Close"].dropna().shape[0] > 2:
            ret = data["Close"].pct_change(); st.line_chart(ret, height=220)

        vol = pd.to_numeric(data["Volume"], errors="coerce")
        vol_year = vol.groupby(data.index.year).sum(min_count=1)
        st.bar_chart(vol_year, height=220)

        st.dataframe(data.tail(200))
    else:
        st.info("Configure acima e clique em **Analisar na Tela**.")

# =============================== TAB 2 ========================================
with tab2:
    st.header("Treinar modelos (SVR ou LSTM) com ou sem otimizador — direto neste arquivo")

    cM1, cM2 = st.columns([1.2, 2.2])
    with cM1:
        model_kind = st.selectbox("Modelo", ["SVR", "LSTM"], key="sel_model_kind")
    with cM2:
        optim_choice = st.selectbox(
            "Otimização",
            ["Sem otimizador (manual)"] + (["Bayes", "Grid", "Genético", "PSO"] if model_kind == "SVR" else ["Bayes", "Genético", "PSO"]),
            key="sel_optim_choice"
        )

    show_explanations(model_kind, optim_choice)
    st.markdown("---")

    with st.form("form_treino_inline"):
        c0, c1, c2 = st.columns([2, 1.2, 1.2])
        with c0: tk_train = st.text_input("Ticker para Treino", value="PETR4.SA")
        with c1: q = st.number_input("Quantidade", min_value=1, value=1, step=1)
        with c2: unit = st.selectbox("Unidade", list(TIME_UNIT.keys()), index=0)
        period_train = f"{q}{TIME_UNIT[unit]}"

        mode_tag = "Manual" if optim_choice == "Sem otimizador (manual)" else optim_choice
        default_display = f"{model_kind}_{mode_tag}_{tk_train.replace('.', '-')}"
        display_name = st.text_input("Nome do modelo (salvar)", value=default_display)

        st.markdown("##### Parâmetros")
        opt_params = {}

        # ===== Campos do modo MANUAL =====
        if optim_choice == "Sem otimizador (manual)":
            if model_kind == "SVR":
                c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
                with c5: look_back = st.number_input("look_back (janelas)", 3, 120, 15, 1)
                with c6: C_val      = st.number_input("C", 0.001, 5000.0, 100.0, 0.001, format="%.3f")
                with c7: gamma_val  = st.number_input("gamma", 0.00001, 1.0, 0.01, 0.00001, format="%.5f")
                with c8: eps_val    = st.number_input("epsilon", 0.0001, 1.0, 0.05, 0.0001, format="%.4f")
            else:  # LSTM manual
                c5, c6, c7 = st.columns([1, 1, 1])
                with c5: look_back   = st.number_input("look_back (janelas)", 10, 200, 60, 1)
                with c6:
                    lstm_units   = st.number_input("LSTM units", 8, 256, 64, 1)
                    dropout_rate = st.slider("dropout", 0.0, 0.9, 0.2, 0.05)
                with c7:
                    rec_dropout = st.slider("recurrent_dropout", 0.0, 0.9, 0.1, 0.05)
                    batch       = st.selectbox("batch_size", [16, 32, 64, 128], index=1)
                epochs = st.slider("epochs", 5, 200, 20, 5)

        # ===== Campos dos OTIMIZADORES =====
        else:
            if model_kind == "SVR":
                if optim_choice == "Bayes":
                    look_back = st.number_input("look_back (janelas)", 3, 120, 15, 1)
                    opt_params["bayes_n_iter"] = st.number_input("Iterações (Bayes)", 5, 300, 32, 1)
                elif optim_choice == "Grid":
                    c_list    = st.text_input("Grid C", "10,100,1000")
                    gamma_l   = st.text_input("Grid gamma", "0.1,0.01,0.001")
                    eps_list  = st.text_input("Grid epsilon", "0.1,0.05,0.01")
                    opt_params["grid_C"]     = [float(x) for x in c_list.replace(" ","").split(",") if x]
                    opt_params["grid_gamma"] = [float(x) for x in gamma_l.replace(" ","").split(",") if x]
                    opt_params["grid_eps"]   = [float(x) for x in eps_list.replace(" ","").split(",") if x]
                    look_back = st.number_input("look_back (janelas)", 3, 120, 15, 1)
                elif optim_choice == "Genético":
                    col = st.columns(3)
                    with col[0]:
                        opt_params["ga_pop"]   = st.number_input("População (GA)", 4, 500, 10, 1)
                        opt_params["ga_elit"]  = st.number_input("Elit ratio", 0.0, 0.5, 0.02, 0.01, format="%.2f")
                        opt_params["ga_parents"] = st.number_input("Parents portion", 0.05, 0.9, 0.3, 0.05, format="%.2f")
                    with col[1]:
                        opt_params["ga_iters"] = st.number_input("Iterações (GA)", 5, 2000, 60, 1)
                        opt_params["ga_mut"]   = st.number_input("Prob. mutação", 0.0, 1.0, 0.35, 0.01, format="%.2f")
                    with col[2]:
                        opt_params["ga_cross"] = st.number_input("Prob. crossover", 0.0, 1.0, 0.50, 0.01, format="%.2f")
                    look_back = st.number_input("look_back (janelas)", 3, 120, 15, 1)
                elif optim_choice == "PSO":
                    col = st.columns(3)
                    with col[0]:
                        opt_params["pso_particles"] = st.number_input("Partículas (PSO)", 4, 500, 12, 1)
                        opt_params["pso_c1"]        = st.number_input("c1", 0.0, 3.0, 0.5, 0.05)
                    with col[1]:
                        opt_params["pso_iters"]     = st.number_input("Iterações (PSO)", 5, 2000, 25, 1)
                        opt_params["pso_c2"]        = st.number_input("c2", 0.0, 3.0, 0.3, 0.05)
                    with col[2]:
                        opt_params["pso_w"]         = st.number_input("w", 0.0, 1.5, 0.9, 0.05)
                    st.caption("Limites de busca: C, gamma, epsilon (mínimos e máximos)")
                    b1, b2 = st.columns(2)
                    with b1:
                        low_txt  = st.text_input("Bounds mínimos (C,γ,ε)",  "1,0.001,0.01")
                    with b2:
                        high_txt = st.text_input("Bounds máximos (C,γ,ε)", "1000,0.1,0.1")
                    try:
                        low  = [float(x) for x in low_txt.replace(" ","").split(",")]
                        high = [float(x) for x in high_txt.replace(" ","").split(",")]
                    except:
                        low, high = [1.0, 0.001, 0.01], [1000.0, 0.1, 0.1]
                    opt_params["pso_bounds_low"]  = tuple(low)
                    opt_params["pso_bounds_high"] = tuple(high)
                    # >>> FIX: definir look_back no PSO (antes não existia)
                    look_back = st.number_input("look_back (janelas)", 3, 120, 15, 1)

            else:  # LSTM otimizadores
                if optim_choice == "Bayes":
                    col = st.columns(3)
                    with col[0]:
                        opt_params["l_b_units"]   = st.number_input("Units mín", 16, 256, 32, 1)
                        opt_params["l_b_epochs0"] = st.number_input("Epochs mín", 3, 100, 5, 1)
                    with col[1]:
                        opt_params["l_t_units"]   = st.number_input("Units máx", 32, 512, 128, 1)
                        opt_params["l_t_epochs0"] = st.number_input("Epochs máx", 5, 400, 50, 1)
                    with col[2]:
                        opt_params["l_calls"]     = st.number_input("n_calls (Bayes)", 5, 200, 15, 1)
                    col2 = st.columns(2)
                    with col2[0]:
                        opt_params["l_b_dropout"] = st.number_input("Dropout mín", 0.0, 0.9, 0.0, 0.05)
                    with col2[1]:
                        opt_params["l_t_dropout"] = st.number_input("Dropout máx", 0.05, 0.9, 0.5, 0.05)
                    look_back = st.number_input("look_back (janelas)", 10, 200, 60, 1)
                elif optim_choice == "Genético":
                    opt_params["lstm_ga_pop"]  = st.number_input("População (GA LSTM)", 4, 200, 10, 1)
                    opt_params["lstm_ga_gens"] = st.number_input("Gerações (GA LSTM)", 1, 200, 5, 1)
                    look_back = st.number_input("look_back (janelas)", 10, 200, 60, 1)
                elif optim_choice == "PSO":
                    opt_params["lstm_pso_trials"] = st.number_input("Tentativas (PSO LSTM)", 5, 300, 20, 1)
                    look_back = st.number_input("look_back (janelas)", 10, 200, 60, 1)

        submitted = st.form_submit_button("Treinar")

    ph_plots_holder = st.empty()
    ph_save = st.empty()

    if not submitted:
        st.info("Preencha as opções e clique em **Treinar**.")
    else:
        st.session_state["train_artifacts"] = {"ticker": tk_train, "period": period_train, "model_files": []}

        status = st.empty()
        df = fetch_with_fallback(tk_train, period_train, order=["Yahoo","BRAPI","Stooq"], insist=False, status=status)
        if df is None or df.empty:
            base = 10.0
            vals = np.clip(base + np.cumsum(np.random.normal(0, 0.1, size=120)), 1e-6, None)
            idx = pd.date_range(end=datetime.today(), periods=len(vals), freq="B")
            df = pd.DataFrame({"Open": vals, "High": vals*1.01, "Low": vals*0.99, "Close": vals, "Volume": 1_000_000}, index=idx)
            st.warning("Fontes sem dados; usando série sintética (demonstração).")

        # checar dependências
        missing = _missing_modules(_required_for(model_kind, optim_choice))
        if missing:
            st.error("Dependências ausentes para o modo selecionado:\n\n```bash\npip install " + " ".join(missing) + "\n```")
            st.stop()

        # >>> FIX: guard para look_back fora do modo Manual (evita NameError)
        if model_kind == "SVR" and optim_choice != "Sem otimizador (manual)":
            try:
                look_back
            except NameError:
                look_back = 15  # default seguro

        ph_plots_holder.empty()
        cont = ph_plots_holder.container()
        prog = st.progress(0, text="Preparando…")

        try:
            if model_kind == "SVR":
                if optim_choice == "Sem otimizador (manual)":
                    prog.progress(0.35, text="Treinando SVR (manual)…")
                    out = train_svr_manual(df, display_name, C_val, gamma_val, eps_val, look_back, period_train, tk_train)
                elif optim_choice == "Bayes":
                    prog.progress(0.25, text="Rodando SVR + Bayes…")
                    out = train_svr_bayes(df, display_name, look_back, period_train, tk_train,
                                          n_iter=int(opt_params["bayes_n_iter"]))
                elif optim_choice == "Grid":
                    prog.progress(0.25, text="Rodando SVR + Grid…")
                    out = train_svr_grid(df, display_name, look_back, period_train, tk_train,
                                         C_list=opt_params["grid_C"],
                                         gamma_list=opt_params["grid_gamma"],
                                         epsilon_list=opt_params["grid_eps"])
                elif optim_choice == "Genético":
                    prog.progress(0.25, text="Rodando SVR + GA…")
                    out = train_svr_ga(df, display_name, look_back, period_train, tk_train,
                                       population_size=int(opt_params["ga_pop"]),
                                       max_num_iteration=int(opt_params["ga_iters"]),
                                       mutation_probability=float(opt_params["ga_mut"]),
                                       crossover_probability=float(opt_params["ga_cross"]),
                                       elit_ratio=float(opt_params["ga_elit"]),
                                       parents_portion=float(opt_params["ga_parents"]))
                elif optim_choice == "PSO":
                    prog.progress(0.25, text="Rodando SVR + PSO…")
                    out = train_svr_pso(df, display_name, look_back, period_train, tk_train,
                                        n_particles=int(opt_params["pso_particles"]),
                                        iters=int(opt_params["pso_iters"]),
                                        c1=float(opt_params["pso_c1"]),
                                        c2=float(opt_params["pso_c2"]),
                                        w=float(opt_params["pso_w"]),
                                        bounds_low=tuple(opt_params["pso_bounds_low"]),
                                        bounds_high=tuple(opt_params["pso_bounds_high"]))
                else:
                    st.error("Modo não suportado."); st.stop()

                prog.progress(0.7, text="Gerando gráficos…")
                fig, ax = plt.subplots(figsize=(9,3.4))
                ax.plot(out["yte"], label="Real (teste)"); ax.plot(out["pred"], label="Previsto (teste)")
                ax.set_title("SVR — Comparativo (teste)"); ax.grid(True); ax.legend()
                cont.image(fig_to_png_bytes(fig), use_container_width=True)
                prog.progress(1.0, text="Concluído")
                st.success(f"SVR treinado • RMSE: {out['rmse']:,.4f} | MAE: {out['mae']:,.4f} | R²: {out['r2'] if not np.isnan(out['r2']) else '—'}")
                if "best_params" in out:
                    st.caption(f"Melhores hiperparâmetros: {out['best_params']}")
                new_files = []
                for p in glob.glob(os.path.join(MODELS_DIR, f"{display_name}_*.svr.pkl")):
                    new_files.append(p)
                st.session_state["train_artifacts"]["model_files"] = list(dict.fromkeys(new_files))

            else:  # LSTM
                if optim_choice == "Sem otimizador (manual)":
                    prog.progress(0.35, text="Treinando LSTM (manual)…")
                    out = train_lstm_manual(df, display_name, look_back, lstm_units, float(dropout_rate),
                                            float(rec_dropout), int(epochs), int(batch), period_train, tk_train)
                elif optim_choice == "Bayes":
                    prog.progress(0.25, text="Rodando LSTM + Bayes…")
                    out = train_lstm_bayes(df, display_name, look_back, period_train, tk_train,
                                           n_calls=int(opt_params["l_calls"]),
                                           units_range=(int(opt_params["l_b_units"]), int(opt_params["l_t_units"])),
                                           dropout_range=(float(opt_params["l_b_dropout"]), float(opt_params["l_t_dropout"])),
                                           epochs_range=(int(opt_params["l_b_epochs0"]), int(opt_params["l_t_epochs0"])))
                elif optim_choice == "Genético":
                    prog.progress(0.25, text="Rodando LSTM + GA…")
                    out = train_lstm_ga(df, display_name, look_back, period_train, tk_train,
                                        population=int(opt_params["lstm_ga_pop"]),
                                        generations=int(opt_params["lstm_ga_gens"]))
                elif optim_choice == "PSO":
                    prog.progress(0.25, text="Rodando LSTM + PSO…")
                    out = train_lstm_pso(df, display_name, look_back, period_train, tk_train,
                                         trials=int(opt_params["lstm_pso_trials"]))
                else:
                    st.error("Modo não suportado."); st.stop()

                prog.progress(0.7, text="Gerando gráficos…")
                fig1, ax1 = plt.subplots(figsize=(9,3.4))
                ax1.plot(out["yte"], label="Real (teste)"); ax1.plot(out["pred"], label="Previsto (teste)")
                ax1.set_title("LSTM — Comparativo (teste)"); ax1.grid(True); ax1.legend()
                cont.image(fig_to_png_bytes(fig1), use_container_width=True)
                if "hist" in out and out["hist"]:
                    fig2, ax2 = plt.subplots(figsize=(9,3.0))
                    ax2.plot(out["hist"].get("loss", []), label="loss")
                    if "val_loss" in out["hist"]: ax2.plot(out["hist"]["val_loss"], label="val_loss")
                    ax2.set_title("LSTM — Evolução por época"); ax2.grid(True); ax2.legend()
                    cont.image(fig_to_png_bytes(fig2), use_container_width=True)
                prog.progress(1.0, text="Concluído")
                st.success(f"LSTM treinado • RMSE: {out['rmse']:,.4f} | MAE: {out['mae']:,.4f} | R²: {out['r2'] if not np.isnan(out['r2']) else '—'}")
                new_files = []
                for ext in ("*.keras","*.pkl"):
                    new_files += glob.glob(os.path.join(MODELS_DIR, f"{display_name}_*{ext}"))
                st.session_state["train_artifacts"]["model_files"] = list(dict.fromkeys(new_files))

        except Exception as e:
            st.error(f"Falha na execução do treino/otimizador: {e}\nAplicando **fallback interno rápido (mini-grid SVR)**.")
            try:
                scaler, X_tr, X_te, y_tr, y_te, _ = _svr_dataset_from_close(df["Close"], look_back=15, test_ratio=0.2)
                grid_C = [1.0, 10.0, 100.0]; grid_gamma = [0.001, 0.01, 0.1]; grid_eps = [0.01, 0.05]
                best = None; best_rmse = float("inf")
                for C_ in grid_C:
                    for g_ in grid_gamma:
                        for e_ in grid_eps:
                            m = SVR(kernel="rbf", C=C_, gamma=g_, epsilon=e_)
                            m.fit(X_tr, y_tr)
                            yhat = m.predict(X_te).reshape(-1,1)
                            yte = scaler.inverse_transform(np.array(y_te).reshape(-1,1)).ravel()
                            pred = scaler.inverse_transform(yhat).ravel()
                            rmse = np.sqrt(mean_squared_error(yte, pred))
                            if rmse < best_rmse:
                                best_rmse = rmse; best = (m, C_, g_, e_, pred, yte)
                m, C_, g_, e_, pred, yte = best
                fig, ax = plt.subplots(figsize=(9,3.4))
                ax.plot(yte, label="Real (teste)"); ax.plot(pred, label="Previsto (teste)")
                ax.set_title(f"SVR (fallback mini-grid) — teste | C={C_}, γ={g_}, ε={e_}"); ax.grid(True); ax.legend()
                cont.image(fig_to_png_bytes(fig), use_container_width=True)
                params = {"ticker": tk_train, "period": period_train, "lookback": 15, "horizon": 5, "optim": f"fallback-{optim_choice}"}
                fb_name = f"{display_name}_Fallback"
                save_svr_model(m, scaler, params, fb_name)
                st.session_state["train_artifacts"]["model_files"] += [
                    max(glob.glob(os.path.join(MODELS_DIR, f"{fb_name}_*.svr.pkl")), key=os.path.getmtime)
                ]
            except Exception as e2:
                st.error(f"Fallback também falhou: {e2}")

    with ph_save:
        arts = st.session_state.get("train_artifacts", {})
        if arts and arts.get("model_files"):
            pkg, fname = build_training_zip(arts)
            st.download_button("Salvar pacote do treino (apenas modelos)", data=pkg, file_name=fname, mime="application/zip")

# =============================== TAB 3 ========================================
with tab3:
    st.header("Fazer Previsão com Modelo Salvo")
    df_models = list_saved_models()
    if df_models.empty:
        st.info(f"Nenhum modelo salvo em `./{MODELS_DIR}/`. Treine um na aba anterior.")
    else:
        def _fmt(row):
            created = row.get("created_at", "")
            return f"{row.get('display_name','?')} • {row.get('model_type','?')} • {row.get('ticker','?')} • {created}"
        options = { _fmt(r): r["id"] for _, r in df_models.iterrows() }
        chosen = st.selectbox("Selecione o modelo salvo", list(options.keys()))
        model_id = options.get(chosen, df_models.iloc[0]["id"])

        def _safe_get(df: pd.DataFrame, model_id: str, col: str, default):
            try:
                if col not in df.columns: return default
                rows = df.loc[df["id"] == model_id, col]
                if rows.empty: return default
                val = rows.iloc[0]
                if pd.isna(val): return default
                return int(float(val)) if isinstance(default, int) else val
            except Exception:
                return default

        default_tk = _safe_get(df_models, model_id, "ticker", "PETR4.SA")
        default_h  = _safe_get(df_models, model_id, "horizon", 5)

        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            tk_pred = st.text_input("Ticker para previsão (pode ser outro)", value=default_tk, key=f"tk_pred_{model_id}")
        with c2:
            qh = st.number_input("Horizonte (dias úteis)", min_value=1, max_value=30, value=int(default_h or 5), key=f"qh_{model_id}")
        with c3:
            period_pred = st.selectbox("Período para carregar", ["6mo","1y","2y","5y"], index=1, key=f"period_pred_{model_id}")

        if st.button("Rodar Previsão", key=f"btn_pred_{model_id}"):
            status = st.empty()
            df = fetch_with_fallback(tk_pred, period_pred, order=["Yahoo","BRAPI","Stooq"], insist=False, status=status)
            if df is None or df.empty:
                st.error("Não foi possível carregar dados para previsão."); st.stop()

            meta, payload = load_any_model(model_id)
            with st.spinner(f"Prevendo com {meta.get('model_type','?')}..."):
                if meta.get("model_type") == "LSTM":
                    lookback = int(meta.get("lookback", 60))
                    preds = forecast_with_lstm(payload, df, horizon=int(qh), lookback=lookback)
                elif meta.get("model_type") == "SVR":
                    preds = forecast_with_svr(payload, df, horizon=int(qh))
                else:
                    st.error(f"Modelo '{meta.get('model_type')}' não suportado."); st.stop()

            st.success("Previsão concluída.")
            hist = df["Close"].iloc[-120:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico"))
            fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name="Previsão"))
            fig.update_layout(title=f"{tk_pred} • Preço de Fechamento (hist + {len(preds)}d previstos)")
            st.plotly_chart(fig, use_container_width=True)

            tab = pd.DataFrame({"Previsto": preds})
            st.download_button(
                "Baixar CSV da previsão",
                data=tab.to_csv().encode("utf-8"),
                file_name=f"forecast_{tk_pred.replace('.','-')}_{int(qh)}d_{int(time.time())}.csv",
                mime="text/csv",
                key=f"dl_pred_{model_id}"
            )

# -----------------------------------------------------------------------------
# Rodapé
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.custom-footer {
  margin-top: 32px; padding: 10px 16px;
  border-top: 1px solid rgba(229,231,235,0.6);
  background: rgba(31,41,55,0.08); color: inherit;
  font-size: 13px; line-height: 1.35;
}
.custom-footer a { color: #1d4ed8; text-decoration: none; }
.custom-footer a:hover { text-decoration: underline; }
@media (prefers-color-scheme: dark){
  .custom-footer{
    background: rgba(255,255,255,0.06);
    border-top-color: rgba(229,231,235,0.25);
  }
  .custom-footer a { color: #93c5fd; }
}
</style>
<div class="custom-footer">
  Feito por <strong>João Henrique Silva de Miranda</strong> — com apoio e financiamento do
  <strong>Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq)</strong> e da
  <strong>Pontifícia Universidade Católica de Goiás (PUC Goiás)</strong>.
  &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/joao-henrique-silva-de-miranda" target="_blank" rel="noopener">LinkedIn</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/JoaoHMiranda" target="_blank" rel="noopener">GitHub</a>
</div>
""", unsafe_allow_html=True)
