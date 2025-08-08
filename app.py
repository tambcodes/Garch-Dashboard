import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(page_title="Return Distribution & EGARCH", layout="wide")
st.title("📈 Return Distribution & EGARCH Dashboard")

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Expected columns: Date, Close (others optional like Open, High, Low, Volume, USD Rate)")
    st.divider()
    date_col = st.text_input("Date column name", value="Date")
    price_col = st.text_input("Price column for returns", value="Close")
    usd_col = st.text_input("USD Rate column (optional)", value="USD Rate")
    apply_usd = st.checkbox("Convert prices using USD Rate", value=False)
    return_type = st.selectbox("Return type", ["log", "simple"], index=0)
    dist = st.selectbox("Error distribution", ["normal", "t"], index=1)
    st.caption("EGARCH(p,o,q) orders")
    p = st.number_input("p (GARCH)", min_value=1, max_value=5, value=1, step=1)
    o = st.number_input("o (Asymmetry)", min_value=0, max_value=5, value=1, step=1)
    q = st.number_input("q (ARCH)", min_value=1, max_value=5, value=1, step=1)
    horizon = st.slider("Forecast horizon (days)", 1, 60, 20)
    forecast_method = st.selectbox(
        "Forecast method",
        ["auto (EGARCH→simulation if h>1)", "simulation", "analytic (1-step only)"],
        index=0,
        help="EGARCH analytic forecasting only supports 1-step. Multi-step requires simulation."
    )
    sims = st.slider("Simulation paths (if simulation)", 200, 5000, 2000, 100)
    bins = st.slider("Histogram bins", 10, 200, 60)
    st.divider()
    st.caption("Tip: Use the date range filter below the charts to zoom in.")

# ───────────────────────── Helpers ─────────────────────────
def compute_returns(df, date_col, price_col, usd_col=None, apply_usd=False, rtype="log"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    price = df[price_col].astype(float)
    if apply_usd and (usd_col is not None) and (usd_col in df.columns):
        price = price * df[usd_col].astype(float)
    ret = (np.log(price).diff() if rtype == "log" else price.pct_change()).dropna()
    out = pd.DataFrame({date_col: df.loc[ret.index, date_col].values, "return": ret.values})
    return out

def fit_egarch(returns, dist="t", p=1, o=1, q=1):
    dist_map = {"t": "t", "normal": "normal"}
    am = arch_model(returns, mean="Constant", vol="EGARCH",
                    p=int(p), o=int(o), q=int(q), dist=dist_map.get(dist, "t"))
    res = am.fit(disp="off")
    return res

def download_button_bytes(data_bytes, label, file_name, mime="application/octet-stream"):
    st.download_button(label, data=data_bytes, file_name=file_name, mime=mime)

# ───────────────────────── Guard: need a file ─────────────────────────
if uploaded is None:
    st.info("Upload a CSV in the sidebar to get started. You can test with your file example.", icon="👈")
    st.stop()

# ───────────────────────── Read & validate ─────────────────────────
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Could not read CSV.")
    st.exception(e)
    st.stop()

for col in [date_col, price_col]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in CSV.")
        st.stop()

# ───────────────────────── Compute returns ─────────────────────────
rets = compute_returns(
    df, date_col, price_col,
    usd_col if usd_col in df.columns else None,
    apply_usd, return_type
)

# ───────────────────────── Tabs ─────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Return Distribution", "⚙️ EGARCH Model", "⬇️ Downloads"])

with tab1:
    st.subheader("Return Distribution")
    mu = rets["return"].mean()
    sigma = rets["return"].std(ddof=1)
    k = rets["return"].kurtosis()
    s = rets["return"].skew()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{mu:.6f}")
    c2.metric("Std Dev", f"{sigma:.6f}")
    c3.metric("Skew", f"{s:.3f}")
    c4.metric("Excess Kurtosis", f"{k:.3f}")

    fig_hist = px.histogram(rets, x="return", nbins=bins, marginal="box", title="Return Histogram")
    fig_hist.update_layout(bargap=0.05, height=500)
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_line = px.line(rets, x=date_col, y="return", title="Time Series of Returns")
    fig_line.update_layout(height=500)
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.subheader("EGARCH Fit & Forecast")

    try:
        with st.spinner("Fitting EGARCH..."):
            res = fit_egarch(rets["return"] * 100, dist=dist, p=p, o=o, q=q)  # scale to percent
    except Exception as e:
        st.error("EGARCH fitting failed.")
        st.exception(e)
        st.stop()

    st.write("**Model Summary**")
    st.text(res.summary().as_text())

    # Conditional volatility
    try:
        cond_vol = res.conditional_volatility  # percent
        vol_df = pd.DataFrame({
            date_col: rets[date_col].iloc[-len(cond_vol):].values,
            "cond_vol_%": cond_vol.values
        })
        fig_vol = px.line(vol_df, x=date_col, y="cond_vol_%", title="Conditional Volatility (σₜ) %")
        fig_vol.update_layout(height=500)
        st.plotly_chart(fig_vol, use_container_width=True)
    except Exception as e:
        st.warning("Could not compute conditional volatility series.")
        st.exception(e)

    # Forecast (EGARCH analytic supports only h=1; use simulation for h>1)
    try:
        method = forecast_method
        if method.startswith("auto"):
            use_sim = horizon > 1
        elif method == "simulation":
            use_sim = True
        else:  # analytic
            use_sim = False

        if use_sim:
            f = res.forecast(
                horizon=horizon,
                reindex=True,
                method="simulation",
                simulations=int(sims),
                random_state=42,
            )
            eff_h = horizon
        else:
            eff_h = horizon
            if eff_h > 1:
                st.info("Analytic forecasts only support 1-step for EGARCH. Reducing horizon to 1.")
                eff_h = 1
            f = res.forecast(horizon=eff_h, reindex=True)

        last = f.variance.iloc[-1]
        var_fc = last.to_numpy() if hasattr(last, "to_numpy") else np.asarray(last)
        vol_fc = np.sqrt(var_fc)  # percent

        fc_df = pd.DataFrame({"h": np.arange(1, len(vol_fc) + 1), "vol_forecast_%": vol_fc})
        title = f"Forecasted Volatility (next {len(vol_fc)} day(s), %)"
        fig_fc = px.line(fc_df, x="h", y="vol_forecast_%", markers=True, title=title)
        fig_fc.update_layout(height=400)
        st.plotly_chart(fig_fc, use_container_width=True)
    except Exception as e:
        st.error("Volatility forecast failed.")
        st.exception(e)

with tab3:
    st.subheader("Downloads")
    # Return data
    csv_rets = rets.to_csv(index=False).encode("utf-8")
    download_button_bytes(csv_rets, "Download Returns CSV", "returns.csv", "text/csv")

    # Volatility series and forecast (if available)
    try:
        cond_vol = res.conditional_volatility
        vol_series = pd.DataFrame({"cond_vol_%": cond_vol})
        csv_vol = vol_series.to_csv(index=False).encode("utf-8")
        download_button_bytes(csv_vol, "Download Conditional Volatility CSV", "conditional_volatility.csv", "text/csv")
    except Exception:
        pass

    # Model parameters
    try:
        params_df = res.params.to_frame(name="value")
        params_df["tval"] = res.tvalues
        params_df["pval"] = res.pvalues
        csv_params = params_df.to_csv().encode("utf-8")
        download_button_bytes(csv_params, "Download Model Parameters CSV", "egarch_params.csv", "text/csv")
    except Exception:
        pass

st.success("Ready. Explore the tabs above for distribution, model fit, and downloads.")
