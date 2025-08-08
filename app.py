
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(page_title="Return Distribution & EGARCH", layout="wide")

st.title("📈 Return Distribution & EGARCH Dashboard")

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
    bins = st.slider("Histogram bins", 10, 200, 60)
    st.divider()
    st.caption("Tip: Use the date range filter below the charts to zoom in.")

def compute_returns(df, date_col, price_col, usd_col=None, apply_usd=False, rtype="log"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    price = df[price_col].astype(float)
    if apply_usd and (usd_col is not None) and (usd_col in df.columns):
        price = price * df[usd_col].astype(float)
    if rtype == "log":
        ret = np.log(price).diff().dropna()
    else:
        ret = price.pct_change().dropna()
    out = pd.DataFrame({date_col: df.loc[ret.index, date_col].values, "return": ret.values})
    return out

def fit_egarch(returns, dist="t", p=1, o=1, q=1):
    dist_map = {"t":"t", "normal":"normal"}
    am = arch_model(returns, mean="Constant", vol="EGARCH", p=int(p), o=int(o), q=int(q), dist=dist_map.get(dist, "t"))
    res = am.fit(disp="off")
    return res

def download_button_bytes(data_bytes, label, file_name, mime="application/octet-stream"):
    st.download_button(label, data=data_bytes, file_name=file_name, mime=mime)

if uploaded is None:
    st.info("Upload a CSV in the sidebar to get started. You can test with your file example.", icon="👈")
    st.stop()

# Read and validate
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

for col in [date_col, price_col]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in CSV.")
        st.stop()

# Compute returns
rets = compute_returns(df, date_col, price_col, usd_col if usd_col in df.columns else None, apply_usd, return_type)

# Layout
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
    with st.spinner("Fitting EGARCH..."):
        try:
            res = fit_egarch(rets["return"]*100, dist=dist, p=p, o=o, q=q)  # scale to percent
        except Exception as e:
            st.error(f"EGARCH fitting failed: {e}")
            st.stop()

    st.write("**Model Summary**")
    st.text(res.summary().as_text())

    cond_vol = res.conditional_volatility  # in percent terms due to scaling
    vol_df = pd.DataFrame({
        date_col: rets[date_col].iloc[-len(cond_vol):].values,
        "cond_vol_%": cond_vol.values
    })

    fig_vol = px.line(vol_df, x=date_col, y="cond_vol_%", title="Conditional Volatility (σₜ) %")
    fig_vol.update_layout(height=500)
    st.plotly_chart(fig_vol, use_container_width=True)

    # Forecast
    f = res.forecast(horizon=horizon, reindex=True)
    # Extract variance forecasts of the last point
    try:
        var_fc = f.variance.iloc[-1].values  # variance in percent^2
    except Exception:
        var_fc = f.variance.values[-1] if hasattr(f.variance, "values") else np.array([])
    vol_fc = np.sqrt(var_fc)  # percent
    fc_df = pd.DataFrame({"h": np.arange(1, len(vol_fc)+1), "vol_forecast_%": vol_fc})
    fig_fc = px.line(fc_df, x="h", y="vol_forecast_%", markers=True, title="Forecasted Volatility (next days, %)")
    fig_fc.update_layout(height=400)
    st.plotly_chart(fig_fc, use_container_width=True)

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
