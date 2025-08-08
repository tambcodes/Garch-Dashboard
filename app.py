import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Return Distribution & EGARCH", layout="wide")

# Try importing arch; if it fails, show a clear message and stop
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception as e:
    ARCH_AVAILABLE = False
    ARCH_IMPORT_ERROR = e

def env_panel():
    import platform
    try:
        import scipy, statsmodels, plotly, numpy, pandas
        st.caption(
            "Env → "
            f"py {platform.python_version()} | "
            f"numpy {numpy.__version__} | pandas {pandas.__version__} | "
            f"scipy {scipy.__version__} | statsmodels {statsmodels.__version__} | "
            f"plotly {plotly.__version__}"
        )
    except Exception:
        pass

def compute_returns(df, date_col, price_col, usd_col=None, apply_usd=False, rtype="log"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    price = df[price_col].astype(float)
    if apply_usd and (usd_col is not None) and (usd_col in df.columns):
        price = price * df[usd_col].astype(float)
    ret = (np.log(price).diff() if rtype == "log" else price.pct_change()).dropna()
    return pd.DataFrame({date_col: df.loc[ret.index, date_col].values, "return": ret.values})

def fit_egarch(returns, dist="t", p=1, o=1, q=1):
    dist_map = {"t": "t", "normal": "normal"}
    am = arch_model(returns, mean="Constant", vol="EGARCH",
                    p=int(p), o=int(o), q=int(q), dist=dist_map.get(dist, "t"))
    res = am.fit(disp="off")
    return res

def main():
    st.title("📈 Return Distribution & EGARCH Dashboard")
    env_panel()

    # Sidebar
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
        # Keep simulations modest to avoid OOM on Cloud
        sims = st.slider("Simulation paths (if simulation is used)", 200, 3000, 500, 100)
        bins = st.slider("Histogram bins", 10, 200, 60)
        st.divider()
        st.caption("Note: EGARCH multi‑step forecasts require simulation; 1‑step can be analytic.")

    # Guard: need arch
    if not ARCH_AVAILABLE:
        st.error("The `arch` package could not be imported. The app can’t run EGARCH without it.")
        st.exception(ARCH_IMPORT_ERROR)
        st.stop()

    # Guard: need a file
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to get started. 👈")
        st.stop()

    # Read CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Could not read CSV.")
        st.exception(e)
        st.stop()

    # Validate columns
    for col in [date_col, price_col]:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in CSV.")
            st.stop()

    # Compute returns
    rets = compute_returns(
        df, date_col, price_col,
        usd_col if usd_col in df.columns else None,
        apply_usd, return_type
    )

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

        # Fit EGARCH on percent returns
        try:
            with st.spinner("Fitting EGARCH..."):
                res = fit_egarch(rets["return"] * 100, dist=dist, p=p, o=o, q=q)
        except Exception as e:
            st.error("EGARCH fitting failed.")
            st.exception(e)
            st.stop()

        st.write("**Model Summary**")
        try:
            st.text(res.summary().as_text())
        except Exception as e:
            st.warning("Could not render summary.")
            st.exception(e)

        # Conditional volatility plot
        try:
            cond_vol = res.conditional_volatility  # percent
            n = len(cond_vol)
            dates = rets[date_col].iloc[-n:].values
            vol_df = pd.DataFrame({date_col: dates, "cond_vol_%": np.asarray(cond_vol)})
            fig_vol = px.line(vol_df, x=date_col, y="cond_vol_%", title="Conditional Volatility (σₜ) %")
            fig_vol.update_layout(height=500)
            st.plotly_chart(fig_vol, use_container_width=True)
        except Exception as e:
            st.warning("Could not compute conditional volatility.")
            st.exception(e)

        # Forecasts — use analytic only for 1-step; else simulation
        try:
            if int(horizon) > 1:
                f = res.forecast(
                    horizon=int(horizon),
                    reindex=True,
                    method="simulation",
                    simulations=int(sims),
                    random_state=42,
                )
            else:
                f = res.forecast(horizon=1, reindex=True)

            last = f.variance.iloc[-1]
            var_fc = last.to_numpy().ravel() if hasattr(last, "to_numpy") else np.asarray(last).ravel()
            vol_fc = np.sqrt(var_fc)  # percent

            fc_df = pd.DataFrame({"h": np.arange(1, len(vol_fc) + 1), "vol_forecast_%": vol_fc})
            fig_fc = px.line(fc_df, x="h", y="vol_forecast_%", markers=True,
                             title=f"Forecasted Volatility (next {len(vol_fc)} day(s), %)")
            fig_fc.update_layout(height=400)
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.error("Volatility forecast failed.")
            st.exception(e)

    with tab3:
        st.subheader("Downloads")
        try:
            st.download_button("Download Returns CSV",
                               data=rets.to_csv(index=False).encode("utf-8"),
                               file_name="returns.csv", mime="text/csv")
        except Exception:
            pass
        try:
            vol_series = pd.DataFrame({"cond_vol_%": np.asarray(res.conditional_volatility)})
            st.download_button("Download Conditional Volatility CSV",
                               data=vol_series.to_csv(index=False).encode("utf-8"),
                               file_name="conditional_volatility.csv", mime="text/csv")
        except Exception:
            pass
        try:
            params_df = res.params.to_frame(name="value")
            params_df["tval"] = res.tvalues
            params_df["pval"] = res.pvalues
            st.download_button("Download Model Parameters CSV",
                               data=params_df.to_csv().encode("utf-8"),
                               file_name="egarch_params.csv", mime="text/csv")
        except Exception:
            pass

    st.success("Ready. Explore the tabs above for distribution, model fit, and downloads.")

# Top-level catch to avoid the generic Streamlit error page
try:
    main()
except Exception as e:
    st.error("A fatal error occurred before the app could render:")
    env_panel()
    st.exception(e)
