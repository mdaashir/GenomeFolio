"""Streamlit UI for GenomeFolio.

Shows the output of every pipeline phase — data, symbolization, raw alignment scores,
pairwise distance, clustering, forecast and band stability — with a graph and a table for
each, then offers the results for download.

Run:

    uv run --group ui streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import replace

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import plots
from src.config import load_config
from src.data import align_to_common_days, get_prices, load_fixture, to_returns
from src.pipeline import run_pipeline

st.set_page_config(page_title="GenomeFolio", page_icon="📈", layout="wide")

CUTOFF_PRESETS = {
    "20/40/60/80": (0.20, 0.40, 0.60, 0.80),
    "15/35/65/85": (0.15, 0.35, 0.65, 0.85),
}


@st.cache_data(show_spinner=False)
def load_returns(use_fixture: bool, config_path: str | None) -> pd.DataFrame:
    cfg = load_config(config_path)
    prices = load_fixture(cfg) if use_fixture else get_prices(cfg)
    return align_to_common_days(to_returns(prices))


def show_figure(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    base = load_config()
    st.title("GenomeFolio")
    st.caption(
        "Sequence-alignment clustering of return patterns, followed by a per-cluster forecast."
    )

    with st.sidebar:
        st.header("Configuration")
        use_fixture = st.checkbox("Use the offline fixture", value=True)
        train_days = st.slider(
            "Training window (days)", 120, 504, int(base.window.train_days), step=21
        )
        horizon = st.slider("Forecast horizon (days)", 5, 63, int(base.window.forecast_horizon))
        scope = st.selectbox("Quantile scope", ["per_stock", "pooled"], index=0)
        preset = st.selectbox("Symbol cutoffs", list(CUTOFF_PRESETS), index=0)
        market_neutral = st.checkbox("Market-neutral returns", value=False)
        band = st.slider("Alignment band", 1, 20, int(base.align.band))
        min_size = st.slider(
            "Minimum cluster size", 2, 8, int(base.cluster.min_cluster_size)
        )
        k_divisor = st.slider("k cap divisor", 2, 6, int(base.cluster.k_cap_divisor))
        model = st.selectbox("Forecast model", ["arima", "spline"], index=0)
        run = st.button("Run pipeline", type="primary", width="stretch")

    cfg = replace(
        base,
        symbolize=replace(
            base.symbolize,
            cutoffs=CUTOFF_PRESETS[preset],
            quantile_scope=scope,
            market_neutral=market_neutral,
        ),
        align=replace(base.align, band=band),
        cluster=replace(base.cluster, min_cluster_size=min_size, k_cap_divisor=k_divisor),
        window=replace(base.window, train_days=train_days, forecast_horizon=horizon),
        forecast=replace(base.forecast, model=model),
    )

    try:
        returns = load_returns(use_fixture, str(base.source_path))
    except Exception as exc:  # network / cache errors
        st.error(f"Could not load prices: {exc}")
        return

    if len(returns) < train_days:
        st.error(f"Need at least {train_days} days of returns, got {len(returns)}.")
        return

    if run:
        window = returns.iloc[-train_days:]
        with st.spinner("Running symbolization, alignment, clustering and forecasting…"):
            st.session_state["result"] = run_pipeline(window, cfg)
            st.session_state["window"] = window
            st.session_state["returns"] = returns
            st.session_state["cfg"] = cfg

    result = st.session_state.get("result")
    if result is None:
        st.info("Set the parameters in the sidebar and press **Run pipeline**.")
        return

    window = st.session_state["window"]
    cfg = st.session_state["cfg"]
    full_returns = st.session_state["returns"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks", window.shape[1])
    col2.metric("Window (days)", len(window))
    col3.metric("Clusters", result.n_clusters)
    col4.metric("Silhouette", f"{result.silhouette:.3f}")
    if not result.min_size_ok:
        st.warning("No k satisfied the minimum cluster size; showing the best-silhouette cut.")

    # Phase 1 -------------------------------------------------------------------------
    st.header("1. Data")
    st.write(
        f"Training window **{window.index.min().date()} → {window.index.max().date()}** "
        f"({len(window)} trading days, {window.shape[1]} stocks)."
    )
    show_figure(plots.cumulative_returns(full_returns))
    st.caption("Full downloaded history; the pipeline uses only the trailing training window.")
    st.dataframe(window.tail(5), width="stretch")

    # Phase 2 -------------------------------------------------------------------------
    st.header("2. Symbolization")
    st.write(
        "Each stock's returns are cut into five symbols using training-window quantiles "
        "(big-down, small-down, flat, small-up, big-up)."
    )
    show_figure(plots.symbol_heatmap(result.symbols))
    show_figure(plots.symbol_distribution(result.symbols))
    st.dataframe(result.symbols.tail(8), width="stretch")

    # Phase 3 -------------------------------------------------------------------------
    st.header("3. Alignment → pairwise distance")
    st.write("Raw pairwise alignment scores, then the normalised distance used for clustering:")
    st.latex(
        r"d(a,b) = 1 - \frac{\mathrm{score}(a,b)}"
        r"{\sqrt{\mathrm{self}(a)\,\mathrm{self}(b)}}"
    )
    show_figure(plots.score_heatmap(result.scores))
    st.dataframe(result.scores.round(1), width="stretch")
    show_figure(plots.distance_heatmap(result.distance))
    st.dataframe(result.distance.round(3), width="stretch")

    # Phase 4 -------------------------------------------------------------------------
    st.header("4. Clustering")
    show_figure(plots.cluster_sizes(result.clusters))
    show_figure(plots.silhouette_curve(result.distance, cfg.cluster))
    show_figure(plots.dendrogram(result.distance, cfg.cluster.linkage))
    labels = pd.DataFrame(
        {
            "stock": list(window.columns),
            "cluster": [
                next(cid for cid, members in result.clusters.items() if t in members)
                for t in window.columns
            ],
        }
    ).sort_values("cluster")
    st.dataframe(labels, width="stretch")

    # Phase 5 -------------------------------------------------------------------------
    st.header("5. Forecast")
    show_figure(
        plots.forecast_panels(result.composite_returns, result.forecasts, cfg.window.forecast_horizon)
    )
    rows = [
        {
            "cluster": cid,
            "stocks": ", ".join(result.clusters[cid]),
            "expected_return": fc.expected_return,
            "model": fc.model,
            "order": fc.order,
            "fallback": fc.fallback,
        }
        for cid, fc in sorted(result.forecasts.items())
    ]
    forecast_df = pd.DataFrame(rows)
    st.dataframe(forecast_df, width="stretch")

    # Stability -----------------------------------------------------------------------
    st.header("6. Band stability")
    st.write("Re-clustering at several alignment band widths to check the solution is not an artefact of one band.")
    fig, summary, ari = plots.band_stability(
        result.symbols, cfg.align, cfg.cluster, cfg.align.band_sweep
    )
    show_figure(fig)
    st.dataframe(summary.round(3), width="stretch")
    st.dataframe(ari.round(3), width="stretch")

    # Downloads -----------------------------------------------------------------------
    payload = {
        "n_clusters": result.n_clusters,
        "silhouette": result.silhouette,
        "min_size_ok": result.min_size_ok,
        "forecast_horizon": cfg.window.forecast_horizon,
        "clusters": {str(cid): list(m) for cid, m in result.clusters.items()},
        "forecasts": {
            str(cid): {
                "expected_return": fc.expected_return,
                "model": fc.model,
                "order": list(fc.order) if fc.order else None,
                "fallback": fc.fallback,
                "path": fc.path,
            }
            for cid, fc in result.forecasts.items()
        },
    }
    left, right = st.columns(2)
    left.download_button(
        "Download clusters + forecasts (CSV)",
        forecast_df.to_csv(index=False),
        file_name="clusters_forecasts.csv",
        mime="text/csv",
        width="stretch",
    )
    right.download_button(
        "Download full result (JSON)",
        json.dumps(payload, indent=2, default=str),
        file_name="forecasts.json",
        mime="application/json",
        width="stretch",
    )


if __name__ == "__main__":
    main()
