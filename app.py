
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Mutual Fund VaR / CVaR Dashboard", layout="wide")


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_price_data(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["as_of"] = pd.to_datetime(raw["as_of"])
    prices = raw.set_index("as_of").sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices


@st.cache_data
def load_sector_data(path: str, ts: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["as_of"] = pd.to_datetime(raw["as_of"])
    raw = raw.rename(columns={"ask_id": "fund"})

    sector_cols = [c for c in raw.columns if "_pct_net" in c]
    if not sector_cols:
        raise ValueError("No sector exposure columns ending in '_pct_net' were found.")

    ts_dt = pd.to_datetime(ts)
    usable = raw.loc[raw["as_of"] <= ts_dt].copy()
    if usable.empty:
        raise ValueError("No sector exposure rows exist on or before the selected start date.")

    usable = usable.sort_values(["fund", "as_of"])
    latest = usable.groupby("fund").tail(1)

    sectors = latest.set_index("fund")[sector_cols].copy()

    def clean_sector_name(col: str) -> str:
        col = col.replace("equity_econ_sector_", "")
        col = col.replace("_pct_net", "")
        col = col.replace("_", " ")
        return col.title()

    sectors.columns = [clean_sector_name(c) for c in sectors.columns]
    sectors = sectors.apply(pd.to_numeric, errors="coerce").fillna(0) / 100.0
    return sectors


# -----------------------------
# Core analytics
# -----------------------------
def rolling_horizon_returns(prices: pd.DataFrame, tau: int, delta: int) -> pd.DataFrame:
    returns = []
    dates = []

    if len(prices) <= tau:
        return pd.DataFrame(columns=prices.columns)

    for i in range(0, len(prices) - tau + 1, delta):
        p_t = prices.iloc[i]
        p_t_tau = prices.iloc[i + tau - 1]
        r = (p_t_tau - p_t) / p_t
        returns.append(r)
        dates.append(prices.index[i])

    return pd.DataFrame(returns, index=dates)


def var_cvar(returns, alpha: float):
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        raise ValueError("No valid returns available for VaR/CVaR.")

    sorted_returns = np.sort(arr)
    idx = int(np.floor((1 - alpha) * len(sorted_returns)))
    idx = max(idx, 1)

    var_return = sorted_returns[idx - 1]
    cvar_return = sorted_returns[:idx].mean()

    return -var_return, -cvar_return


def select_funds(
    prices_subset: pd.DataFrame,
    d: int,
    mode: str,
    manual_funds: list[str] | None = None,
    min_coverage: float = 0.90,
    random_seed: int = 42,
) -> list[str]:
    min_non_missing_count = int(min_coverage * len(prices_subset))
    available_funds = [
        col for col in prices_subset.columns
        if prices_subset[col].notna().sum() >= min_non_missing_count
    ]

    if mode != "Manual" and len(available_funds) < d:
        raise ValueError(
            f"Only {len(available_funds)} funds meet the {min_coverage:.0%} coverage requirement, but d={d}."
        )

    if mode == "First d funds":
        return available_funds[:d]

    if mode == "Random d funds":
        rng = np.random.default_rng(random_seed)
        return list(rng.choice(available_funds, size=d, replace=False))

    if mode == "Manual":
        selected = list(manual_funds or [])
        if len(selected) != d:
            raise ValueError(f"Manual selection must contain exactly d={d} funds.")
        missing = [f for f in selected if f not in prices_subset.columns]
        if missing:
            raise ValueError(f"Funds not found in price data: {missing}")
        return selected

    raise ValueError("Invalid selection mode.")


def compute_fund_shocks(sector_matrix: pd.DataFrame, sector_shocks: dict[str, float]) -> pd.Series:
    shock_series = pd.Series(sector_shocks, dtype=float)
    common = sector_matrix.columns.intersection(shock_series.index)
    if len(common) == 0:
        return pd.Series(0.0, index=sector_matrix.index)
    return sector_matrix[common].fillna(0) @ shock_series[common]


def run_analysis(
    prices: pd.DataFrame,
    sectors: pd.DataFrame,
    ts: str,
    te: str,
    d: int,
    tau: int,
    delta: int,
    alpha: float,
    initial_capital: float,
    selection_mode: str,
    manual_funds: list[str] | None,
    sector_shocks: dict[str, float],
):
    prices_subset = prices.loc[ts:te].copy()

    if prices_subset.empty:
        raise ValueError("No price data in the selected date range.")
    if len(prices_subset) <= tau:
        raise ValueError("Not enough observations in the selected date range for the chosen tau.")

    common_funds = sorted(set(prices_subset.columns).intersection(set(sectors.index)))
    prices_subset = prices_subset[common_funds].copy()
    sectors = sectors.loc[common_funds].copy()

    selected_funds = select_funds(
        prices_subset=prices_subset,
        d=d,
        mode=selection_mode,
        manual_funds=manual_funds,
    )

    prices_d = prices_subset[selected_funds].copy().dropna()
    sectors_d = sectors.loc[selected_funds].copy()

    if len(prices_d) <= tau:
        raise ValueError("Not enough cleaned observations after dropping missing values for chosen tau.")

    fund_returns = rolling_horizon_returns(prices_d, tau=tau, delta=delta)
    if fund_returns.empty:
        raise ValueError("No rolling horizon returns were generated.")

    weights = np.ones(len(selected_funds)) / len(selected_funds)
    portfolio_returns = fund_returns @ weights

    base_var, base_cvar = var_cvar(portfolio_returns.values, alpha)

    fund_shocks = compute_fund_shocks(sectors_d, sector_shocks)
    stressed_fund_returns = fund_returns.add(fund_shocks, axis=1)
    stressed_portfolio_returns = stressed_fund_returns @ weights
    stress_var, stress_cvar = var_cvar(stressed_portfolio_returns.values, alpha)

    return {
        "selected_funds": selected_funds,
        "prices_d": prices_d,
        "sectors_d": sectors_d,
        "fund_returns": fund_returns,
        "weights": weights,
        "portfolio_returns": portfolio_returns,
        "stressed_portfolio_returns": stressed_portfolio_returns,
        "fund_shocks": fund_shocks,
        "base_var": base_var,
        "base_cvar": base_cvar,
        "stress_var": stress_var,
        "stress_cvar": stress_cvar,
        "base_var_dollars": base_var * initial_capital,
        "base_cvar_dollars": base_cvar * initial_capital,
        "stress_var_dollars": stress_var * initial_capital,
        "stress_cvar_dollars": stress_cvar * initial_capital,
    }


def bootstrap_var_cvar(portfolio_returns: pd.Series, alpha: float, n_sim: int = 10000):
    rng = np.random.default_rng(42)
    simulated_returns = rng.choice(portfolio_returns.values, size=n_sim, replace=True)
    var_sim, cvar_sim = var_cvar(simulated_returns, alpha)
    return simulated_returns, var_sim, cvar_sim


def sensitivity_by_d(prices_subset, available_funds, alpha, tau, delta):
    d_values = [1, 5, 10, 15, 20, 25, 30]
    rows = []
    for d_test in d_values:
        if len(available_funds) < d_test:
            rows.append({"d": d_test, "VaR": np.nan, "CVaR": np.nan})
            continue

        prices_test = prices_subset[available_funds[:d_test]].copy().dropna()
        if len(prices_test) <= tau:
            rows.append({"d": d_test, "VaR": np.nan, "CVaR": np.nan})
            continue

        fund_ret_test = rolling_horizon_returns(prices_test, tau=tau, delta=delta)
        if fund_ret_test.empty:
            rows.append({"d": d_test, "VaR": np.nan, "CVaR": np.nan})
            continue

        weights_test = np.ones(d_test) / d_test
        port_ret_test = fund_ret_test @ weights_test
        var_test, cvar_test = var_cvar(port_ret_test.values, alpha)
        rows.append({"d": d_test, "VaR": var_test, "CVaR": cvar_test})

    return pd.DataFrame(rows)


def sensitivity_by_tau(prices_d, alpha, delta, tau_values=(252, 756, 1260)):
    rows = []
    d = prices_d.shape[1]
    weights = np.ones(d) / d
    for tau_test in tau_values:
        if len(prices_d) <= tau_test:
            rows.append({"tau": tau_test, "VaR": np.nan, "CVaR": np.nan})
            continue

        fund_ret_test = rolling_horizon_returns(prices_d, tau=tau_test, delta=delta)
        if fund_ret_test.empty:
            rows.append({"tau": tau_test, "VaR": np.nan, "CVaR": np.nan})
            continue

        port_ret_test = fund_ret_test @ weights
        var_test, cvar_test = var_cvar(port_ret_test.values, alpha)
        rows.append({"tau": tau_test, "VaR": var_test, "CVaR": cvar_test})

    return pd.DataFrame(rows)


def sensitivity_by_delta(prices_d, alpha, tau, delta_values=(1, 5, 22, 66)):
    rows = []
    d = prices_d.shape[1]
    weights = np.ones(d) / d
    for delta_test in delta_values:
        fund_ret_test = rolling_horizon_returns(prices_d, tau=tau, delta=delta_test)
        if fund_ret_test.empty:
            rows.append({"delta": delta_test, "VaR": np.nan, "CVaR": np.nan})
            continue

        port_ret_test = fund_ret_test @ weights
        var_test, cvar_test = var_cvar(port_ret_test.values, alpha)
        rows.append({"delta": delta_test, "VaR": var_test, "CVaR": cvar_test})

    return pd.DataFrame(rows)


def sensitivity_heatmap(prices_d, alpha, tau_values=(252, 756, 1260), delta_values=(1, 5, 22, 66)):
    d = prices_d.shape[1]
    weights = np.ones(d) / d
    heatmap = pd.DataFrame(index=tau_values, columns=delta_values, dtype=float)

    for tau_test in tau_values:
        for delta_test in delta_values:
            if len(prices_d) <= tau_test:
                heatmap.loc[tau_test, delta_test] = np.nan
                continue

            fund_ret_test = rolling_horizon_returns(prices_d, tau=tau_test, delta=delta_test)
            if fund_ret_test.empty:
                heatmap.loc[tau_test, delta_test] = np.nan
                continue

            port_ret_test = fund_ret_test @ weights
            var_test, _ = var_cvar(port_ret_test.values, alpha)
            heatmap.loc[tau_test, delta_test] = var_test

    return heatmap


# -----------------------------
# Plot helpers
# -----------------------------
def distribution_plot(base_returns, stressed_returns, base_var, stress_var, alpha):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=base_returns,
        name="Base",
        opacity=0.6,
        nbinsx=40
    ))
    fig.add_trace(go.Histogram(
        x=stressed_returns,
        name="Stressed",
        opacity=0.6,
        nbinsx=40
    ))

    fig.add_vline(x=-base_var, line_dash="dash", annotation_text=f"Base VaR ({alpha:.0%})")
    fig.add_vline(x=-stress_var, line_dash="dot", annotation_text=f"Stress VaR ({alpha:.0%})")

    fig.update_layout(
        title="Base vs Stressed Portfolio Return Distribution",
        xaxis_title="Portfolio Return",
        yaxis_title="Frequency",
        barmode="overlay",
        height=500,
    )
    return fig


def line_chart(df: pd.DataFrame, x_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df["VaR"], mode="lines+markers", name="VaR"))
    fig.add_trace(go.Scatter(x=df[x_col], y=df["CVaR"], mode="lines+markers", name="CVaR"))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title="Risk")
    return fig


def heatmap_plot(heatmap_df: pd.DataFrame):
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=[str(c) for c in heatmap_df.columns],
        y=[str(i) for i in heatmap_df.index],
        text=np.round(heatmap_df.values, 4),
        texttemplate="%{text}",
        hovertemplate="tau=%{y}<br>delta=%{x}<br>VaR=%{z:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="VaR Heatmap (tau vs delta)",
        xaxis_title="delta",
        yaxis_title="tau",
        height=420,
    )
    return fig


# -----------------------------
# App UI
# -----------------------------
st.title("Mutual Fund VaR / CVaR Dashboard")
st.caption("Historical simulation, bootstrap simulation, and sector-based stress testing.")

with st.sidebar:
    st.header("Data")
    price_path = st.text_input("Price CSV", value="data/us_equity_adj_close.csv")
    sector_path = st.text_input("Sector CSV", value="data/us_equity_sectors.csv")

    st.header("Parameters")
    ts = st.date_input("Start date", value=pd.Timestamp("2017-01-01"))
    te = st.date_input("End date", value=pd.Timestamp("2024-12-31"))
    d = st.slider("Number of funds (d)", 1, 50, 10)
    tau = st.selectbox("Horizon (tau)", options=[252, 756, 1260], index=0)
    delta = st.selectbox("Rolling step (delta)", options=[1, 5, 22, 66], index=2)
    alpha = st.slider("Confidence level (alpha)", 0.90, 0.99, 0.95, 0.01)
    initial_capital = st.number_input("Initial capital", min_value=1000, value=1_000_000, step=10000)

    selection_mode = st.selectbox(
        "Fund selection",
        options=["First d funds", "Random d funds", "Manual"],
    )

# Load data after ts is defined so sector exposures can be snapped to the latest row <= ts
try:
    prices = load_price_data(price_path)
    sectors_matrix = load_sector_data(sector_path, str(ts))
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

common_funds = sorted(set(prices.columns).intersection(set(sectors_matrix.index)))
prices = prices[common_funds].copy()
sectors_matrix = sectors_matrix.loc[common_funds].copy()

with st.sidebar:
    all_funds = list(prices.columns)
    default_manual = all_funds[: min(d, len(all_funds))]
    manual_funds = st.multiselect("Manual funds", options=all_funds, default=default_manual)

    st.header("Sector shocks")
    sector_shocks = {}
    for sector in sectors_matrix.columns:
        sector_shocks[sector] = st.slider(
            sector, min_value=-0.30, max_value=0.30, value=0.0, step=0.01
        )

run_clicked = st.button("Run analysis", type="primary")

if run_clicked:
    try:
        results = run_analysis(
            prices=prices,
            sectors=sectors_matrix,
            ts=str(ts),
            te=str(te),
            d=d,
            tau=tau,
            delta=delta,
            alpha=alpha,
            initial_capital=initial_capital,
            selection_mode=selection_mode,
            manual_funds=manual_funds if selection_mode == "Manual" else None,
            sector_shocks=sector_shocks,
        )

        simulated_returns, var_sim, cvar_sim = bootstrap_var_cvar(
            results["portfolio_returns"], alpha, n_sim=10000
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Base VaR", f"{results['base_var']:.2%}", f"${results['base_var_dollars']:,.0f}")
        c2.metric("Base CVaR", f"{results['base_cvar']:.2%}", f"${results['base_cvar_dollars']:,.0f}")
        c3.metric("Stress VaR", f"{results['stress_var']:.2%}", f"${results['stress_var_dollars']:,.0f}")
        c4.metric("Stress CVaR", f"{results['stress_cvar']:.2%}", f"${results['stress_cvar_dollars']:,.0f}")

        st.subheader("Selected funds")
        st.write(results["selected_funds"])

        st.subheader("Implied fund shocks")
        st.dataframe(
            results["fund_shocks"].rename("Implied Shock").sort_values(),
            use_container_width=True,
        )

        st.subheader("Base vs stressed distribution")
        st.plotly_chart(
            distribution_plot(
                results["portfolio_returns"],
                results["stressed_portfolio_returns"],
                results["base_var"],
                results["stress_var"],
                alpha,
            ),
            use_container_width=True,
        )

        bootstrap_df = pd.DataFrame({
            "Method": ["Historical", "Bootstrap Simulation"],
            "VaR": [results["base_var"], var_sim],
            "CVaR": [results["base_cvar"], cvar_sim],
            "VaR ($)": [results["base_var_dollars"], var_sim * initial_capital],
            "CVaR ($)": [results["base_cvar_dollars"], cvar_sim * initial_capital],
        })

        summary_df = pd.DataFrame({
            "Metric": ["VaR", "CVaR"],
            "Base Return": [results["base_var"], results["base_cvar"]],
            "Base Dollars": [results["base_var_dollars"], results["base_cvar_dollars"]],
            "Stress Return": [results["stress_var"], results["stress_cvar"]],
            "Stress Dollars": [results["stress_var_dollars"], results["stress_cvar_dollars"]],
        })

        left, right = st.columns(2)
        with left:
            st.subheader("Stress summary")
            st.dataframe(summary_df, use_container_width=True)
        with right:
            st.subheader("Bootstrap comparison")
            st.dataframe(bootstrap_df, use_container_width=True)

        # Sensitivity analysis
        st.subheader("Sensitivity analysis")
        prices_subset = prices.loc[str(ts):str(te)].copy()
        min_non_missing_count = int(0.90 * len(prices_subset))
        available_funds = [
            col for col in prices_subset.columns
            if prices_subset[col].notna().sum() >= min_non_missing_count
        ]

        sens_d = sensitivity_by_d(prices_subset, available_funds, alpha, tau, delta)
        sens_tau = sensitivity_by_tau(results["prices_d"], alpha, delta)
        sens_delta = sensitivity_by_delta(results["prices_d"], alpha, tau)
        heatmap_df = sensitivity_heatmap(results["prices_d"], alpha)

        tab1, tab2, tab3, tab4 = st.tabs([
            "By d", "By tau", "By delta", "tau × delta heatmap"
        ])
        with tab1:
            st.plotly_chart(line_chart(sens_d, "d", "VaR and CVaR vs Number of Funds"), use_container_width=True)
            st.dataframe(sens_d, use_container_width=True)
        with tab2:
            st.plotly_chart(line_chart(sens_tau, "tau", "VaR and CVaR vs Horizon (tau)"), use_container_width=True)
            st.dataframe(sens_tau, use_container_width=True)
        with tab3:
            st.plotly_chart(line_chart(sens_delta, "delta", "VaR and CVaR vs Rolling Step (delta)"), use_container_width=True)
            st.dataframe(sens_delta, use_container_width=True)
        with tab4:
            st.plotly_chart(heatmap_plot(heatmap_df), use_container_width=True)
            st.dataframe(heatmap_df, use_container_width=True)

        with st.expander("Sector exposures for selected funds"):
            st.dataframe(results["sectors_d"], use_container_width=True)

        with st.expander("How to interpret this"):
            st.markdown(
                """
- **VaR** is the loss threshold exceeded only in the worst tail of outcomes.
- **CVaR** is the average loss conditional on being in that tail.
- **Stress testing** translates sector shocks into fund shocks using each selected fund's sector exposures.
- **Sensitivity analysis** shows how risk changes with diversification (`d`), horizon (`tau`), and rolling step (`delta`).
                """
            )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Choose inputs in the sidebar and click **Run analysis**.")
