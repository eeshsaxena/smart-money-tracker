"""Markowitz Mean-Variance Optimization and Efficient Frontier."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize

from config import RISK_FREE_RATE


def fetch_stock_prices(symbols: list[str], period: str = "2y") -> pd.DataFrame:
    prices = pd.DataFrame()
    for sym in symbols[:20]:
        try:
            data = yf.download(f"{sym}.NS", period=period, progress=False)
            if not data.empty:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                prices[sym] = close
        except Exception:
            continue
    return prices.dropna(axis=1, how="all").ffill().dropna()


def compute_efficient_frontier(
    prices: pd.DataFrame,
    num_portfolios: int = 5000,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    if prices.empty or prices.shape[1] < 2:
        return {"portfolios": pd.DataFrame(), "optimal": {}, "min_vol": {}}

    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(mean_returns)

    results = []
    np.random.seed(42)

    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        port_return = np.dot(weights, mean_returns) * 100
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100
        sharpe = (port_return / 100 - risk_free_rate) / (port_vol / 100) if port_vol > 0 else 0

        results.append({
            "return_pct": round(port_return, 2),
            "volatility_pct": round(port_vol, 2),
            "sharpe_ratio": round(sharpe, 2),
            **{f"w_{col}": round(w * 100, 2) for col, w in zip(prices.columns, weights)},
        })

    df = pd.DataFrame(results)

    optimal_idx = df["sharpe_ratio"].idxmax()
    optimal = df.iloc[optimal_idx].to_dict()

    min_vol_idx = df["volatility_pct"].idxmin()
    min_vol = df.iloc[min_vol_idx].to_dict()

    return {"portfolios": df, "optimal": optimal, "min_vol": min_vol}


def optimize_portfolio(
    prices: pd.DataFrame,
    target: str = "max_sharpe",
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    if prices.empty or prices.shape[1] < 2:
        return {}

    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(mean_returns)

    def neg_sharpe(weights):
        port_ret = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 0.40) for _ in range(n))
    initial = np.array([1 / n] * n)

    if target == "max_sharpe":
        result = minimize(neg_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        result = minimize(portfolio_vol, initial, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        return {}

    weights = result.x
    port_ret = np.dot(weights, mean_returns) * 100
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100
    sharpe = (port_ret / 100 - risk_free_rate) / (port_vol / 100) if port_vol > 0 else 0

    allocation = {col: round(w * 100, 2) for col, w in zip(prices.columns, weights) if w > 0.5 / 100}

    return {
        "return_pct": round(port_ret, 2),
        "volatility_pct": round(port_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "allocation": allocation,
    }


def create_efficient_frontier_chart(
    frontier_data: dict,
    fund_name: str = "",
) -> go.Figure:
    df = frontier_data.get("portfolios", pd.DataFrame())
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["volatility_pct"],
        y=df["return_pct"],
        mode="markers",
        marker=dict(
            size=4,
            color=df["sharpe_ratio"],
            colorscale="Viridis",
            colorbar=dict(title="Sharpe"),
            opacity=0.6,
        ),
        name="Random Portfolios",
        text=df["sharpe_ratio"].apply(lambda s: f"Sharpe: {s:.2f}"),
    ))

    opt = frontier_data.get("optimal", {})
    if opt:
        fig.add_trace(go.Scatter(
            x=[opt["volatility_pct"]],
            y=[opt["return_pct"]],
            mode="markers+text",
            marker=dict(size=18, color="#ff1744", symbol="star"),
            name=f"Max Sharpe ({opt.get('sharpe_ratio', 0):.2f})",
            text=["Max Sharpe"],
            textposition="top center",
        ))

    mv = frontier_data.get("min_vol", {})
    if mv:
        fig.add_trace(go.Scatter(
            x=[mv["volatility_pct"]],
            y=[mv["return_pct"]],
            mode="markers+text",
            marker=dict(size=18, color="#00bfa5", symbol="diamond"),
            name=f"Min Vol ({mv.get('volatility_pct', 0):.1f}%)",
            text=["Min Volatility"],
            textposition="top center",
        ))

    fig.update_layout(
        title=f"Efficient Frontier — {fund_name}" if fund_name else "Markowitz Efficient Frontier",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Expected Annual Return (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_allocation_pie(allocation: dict, title: str = "Optimal Allocation") -> go.Figure:
    if not allocation:
        return go.Figure()

    labels = list(allocation.keys())
    values = list(allocation.values())

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        textinfo="label+percent",
        hole=0.4,
    ))
    fig.update_layout(title=title, height=400)
    return fig
