import math
import time
import datetime
from functools import partial
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st

try:
    import tickertape
    tickertape.display_ticker_tape()
except Exception:
    pass

st.set_page_config(page_title="Stock Portfolio Optimization", layout="centered")

TODAY = datetime.date.today()
DEFAULT_START = TODAY - datetime.timedelta(days=372)
DEFAULT_END = TODAY - datetime.timedelta(days=7)

@st.cache_data(show_spinner=False, ttl=60*15)
def _yf_download(tickers, start_date, end_date):
    tickers = [t.strip().upper() for t in tickers]
    def _go():
        return yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=True,
            interval="1d",
            group_by="column",
        )
    last_err = None
    for _ in range(3):
        try:
            df = _go()
            if df is None or df.empty:
                raise ValueError("No price data returned from yfinance.")
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"yfinance download failed after retries: {last_err}")

def _extract_adj_close(df):
    adj = None
    try:
        adj = df["Adj Close"]
    except Exception:
        if "Adj Close" in df.columns:
            adj = df[["Adj Close"]]
        elif "Close" in df.columns:
            adj = df[["Close"]]
        else:
            try:
                candidates = [c for c in df.columns if isinstance(c, tuple) and c[0] == "Adj Close"]
                if candidates:
                    adj = df.loc[:, candidates]
            except Exception:
                pass
    if adj is None:
        raise ValueError("Could not find 'Adj Close' (or 'Close') in the returned data.")
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    if isinstance(adj.columns, pd.MultiIndex):
        if adj.columns.nlevels >= 2:
            try:
                adj.columns = adj.columns.get_level_values(1)
            except Exception:
                adj.columns = [c[-1] for c in adj.columns]
    adj = adj.dropna(how="all")
    if adj.empty:
        raise ValueError("All adjusted prices are NaN for the selected range.")
    return adj

def get_stock_data(tickers, start_date, end_date):
    tickers = [t.strip().upper() for t in tickers]
    if any(t == "" for t in tickers):
        raise ValueError("Ticker names cannot be empty.")
    if end_date > TODAY:
        raise ValueError("End date should be no later than today.")
    if start_date > end_date:
        raise ValueError("Start date should be earlier than end date.")
    raw = _yf_download(tickers, start_date, end_date)
    data = _extract_adj_close(raw)
    existing = [c for c in data.columns if c in tickers]
    if not existing:
        raise ValueError("None of the requested tickers returned adjusted prices.")
    data = data[existing].dropna()
    if data.empty:
        raise ValueError("No usable price data for the tickers and date range provided.")
    return data

def daily_returns_from_prices(prices_df):
    return prices_df.pct_change().dropna()

def portfolio_performance(weights, returns):
    mean_daily = returns.mean().values
    cov_daily = returns.cov().values
    ret_annual = float(np.dot(weights, mean_daily) * 252.0)
    vol_annual = float(np.sqrt(weights @ (cov_daily * 252.0) @ weights))
    sharpe = ret_annual / vol_annual if vol_annual > 0 else np.nan
    return ret_annual, vol_annual, sharpe

def obj_sharpe(weights, returns):
    return -portfolio_performance(weights, returns)[2]

def obj_cvar(weights, returns, alpha=0.05):
    p = returns.values @ weights
    mu = p.mean()
    sigma = p.std(ddof=1)
    z = norm.ppf(alpha)
    cvar_proxy = mu - sigma * z
    return -cvar_proxy

def obj_sortino(weights, returns):
    p = returns.values @ weights
    downside = p[p < 0]
    ds = downside.std(ddof=1) if len(downside) > 0 else 0.0
    if ds == 0:
        return np.inf
    sortino = p.mean() / ds
    return -sortino

def obj_variance(weights, returns):
    cov_daily = returns.cov().values
    return float(weights @ (cov_daily * 252.0) @ weights)

def optimize_portfolio(tickers, initial_allocations, start_date, end_date, optimization_criterion='sharpe'):
    prices = get_stock_data(tickers, start_date, end_date)
    returns = daily_returns_from_prices(prices)
    cons = ({'type': 'eq', 'fun': lambda x: float(np.sum(x) - 1.0)},)
    bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
    x0 = np.array(initial_allocations, dtype=float)
    if optimization_criterion == 'sharpe':
        objective = obj_sharpe
        criterion_name = "Highest Sharpe Ratio"
    elif optimization_criterion == 'cvar':
        objective = obj_cvar
        criterion_name = "Highest CVaR Proxy"
    elif optimization_criterion == 'sortino':
        objective = obj_sortino
        criterion_name = "Highest Sortino Ratio"
    elif optimization_criterion == 'volatility':
        objective = obj_variance
        criterion_name = "Lowest Volatility"
    else:
        raise ValueError("Invalid optimization criterion.")
    res = minimize(objective, x0, args=(returns,), method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    w_opt = res.x
    return x0, w_opt, returns, criterion_name

def portfolio_performance_with_returns(weights, returns):
    return portfolio_performance(weights, returns)

def plot_efficient_frontier(tickers, returns, optimized_weights, criterion_name, optimization_criterion):
    np.random.seed(42)
    num_portfolios = 8000
    mean_daily = returns.mean().values
    cov_annual = returns.cov().values * 252.0
    mean_annual = mean_daily * 252.0
    n = len(tickers)
    W = np.random.rand(num_portfolios, n)
    W = W / W.sum(axis=1, keepdims=True)
    rets = W @ mean_annual
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, cov_annual, W))
    sharpes = np.divide(rets, vols, out=np.full_like(rets, np.nan), where=vols > 0)
    max_sharpe_idx = np.nanargmax(sharpes)
    sdp, rp = float(vols[max_sharpe_idx]), float(rets[max_sharpe_idx])
    fig, ax = plt.subplots()
    sc = ax.scatter(vols, rets, c=sharpes, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
    if optimization_criterion != 'sharpe':
        ax.scatter(sdp, rp, marker='*', s=120, label='Maximum Sharpe Ratio')
    opt_ret, opt_vol, _ = portfolio_performance(optimized_weights, returns)
    ax.scatter(opt_vol, opt_ret, marker='o', s=120, label=f'Optimized ({criterion_name})')
    ax.set_xlabel('Volatility (annual)')
    ax.set_ylabel('Return (annual)')
    ax.legend(loc='upper left')
    st.pyplot(fig)

def display_results(tickers, initial_guess, optimized_weights, returns, criterion_name):
    init_ret, init_vol, init_sharpe = portfolio_performance(initial_guess, returns)
    opt_ret, opt_vol, opt_sharpe = portfolio_performance(optimized_weights, returns)
    st.markdown("### Initial Portfolio Performance")
    st.markdown(f"**Return:** {init_ret:.2f}")
    st.markdown(f"**Volatility:** {init_vol:.2f}")
    st.markdown(f"**Sharpe Ratio:** {init_sharpe:.2f}")
    st.markdown(f"### Optimized Portfolio Performance ({criterion_name})")
    st.markdown(f"**Return:** {opt_ret:.2f}")
    st.markdown(f"**Volatility:** {opt_vol:.2f}")
    st.markdown(f"**Sharpe Ratio:** {opt_sharpe:.2f}")
    initial_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Initial Weight (%)': np.array(initial_guess) * 100.0
    })
    st.markdown("### Initial Weights")
    st.table(initial_weights_df)
    optimized_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Optimized Weight (%)': np.array(optimized_weights) * 100.0
    })
    st.markdown("### Optimized Weights")
    st.table(optimized_weights_df)

st.title("Stock Portfolio Optimization App")
st.subheader("Akaash Mitsakakis-Nath")
st.markdown("""
This app performs portfolio optimization using different optimization techniques. 
Enter your stock tickers, initial allocations, a historical date range, and the desired optimization criterion to get started.
""")
st.markdown("""
### Optimization Techniques
- **Sharpe Ratio**
- **CVaR (proxy)**
- **Sortino Ratio**
- **Volatility**
""")

default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

with st.expander("Enter your portfolio details:"):
    tickers = []
    allocations = []
    num_tickers = st.number_input("Number of tickers:", min_value=1, max_value=10, value=5, step=1)
    for i in range(num_tickers):
        col1, col2 = st.columns(2)
        default_t = default_tickers[i] if i < len(default_tickers) else ''
        ticker = col1.text_input(f"Ticker {i+1}:", default_t)
        default_alloc = 100.0 / num_tickers
        allocation = col2.number_input(f"Allocation for Ticker {i+1} (%):", value=float(default_alloc), format='%.2f', step=0.01)
        allocations.append(allocation / 100.0)
        tickers.append(ticker)

if not math.isclose(sum(allocations), 1.0, rel_tol=1e-5, abs_tol=1e-8):
    st.error("The sum of allocations must be approximately **100%**.")
else:
    with st.expander("Enter the date range and optimization criterion:"):
        start_date = st.date_input("Start date", DEFAULT_START)
        end_date = st.date_input("End date", DEFAULT_END)
        optimization_criterion = st.selectbox("Optimization criterion:", ['sharpe', 'cvar', 'sortino', 'volatility'])
    if st.button("Optimize Portfolio"):
        try:
            initial_guess, optimized_weights, returns, criterion_name = optimize_portfolio(
                tickers, allocations, start_date, end_date, optimization_criterion
            )
            st.markdown("---")
            st.header("Portfolio Performance")
            display_results(tickers, initial_guess, optimized_weights, returns, criterion_name)
            st.markdown("---")
            st.header("Efficient Frontier")
            with st.spinner('Calculating and plotting the efficient frontier...'):
                plot_efficient_frontier(tickers, returns, optimized_weights, criterion_name, optimization_criterion)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An error occurred during portfolio optimization: {e}")
