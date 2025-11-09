import math
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import norm

st.set_page_config(page_title="Stock Portfolio Optimization", layout="centered")

def _today(tz="America/New_York"):
    return pd.Timestamp.now(tz=tz).date()

def coerce_symbols(raw):
    if isinstance(raw, (np.ndarray, pd.Index, set, tuple)):
        raw = list(raw)
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for x in raw:
        if x is None:
            s = ""
        elif isinstance(x, str):
            s = x
        elif isinstance(x, (np.floating, float)):
            s = "" if (isinstance(x, (np.floating, float)) and pd.isna(x)) else str(x)
        elif isinstance(x, (np.integer, int)):
            s = str(x)
        else:
            s = str(x)
        out.append(s.strip().upper())
    return out

def validate_symbols(symbols):
    bad_idx = [i+1 for i, s in enumerate(symbols) if s == ""]
    if bad_idx:
        raise ValueError(f"Ticker names cannot be empty. Empty at positions: {bad_idx}")
    return symbols

@st.cache_data(show_spinner=False, ttl=60*10)
def load_prices(symbols, start_date, end_date):
    symbols = validate_symbols(coerce_symbols(symbols))
    df = yf.download(
        " ".join(symbols),
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=False,
        interval="1d",
        group_by="column",
    )
    
    try:
        adj = df["Adj Close"]
    except Exception:
        adj = df.get("Adj Close", df.get("Close"))
    
    if adj is None or adj.empty:
        raise RuntimeError("No usable price data.")
    
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    
    if isinstance(adj.columns, pd.MultiIndex) and adj.columns.nlevels >= 2:
        adj.columns = adj.columns.get_level_values(1)
    
    return adj.dropna(how="all")

def get_stock_data(tickers, start_date, end_date, tz):
    symbols = validate_symbols(coerce_symbols(tickers))
    if end_date > _today(tz):
        raise ValueError("End date should be no later than today.")
    if start_date > end_date:
        raise ValueError("Start date should be earlier than end date.")
    
    raw = load_prices(symbols, start_date, end_date)
    keep = [c for c in raw.columns if c in symbols]
    data = raw[keep] if keep else raw
    data = data.dropna(axis=1, how="all").dropna()
    
    if data.empty:
        raise ValueError("No usable price data for the tickers/date range.")
    
    dropped = [s for s in symbols if s not in data.columns]
    return data, dropped

def daily_returns(prices):
    return prices.pct_change().dropna()

def perf(weights, rets):
    m = rets.mean().values
    C = rets.cov().values
    r = float(np.dot(weights, m) * 252.0)
    v = float(np.sqrt(weights @ (C * 252.0) @ weights))
    s = r / v if v > 0 else np.nan
    return r, v, s

def obj_sharpe(w, R): 
    return -perf(w, R)[2]

def obj_cvar(w, R, a=0.05):
    p = R.values @ w
    mu = p.mean()
    sd = p.std(ddof=1)
    z = norm.ppf(a)
    return -(mu - sd * z)

def obj_sortino(w, R):
    p = R.values @ w
    d = p[p < 0]
    ds = d.std(ddof=1) if len(d) else 0.0
    return np.inf if ds == 0 else -(p.mean() / ds)

def obj_var(w, R):
    C = R.cov().values
    return float(w @ (C * 252.0) @ w)

def optimize_portfolio(tickers, allocs, start_date, end_date, crit, tz):
    with st.status("Downloading data...", expanded=False) as s:
        prices, dropped = get_stock_data(tickers, start_date, end_date, tz)
        s.update(label="Computing returns…")
        R = daily_returns(prices)
        used_tickers = list(R.columns)
        
        if len(used_tickers) != len(tickers):
            kept_mask = [t in used_tickers for t in tickers]
            allocs = list(np.array(allocs)[kept_mask])
            tot = sum(allocs)
            if tot <= 0:
                raise ValueError("All tickers lacked data; nothing to optimize.")
            allocs = [a/tot for a in allocs]
            tickers = used_tickers
        
        s.update(label="Optimizing…")
        cons = ({'type': 'eq', 'fun': lambda x: float(np.sum(x) - 1.0)},)
        bnds = tuple((0.0, 1.0) for _ in range(len(tickers)))
        x0 = np.array(allocs, float)
        
        if crit == 'sharpe':
            f, name = obj_sharpe, "Highest Sharpe Ratio"
        elif crit == 'cvar':
            f, name = obj_cvar, "Highest CVaR Proxy"
        elif crit == 'sortino':
            f, name = obj_sortino, "Highest Sortino Ratio"
        elif crit == 'volatility':
            f, name = obj_var, "Lowest Volatility"
        else:
            raise ValueError("Invalid optimization criterion.")
        
        res = minimize(f, x0, args=(R,), method='SLSQP', bounds=bnds, constraints=cons, 
                      options={'maxiter': 400, 'ftol': 1e-9, 'disp': False})
        
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        
        s.update(label="Done", state="complete")
    
    return x0, res.x, R, name, dropped, prices

def plot_ef(tickers, R, w_opt, name, crit):
    np.random.seed(42)
    n = len(tickers)
    N = 3500
    m = R.mean().values * 252.0
    C = R.cov().values * 252.0
    W = np.random.rand(N, n)
    W /= W.sum(axis=1, keepdims=True)
    rets = W @ m
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, C, W))
    sh = np.divide(rets, vols, out=np.full_like(rets, np.nan), where=vols > 0)
    i = np.nanargmax(sh)
    sdp, rp = float(vols[i]), float(rets[i])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(vols, rets, c=sh, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
    
    if crit != 'sharpe':
        ax.scatter(sdp, rp, marker='*', s=200, c='red', label='Max Sharpe', edgecolors='black', zorder=5)
    
    r_opt, v_opt, _ = perf(w_opt, R)
    ax.scatter(v_opt, r_opt, marker='o', s=200, c='orange', label=f'Optimized ({name})', edgecolors='black', zorder=5)
    
    ax.set_xlabel('Volatility (Annual)', fontsize=12)
    ax.set_ylabel('Return (Annual)', fontsize=12)
    ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

def backtest_portfolios(prices, initial_weights, optimized_weights, start_date, end_date, tz):
    """Backtest initial portfolio, optimized portfolio, and S&P 500."""
    
    # Get S&P 500 data
    sp500_prices, _ = get_stock_data(['SPY'], start_date, end_date, tz)
    
    # Normalize prices to start at 100
    initial_portfolio = (prices @ initial_weights).values
    optimized_portfolio = (prices @ optimized_weights).values
    sp500 = sp500_prices['SPY'].values
    
    # Normalize all to 100 at start
    initial_portfolio = 100 * initial_portfolio / initial_portfolio[0]
    optimized_portfolio = 100 * optimized_portfolio / optimized_portfolio[0]
    sp500 = 100 * sp500 / sp500[0]
    
    # Calculate returns
    initial_return = (initial_portfolio[-1] / initial_portfolio[0] - 1) * 100
    optimized_return = (optimized_portfolio[-1] / optimized_portfolio[0] - 1) * 100
    sp500_return = (sp500[-1] / sp500[0] - 1) * 100
    
    # Calculate volatility (annualized)
    initial_vol = np.std(np.diff(initial_portfolio) / initial_portfolio[:-1]) * np.sqrt(252) * 100
    optimized_vol = np.std(np.diff(optimized_portfolio) / optimized_portfolio[:-1]) * np.sqrt(252) * 100
    sp500_vol = np.std(np.diff(sp500) / sp500[:-1]) * np.sqrt(252) * 100
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    initial_sharpe = initial_return / initial_vol if initial_vol > 0 else 0
    optimized_sharpe = optimized_return / optimized_vol if optimized_vol > 0 else 0
    sp500_sharpe = sp500_return / sp500_vol if sp500_vol > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = prices.index
    ax.plot(dates, initial_portfolio, label='Initial Portfolio', linewidth=2, color='blue')
    ax.plot(dates, optimized_portfolio, label='Optimized Portfolio', linewidth=2, color='green')
    ax.plot(dates, sp500, label='S&P 500 (SPY)', linewidth=2, color='red', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Backtest: Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    st.pyplot(fig)
    
    # Display metrics
    st.markdown("### 📊 Backtest Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Initial Portfolio")
        st.metric("Total Return", f"{initial_return:.2f}%")
        st.metric("Volatility", f"{initial_vol:.2f}%")
        st.metric("Sharpe Ratio", f"{initial_sharpe:.3f}")
    
    with col2:
        st.markdown("#### Optimized Portfolio")
        st.metric("Total Return", f"{optimized_return:.2f}%", 
                 delta=f"{(optimized_return - initial_return):.2f}%")
        st.metric("Volatility", f"{optimized_vol:.2f}%", 
                 delta=f"{(optimized_vol - initial_vol):.2f}%",
                 delta_color="inverse")
        st.metric("Sharpe Ratio", f"{optimized_sharpe:.3f}", 
                 delta=f"{(optimized_sharpe - initial_sharpe):.3f}")
    
    with col3:
        st.markdown("#### S&P 500")
        st.metric("Total Return", f"{sp500_return:.2f}%")
        st.metric("Volatility", f"{sp500_vol:.2f}%")
        st.metric("Sharpe Ratio", f"{sp500_sharpe:.3f}")
    
    # Summary comparison
    st.markdown("### 🏆 Performance Summary")
    
    best_return = max(initial_return, optimized_return, sp500_return)
    best_sharpe = max(initial_sharpe, optimized_sharpe, sp500_sharpe)
    lowest_vol = min(initial_vol, optimized_vol, sp500_vol)
    
    summary = []
    if optimized_return == best_return:
        summary.append("✅ **Optimized portfolio** achieved the highest return")
    if optimized_sharpe == best_sharpe:
        summary.append("✅ **Optimized portfolio** has the best risk-adjusted return (Sharpe)")
    if optimized_vol == lowest_vol:
        summary.append("✅ **Optimized portfolio** has the lowest volatility")
    
    if summary:
        for s in summary:
            st.markdown(s)
    else:
        st.info("The optimization improved certain metrics but didn't achieve the best overall performance compared to benchmarks.")

def show_results(tickers, w0, w, R, name, dropped):
    ir, iv, is_ = perf(w0, R)
    or_, ov, os_ = perf(w, R)
    
    st.markdown("### Initial Portfolio Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Return", f"{ir:.2f}")
    with col2:
        st.metric("Volatility", f"{iv:.2f}")
    with col3:
        st.metric("Sharpe Ratio", f"{is_:.2f}")
    
    st.markdown(f"### Optimized Portfolio Performance ({name})")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Return", f"{or_:.2f}", delta=f"{(or_ - ir):.2f}")
    with col2:
        st.metric("Volatility", f"{ov:.2f}", delta=f"{(ov - iv):.2f}", delta_color="inverse")
    with col3:
        st.metric("Sharpe Ratio", f"{os_:.2f}", delta=f"{(os_ - is_):.2f}")
    
    st.markdown("### Portfolio Weights Comparison")
    
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Initial Weight (%)': np.array(w0)*100.0,
        'Optimized Weight (%)': np.array(w)*100.0,
        'Change (%)': (np.array(w) - np.array(w0))*100.0
    })
    
    st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    if dropped:
        st.warning(f"⚠️ Dropped due to no data: {', '.join(dropped)}")

# ==================== STREAMLIT APP ====================

st.title("Stock Portfolio Optimization App")
st.subheader("Akaash Mitsakakis-Nath")
st.markdown("Optimize a stock portfolio using Sharpe, CVaR proxy, Sortino, or Volatility.")

default_tickers = ['AAPL','MSFT','GOOG','AMZN','META']

with st.expander("Enter your portfolio details:", expanded=True):
    tickers, allocations = [], []
    num_tickers = st.number_input("Number of tickers:", 1, 20, 5, 1)
    for i in range(num_tickers):
        c1, c2 = st.columns(2)
        tdef = default_tickers[i] if i < len(default_tickers) else ''
        t = c1.text_input(f"Ticker {i+1}:", tdef, key=f"ticker_{i}")
        a = c2.number_input(f"Allocation for Ticker {i+1} (%):", value=float(100.0/num_tickers), format='%.2f', step=0.01, key=f"alloc_{i}")
        tickers.append(t)
        allocations.append(a/100.0)

with st.expander("Enter the date range and optimization criterion:", expanded=True):
    tz = st.selectbox("Timezone:", ["America/New_York","UTC","Europe/London","Europe/Paris","Asia/Tokyo"], index=0)
    local_today = _today(tz)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", local_today - datetime.timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", local_today)
    
    crit = st.selectbox("Optimization criterion:", ['sharpe','cvar','sortino','volatility'])

alloc_sum = sum(allocations)*100
symbols_preview = coerce_symbols(tickers)

st.markdown("### 📊 Preview")
col1, col2 = st.columns(2)
with col1:
    st.caption(f"**Allocation sum:** {alloc_sum:.2f}%")
    st.caption(f"**Date range:** {(end_date - start_date).days} days")
with col2:
    st.caption(f"**Symbols:** {', '.join(symbols_preview)}")
    st.caption(f"**Period:** {start_date.isoformat()} → {end_date.isoformat()}")

if not math.isclose(alloc_sum, 100.0, rel_tol=1e-3, abs_tol=0.01):
    st.warning(f"⚠️ Allocation sum is {alloc_sum:.2f}%. Should be 100%.")

if (end_date - start_date).days < 30:
    st.info("ℹ️ Consider using at least 30 days of data for better optimization results.")

run = st.button("🚀 Optimize Portfolio", type="primary")

if run:
    try:
        symbols = validate_symbols(symbols_preview)
        if not math.isclose(sum(allocations), 1.0, rel_tol=1e-5, abs_tol=1e-8):
            st.error("The sum of allocations must be approximately 100%.")
            st.stop()
        
        w0, w, R, name, dropped, prices = optimize_portfolio(symbols, allocations, start_date, end_date, crit, tz)
        
        st.markdown("---")
        st.header("📈 Portfolio Performance")
        show_results(symbols, w0, w, R, name, dropped)
        
        st.markdown("---")
        st.header("📉 Efficient Frontier")
        with st.spinner('Calculating and plotting the efficient frontier...'):
            plot_ef(R.columns.tolist(), R, w, name, crit)
        
        st.markdown("---")
        st.header("📊 Backtest Analysis")
        with st.spinner('Running backtest against S&P 500...'):
            backtest_portfolios(prices, w0, w, start_date, end_date, tz)
            
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        st.info("💡 Try the following:\n- Increase the date range to at least 60-90 days\n- Verify ticker symbols are correct\n- Check your internet connection")