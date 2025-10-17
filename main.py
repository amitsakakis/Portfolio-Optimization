import math, time, datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st

try:
    import tickertape
    tickertape.display_ticker_tape()
except Exception:
    pass

st.set_page_config(page_title="Stock Portfolio Optimization", layout="centered")

def _today(tz="America/Toronto"):
    return pd.Timestamp.now(tz=tz).date()

def _yf_download_with_timeout(tickers, start_date, end_date, timeout_sec=30):
    def _call():
        return yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
            interval="1d",
            group_by="column",
        )
    last_err = None
    for _ in range(2):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call)
                return fut.result(timeout=timeout_sec)
        except TimeoutError:
            last_err = TimeoutError(f"yfinance download exceeded {timeout_sec}s")
        except Exception as e:
            last_err = e
        time.sleep(1.2)
    raise RuntimeError(f"yfinance download failed: {last_err}")

@st.cache_data(show_spinner=False, ttl=60*10)
def _load_prices(tickers, start_date, end_date):
    df = _yf_download_with_timeout(tickers, start_date, end_date)
    if df is None or df.empty:
        raise ValueError("No price data returned.")
    try:
        adj = df["Adj Close"]
    except Exception:
        if "Adj Close" in df.columns:
            adj = df[["Adj Close"]]
        elif "Close" in df.columns:
            adj = df[["Close"]]
        else:
            raise ValueError("Missing 'Adj Close'/'Close' in data.")
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    if isinstance(adj.columns, pd.MultiIndex):
        if adj.columns.nlevels >= 2:
            adj.columns = adj.columns.get_level_values(1)
    adj = adj.dropna(how="all")
    if adj.empty:
        raise ValueError("All adjusted prices are NaN for the selected range.")
    return adj

def sanitize_tickers(raw):
    cleaned = []
    for t in raw:
        if t is None:
            s = ""
        elif isinstance(t, str):
            s = t
        elif isinstance(t, (float, int, np.floating, np.integer)):
            s = "" if pd.isna(t) else str(t)
        else:
            s = str(t)
        s = s.strip().upper()
        cleaned.append(s)
    return cleaned

def get_stock_data(tickers, start_date, end_date, tz):
    tickers = sanitize_tickers(tickers)
    if any(t == "" for t in tickers):
        raise ValueError("Ticker names cannot be empty.")
    if end_date > _today(tz):
        raise ValueError("End date should be no later than today.")
    if start_date > end_date:
        raise ValueError("Start date should be earlier than end date.")
    raw = _load_prices(tickers, start_date, end_date)
    cols = [c for c in raw.columns if c in tickers]
    if not cols:
        raise ValueError("None of the requested tickers returned prices.")
    data = raw[cols].dropna()
    if data.empty:
        raise ValueError("No usable price data for the tickers/date range.")
    return data

def daily_returns(prices):
    return prices.pct_change().dropna()

def perf(weights, rets):
    m = rets.mean().values
    C = rets.cov().values
    r = float(np.dot(weights, m) * 252.0)
    v = float(np.sqrt(weights @ (C * 252.0) @ weights))
    s = r / v if v > 0 else np.nan
    return r, v, s

def obj_sharpe(w, R): return -perf(w, R)[2]
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
        prices = get_stock_data(tickers, start_date, end_date, tz)
        s.update(label="Computing returns...")
        R = daily_returns(prices)
        s.update(label="Optimizing...")
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
    return x0, res.x, R, name

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
    fig, ax = plt.subplots()
    sc = ax.scatter(vols, rets, c=sh, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Sharpe')
    if crit != 'sharpe':
        ax.scatter(sdp, rp, marker='*', s=120, label='Max Sharpe')
    r_opt, v_opt, _ = perf(w_opt, R)
    ax.scatter(v_opt, r_opt, marker='o', s=120, label=f'Optimized ({name})')
    ax.set_xlabel('Volatility (annual)')
    ax.set_ylabel('Return (annual)')
    ax.legend(loc='upper left')
    st.pyplot(fig)

def show_results(tickers, w0, w, R, name):
    ir, iv, is_ = perf(w0, R)
    or_, ov, os_ = perf(w, R)
    st.markdown("### Initial Portfolio Performance")
    st.markdown(f"**Return:** {ir:.2f}")
    st.markdown(f"**Volatility:** {iv:.2f}")
    st.markdown(f"**Sharpe Ratio:** {is_:.2f}")
    st.markdown(f"### Optimized Portfolio Performance ({name})")
    st.markdown(f"**Return:** {or_:.2f}")
    st.markdown(f"**Volatility:** {ov:.2f}")
    st.markdown(f"**Sharpe Ratio:** {os_:.2f}")
    st.markdown("### Initial Weights")
    st.table(pd.DataFrame({'Ticker': tickers,'Initial Weight (%)': np.array(w0)*100.0}))
    st.markdown("### Optimized Weights")
    st.table(pd.DataFrame({'Ticker': tickers,'Optimized Weight (%)': np.array(w)*100.0}))

st.title("Stock Portfolio Optimization App")
st.subheader("Akaash Mitsakakis-Nath")
st.markdown("Optimize a stock portfolio over a recent window using Sharpe, CVaR proxy, Sortino, or Volatility.")

default_tickers = ['AAPL','MSFT','GOOG','AMZN','META']

with st.expander("Enter your portfolio details:"):
    tickers, allocations = [], []
    num_tickers = st.number_input("Number of tickers:", 1, 10, 5, 1)
    for i in range(num_tickers):
        c1, c2 = st.columns(2)
        tdef = default_tickers[i] if i < len(default_tickers) else ''
        t = c1.text_input(f"Ticker {i+1}:", tdef)
        a = c2.number_input(f"Allocation for Ticker {i+1} (%):", value=float(100.0/num_tickers), format='%.2f', step=0.01)
        tickers.append(t); allocations.append(a/100.0)

with st.expander("Enter the date range and optimization criterion:"):
    tz = st.selectbox("Timezone:", ["America/Toronto","America/New_York","UTC","Europe/London","Europe/Paris","Asia/Tokyo"], index=0)
    local_today = _today(tz)
    start_date = st.date_input("Start date", local_today - datetime.timedelta(days=7))
    end_date = st.date_input("End date", local_today)
    crit = st.selectbox("Optimization criterion:", ['sharpe','cvar','sortino','volatility'])

alloc_sum = sum(allocations)*100
st.caption(f"Allocation sum: {alloc_sum:.4f}%")

run = st.button("Optimize Portfolio")

if run:
    try:
        tickers = sanitize_tickers(tickers)
        invalid_positions = [i+1 for i, t in enumerate(tickers) if t == ""]
        if invalid_positions:
            st.error(f"Please fill all ticker fields. Empty/invalid at: {invalid_positions}")
            st.stop()
        if not math.isclose(sum(allocations), 1.0, rel_tol=1e-5, abs_tol=1e-8):
            st.error("The sum of allocations must be approximately 100%.")
            st.stop()
        w0, w, R, name = optimize_portfolio(tickers, allocations, start_date, end_date, crit, tz)
        st.markdown("---")
        st.header("Portfolio Performance")
        show_results(tickers, w0, w, R, name)
        st.markdown("---")
        st.header("Efficient Frontier")
        with st.spinner('Calculating and plotting the efficient frontier...'):
            plot_ef(tickers, R, w, name, crit)
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {e}")
