import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st
import datetime
from multiprocessing import Pool
from functools import partial
import time
import math

# Function to get stock data
def get_stock_data(tickers, start_date, end_date):
    if end_date > datetime.date.today():
        raise ValueError("End date should be no later than today.")
    if start_date > end_date:
        raise ValueError("Start date should be earlier than end date.")
    
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.dropna()
    if data.empty:
        raise ValueError("No data available for the entered stock tickers and date range. Please validate your inputs, or try increasing the data range.")
    
    return data

# Function to get largest 20 tickers by market cap
def get_top_20_tickers():
    # Use predefined list of largest 20 tickers by market cap
    top_20_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'TSM', 'NVDA', 'JNJ',
        'WMT', 'JPM', 'V', 'PG', 'UNH', 'HD', 'DIS', 'MA', 'PYPL', 'BAC'
    ]
    return top_20_tickers

# Calculate portfolio performance
def portfolio_performance(weights, returns):
    portfolio_return = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Define the negative Sharpe ratio objective function
def sharpe(weights, returns):
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(weights, returns)
    return -sharpe_ratio

# Define additional objective functions
def cvar(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    conf_level = 0.05
    cvar = portfolio_mean - portfolio_std * norm.ppf(conf_level)
    return cvar

def sortino(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = portfolio_returns.mean() / downside_std
    return -sortino_ratio

def variance(weights, returns):
    return np.dot(weights.T, np.dot(returns.cov() * 252, weights))

# Main function to perform optimization
def optimize_portfolio(tickers, initial_allocations, start_date, end_date, optimization_criterion='sharpe'):
    data = get_stock_data(tickers, start_date, end_date)
    returns = data.pct_change().dropna()
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    initial_guess = np.array(initial_allocations)
    
    if optimization_criterion == 'sharpe':
        objective_function = sharpe
        criterion_name = "Highest Sharpe Ratio"
    elif optimization_criterion == 'cvar':
        objective_function = cvar
        criterion_name = "Lowest Conditional Value at Risk (CVaR)"
    elif optimization_criterion == 'sortino':
        objective_function = sortino
        criterion_name = "Highest Sortino Ratio"
    elif optimization_criterion == 'volatility':
        objective_function = variance
        criterion_name = "Lowest Volatility"
    else:
        raise ValueError("Invalid optimization criterion.")
    
    optimized_results = minimize(objective_function, initial_guess, args=(returns,),
                                 method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimized_weights = optimized_results.x
    
    return initial_guess, optimized_weights, returns, criterion_name

def portfolio_performance_with_returns(weights, returns):
    return portfolio_performance(weights, returns)

def plot_efficient_frontier(tickers, returns, optimized_weights, criterion_name, optimization_criterion):
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    pool = Pool()
    weights = [np.random.random(len(tickers)) for _ in range(num_portfolios)]
    weights = [weight / np.sum(weight) for weight in weights]  # Normalize weights
    performance_function = partial(portfolio_performance_with_returns, returns=returns)
    for i, result in enumerate(pool.imap_unordered(performance_function, weights)):
        portfolio_return, portfolio_volatility, sharpe_ratio = result
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        weights_record.append(weights[i])
    pool.close()
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    if optimization_criterion != 'sharpe':
        ax.scatter(sdp, rp, marker='*', color='r', s=100, label='Maximum Sharpe Ratio')
    
    opt_return, opt_volatility, _ = portfolio_performance(optimized_weights, returns)
    ax.scatter(opt_volatility, opt_return, marker='o', color='blue', s=100, label=f'Optimized Portfolio ({criterion_name})')
    
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend(loc='upper left')
    st.pyplot(fig)

# Function to display results
def display_results(initial_guess, optimized_weights, returns, criterion_name):
    initial_performance = portfolio_performance(initial_guess, returns)
    optimized_performance = portfolio_performance(optimized_weights, returns)
    
    st.markdown("### Initial Portfolio Performance:")
    st.markdown(f"**Return:** {initial_performance[0]:.2f}")
    st.markdown(f"**Volatility:** {initial_performance[1]:.2f}")
    st.markdown(f"**Sharpe Ratio:** {initial_performance[2]:.2f}")
    
    st.markdown(f"### Optimized Portfolio Performance ({criterion_name}):")
    st.markdown(f"**Return:** {optimized_performance[0]:.2f}")
    st.markdown(f"**Volatility:** {optimized_performance[1]:.2f}")
    st.markdown(f"**Sharpe Ratio:** {optimized_performance[2]:.2f}")
    
    initial_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Initial Weight (%)': initial_guess * 100
    })
    st.markdown("### Initial Weights:")
    st.table(initial_weights_df)
    
    optimized_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Optimized Weight (%)': optimized_weights * 100
    })
    st.markdown("### Optimized Weights:")
    st.table(optimized_weights_df)

# Function to fetch stock prices for the ticker tape
def fetch_stock_data(tickers):
    quote_table = yf.Tickers(tickers).history(period="1d")
    prices = quote_table['Close'].iloc[-1].values.flatten()
    previous_closes = quote_table['Close'].iloc[-2].values.flatten()
    valid_data = {tickers[i]: (prices[i], previous_closes[i]) for i in range(len(tickers)) if not np.isnan(prices[i]) and not np.isnan(previous_closes[i])}
    return valid_data

# Function to create ticker tape HTML
def create_ticker_tape(data):
    html = '<div style="background-color:black;color:white;overflow:hidden;width:100%;"><marquee behavior="scroll" direction="left" scrollamount="5">'
    for ticker, (price, prev_close) in data.items():
        color = 'green' if price > prev_close else 'red'
        html += f'<span style="margin-right:50px;color:{color};">{ticker}: ${price:.2f}</span>'
    html += '</marquee></div>'
    return html

st.title("Stock Portfolio Optimization App")
st.subheader("Akaash Mitsakakis-Nath")
st.markdown(""" This app performs portfolio optimization using different optimization techniques. Enter your Stock tickers, initial allocations, historical data range, and finally the desired optimization criterion to get started.""")
st.markdown("""
### Optimization Techniques
- **Sharpe Ratio**: Measures the performance of the portfolio compared to a risk-free asset, after adjusting for its risk. The higher the Sharpe ratio, the better the portfolio's risk-adjusted performance.
- **CVaR (Conditional Value at Risk)**: Estimates the expected loss of an investment in the worst-case scenario beyond the VaR (Value at Risk) threshold. It helps in managing extreme risks.
- **Sortino Ratio**: Similar to the Sharpe ratio but only considers downside risk (negative returns). It is useful for investors who are primarily concerned with the negative variability of returns.
- **Volatility**: Represents the degree of variation of trading prices. Lower volatility is often preferred as it indicates a more stable investment.
""")

default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

with st.expander("Enter your portfolio details:"):
    tickers = []
    allocations = []
    num_tickers = st.number_input("Number of tickers:", min_value=1, max_value=10, value=5, step=1)
    for i in range(num_tickers):
        col1, col2 = st.columns(2)
        ticker = col1.text_input(f"Ticker {i+1}:", default_tickers[i] if i < len(default_tickers) else '')
        if i < min(len(default_tickers), num_tickers):
            allocation = col2.number_input(f"Allocation for Ticker {i+1} (%):", value=float(100/min(len(default_tickers), num_tickers)), format='%.2f', step=0.01)
        else:
            allocation = col2.number_input(f"Allocation for Ticker {i+1} (%):", value=0.0, format='%.2f', step=0.01)
        allocations.append(allocation / 100)
        tickers.append(ticker)

if not math.isclose(sum(allocations), 1.0, rel_tol=1e-5):
    st.error("The sum of allocations must be approximately 100%.")
else:
    with st.expander("Enter the date range and optimization criterion:"):
        start_date = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=365))
        end_date = st.date_input("End date", datetime.date.today())
        optimization_criterion = st.selectbox("Optimization criterion:", ['sharpe', 'cvar', 'sortino', 'volatility'])

if st.button("Optimize Portfolio"):
    if start_date and end_date:
        try:
            data = get_stock_data(tickers, start_date, end_date)
            
            initial_guess, optimized_weights, returns, criterion_name = optimize_portfolio(tickers, allocations, start_date, end_date, optimization_criterion)
            
            st.markdown("---")
            st.header("Portfolio Performance")
            display_results(initial_guess, optimized_weights, returns, criterion_name)

            st.markdown("---")
            st.header("Efficient Frontier")
            with st.spinner('Calculating and plotting the efficient frontier...'):
                plot_efficient_frontier(tickers, returns, optimized_weights, criterion_name, optimization_criterion)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An error occurred during portfolio optimization: {str(e)}")

# Fetch and display stock ticker tape for the largest 20 tickers
top_20_tickers = get_top_20_tickers()
data = fetch_stock_data(top_20_tickers)
ticker_tape_html = create_ticker_tape(data)
st.components.v1.html(ticker_tape_html, height=50)

