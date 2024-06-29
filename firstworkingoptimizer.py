##NOTE: THIS IS JUST AN OLDER VERSION OF THE CODE HERE FOR SHOW AND BECAUSE WHY NOT. THE APP DOESN'T USE THIS

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st

# Function to get stock data
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Function to calculate portfolio performance
def portfolio_performance(weights, returns):
    portfolio_return = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function to minimize negative Sharpe ratio
def negative_sharpe_ratio(weights, returns):
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(weights, returns)
    return -sharpe_ratio

# Main function to perform optimization
def optimize_portfolio(tickers, initial_allocations, start_date, end_date):
    # Get stock data
    data = get_stock_data(tickers, start_date, end_date)
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    
    # Initial guess
    initial_guess = np.array(initial_allocations)
    
    # Optimize portfolio
    optimized_results = minimize(negative_sharpe_ratio, initial_guess, args=(returns,),
                                 method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimized_weights = optimized_results.x
    
    return initial_guess, optimized_weights, returns

def plot_efficient_frontier(returns):
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(weights, returns)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        weights_record.append(weights)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    ax.scatter(sdp, rp, marker='*', color='r', s=100, label='Maximum Sharpe Ratio')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend(loc='upper left')
    st.pyplot(fig)

# Function to display results
def display_results(initial_guess, optimized_weights, returns):
    initial_performance = portfolio_performance(initial_guess, returns)
    optimized_performance = portfolio_performance(optimized_weights, returns)
    
    st.markdown("### Initial Portfolio Performance:")
    st.markdown(f"**Return:** {initial_performance[0]:.2f}")
    st.markdown(f"**Volatility:** {initial_performance[1]:.2f}")
    st.markdown(f"**Sharpe Ratio:** {initial_performance[2]:.2f}")
    
    st.markdown("### Optimized Portfolio Performance:")
    st.markdown(f"**Return:** {optimized_performance[0]:.2f}")
    st.markdown(f"**Volatility:** {optimized_performance[1]:.2f}")
    st.markdown(f"**Sharpe Ratio:** {optimized_performance[2]:.2f}")
    
    initial_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Initial Weight (%)': initial_guess * 100  # Convert to percentage
    })
    st.markdown("### Initial Weights:")
    st.table(initial_weights_df)
    
    optimized_weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Optimized Weight (%)': optimized_weights * 100  # Convert to percentage
    })
    st.markdown("### Optimized Weights:")
    st.table(optimized_weights_df)

# User inputs
default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
default_allocations = [0.2, 0.2, 0.2, 0.2, 0.2]  # Example initial allocations

# Get user input for tickers and allocations
user_tickers = st.text_input("Enter your tickers, separated by commas:", ', '.join(default_tickers))
user_allocations = st.text_input("Enter your initial allocations, separated by commas:", ', '.join(map(str, default_allocations)))

# Convert user input to lists
tickers = [ticker.strip() for ticker in user_tickers.split(',')]
initial_allocations = [float(allocation.strip()) for allocation in user_allocations.split(',')]

# Check if the sum of allocations is 1
if sum(initial_allocations) != 1.0:
    st.error("The sum of allocations must be 1.")
else:
    start_date = st.text_input("Enter start date(ex:'2023-01-01'): ")
    end_date = st.text_input("Enter end date(ex: 2024-01-01'): ")

    # Optimize portfolio and display results
    if start_date and end_date:
        initial_guess, optimized_weights, returns = optimize_portfolio(tickers, initial_allocations, start_date, end_date)
        display_results(initial_guess, optimized_weights, returns)

        # Plot the efficient frontier
        plot_efficient_frontier(returns)





