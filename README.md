<h1 align="center">Stock Portfolio Optimization App</h1>

## Project Description

This project is a Stock Portfolio Optimization Calculator utilizing various optimization techniques to maximize portfolio performance. It includes graphical visualization of the efficient frontier and portfolio metrics. It is a web-hosted interactive application, hosted using [Streamlit's](https://streamlit.io) functionality.

The project can be found here: [Link to Project](https://stockportfoliooptimizationapp.streamlit.app)

## Portfolio Optimization

The **Portfolio Optimization** process aims to allocate assets in a way that maximizes returns and minimizes risk based on specific criteria. This application supports optimization using:

- **Sharpe Ratio**: Measures the performance of the portfolio compared to a risk-free asset, adjusted for risk.
- **CVaR (Conditional Value at Risk)**: Estimates the expected loss of an investment in the worst-case scenario.
- **Sortino Ratio**: Similar to the Sharpe ratio but only considers downside risk (negative returns).
- **Volatility**: Represents the degree of variation in trading prices, indicating stability.

## Portfolio Optimization Assumptions

Portfolio Optimization Model Assumptions:

- Historical Returns as Predictors: The model assumes that historical returns are indicative of future performance, using past stock data to estimate future returns.
- Normal Distribution of Returns: The model assumes that asset returns follow a normal distribution, which simplifies the calculation of portfolio performance metrics.
- No Transaction Costs: The model assumes there are no transaction costs associated with buying or selling assets, which could otherwise impact the optimization results.
- Full Investment: The model assumes that all available capital is fully invested in the portfolio, with no cash reserves.
- Fixed Time Horizon: The model assumes a fixed investment period, as specified by the start and end dates provided.

## The Efficient Frontier

The **Efficient Frontier** represents the set of optimal portfolios that offer the highest expected return for a defined level of risk. It is a key concept in Modern Portfolio Theory. This application leverages **multiprocessing** to efficiently calculate and plot the efficient frontier.

![Efficient Frontier](efficient-frontier.png)

## 📝 License

© 2024 [Akaash Mitsakakis-Nath](https://github.com/amitsakakis).<br />
This project is MIT licensed.

$$
$$
