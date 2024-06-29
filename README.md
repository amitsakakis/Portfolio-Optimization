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

The portfolio optimization model makes several assumptions:

- Historical returns are indicative of future performance.
- Asset returns are normally distributed.
- The risk-free rate is known and constant.
- There are no transaction costs.

## Optimization Criteria Formulas

### Sharpe Ratio

The Sharpe Ratio is defined as:
\[
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
\]
Where:

- \( R_p \) is the portfolio return
- \( R_f \) is the risk-free rate
- \( \sigma_p \) is the portfolio volatility

### CVaR (Conditional Value at Risk)

The CVaR is defined as the expected loss beyond the Value at Risk (VaR) threshold:
\[
\text{CVaR} = \mathbb{E}[L | L > \text{VaR}]
\]

### Sortino Ratio

The Sortino Ratio is defined as:
\[
\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d}
\]
Where:

- \( \sigma_d \) is the downside deviation

### Volatility

Volatility is defined as the standard deviation of portfolio returns:
\[
\sigma_p
\]

## The Efficient Frontier

The **Efficient Frontier** represents the set of optimal portfolios that offer the highest expected return for a defined level of risk. It is a key concept in Modern Portfolio Theory.

![Efficient Frontier](efficient-frontier.png)

## The Ticker Tape

The application includes a real-time ticker tape displaying the current prices and changes of top stocks. This feature is implemented using yfinance and Streamlit.

## 📝 License

© 2024 [Akaash Mitsakakis-Nath](https://github.com/amitsakakis).<br />
This project is MIT licensed.
