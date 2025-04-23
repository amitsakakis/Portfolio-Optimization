import streamlit as st
import yfinance as yf
import pandas as pd

def get_top_stocks():
    # List of top  stock tickers - can add to list whenever its just for visual
    tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META',
        'TSLA', 'NVDA', 'BRK-B', 'JNJ', 'WMT',
        'JPM', 'V', 'PG', 'DIS', 'MA',
        'PYPL', 'NFLX', 'INTC', 'ADBE', 'CSCO'
    ]

    # Fetch current data for the tickers
    data = yf.download(tickers, period='5d', interval='1h')
    
    # Handle NaN values by forward filling
    data = data.ffill().bfill()
    
    current_data = data['Adj Close'].iloc[-1].reset_index()
    previous_close_data = data['Adj Close'].iloc[-2].reset_index()
    
    current_data.columns = ['Ticker', 'Price']
    previous_close_data.columns = ['Ticker', 'Previous Price']

    combined_data = pd.merge(current_data, previous_close_data, on='Ticker')
    combined_data['Change'] = combined_data['Price'] - combined_data['Previous Price']

    return combined_data

def display_ticker_tape():
    st.markdown(
        """
        <style>
        .ticker-tape {
            white-space: nowrap;
            overflow: hidden;
            box-sizing: border-box;
            padding: 10px 0;
        }
        .ticker-tape div {
            display: inline-block;
            padding-left: 100%;
            animation: ticker 60s linear infinite;
        }
        @keyframes ticker {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        .ticker-tape .up { color: green; }
        .ticker-tape .down { color: red; }
        </style>
        """,
        unsafe_allow_html=True
    )

    stock_data = get_top_stocks()
    stock_data['Display'] = stock_data.apply(
        lambda row: f"<span class='{ 'up' if row['Change'] >= 0 else 'down' }'>{row['Ticker']}: ${row['Price']:.2f} ({row['Change']:+.2f})</span>", axis=1
    )

    ticker_html = "<div class='ticker-tape'><div>" + " &nbsp; | &nbsp; ".join(stock_data['Display']) + "</div></div>"
    st.markdown(ticker_html, unsafe_allow_html=True)
