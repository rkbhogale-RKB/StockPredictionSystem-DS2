import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import datetime as dt
import time

pio.templates.default = "plotly_white"  # Fix blank/black graphs

# Cache data fetch
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# Load XGBoost model (upload 'xgboost_stock_model.pkl' to repo)
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_stock_model.pkl')
    except Exception as e:
        st.error(f"Model load failed: {e}. Upload xgboost_stock_model.pkl")
        return None

model = load_model()
if model is None:
    st.stop()

# Expanded stock list
stocks_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "SBI": "SBIN.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "HUL": "HINDUNILVR.NS",
    "L&T": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Tech": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "NTPC": "NTPC.NS",
    "Nifty 50 Index": "^NSEI"
}

st.set_page_config(page_title="NSE Stock Predictor - XGBoost", layout="wide")
st.title("📈 NSE Stock Direction Predictor (XGBoost)")
st.caption("Universal model trained on 20 top NSE stocks • Predicts Up/Down in 5 days • Educational only")

selected = st.selectbox("Select Stock", list(stocks_dict.keys()))
ticker = stocks_dict[selected]

if st.button("Analyze & Predict"):
    with st.spinner("Fetching data & predicting..."):
        df = fetch_stock_data(ticker)
        if df.empty or len(df) < 200:
            st.error("Not enough data. Try another stock.")
        else:
            # Current price info
            latest_close = df['Close'].iloc[-1]
            change = df['Close'].pct_change().iloc[-1] * 100
            st.metric("Latest Close", f"₹{latest_close:,.2f}", f"{change:.2f}%")

            # Feature engineering (same as training)
            df['SMA_Ratio'] = df['Close'].rolling(20).mean() / df['Close'].rolling(200).mean()
            df['EMA_9_21_Gap'] = (df['Close'].ewm(span=9).mean() - df['Close'].ewm(span=21).mean()) / df['Close']
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            latest_features = df[['SMA_Ratio', 'EMA_9_21_Gap', 'RSI']].iloc[-1:].dropna()

            if latest_features.empty:
                st.warning("Not enough data for features (need 200+ days).")
            else:
                prob_up = model.predict_proba(latest_features)[0][1]  # prob of class 1 (Up)
                signal = "UP" if prob_up > 0.6 else "DOWN" if prob_up < 0.4 else "NEUTRAL"
                confidence = f"{prob_up*100:.1f}%" if prob_up >= 0.5 else f"{(1-prob_up)*100:.1f}%"

                st.subheader(f"5-Day Direction Prediction: **{signal}** ({confidence} confidence)")

                # Simple chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index[-150:], y=df['Close'][-150:], name='Close', line=dict(color='blue')))
                fig.update_layout(title=f"{selected} Recent Price", height=400, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                st.info("Model trained on multiple stocks using SMA ratio, EMA gap, RSI → predicts if price higher in 5 days.")
