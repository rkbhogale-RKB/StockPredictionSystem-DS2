import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import datetime as dt

# Dark theme + better visibility
pio.templates.default = "plotly_dark"

# ────────────────────────────────────────────────
# Cache data fetch
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# Load model
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

# ────────────────────────────────────────────────
st.set_page_config(page_title="NSE Smart Predictor", layout="wide", page_icon="📈")

# Minimal custom style – dark friendly
st.markdown("""
    <style>
    .big-signal { 
        font-size: 3.2rem; 
        font-weight: bold; 
        text-align: center; 
        margin: 20px 0;
    }
    .strong-buy  { color: #00ff9d; }
    .buy         { color: #40c4ff; }
    .hold        { color: #b0bec5; }
    .sell        { color: #ffca28; }
    .strong-sell { color: #ff5252; }
    .confidence  { font-size: 1.6rem; color: #aaa; }
    .price-metric { font-size: 1.5rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 NSE Smart Predictor")
st.caption("XGBoost • Multi-stock trained • 5-day direction forecast • Educational only")

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
    "Nifty 50 Index": "^NSEI"
}

col1, col2 = st.columns([3, 1])
with col1:
    selected = st.selectbox("Select Stock", list(stocks_dict.keys()), index=2)
with col2:
    if st.button("Analyze", type="primary", use_container_width=True):
        pass

ticker = stocks_dict[selected]

# ────────────────────────────────────────────────
if st.button("Analyze & Predict", type="primary", key="predict_btn"):
    with st.spinner(f"Loading {selected}..."):
        df = fetch_stock_data(ticker)
        if df.empty or len(df) < 200:
            st.error("Not enough data — try another stock.")
        else:
            # Latest day info
            last_row = df.iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_row['Close']
            change_pct = (last_row['Close'] - prev_close) / prev_close * 100

            # Show today's key prices in one line
            st.markdown(f"""
                <div style="display: flex; justify-content: space-around; background: #1e1e1e; padding: 16px; border-radius: 8px; margin-bottom: 24px;">
                    <div><strong>Open:</strong> ₹{last_row['Open']:.2f}</div>
                    <div><strong>High:</strong> ₹{last_row['High']:.2f}</div>
                    <div><strong>Low:</strong> ₹{last_row['Low']:.2f}</div>
                    <div><strong>Close:</strong> ₹{last_row['Close']:.2f}</div>
                    <div><strong>Change:</strong> {'+' if change_pct >= 0 else ''}{change_pct:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

            # Feature engineering
            df['SMA_Ratio'] = df['Close'].rolling(20).mean() / df['Close'].rolling(200).mean()
            df['EMA_9_21_Gap'] = (df['Close'].ewm(span=9).mean() - df['Close'].ewm(span=21).mean()) / df['Close']
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            latest = df[['SMA_Ratio', 'EMA_9_21_Gap', 'RSI']].iloc[-1:].dropna()

            if latest.empty:
                st.warning("Not enough history for features.")
            else:
                prob_up = model.predict_proba(latest)[0][1]

                # Signal logic
                if prob_up >= 0.70:
                    signal_class = "strong-buy"
                    signal_text = "STRONG BUY"
                elif prob_up >= 0.60:
                    signal_class = "buy"
                    signal_text = "BUY"
                elif prob_up >= 0.40:
                    signal_class = "hold"
                    signal_text = "HOLD"
                elif prob_up >= 0.30:
                    signal_class = "sell"
                    signal_text = "SELL"
                else:
                    signal_class = "strong-sell"
                    signal_text = "STRONG SELL"

                confidence = f"{max(prob_up, 1-prob_up)*100:.1f}%"

                # Big signal display
                st.markdown(f"""
                    <div class="big-signal {signal_class}">
                        {signal_text}
                    </div>
                    <div class="confidence" style="text-align:center;">
                        Confidence: {confidence}
                    </div>
                """, unsafe_allow_html=True)

            # Chart – Candlestick
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df.index[-150:],
                    open=df['Open'][-150:],
                    high=df['High'][-150:],
                    low=df['Low'][-150:],
                    close=df['Close'][-150:],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                )
            ])
            fig.update_layout(
                title=f"{selected} – Recent 150 Days",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=550,
                xaxis_rangeslider_visible=True,
                hovermode="x unified",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("About this prediction"):
                st.markdown("""
                - XGBoost model trained on 20 major NSE stocks  
                - Features: SMA ratio, EMA gap, RSI  
                - Predicts if price will be higher in 5 trading days  
                - **Not financial advice** – education & illustration only
                """)

else:
    st.info("Select a stock and click Analyze & Predict to see the forecast.")
