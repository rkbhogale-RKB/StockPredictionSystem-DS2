import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import datetime as dt
import time

# Use light theme for better visibility (can switch to 'plotly_dark' later if you prefer dark mode)
pio.templates.default = "plotly_white"

# ────────────────────────────────────────────────
# Cache data fetch
@st.cache_data(ttl=1800)  # 30 minutes
def fetch_stock_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# Load XGBoost model
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_stock_model.pkl')
    except Exception as e:
        st.error(f"Model load failed: {e}. Make sure 'xgboost_stock_model.pkl' is in repo root.")
        return None

model = load_model()
if model is None:
    st.stop()

# ────────────────────────────────────────────────
# UI Setup
st.set_page_config(page_title="NSE Smart Predictor", layout="wide", page_icon="📈")

# Custom CSS for nicer look
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card { 
        background: white; 
        border-radius: 12px; 
        padding: 20px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .signal-strong-buy { color: #28a745; font-weight: bold; font-size: 1.8rem; }
    .signal-buy { color: #17a2b8; font-weight: bold; font-size: 1.6rem; }
    .signal-hold { color: #6c757d; font-weight: bold; font-size: 1.6rem; }
    .signal-sell { color: #ffc107; font-weight: bold; font-size: 1.6rem; }
    .signal-strong-sell { color: #dc3545; font-weight: bold; font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 NSE Smart Stock Predictor")
st.caption("XGBoost • Trained on 20 top NSE stocks • 5-day direction forecast • Educational only")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
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
    selected = st.selectbox("Choose Stock", list(stocks_dict.keys()), index=2)
    ticker = stocks_dict[selected]

    st.markdown("---")
    if st.button("Analyze & Predict", type="primary", use_container_width=True):
        pass  # just to trigger rerun

# ────────────────────────────────────────────────
if st.button("Analyze & Predict", type="primary", key="main_predict"):
    with st.spinner(f"Loading {selected} data & running prediction..."):
        df = fetch_stock_data(ticker)
        if df.empty or len(df) < 200:
            st.error("Not enough data — try another stock or wait.")
        else:
            # Latest price
            latest_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            change_pct = (latest_close - prev_close) / prev_close * 100

            # Signal card
            col_signal, col_price = st.columns([2, 1])

            with col_signal:
                # Feature engineering (same as training)
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
                    
                    if prob_up >= 0.70:
                        signal_class = "signal-strong-buy"
                        signal_text = "STRONG BUY"
                        icon = "🚀"
                    elif prob_up >= 0.60:
                        signal_class = "signal-buy"
                        signal_text = "BUY"
                        icon = "📈"
                    elif prob_up >= 0.40:
                        signal_class = "signal-hold"
                        signal_text = "HOLD"
                        icon = "⚖️"
                    elif prob_up >= 0.30:
                        signal_class = "signal-sell"
                        signal_text = "SELL"
                        icon = "📉"
                    else:
                        signal_class = "signal-strong-sell"
                        signal_text = "STRONG SELL"
                        icon = "⚠️"

                    confidence = f"{max(prob_up, 1-prob_up)*100:.1f}%"

                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>5-Day Outlook</h3>
                            <p class="{signal_class}">{icon} {signal_text}</p>
                            <p style="font-size:1.3rem; color:#555;">Confidence: {confidence}</p>
                        </div>
                    """, unsafe_allow_html=True)

            with col_price:
                st.metric(
                    label=f"Latest {selected}",
                    value=f"₹{latest_close:,.2f}",
                    delta=f"{change_pct:.2f}%",
                    delta_color="normal"
                )

            # Chart – Candlestick + Prediction (if you want to predict price, we can extend)
            st.subheader("Price Chart (Recent 150 days)")
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df.index[-150:],
                    open=df['Open'][-150:],
                    high=df['High'][-150:],
                    low=df['Low'][-150:],
                    close=df['Close'][-150:],
                    name='Price',
                    increasing_line_color='green', decreasing_line_color='red'
                )
            ])
            fig.update_layout(
                title=f"{selected} Recent Price Action",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                xaxis_rangeslider_visible=True,
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Extra info
            with st.expander("How this prediction works"):
                st.markdown("""
                - Model trained on 20 major NSE stocks using relative features (SMA ratio, EMA gap, RSI).
                - Predicts direction (UP/DOWN) for next 5 trading days.
                - Confidence reflects how certain the model is.
                - **Not financial advice** — for education only.
                """)

else:
    st.info("Choose a stock and click 'Analyze & Predict' to see the forecast.")
