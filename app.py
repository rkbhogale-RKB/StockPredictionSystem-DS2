import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import datetime as dt

pio.templates.default = "plotly_dark"

# 1. Cache data
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# 2. Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_stock_model.pkl')
    except:
        return None

model = load_model()
if model is None:
    st.error("Model file 'xgboost_stock_model.pkl' not found!")
    st.stop()

# ────────────────────────────────────────────────
st.set_page_config(page_title="NSE Smart Predictor", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .big-signal { font-size: 3.8rem; font-weight: bold; text-align: center; margin: 20px 0; }
    .strong-buy { color: #00ff9d; }
    .buy { color: #40c4ff; }
    .hold { color: #b0bec5; }
    .sell { color: #ffca28; }
    .strong-sell { color: #ff5252; }
    .confidence { font-size: 1.8rem; color: #aaa; text-align: center; }
    .price-bar { background: #1e1e1e; padding: 16px; border-radius: 8px; margin: 20px 0; font-size: 1.3rem; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 NSE Smart Predictor")
st.caption("XGBoost • Multi-stock trained • 5-day direction forecast • Educational only")

# Stock selector
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

selected = st.selectbox("Select Stock", list(stocks_dict.keys()), index=2)
ticker = stocks_dict[selected]

# ────────────────────────────────────────────────
with st.spinner(f"Analyzing {selected}..."):
    df = fetch_stock_data(ticker)
   
    if df.empty or len(df) < 200:
        st.error("Not enough data to generate features.")
    else:
        # Latest Price Info
        last = df.iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change_pct = (last['Close'] - prev_close) / prev_close * 100
        st.markdown(f"""
            <div class="price-bar">
                Open: ₹{last['Open']:.2f} | High: ₹{last['High']:.2f} | Low: ₹{last['Low']:.2f}
                | Close: ₹{last['Close']:.2f} | Change: {'+' if change_pct >= 0 else ''}{change_pct:.2f}%
            </div>
        """, unsafe_allow_html=True)

        # Feature Engineering
        df['SMA_Ratio'] = df['Close'].rolling(20).mean() / df['Close'].rolling(200).mean()
        df['EMA_9_21_Gap'] = (df['Close'].ewm(span=9).mean() - df['Close'].ewm(span=21).mean()) / df['Close']
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        latest_features = df[['SMA_Ratio', 'EMA_9_21_Gap', 'RSI']].iloc[-1:].dropna()
        
        if not latest_features.empty:
            prob_up = model.predict_proba(latest_features)[0][1]

            # Signal Logic
            if prob_up >= 0.70: sig_class, sig_text = "strong-buy", "STRONG BUY"
            elif prob_up >= 0.60: sig_class, sig_text = "buy", "BUY"
            elif prob_up >= 0.40: sig_class, sig_text = "hold", "HOLD"
            elif prob_up >= 0.30: sig_class, sig_text = "sell", "SELL"
            else: sig_class, sig_text = "strong-sell", "STRONG SELL"

            st.markdown(f'<div class="big-signal {sig_class}">{sig_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Model Confidence: {max(prob_up, 1 - prob_up) * 100:.1f}%</div>', unsafe_allow_html=True)

            # 5-Day Outlook Trend
            st.subheader("🗓️ 5-Day Outlook Trend")
            future_dates = pd.bdate_range(start=dt.date.today(), periods=6)  # Today + Next 5 biz days
           
            trend_val = "UP 📈" if prob_up > 0.55 else ("DOWN 📉" if prob_up < 0.45 else "SIDEWAYS ↔️")
           
            trend_data = {
                "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
                "Day": [d.strftime('%A') for d in future_dates],
                "Forecasted Trend": [trend_val] * 6
            }
            trend_df = pd.DataFrame(trend_data)
            st.table(trend_df)

            # ─── IMPROVED CHART ───
            st.subheader(f"{selected} Price & Trend Lines (Last 200 Days)")

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index[-200:],
                open=df['Open'][-200:],
                high=df['High'][-200:],
                low=df['Low'][-200:],
                close=df['Close'][-200:],
                name='Price',
                increasing_line_color='#00cc96',
                decreasing_line_color='#ff4d4d'
            ))

            # Moving Averages
            fig.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['Close'].rolling(20).mean()[-200:],
                name='20-day SMA',
                line=dict(color='orange', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['Close'].rolling(200).mean()[-200:],
                name='200-day SMA',
                line=dict(color='purple', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['Close'].ewm(span=9).mean()[-200:],
                name='9-day EMA',
                line=dict(color='#4fc3f7', width=1.5)
            ))

            fig.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['Close'].ewm(span=21).mean()[-200:],
                name='21-day EMA',
                line=dict(color='#ffca28', width=1.5)
            ))

            fig.update_layout(
                title=f"{selected} – Recent Price Action & Key Moving Averages",
                height=600,
                template="plotly_dark",
                xaxis_rangeslider_visible=True,
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built by Rohit K.Bhogale • Educational project • Data via yfinance")
