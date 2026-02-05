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
    .strong-buy  { color: #00ff9d; }
    .buy         { color: #40c4ff; }
    .hold        { color: #b0bec5; }
    .sell        { color: #ffca28; }
    .strong-sell { color: #ff5252; }
    .confidence  { font-size: 1.8rem; color: #aaa; text-align: center; }
    .price-bar   { background: #1e1e1e; padding: 16px; border-radius: 8px; margin: 20px 0; font-size: 1.3rem; text-align: center; }
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
                Open: ₹{last['Open']:.2f}  |  High: ₹{last['High']:.2f}  |  Low: ₹{last['Low']:.2f}  
                |  Close: ₹{last['Close']:.2f}  |  Change: {'+' if change_pct >= 0 else ''}{change_pct:.2f}%
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

            # ─── NEW ADDITION: 5-DAY TREND TABLE ───
            st.subheader("🗓️ 5-Day Outlook Trend")
            future_dates = pd.bdate_range(start=dt.date.today(), periods=6) # Today + Next 5 biz days
            
            trend_val = "UP 📈" if prob_up > 0.55 else ("DOWN 📉" if prob_up < 0.45 else "SIDEWAYS ↔️")
            
            trend_data = {
                "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
                "Day": [d.strftime('%A') for d in future_dates],
                "Forecasted Trend": [trend_val] * 6
            }
            trend_df = pd.DataFrame(trend_data)
            st.table(trend_df)

            # ─── CHART ───
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index[-100:], open=df['Open'][-100:], high=df['High'][-100:],
                low=df['Low'][-100:], close=df['Close'][-100:], name='Price'
            ))
            fig.update_layout(title=f"{selected} Price Action", height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # ─── NEW ADDITION: NEWS TICKER ───
            # ─── FIXED NEWS TICKER ───
st.subheader(f"📰 Recent News: {selected}")
ticker_obj = yf.Ticker(ticker)

try:
    news_list = ticker_obj.news
    # Yahoo news can be tricky; let's ensure it's a list and not empty
    if news_list and len(news_list) > 0:
        for article in news_list[:5]: # Show top 5
            col1, col2 = st.columns([1, 4])
            
            # Safely get title and link using .get() to avoid KeyError
            title = article.get('title', 'No Title Available')
            link = article.get('link', '#')
            publisher = article.get('publisher', 'Unknown Source')
            
            with col1:
                # Some articles don't have thumbnails; check carefully
                thumb = article.get('thumbnail', {})
                if thumb and 'resolutions' in thumb:
                    st.image(thumb['resolutions'][0]['url'], width=150)
                else:
                    st.write("📷 No Image")
                    
            with col2:
                st.markdown(f"**[{title}]({link})**")
                st.caption(f"Source: {publisher}")
    else:
        st.info("No recent news found for this stock.")
except Exception as e:
    st.warning("Could not load news at this moment.")

st.markdown("---")
st.caption("Built by Rohit K.Bhogale • Educational project • Data via yfinance")
