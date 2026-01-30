import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import datetime as dt

pio.templates.default = "plotly_dark"

# Cache data
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_stock_model.pkl')
    except:
        return None

model = load_model()
if model is None:
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
    .select-row  { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }
    .select-row select { flex: 1; }
    .select-row button { flex: 0 0 auto; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 NSE Smart Predictor")
st.caption("XGBoost • Multi-stock trained • 5-day direction forecast • Educational only")

# Select + Button in one clean row
st.markdown('<div class="select-row">', unsafe_allow_html=True)
col_select, col_btn = st.columns([5, 2])
with col_select:
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

with col_btn:
    analyze_clicked = st.button("Analyze & Predict", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
if analyze_clicked:
    with st.spinner(f"Loading {selected}..."):
        df = fetch_stock_data(ticker)
        if df.empty or len(df) < 200:
            st.error("Not enough data.")
        else:
            # Today's prices
            last = df.iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else last['Close']
            change_pct = (last['Close'] - prev_close) / prev_close * 100

            st.markdown(f"""
                <div class="price-bar">
                    Open: ₹{last['Open']:.2f}  |  High: ₹{last['High']:.2f}  |  Low: ₹{last['Low']:.2f}  
                    |  Close: ₹{last['Close']:.2f}  |  Change: {'+' if change_pct >= 0 else ''}{change_pct:.2f}%
                </div>
            """, unsafe_allow_html=True)

            # Features
            df['SMA_Ratio'] = df['Close'].rolling(20).mean() / df['Close'].rolling(200).mean()
            df['EMA_9_21_Gap'] = (df['Close'].ewm(span=9).mean() - df['Close'].ewm(span=21).mean()) / df['Close']
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            latest = df[['SMA_Ratio', 'EMA_9_21_Gap', 'RSI']].iloc[-1:].dropna()

            if latest.empty:
                st.warning("Not enough history.")
            else:
                prob_up = model.predict_proba(latest)[0][1]

                if prob_up >= 0.70:
                    sig_class = "strong-buy"
                    sig_text = "STRONG BUY"
                elif prob_up >= 0.60:
                    sig_class = "buy"
                    sig_text = "BUY"
                elif prob_up >= 0.40:
                    sig_class = "hold"
                    sig_text = "HOLD"
                elif prob_up >= 0.30:
                    sig_class = "sell"
                    sig_text = "SELL"
                else:
                    sig_class = "strong-sell"
                    sig_text = "STRONG SELL"

                confidence = f"{max(prob_up, 1 - prob_up) * 100:.1f}%"

                st.markdown(f'<div class="big-signal {sig_class}">{sig_text}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">Confidence: {confidence}</div>', unsafe_allow_html=True)

            # Chart – fixed scaling + subtle future zone
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index[-150:],
                open=df['Open'][-150:],
                high=df['High'][-150:],
                low=df['Low'][-150:],
                close=df['Close'][-150:],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))

            # Future shaded zone (much tighter range so it doesn't squash y-axis)
            last_close = last['Close']
            future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)

            # Zone height = 5–10% of current price, opacity based on confidence
            range_pct = 0.08 if prob_up > 0.7 or prob_up < 0.3 else 0.05
            opacity = 0.4 if abs(prob_up - 0.5) > 0.15 else 0.2

            if prob_up > 0.5:
                color = f'rgba(0, 255, 157, {opacity})'  # green
                y_center = last_close * 1.02
            elif prob_up < 0.5:
                color = f'rgba(255, 82, 82, {opacity})'  # red
                y_center = last_close * 0.98
            else:
                color = f'rgba(176, 190, 197, {opacity})'  # gray
                y_center = last_close

            y_high = y_center * (1 + range_pct)
            y_low = y_center * (1 - range_pct)

            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=[y_high] * len(future_dates) + [y_low] * len(future_dates),
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0)'),
                name='Prediction Zone',
                showlegend=True
            ))

            fig.update_layout(
                title=f"{selected} – Recent 150 Days + Outlook Zone",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=600,
                xaxis_rangeslider_visible=True,
                hovermode="x unified",
                template="plotly_dark",
                yaxis=dict(autorange=True, gridcolor='#444')  # force proper auto-range
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("About this forecast"):
                st.markdown("""
                - Model trained on 20 major NSE stocks  
                - Uses relative features (SMA ratio, EMA gap, RSI)  
                - Predicts if price likely higher in 5 trading days  
                - Shaded zone shows confidence-based direction & strength  
                - **Educational only** — not trading advice
                """)

else:
    st.info("Choose a stock and click Analyze & Predict.")
