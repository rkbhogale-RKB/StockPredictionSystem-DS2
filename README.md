# Stock Prediction System using XGBoost

A **machine learning-based stock price direction predictor** for the top 20 NSE (National Stock Exchange of India) stocks. The model predicts whether a stock's closing price will go **up** or **not** in the next 5 trading days.

The project includes:
- Data collection & feature engineering from real market data (via yfinance)
- Training an XGBoost classifier on multiple stocks (universal model)
- A **live web application** built with **Streamlit**

---

## 🚀 Live Demo

The app is already deployed on **Streamlit Cloud**:

👉 **[Open the Stock Prediction App](https://your-streamlit-app-url.streamlit.app)**  
*(Replace this line with your actual deployed Streamlit URL)*

---

## 📊 Features

- Predicts **5-day price direction** (Up / Not Up) for top NSE stocks
- Uses **scale-agnostic technical indicators**:
  - SMA Ratio (20-day vs 200-day)
  - EMA 9-21 Gap (normalized)
  - RSI (14-day)
- Trained on **5 years** of historical data from 20 major Indian stocks
- Fast inference with **XGBoost**
- Clean and interactive Streamlit dashboard

---

## 🛠️ Tech Stack

- **Python**
- **yfinance** – Real-time & historical stock data
- **pandas / numpy** – Data processing
- **XGBoost** – Machine learning model
- **Streamlit** – Web application
- **joblib** – Model serialization

---

## 📁 Project Structure

