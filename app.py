import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(page_title="BTC Price Prediction", layout="wide")
st.title("💹 Bitcoin Price Prediction on Recent Data (2022-2025)")

# ------------------------------
# Fetch BTC Data (2014-2018)
# ------------------------------
@st.cache_data
def load_btc_data(ticker="BTC-USD", start="2022-01-01", end="2025-10-31"):
    btc = yf.Ticker(ticker)
    df = btc.history(start=start, end=end)
    if df.empty:
        return None
    df = df.reset_index()
    if 'Close' not in df.columns:
        if 'close' in df.columns:
            df.rename(columns={'close':'Close'}, inplace=True)
        else:
            return None
    df = df[['Date','Close']].dropna()
    return df

df_btc = load_btc_data()
if df_btc is None or df_btc.empty:
    st.error("No BTC data available for the specified period.")
    st.stop()

st.subheader("📊 BTC Historical Data (2014-2025)")
st.dataframe(df_btc.tail())

# ------------------------------
# Load LSTM Model and Scaler
# ------------------------------
st.subheader("🧠 Loading LSTM Model and Scaler from folder")
try:
    model = load_model("lstm_model_new.h5")
    st.success("✅ LSTM model loaded successfully!")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# ------------------------------
# Data Preparation: Log + Scale
# ------------------------------
df_btc['Log_Close'] = np.log(df_btc['Close'])
log_close = df_btc['Log_Close'].values.reshape(-1,1)
scaled_data = scaler.transform(log_close)

# ------------------------------
# Train-Test Split (80%-20%)
# ------------------------------
split_idx = int(len(scaled_data) * 0.2)
train_data = scaled_data[:split_idx]
test_data = scaled_data[split_idx:]

# ------------------------------
# Create Rolling Windows
# ------------------------------
def create_rolling(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X_train, y_train = create_rolling(train_data, look_back)
X_test, y_test = create_rolling(test_data, look_back)

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ------------------------------
# Predict on Test Set
# ------------------------------
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred_log = scaler.inverse_transform(y_pred_scaled)
y_pred_prices = np.exp(y_pred_log)

# Actual test prices
y_test_log = scaler.inverse_transform(y_test.reshape(-1,1))
y_test_prices = np.exp(y_test_log)

# Dates for test set
test_dates = df_btc['Date'].iloc[split_idx + look_back:].reset_index(drop=True)

# ------------------------------
# Display Test Predictions
# ------------------------------
forecast_df = pd.DataFrame({
    'Date': test_dates,
    'Actual Price': y_test_prices.flatten(),
    'Predicted Price': y_pred_prices.flatten()
})

st.subheader("📈 BTC Prediction vs Actual (Recent Data)")
st.dataframe(forecast_df.style.format({'Actual Price':'{:.2f}','Predicted Price':'{:.2f}'}))

# ------------------------------
# Plot Actual vs Predicted
# ------------------------------
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(test_dates, y_test_prices, label="Actual Price", color="blue")
ax.plot(test_dates, y_pred_prices, label="Predicted Price", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title("BTC Price Prediction on Test Set")
ax.legend()
st.pyplot(fig)

st.success("✅ Prediction complete on last 20% of BTC data.")


