import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(page_title="BTC Price Forecasting", layout="wide")
st.title("💹 Bitcoin Price Forecasting (Upload CSV)")

# ------------------------------
# Sidebar: Forecast Days
# ------------------------------
forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 7)

# ------------------------------
# Upload CSV
# ------------------------------
st.sidebar.subheader("Upload Kaggle Cryptocurrency CSV")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if csv_file is None:
    st.warning("Please upload the cryptocurrency CSV to continue.")
    st.stop()

# ------------------------------
# Load and Filter BTC Data
# ------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["Date"])
    btc = df[df["Symbol"] == "BTC"].sort_values("Date").reset_index(drop=True)
    return btc

df_btc = load_data(csv_file)

if df_btc.empty:
    st.error("No BTC data found in the CSV.")
    st.stop()

st.subheader("📊 Historical BTC Data")
st.dataframe(df_btc.tail())

# ------------------------------
# Plot Historical Closing Price
# ------------------------------
st.subheader("📈 Historical BTC Closing Price")
fig, ax = plt.subplots()
ax.plot(df_btc['Date'], df_btc['Close'], label="Close Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# ------------------------------
# Upload Model & Scaler
# ------------------------------
st.subheader("🧠 Upload Pre-Trained Model (.keras/.h5) and Scaler (.pkl)")
model_file = st.file_uploader("Model (.keras or .h5)", type=['keras', 'h5'])
scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'])

if model_file is None or scaler_file is None:
    st.warning("Please upload both model and scaler files.")
    st.stop()

# Save uploaded files temporarily
with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_model:
    tmp_model.write(model_file.getbuffer())
    tmp_model_path = tmp_model.name

with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_scaler:
    tmp_scaler.write(scaler_file.getbuffer())
    tmp_scaler_path = tmp_scaler.name

# Load model and scaler
try:
    model = load_model(tmp_model_path, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

with open(tmp_scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.success("✅ Model and scaler loaded successfully!")

# ------------------------------
# Data Preparation: Log + Scaling
# ------------------------------
df_btc['Log_Close'] = np.log1p(df_btc['Close'])
scaled_data = scaler.transform(df_btc[['Log_Close']])

# Train-Test Split (80-20)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_rolling(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X_train, y_train = create_rolling(train_data, look_back)
X_test, y_test = create_rolling(test_data, look_back)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ------------------------------
# Rolling Forecast
# ------------------------------
st.subheader("🔮 Forecasted BTC Prices")
last_60 = scaled_data[-look_back:]
forecast_input = last_60.reshape(1, look_back, 1)
forecast_scaled = []

for _ in range(forecast_days):
    pred = model.predict(forecast_input, verbose=0)
    forecast_scaled.append(pred[0,0])
    forecast_input = np.append(forecast_input[:,1:,:], [[pred]], axis=1)

forecast_scaled = np.array(forecast_scaled).reshape(-1,1)
forecast_log = scaler.inverse_transform(forecast_scaled)
forecast_prices = np.expm1(forecast_log)

future_dates = pd.date_range(df_btc['Date'].iloc[-1], periods=forecast_days+1, freq='D')[1:]
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast_prices.flatten()})

st.dataframe(forecast_df.style.format({'Predicted Price':'{:.2f}'}))

# ------------------------------
# Plot Historical + Forecast
# ------------------------------
fig2, ax2 = plt.subplots()
ax2.plot(df_btc['Date'], df_btc['Close'], label="Historical", color="blue")
ax2.plot(forecast_df['Date'], forecast_df['Predicted Price'], label="Predicted", color="orange")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

st.success(f"✅ Forecast complete for {forecast_days} days of BTC prices.")
