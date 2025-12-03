import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

# ============================
# Load Model
# ============================
model = load_model('Bitcoin_Price_prediction_Model.keras')

st.header('Bitcoin Price Prediction Model')

# ============================
# Load Bitcoin Data
# ============================
st.subheader('Bitcoin Price Data')

data = yf.download('BTC-USD', '2015-01-01', '2023-11-30')

# Flatten MultiIndex columns (Fix for Streamlit)
data.columns = [col if not isinstance(col, tuple) else col[0] for col in data.columns]

data = data.reset_index()
st.write(data)

# ============================
# Line Chart (Close price)
# ============================
st.subheader('Bitcoin Close Price Chart')

chart_df = data[['Date', 'Close']].copy()
chart_df = chart_df.set_index('Date')

st.line_chart(chart_df)

# ============================
# Prepare Data for ML Model
# ============================
model_data = data[['Close']].copy()

train_data = model_data[:-100]
test_data = model_data[-200:]

scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

base_days = 100
x = []
y = []

for i in range(base_days, len(test_scaled)):
    x.append(test_scaled[i-base_days:i])
    y.append(test_scaled[i, 0])

x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape[0], x.shape[1], 1)

# ============================
# Predictions
# ============================
st.subheader('Predicted vs Original Prices')

pred = model.predict(x)
pred = scaler.inverse_transform(pred)
orig = scaler.inverse_transform(y.reshape(-1, 1))

pred_df = pd.DataFrame(pred, columns=['Predicted Price'])
orig_df = pd.DataFrame(orig, columns=['Original Price'])

chart_data = pd.concat([pred_df, orig_df], axis=1)
st.write(chart_data)

st.line_chart(chart_data)

# ============================
# Predict Next 5 Days
# ============================
st.subheader('Predicted Bitcoin Price For Next 5 Days')

last_seq = test_scaled[-base_days:]
future_predictions = []

for _ in range(5):
    seq = last_seq.reshape(1, base_days, 1)
    next_pred = model.predict(seq)
    future_predictions.append(next_pred[0][0])
    last_seq = np.append(last_seq[1:], next_pred).reshape(base_days, 1)

future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler.inverse_transform(future_predictions)

future_df = pd.DataFrame(future_predictions, columns=['Future Price'])

st.write(future_df)
st.line_chart(future_df)
