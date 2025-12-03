import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

model = load_model('Bitcoin_Price_prediction_Model.keras')

st.header('Bitcoin Price Prediction Model')

# ============================
# Load Bitcoin Data
# ============================
st.subheader('Bitcoin Price Data')
data = yf.download('BTC-USD','2015-01-01','2023-11-30')
data = data.reset_index()
st.write(data)

# ============================
# Line Chart (Fixed)
# ============================
st.subheader('Bitcoin Line Chart')

# Keep "Date" for chart
chart_df = data[['Date', 'Close']]
chart_df = chart_df.set_index('Date')

st.line_chart(chart_df)

# ============================
# Prepare Data for Model
# ============================
# Use only Close price for model
model_data = data[['Close']]

train_data = model_data[:-100]
test_data = model_data[-200:]

scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

base_days = 100
x = []
y = []

for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i, 0])

x = np.array(x)
y = np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# ============================
# Predictions
# ============================
st.subheader('Predicted vs Original Prices')

pred = model.predict(x)
pred = scaler.inverse_transform(pred)
ys = scaler.inverse_transform(y.reshape(-1,1))

preds = pd.DataFrame(pred, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat([preds, ys], axis=1)

st.write(chart_data)

# Chart
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(chart_data)

# ============================
# Predict Future Days
# ============================

st.subheader('Predicted Future Days Bitcoin Price')

last_sequence = test_data_scale[-base_days:]
future_predictions = []

for _ in range(5):   # future_days = 5
    seq = last_sequence.reshape(1, base_days, 1)
    pred = model.predict(seq)
    future_predictions.append(pred[0][0])
    last_sequence = np.append(last_sequence[1:], pred).reshape(base_days, 1)

future_predictions = np.array(future_predictions)
future_predictions = scaler.inverse_transform(future_predictions.reshape(-1,1))

st.line_chart(future_predictions)
