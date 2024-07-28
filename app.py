import os
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the model
model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None
    st.error("Model file not found. Please check the file path.")

st.header('Stock Market Predictor')

# User inputs
stock = st.text_input('Enter Stock Symbol', 'GOOG').upper()
start = st.date_input('Start Date', pd.to_datetime('2012-01-01'))
end = st.date_input('End Date', pd.to_datetime('2022-12-31'))

# Fetch data
try:
    data = yf.download(stock, start, end)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    data = pd.DataFrame()

# Display data
st.subheader('Stock Data')
st.write(data)

if not data.empty and 'Close' in data.columns:
    data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    # Plot Price vs MA50
    st.subheader('Price vs MA50')
    ma_50_days = data['Close'].rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(data['Close'], 'g', label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price vs MA50')
    plt.legend()
    st.pyplot(fig1)

    # Plot Price vs MA50 vs MA100
    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(ma_100_days, 'b', label='MA100')
    plt.plot(data['Close'], 'g', label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price vs MA50 vs MA100')
    plt.legend()
    st.pyplot(fig2)

    # Plot Price vs MA100 vs MA200
    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data['Close'].rolling(200).mean()
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label='MA100')
    plt.plot(ma_200_days, 'b', label='MA200')
    plt.plot(data['Close'], 'g', label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price vs MA100 vs MA200')
    plt.legend()
    st.pyplot(fig3)

    # Prepare data for prediction
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    if model:
        try:
            predict = model.predict(x)

            # Reverse the scaling
            predict = scaler.inverse_transform(np.hstack((predict, np.zeros((predict.shape[0], data_test_scale.shape[1] - 1)))))[:, 0]
            y = scaler.inverse_transform(np.hstack((y.reshape(-1, 1), np.zeros((y.shape[0], data_test_scale.shape[1] - 1)))))[:, 0]

            # Plot Original Price vs Predicted Price
            st.subheader('Original Price vs Predicted Price')
            fig4 = plt.figure(figsize=(8, 6))
            plt.plot(predict, 'r', label='Predicted Price')
            plt.plot(y, 'g', label='Original Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title('Original Price vs Predicted Price')
            plt.legend()
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("No data found for the stock symbol entered. Please check the symbol and try again.")
