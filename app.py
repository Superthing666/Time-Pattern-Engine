
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Time Pattern Engine", layout="wide")

st.title("🧠 Time Pattern Engine 3.0")
st.markdown("Upload biomedical or time-series signal data to analyze, predict, and detect anomalies.")

tab1, tab2, tab3 = st.tabs(["📁 Upload & Preview", "🔮 Predict", "⚠️ Anomaly Detection"])

def create_sequences(data, window):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(x), np.array(y)

with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
        st.write("### Raw Data Preview")
        st.dataframe(df.head())

        signal_col = st.selectbox("Select signal column", df.columns)
        signal = df[signal_col].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        signal_scaled = scaler.fit_transform(signal)

        st.write("### Normalized Signal")
        fig, ax = plt.subplots()
        ax.plot(signal_scaled)
        st.pyplot(fig)
        st.session_state["signal_scaled"] = signal_scaled

with tab2:
    if "signal_scaled" in st.session_state:
        st.subheader("Train & Predict using LSTM")
        window_size = st.slider("Window size", 5, 100, 20)
        epochs = st.slider("Epochs", 1, 50, 10)

        data = st.session_state["signal_scaled"]
        X, y = create_sequences(data, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(50, input_shape=(X.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        with st.spinner("Training LSTM..."):
            model.fit(X, y, epochs=epochs, verbose=0)

        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))

        st.success(f"Model trained. RMSE: {rmse:.4f}")
        fig2, ax2 = plt.subplots()
        ax2.plot(y, label="True")
        ax2.plot(preds, label="Predicted")
        ax2.set_title("Prediction vs Actual")
        ax2.legend()
        st.pyplot(fig2)

with tab3:
    if "signal_scaled" in st.session_state:
        st.subheader("Anomaly Detection (Simple Threshold)")

        threshold = st.slider("Threshold (std deviations)", 2.0, 5.0, 3.0)
        data = st.session_state["signal_scaled"].flatten()
        mean = np.mean(data)
        std = np.std(data)

        anomalies = np.where(np.abs(data - mean) > threshold * std)[0]
        fig3, ax3 = plt.subplots()
        ax3.plot(data, label="Signal")
        ax3.scatter(anomalies, data[anomalies], color="red", label="Anomalies")
        ax3.set_title("Anomaly Detection")
        ax3.legend()
        st.pyplot(fig3)
