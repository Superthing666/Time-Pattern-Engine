
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.set_page_config(page_title="Time Pattern Engine", layout="wide")

# App-wide session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'signal' not in st.session_state:
    st.session_state.signal = None

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Upload & Visualize", "⚙️ Run Engine", "📈 Predictions", "🛠 Anomalies"])

st.title("🧠 Time Pattern Engine")
st.markdown("#### Discover signal patterns. Predict the future. Understand time.")

if page == "📊 Upload & Visualize":
    st.markdown("### Upload Signal File (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("File uploaded successfully.")
        st.write("Preview:", df.head())

        if df.select_dtypes(include=np.number).shape[1] > 0:
            signal_col = st.selectbox("Select signal column", df.columns)
            st.session_state.signal = df[signal_col].dropna().values.reshape(-1, 1)
            st.line_chart(st.session_state.signal)
        else:
            st.warning("No numeric columns found in file.")

elif page == "⚙️ Run Engine":
    st.markdown("### Run Engine")
    if st.session_state.signal is not None:
        st.markdown("#### Normalized Signal Preview")
        scaler = MinMaxScaler()
        signal_scaled = scaler.fit_transform(st.session_state.signal)
        st.line_chart(signal_scaled)
        st.session_state.scaler = scaler
        st.session_state.signal_scaled = signal_scaled
        st.success("Signal processed and ready for prediction.")
    else:
        st.warning("Please upload and select a signal first.")

elif page == "📈 Predictions":
    st.markdown("### Future Signal Predictions")
    if st.session_state.signal_scaled is not None:
        def create_sequences(data, seq_length):
            x, y = [], []
            for i in range(len(data) - seq_length):
                x.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(x), np.array(y)

        SEQ_LEN = 20
        X, y = create_sequences(st.session_state.signal_scaled, SEQ_LEN)

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, verbose=0)

        preds = model.predict(X)
        st.line_chart({
            "Actual": y.flatten(),
            "Predicted": preds.flatten()
        })
        st.success("Prediction complete.")
    else:
        st.warning("Run the engine first to prepare data.")

elif page == "🛠 Anomalies":
    st.markdown("### Anomaly Detection (Coming soon)")
