import requests, json, numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# === Config ===
CHANNEL_ID = '2560122'
READ_API_KEY = 'T90CELFXL2HRBLJA'
NUM_RESULTS = 500
FORECAST_PATH = 'forecast.json'
SEQUENCE_LENGTH = 6  # Number of time steps used for prediction

# === Fetch sensor data ===
def fetch_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={NUM_RESULTS}"
    r = requests.get(url)
    feeds = r.json()['feeds']
    return np.array([
        [float(f['field1']), float(f['field2']), float(f['field3'])]
        for f in feeds if all(f[f'field{i}'] for i in range(1, 4))
    ])

# === Prepare inputoutput for training ===
def prepare_data(data, sequence_length=SEQUENCE_LENGTH)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled) - sequence_length)
        X.append(scaled[ii + sequence_length])
        y.append(scaled[i + sequence_length])  # Predict next timestep
    
    return np.array(X), np.array(y), scaler

# === Train LSTM model ===
def train_model(X, y)
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

# === Forecast next time step ===
def forecast(model, scaler, last_sequence)
    scaled_sequence = scaler.transform(last_sequence)
    input_seq = np.expand_dims(scaled_sequence, axis=0)  # Shape (1, 6, 3)
    prediction_scaled = model.predict(input_seq)[0]
    return scaler.inverse_transform([prediction_scaled])[0]

# === Save forecast.json ===
def save_forecast(prediction)
    with open(FORECAST_PATH, 'w') as f
        json.dump({
            temperature round(prediction[0], 2),
            humidity round(prediction[1], 2),
            air_quality round(prediction[2], 2)
        }, f, indent=2)

# === Main ===
def main():
    print("📡 Fetching data...")
    data = fetch_data()
    print("🧠 Preparing training data...")
    X, y, scaler = prepare_data(data)
    print("📈 Training model...")
    model = train_model(X, y)
    print("🔮 Forecasting...")
    prediction = forecast(model, scaler, data[-1])
    print("💾 Saving forecast.json...")
    save_forecast(prediction)
    print("✅ Done!")
if __name__ == __main__
    main()
