import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta  # Technical Analysis library
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Fetch SMCI stock data
ticker = 'SMCI'
df = yf.download(ticker, start='2015-01-01', end='2024-12-31')

# Print basic info
print(df.head())

# Clean dataframe (ensure no NaNs for indicators)
df = df.dropna()

# Add indicators using ta
df['SMA_20'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=20)
df['EMA_20'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=20)
df['RSI_14'] = ta.momentum.rsi(df['Close'].squeeze(), window=14)
df['MACD'] = ta.trend.macd_diff(df['Close'].squeeze())

# Add more sophisticated indicators
# Bollinger Bands
bb = ta.volatility.BollingerBands(df['Close'].squeeze())
df['BB_upper'] = bb.bollinger_hband()
df['BB_middle'] = bb.bollinger_mavg()
df['BB_lower'] = bb.bollinger_lband()

# Stochastic Oscillator
df['STOCH_k'] = ta.momentum.stoch(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())
df['STOCH_d'] = ta.momentum.stoch_signal(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())

# ATR (Average True Range)
df['ATR'] = ta.volatility.average_true_range(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())

# Williams %R
df['Williams_R'] = ta.momentum.williams_r(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())

# Drop any rows with missing values from indicator computation
df = df.dropna()

# View updated data
print(df[['Close', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD']].tail())

# 1. Select & Scale Features
print("\n=== Feature Scaling ===")

# Features for model input (expanded with more indicators)
features = ['Close', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'STOCH_k', 'STOCH_d', 'ATR', 'Williams_R']
df_model = df[features]

# Scale features to range [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_model)

# Convert back to DataFrame to keep feature names
scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)

print("Original data (last 5 rows):")
print(df_model.tail())
print("\nScaled data (last 5 rows):")
print(scaled_df.tail())

# 2. Create LSTM Sequences
print("\n=== LSTM Sequence Creation ===")

# Define how many past days to use for each prediction
lookback = 60

X = []
y = []

for i in range(lookback, len(scaled_df)):
    X.append(scaled_df.iloc[i - lookback:i].values)   # 60-day window
    y.append(scaled_df.iloc[i]['Close'])              # Target: next day's Close

# Convert to numpy arrays
X, y = np.array(X), np.array(y)

# Print shape to confirm
print("X shape:", X.shape)  # (samples, 60, features)
print("y shape:", y.shape)  # (samples,)

# 3. Train-Test Split
print("\n=== Train-Test Split ===")

# We'll use 80% for training and 20% for testing
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")
print(f"Total samples: {len(X_train) + len(X_test)}")

# 4. Model Building & Training
print("\n=== LSTM Model Training ===")

# Step 1: Define the improved LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(units=1))  # Predicting the next day's Close price

# Step 2: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Train the model with callbacks
print("Training improved LSTM model...")

# Add callbacks for better training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stop, reduce_lr], verbose=1)

# 5. Model Evaluation & Performance Check
print("\n=== Model Performance Evaluation ===")

# Get model predictions
predicted_prices = model.predict(X_test)

# Undo scaling to get actual price values
# We need to create the full feature array for inverse transform
predicted_full = np.zeros((len(predicted_prices), 12))  # Updated for 12 features
predicted_full[:, 0] = predicted_prices.flatten()  # Close price is first feature
predicted_prices_actual = scaler.inverse_transform(predicted_full)[:, 0]

actual_full = np.zeros((len(y_test), 12))  # Updated for 12 features
actual_full[:, 0] = y_test.flatten()  # Close price is first feature
actual_prices = scaler.inverse_transform(actual_full)[:, 0]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices_actual))
print("Root Mean Squared Error (RMSE):", rmse)

# Plot
plt.figure(figsize=(14,6))
plt.plot(actual_prices, color='blue', label='Actual SMCI Price')
plt.plot(predicted_prices_actual, color='red', label='Predicted SMCI Price')
plt.title('SMCI Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 6. Predict Future Stock Price
print("\n=== Future Price Prediction ===")

# Save the trained model
model.save("smci_lstm_model.h5")
print("Model saved as 'smci_lstm_model.h5'")

# Download latest SMCI data for prediction
print("Downloading latest SMCI data...")
df_latest = yf.download("SMCI", period="2y", interval="1d")
close_prices = df_latest['Close'].values.reshape(-1, 1)

# Normalize using MinMaxScaler
scaler_latest = MinMaxScaler()
scaled_data_latest = scaler_latest.fit_transform(close_prices)

# Prepare input for prediction
window_size = 60  # should match what you used in training
last_window = scaled_data_latest[-window_size:]  # shape = (60, 1)
X_input = np.reshape(last_window, (1, window_size, 1))

# Predict
predicted_price_scaled = model.predict(X_input)
predicted_price = scaler_latest.inverse_transform(predicted_price_scaled)

print("📈 Predicted next day's closing price for SMCI:", predicted_price[0][0])
print("💰 Current SMCI price:", df_latest['Close'].iloc[-1])
print("📊 Price change prediction:", predicted_price[0][0] - df_latest['Close'].iloc[-1])

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='SMCI Close Price', color='blue')
plt.title('SMCI Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Optional Plotting (just to see if indicators make sense):
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close')
plt.plot(df['SMA_20'], label='SMA 20')
plt.plot(df['EMA_20'], label='EMA 20')
plt.title('SMCI Price with SMA & EMA')
plt.legend()
plt.grid(True)
plt.show()
