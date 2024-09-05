import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from prophet import Prophet
import matplotlib.pyplot as plt

# Define the path to your MIT-BIH Arrhythmia Database folder
data_folder = 'C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/mit-bih-arrhythmia-database-1.0.0'

# Function to load ECG data and labels from CSV files
def load_ecg_data(record_name):
    ecg_file = os.path.join(data_folder, f'{record_name}_ecg_data.csv')
    labels_file = os.path.join(data_folder, f'{record_name}_labels.csv')
    
    ecg_data = pd.read_csv(ecg_file)
    labels_data = pd.read_csv(labels_file)
    
    return ecg_data, labels_data

# Prepare data for CNN model
def prepare_cnn_data(ecg_data, labels_data):
    if 'label' not in labels_data.columns:
        raise ValueError("labels_data must contain a 'label' column")
    
    # Merge ECG data with labels_data based on timestamp
    merged_data = pd.merge(ecg_data, labels_data, on='timestamp', how='left')
    
    if 'label' not in merged_data.columns:
        raise ValueError("Merged data must contain a 'label' column")
    
    # Convert labels to numeric values
    label_encoder = LabelEncoder()
    merged_data['label'] = label_encoder.fit_transform(merged_data['label'])
    
    X = merged_data['signal'].values
    y = merged_data['label'].values
    X = X.reshape(-1, 1)  # Reshape for Conv1D
    scaler = StandardScaler()
    X = scaler.fit_transform(X).reshape(-1, 1, 1)
    
    return X, y

# Create and train CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=1, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data for Prophet model
def prepare_prophet_data(ecg_data):
    df = pd.DataFrame({
        'ds': pd.to_datetime(ecg_data['timestamp'], unit='s'),
        'y': ecg_data['signal']
    })
    return df

# Generate CSV file for time series data
def generate_timeseries_csv(ecg_data, filename='time_series_data.csv'):
    ecg_data[['timestamp', 'signal']].to_csv(os.path.join(data_folder, filename), index=False)

# Example record to use (you can loop over all records if needed)
record_name = '100'
ecg_data, labels_data = load_ecg_data(record_name)

# Generate time-series CSV if needed
generate_timeseries_csv(ecg_data)

# Prepare data for CNN
X, y = prepare_cnn_data(ecg_data, labels_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CNN model
model = create_cnn_model(X_train.shape[1:], len(np.unique(y)))
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc}')

# Save the trained model
model.save('trained_ecg_model.h5')

# Prepare data for Prophet
time_series_file = os.path.join(data_folder, 'time_series_data.csv')
time_series_data = pd.read_csv(time_series_file)
df_prophet = prepare_prophet_data(time_series_data)

# Create and fit Prophet model
model_prophet = Prophet()
model_prophet.fit(df_prophet)

# Forecast and plot
future = model_prophet.make_future_dataframe(periods=365)
forecast = model_prophet.predict(future)
model_prophet.plot(forecast)
plt.show()
