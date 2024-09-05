import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import os

# Function to load and preprocess data from CSV files
def load_and_preprocess_data(csv_dir):
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    print("CSV Files Found:", csv_files)  # Debug line
    
    if not csv_files:
        raise ValueError("No CSV files found in the directory.")
    
    all_data = []
    all_labels = []
    
    for csv_file in csv_files:
        # Load the CSV file
        try:
            data = pd.read_csv(os.path.join(csv_dir, csv_file))
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        
        # Check if the CSV file has the correct format
        if data.shape[1] < 2:
            print(f"CSV file {csv_file} does not have enough columns.")
            continue
        
        # Assume last column is the label
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Target labels
        
        all_data.append(X)
        all_labels.append(y)
    
    # Concatenate all data
    if not all_data:
        raise ValueError("No data to concatenate.")
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder

# Define CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to train and save the model
def main():
    # Corrected path to CSV files
    csv_dir = 'C:/Users/praga/OneDrive/Desktop/MINIPROJECT/disease_prediction/predict/mit-bih-arrhythmia-database-1.0.0'
    
    # Load and preprocess data
    try:
        X, y, label_encoder = load_and_preprocess_data(csv_dir)
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape for CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Create and train model
    model = create_cnn_model(X_train.shape[1:], len(np.unique(y)))
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))
    
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, class_weight=class_weights)
    
    # Save model and label encoder
    model.save('ecg_cnn_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

if __name__ == '__main__':
    main()
