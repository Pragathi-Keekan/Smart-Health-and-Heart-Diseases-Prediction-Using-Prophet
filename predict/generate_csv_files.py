import os
import pandas as pd
import wfdb
import numpy as np

# Define the path to your MIT-BIH Arrhythmia Database folder
data_folder = 'C:/Users/praga/OneDrive/Desktop/MINIPROJECT/disease_prediction/predict/mit-bih-arrhythmia-database-1.0.0'

# Load ECG data and labels from MIT-BIH dataset
def load_data(record_name):
    # Load the ECG signal and the associated annotation file
    record = wfdb.rdrecord(os.path.join(data_folder, record_name))
    annotation = wfdb.rdann(os.path.join(data_folder, record_name), 'atr')
    
    # Extract the signal and annotation
    signal = record.p_signal[:, 0]  # Assuming single channel ECG data
    timestamps = record.p_signal[:, 0]  # Using the signal itself as a placeholder for timestamp
    labels = annotation.symbol
    label_indices = annotation.sample
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'signal': signal
    })
    
    # Create labels DataFrame
    labels_df = pd.DataFrame({
        'timestamp': [timestamps[i] for i in label_indices],
        'label': [labels[i] for i in range(len(labels))]
    })
    
    return df, labels_df

# Generate CSV files for each record
def generate_csv_files():
    # List of record names in the MIT-BIH Arrhythmia Database
    record_names = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '205', '207', '208', '209', '210']
    
    # Process each record
    for record_name in record_names:
        ecg_data, labels_data = load_data(record_name)
        
        # Save to CSV
        ecg_data.to_csv(os.path.join(data_folder, f'{record_name}_ecg_data.csv'), index=False)
        labels_data.to_csv(os.path.join(data_folder, f'{record_name}_labels.csv'), index=False)
        
        print(f'Generated CSV files for record {record_name}')

generate_csv_files()
