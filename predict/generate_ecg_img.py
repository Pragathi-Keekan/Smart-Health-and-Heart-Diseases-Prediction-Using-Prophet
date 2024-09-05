import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define the output directory for generated images
output_dir = 'C:/Users/praga/OneDrive/Desktop/MINIPROJECT/disease_prediction/predict/synthetic_ecg_images'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

def generate_synthetic_ecg_image(filename):
    # Generate synthetic ECG-like data
    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x) + 0.5 * np.sin(5 * x) + 0.2 * np.random.normal(size=x.shape)
    
    # Plot the synthetic ECG-like data
    plt.figure(figsize=(10, 2))
    plt.plot(x, y, color='black')
    plt.axis('off')
    
    # Save the plot as an image file
    image_path = os.path.join(output_dir, filename)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Generate 10 synthetic ECG images
for i in range(10):
    filename = f'synthetic_ecg_image_{i+1}.png'
    generate_synthetic_ecg_image(filename)
    print(f"Generated {filename}")
