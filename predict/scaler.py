# fit_and_save_scaler.py
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import joblib

# Load some example data (replace this with your actual training data)
data = load_iris()
X = data.data

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
