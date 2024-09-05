import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your data
data = pd.read_csv('C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/Testing.csv')

# Check columns and first few rows
print("Columns in data:", data.columns)
print("First few rows of data:")
print(data.head())

# Handle missing values if necessary (e.g., fill with mean or drop)
# data.fillna(data.mean(), inplace=True)  # Example for numeric data
# data.dropna(inplace=True)  # Example to drop rows with missing values

# Separate features and target variable
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(eval_metric='mlogloss')

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the model
joblib.dump(model, 'xgboost_model.pkl')

# Save the accuracy to a file
with open('model_accuracies.txt', 'w') as f:
    f.write(f'XGBoost: {accuracy:.4f}\n')

print("XGBoost model trained and evaluated. Accuracy saved to model_accuracies.txt")
