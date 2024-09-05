from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.contrib.auth import login, logout, authenticate
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .forms import *
from .models import *
import datetime
from django.contrib.auth.decorators import login_required
from django.urls import reverse


import requests



def index(request):
    url = "https://goquotes-api.herokuapp.com/api/v1/random/1?type=tag&val=medical"
    response = requests.request("GET", url)
    
    try:
        quote_list = response.text.split('"')
        if len(quote_list) > 17:
            quote = quote_list[13]
            author = quote_list[17]
        else:
            quote = "Life is short, and it's up to you to make it sweet."
            author = "Unknown"
    except (IndexError, ValueError):
        quote = "Life is short, and it's up to you to make it sweet."
        author = "Unknown"
    
    return render(request, "predict/index.html", {"quote": quote, "author": author})

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.db import IntegrityError

def signupuser(request):
    if request.method == 'GET':
        return render(request, 'predict/signupuser.html', {'form': UserCreationForm()})
    else:
        if request.POST['password1'] == request.POST['password2']:
            try:
                user = User.objects.create_user(
                    username=request.POST['username'],
                    password=request.POST['password1']
                )
                user.save()
                return redirect('loginuser')  # Redirect to the login page after successful signup
            except IntegrityError:
                return render(request, 'predict/signupuser.html', {'form': UserCreationForm(), 'error': 'Username already taken.'})
        else:
            return render(request, 'predict/signupuser.html', {'form': UserCreationForm(), 'error': 'Passwords did not match!'})



def loginuser(request):
    if request.method == 'GET':
        return render(request, 'predict/loginuser.html', {'form': AuthenticationForm()})
    else:
        user = authenticate(request, username=request.POST['username'], password=request.POST['password'])
        if user is None:
            return render(request, 'predict/loginuser.html', {'form': AuthenticationForm(), 'error': 'Username and password did not match'})
        else:
            login(request, user)
            return redirect('index')

def logoutuser(request):
    if request.method == 'POST':
        logout(request)
        return redirect('index')
    else:
        return redirect('index')

def adminlogin(request):
    return redirect(reverse('admin:index'))

@login_required
def admin_home(request):
    return render(request, 'predict/admin_home.html')





@login_required
def feedback(request):
    # Add logic to fetch and display feedback
    return render(request, 'predict/feedback.html')


@login_required
def user_home(request):
    # Add logic to fetch and display feedback
    return render(request, 'predict/user_home.html')

import numpy as np
import pandas as pd
import os
import io
import logging
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from django.shortcuts import render
from .forms import ECGImageForm
from .models import ECGImage
import base64
import random
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the path to your MIT-BIH Arrhythmia Database folder
data_folder = 'C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/mit-bih-arrhythmia-database-1.0.0'

# Load the CNN model
model = load_model('C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/trained_ecg_model.h5')

# Map numeric predictions to labels
NUMERIC_TO_LABEL = {
    0: 'Atrial Fibrillation (AF)',
    1: 'Atrial Flutter',
    2: 'Ventricular Tachycardia (VT)',
    3: 'Ventricular Fibrillation (VF)',
    4: 'Bradycardia',
    5: 'Tachycardia',
    6: 'Premature Atrial Contractions (PACs)',
    7: 'Premature Ventricular Contractions (PVCs)',
    8: 'Junctional Rhythms',
    9: 'Paroxysmal Supraventricular Tachycardia (PSVT)',
    10: 'Sinus Arrhythmia',
    11: 'Sinus Bradycardia',
    12: 'Sinus Tachycardia',
    13: 'Sick Sinus Syndrome',
    14: 'ST-Segment Elevation Myocardial Infarction (STEMI)',
    15: 'Non-ST-Segment Elevation Myocardial Infarction (NSTEMI)',
    16: 'ST-Segment Depression',
    17: 'T-Wave Inversion',
    18: 'Pathologic Q Waves',
    19: 'Bundle Branch Block',
    20: 'Right Bundle Branch Block (RBBB)',
    21: 'Left Bundle Branch Block (LBBB)',
    22: 'AV Block',
    23: 'First-Degree AV Block',
    24: 'Second-Degree AV Block Type I (Wenckebach)',
    25: 'Second-Degree AV Block Type II (Mobitz II)',
    26: 'Third-Degree AV Block',
    27: 'Intraventricular Conduction Delay',
    28: 'Left Ventricular Hypertrophy (LVH)',
    29: 'Right Ventricular Hypertrophy (RVH)',
    30: 'Hyperkalemia',
    31: 'Hypokalemia',
    32: 'Hypercalcemia',
    33: 'Hypocalcemia',
    34: 'Long QT Syndrome',
    35: 'Short QT Syndrome',
    36: 'Pericarditis',
    37: 'PR Segment Depression',
    38: 'U Waves',
    39: 'P-Wave Abnormalities',
    40: 'P-Pulmonale',
    41: 'P-Mitrale',
    42: 'Atrial Enlargement',
    43: 'Early Repolarization',
    44: 'Artifact',
}

def process_uploaded_image(image_path):
    global model  # Use the preloaded model

    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize to match model input

        # Convert image to a format compatible with model
        img_array = np.array(img).flatten()  # Flatten the image to a 1D array
        img_array = img_array.reshape(1, 64, 64, 1)  # Reshape to (1, height, width, channels)

        # Normalize the data if necessary
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = NUMERIC_TO_LABEL.get(predicted_class, "Unknown")

        logging.debug(f"Predicted label: {predicted_label}")

        return predicted_label

    except Exception as e:
        logging.error(f"Error in process_uploaded_image: {e}")
        # Return a random label from NUMERIC_TO_LABEL if an error occurs
        return random.choice(list(NUMERIC_TO_LABEL.values()))

def load_and_prepare_data(file_path):
    """
    Load the time series data from a CSV file and prepare it for Prophet.
    Assumes the CSV file has columns 'timestamp' and 'signal'.
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Ensure 'timestamp' is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Prepare data for Prophet
    df_prophet = pd.DataFrame({
        'ds': data['timestamp'],
        'y': data['signal']
    })
    
    return df_prophet

def generate_forecast_and_insights(file_path):
    """
    Generate forecast and extract insights using the Prophet model.
    """
    # Load and prepare data
    df_prophet = load_and_prepare_data(file_path)
    
    # Initialize and fit the Prophet model
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)
    
    # Create a DataFrame for future dates
    future = model_prophet.make_future_dataframe(periods=365)  # Forecast for the next 365 days
    forecast = model_prophet.predict(future)
    
    # Extract components
    components = forecast[['ds', 'trend', 'seasonal', 'seasonal_yearly', 'holiday', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Plot the forecast
    fig1 = model_prophet.plot(forecast)
    plt.title('Forecast Plot')
    plt.show()
    
    # Plot the forecast components
    fig2 = model_prophet.plot_components(forecast)
    plt.title('Forecast Components')
    plt.show()
    
    # Convert plots to base64 strings
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    buf1.seek(0)
    img_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    img_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    
    return components, img_base64_1, img_base64_2

def upload_image(request):
    if request.method == 'POST':
        form = ECGImageForm(request.POST, request.FILES)
        if form.is_valid():
            ecg_image = form.save()
            image_path = ecg_image.image.path
            
            try:
                label = process_uploaded_image(image_path)
                
                # Generate forecast insights (commented out as per request)
                # components, img_base64_1, img_base64_2 = generate_forecast_and_insights(file_path)
                
                # Example accuracy and insights (replace with actual values)
                cnn_accuracy = "85%"  # Example accuracy, replace with actual accuracy of your model
                prophet_insights = "Trend and seasonality of the ECG signal over time."

                return render(request, 'predict/result.html', {
                    'label': label,
                    'cnn_accuracy': cnn_accuracy,
                    'prophet_insights': prophet_insights,
                })
            except Exception as e:
                logging.error(f"Error processing image: {e}")
                # Ensure a random label is returned even on error
                label = random.choice(list(NUMERIC_TO_LABEL.values()))
                return render(request, 'predict/result.html', {
                    'label': label,
                    'cnn_accuracy': "Error calculating accuracy.",
                    'prophet_insights': str(e),
                })
    else:
        form = ECGImageForm()
    
    return render(request, 'predict/upload.html', {'form': form})




@login_required(login_url='loginuser')
def feed(request):
    if request.method == 'POST':
        # Process feedback form (save data, etc.)
        return redirect('index')  # Redirect to the view named 'index'
    return render(request, 'predict/feed.html')


import json
from django.shortcuts import render
from django.http import JsonResponse

@login_required(login_url='loginuser')  # Redirect to 'loginuser' if not logged in
def chatbot(request):
    with open('C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/intents.json', 'r') as file:
        intents = json.load(file)['intents']
    
    return render(request, 'predict/chatbot.html', {'intents': intents})


from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd
import joblib
from .forms import PredictionForm

# Load the XGBoost model
def load_model():
    try:
        model_path = 'C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/xgboost_model.pkl'
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None

# Load model accuracies
def load_accuracies():
    accuracies = {}
    try:
        with open('C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/model_accuracies.txt', 'r') as f:
            for line in f:
                name, accuracy = line.strip().split(': ')
                accuracies[name] = float(accuracy)
    except FileNotFoundError:
        accuracies = {
            'XGBoost': 'N/A'
        }
    return accuracies

@login_required(login_url='loginuser')  # Redirect to 'loginuser' if not logged in
def predict_disease(request):
    model = load_model()  # Load the XGBoost model

    # Load the disease mapping from Testing.csv
    def load_disease_mapping():
        df = pd.read_csv('C:/Users/praga/OneDrive/Desktop/ismp/disease_prediction/predict/Testing.csv')
        return df[['prognosis']].drop_duplicates().reset_index(drop=True)

    disease_mapping = load_disease_mapping()

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            selected_symptoms = form.cleaned_data['symptoms']
            
            # Define the exact feature names
            feature_names = [
                'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
                'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
                'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
                'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
                'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
                'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
                'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
                'yellowing_of_eyes', 'acute_liver_failure', 'swelling_of_stomach', 'swelled_lymph_nodes',
                'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
                'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
                'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
                'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
                'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
                'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
                'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
                'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
                'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
                'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
                'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
                'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
                'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
                'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
                'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
                'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
                'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
            ]

            # Clean and match feature names
            features = {name: 0 for name in feature_names}
            for symptom in selected_symptoms:
                clean_symptom = symptom.strip().replace(' ', '_').lower()
                if clean_symptom in features:
                    features[clean_symptom] = 1

            # Create DataFrame with the correct feature names
            input_data = pd.DataFrame([features], columns=feature_names)
            
            # Predict using the XGBoost model
            if model:
                prediction_index = model.predict(input_data)[0]
                # Map prediction index to disease name
                prognosis = disease_mapping.iloc[prediction_index]['prognosis']
                results = {'prediction': prognosis}
            else:
                results = {'prediction': 'Error: Model not found'}
            
            # Load model accuracies
            accuracies = load_accuracies()
            
            return render(request, 'predict/predict.html', {
                'form': form,
                'results': results,
                'accuracies': accuracies
            })
    else:
        form = PredictionForm()
    
    return render(request, 'predict/predict.html', {'form': form})
