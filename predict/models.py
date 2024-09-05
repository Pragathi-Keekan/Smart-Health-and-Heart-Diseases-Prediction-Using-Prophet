from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Report(models.Model):
    name = models.CharField(max_length=100)
    symptoms = models.JSONField(default=dict)  # Use a default value
    predicted_diseases = models.JSONField(default=dict)


class Prediction(models.Model):
    symptoms = models.JSONField()  # To store symptom data as JSON
    prediction = models.CharField(max_length=255)
    date_predicted = models.DateTimeField(auto_now_add=True)
    algorithm = models.CharField(max_length=255, default='XGBoost')  # Optional field to store the algorithm used

    def __str__(self):
        return f"{self.prediction} on {self.date_predicted} using {self.algorithm}"
    

class SignupLog(models.Model):
    username = models.CharField(max_length=150)
    signup_date = models.DateField(default=timezone.now)
    signup_time = models.TimeField(default=timezone.now)

    def __str__(self):
        return f"{self.username} - {self.signup_date} {self.signup_time}"



class Registration(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15)
    date_registered = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"



class ECGImage(models.Model):
    image = models.ImageField(upload_to='uploaded_ecg_images/')
    created_at = models.DateTimeField(auto_now_add=True)







