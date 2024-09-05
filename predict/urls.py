from django.contrib import admin
from django.urls import path
from predict import views


urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', views.index, name="index"),
    # Auth
    path('signup/', views.signupuser, name="signupuser"),
    path('login/', views.loginuser, name="loginuser"),
    path('logout/', views.logoutuser, name="logoutuser"),
    path('adminlogin/', views.adminlogin, name='adminlogin'),
    path('admin_home/', views.admin_home, name='admin_home'),
path('upload/', views.upload_image, name='upload_image'),
    path('feed/', views.feed, name='feed'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('predict/', views.predict_disease, name='predict_disease'),
]