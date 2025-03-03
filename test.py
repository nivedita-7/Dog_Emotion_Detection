from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import os
import librosa
import tensorflow as tf
import numpy as np
from keras.models import load_model

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'models', 'dog_audio_model.hdf5')
audio_model = load_model(model_path)
audio_class_labels = ['angry', 'happy', 'howling', 'sad']

def classify_audio_emotion(file_path):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_probabilities = audio_model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted_probabilities)
    predicted_class = audio_class_labels[predicted_label]
    return predicted_class

path= r"C:/Users/nived/Downloads/happy14 (10).wav"

label= classify_audio_emotion(path)
print(label)