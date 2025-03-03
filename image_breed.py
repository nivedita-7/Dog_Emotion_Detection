from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

# Load pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to predict the breed of the dog
def predict_breed(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=1)[0]
    return decoded_preds[0][1]

#img_path = r"C:/Users/nived/Downloads/angry_dog.jpeg"

#label= predict_breed(img_path)
#print(label)