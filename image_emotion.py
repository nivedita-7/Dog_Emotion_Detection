from flask import Flask, render_template, request, url_for, redirect
import tensorflow as tf
import os
#import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image


model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'models', 'image_emotion_model.h5')
image_emotion_model = load_model(model_path)

# Define the emotion dictionary
emotion_dict = {0: "angry", 1: "sad", 2: "relaxed", 3: "happy"}

def classify_image_emotion(image_path):
    # Load the image
    image = Image.open(image_path)

    # Resize and convert the image to grayscale
    image = image.resize((48, 48)).convert("L")

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Reshape the image array to match the input shape of the model
    input_image = image_array.reshape(1, 48, 48, 1)

    # Perform inference (prediction)
    predictions = image_emotion_model.predict(input_image)

    # Get the predicted label
    predicted_class = np.argmax(predictions)

    # Map the predicted class index to the corresponding emotion
    predicted_emotion = emotion_dict[predicted_class]
    return predicted_emotion


# Load the image
#img_path = r"C:/Users/nived/Downloads/angry_dog.jpeg"

#emotion= classify_image_emotion(img_path)
#print("Predicted emotion:",emotion)

