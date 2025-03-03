from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from keras.models import load_model
from image_breed import preprocess_image, predict_breed
from image_emotion import classify_image_emotion

app = Flask(__name__)

# Define the upload folder for storing files
UPLOAD_FOLDER = r'E:\dog_sentiment_analysis_project\Dog_Emotion_Detection_Project\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions for image and audio files
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3'}

# Load audio model
MODEL_PATH = 'models/dog_audio_model.hdf5'
audio_model = load_model(MODEL_PATH)
audio_class_labels = ['angry', 'happy', 'howling', 'sad']

# Function to classify emotions in audio
def classify_audio_emotion(file_path):
    # Audio processing logic
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_probabilities = audio_model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted_probabilities)
    predicted_class = audio_class_labels[predicted_label]
    return predicted_class

# Function to classify breed in image
def classify_image_breed(file_path):
    return predict_breed(file_path)

# Function to check if file extension is allowed
def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/', methods=['GET', 'POST'])
def process_form():
    if request.method == 'POST':
        if 'audio' in request.files:
            # Handle audio form submission
            file = request.files['audio']
            if file.filename == '':
                return "No selected file"
            if allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Check if the file is mp3, if so, convert it to wav
                if file_path.endswith('.mp3'):
                    wav_path = file_path[:-4] + '.wav'
                    os.system(f'ffmpeg -i {file_path} {wav_path}')
                    os.remove(file_path)
                    file_path = wav_path
                
                label_audio = classify_audio_emotion(file_path)
                os.remove(file_path)  # Remove the uploaded file
                return redirect(url_for('success', label_audio=label_audio))
            else:
                return "Invalid audio file format"
        elif 'image' in request.files:
            # Handle image form submission
            file = request.files['image']
            if file.filename == '':
                return "No selected file"
            if allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                label_image_breed = classify_image_breed(file_path)
                label_image_emotion = classify_image_emotion(file_path)
                os.remove(file_path)  # Remove the uploaded file
                return redirect(url_for('success', label_image_1=label_image_breed, label_image_2=label_image_emotion))
            else:
                return "Invalid image file format"
        else:
            return "Invalid form submission"
    elif request.method == 'GET':
        return render_template('index.html')


@app.route('/success')
def success():
    label_audio = request.args.get('label_audio')
    label_image_1 = request.args.get('label_image_1')
    label_image_2 = request.args.get('label_image_2')
    if label_audio:
        return render_template('success.html', label=label_audio)
    else:
        return render_template('success.html', label_image_1=label_image_1, label_image_2=label_image_2)

if __name__ == '__main__':
    app.run(debug=True)
