import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, jsonify
from io import BytesIO

# Define a Flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Define a mapping of class indices to human-readable labels
class_labels = [
    'Endek Bali', 
    'Gringsing Bali', 
    'Ikat Flores', 
    'Lurik Jogja', 
    'Songket Lombok', 
    'Songket Minangkabau', 
    'Songket Palembang', 
    'Toraja', 
    'Ulos'
]

def model_predict(img_file, model):
    img = image.load_img(BytesIO(img_file.read()), target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = x / 255.0  # Assuming your model expects images to be normalized in this way

    preds = model.predict(x)
    return preds

@app.route('/')
def hello():
    return jsonify(status='success', message='Hello Ruang Tenun')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        image_file = request.files['image']

        # Make prediction
        preds = model_predict(image_file, model)

        # Assuming preds is a 2D array of shape (1, num_classes)
        # Get the index of the highest probability class
        pred_class_index = np.argmax(preds, axis=1)[0]
        # Map the index to the corresponding label
        result = class_labels[pred_class_index]
        
        return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
