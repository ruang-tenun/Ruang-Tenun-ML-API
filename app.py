import numpy as np
import uuid
from datetime import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

MODEL_PATH = 'models/model.h5'

model = load_model(MODEL_PATH)

classes = [
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

def predictClassification(img_file, model):
    img = image.load_img(BytesIO(img_file.read()), target_size=(150, 150))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = img / 255.0 

    prediction = model.predict(img)
    return prediction

@app.route('/')
def hello():
    return jsonify(status='success', message='Hello Ruang Tenun')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        prediction_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        image_file = request.files['image']

        prediction = predictClassification(image_file, model)

        class_result = np.argmax(prediction, axis=1)[0]
        result = classes[class_result]
        
        confidence_score = float(np.max(prediction))
        
        return jsonify(
            id=prediction_id, 
            result=result, 
            confidence_score=confidence_score,
            created_at=created_at
        )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
