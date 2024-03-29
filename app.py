from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

model = ResNet50(weights='imagenet') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array_expanded_dims)
        
        predictions = model.predict(processed_img)
        result = decode_predictions(predictions, top=1)[0]  
        
        return jsonify({'name': str(result[0][1])}) 

if __name__ == '__main__':
    app.run(debug=True, port=5000)