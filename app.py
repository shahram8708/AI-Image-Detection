from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder='templates', static_folder='static')

model = ResNet50(weights='imagenet')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            img = Image.open(file.stream)
            img = img.convert('RGB')
            img = img.resize((224, 224), Image.NEAREST)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            processed_img = preprocess_input(img_array)

            predictions = model.predict(processed_img)
            result = decode_predictions(predictions, top=1)[0]

            return jsonify({'name': str(result[0][1])})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
