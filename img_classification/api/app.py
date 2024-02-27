from flask import Flask, request, jsonify
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import waitress

app = Flask(__name__)

# Load the pre-trained Keras model (ResNet50)
model = ResNet50(weights='imagenet')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and file.filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((224, 224), Image.NEAREST)

        # Preprocess the image for the model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make the prediction
        predictions = model.predict(x)

        # Decode the predictions
        results = decode_predictions(predictions, top=3)[0]

        # Format the results
        prediction_results = [{'class': result[1], 'confidence': float(result[2])} for result in results]

        return jsonify(prediction_results), 200

    else:
        return jsonify({'error': 'Allowed file types are jpg, jpeg, png'}), 400


if __name__ == '__main__':
    waitress.serve(app, host="0.0.0.0", port=8000)
