import flask
from flask import Flask, jsonify, request
import numpy as np
from tensorflow import keras
from PIL import Image
import io

#Load the model
model = keras.models.load_model('my_model.h5')

#initialize the Flask application
app = Flask(__name__)

#Define a route for the API endpoint
@app.route('/predict', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    data = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            # Read the image in PIL format
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))

            # Resize the image to (28, 28) and convert to grayscale
            image = image.resize((28, 28))
            image = image.convert('L')

            # Convert the PIL image to a numpy array
            image = np.array(image)

            # Reshape the image to (1, 28, 28, 1) to match the input shape of the model
            image = image.reshape((1, 28, 28, 1))
            image = image.astype('float32') / 255.0

            # Make predictions on the image using the loaded model
            y_pred = model.predict(image)
            # Convert predictions from categorical back to numerical values
            y_pred = np.argmax(y_pred, axis=1)
            data['predictions'] = y_pred.tolist()
            data['success'] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)