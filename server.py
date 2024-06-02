import tensorflow as tf
import numpy as np
import json
from flask import Flask, request, jsonify

# Load the trained model
app = Flask(__name__)
model = tf.keras.models.load_model('my_mnist.h5')
feature_model = tf.keras.models.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])

# Preprocess the MNIST dataset for predictions
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255

def get_prediction(image=None):
    if image is None:
        index = np.random.choice(x_test.shape[0])
        image = x_test[index, :].reshape(1, 784)
    else:
        image = np.array(image).reshape(1, 784)
    
    image_arr = feature_model.predict(image)
    return image_arr, image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if 'image' in data:
                image_arr, image = get_prediction(data['image'])
            else:
                image_arr, image = get_prediction()
            
            final_preds = [p.tolist() for p in image_arr]
            response = jsonify({'prediction': final_preds, 'image': image.tolist()})
            response.status_code = 200
            return response
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return 'MNIST server'

if __name__ == '__main__':
    app.run(port=5000)
