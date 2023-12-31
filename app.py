from flask import Flask, request, render_template, jsonify
#scientific computing library for saving, reading, and resizing images

#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
from tensorflow import keras
#system level operations (like loading files)
import sys 
#for reading operating system data
#from keras.models import load_model

import os
import io
import cv2
from PIL import Image
import tensorflow as tf
from gevent.pywsgi import WSGIServer

app = Flask('crack_detection')

# Model saved with Keras model.save()
MODEL_PATH = os.path.join(os.path.dirname(__file__), '_crack_detection.h5')

# Load trained model
#model = tf.keras.models.load_model(MODEL_PATH)
#model._make_predict_function()
#print('Model loaded. Start serving...')


print('Model loaded. Check http://127.0.0.1:8080/')


def predict(graph, session , img, model):
    img = img.resize((128, 640))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    with graph.as_default():
        with session.as_default():
            preds = model.predict(images)
    return preds

graph = tf.Graph()
with graph.as_default():
    session= tf.Session()
    with session.as_default():
        model = tf.keras.models.load_model(MODEL_PATH)

graph_det = graph
sess= session

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image_class():
    img = request.files['file'].read()
    
    img = Image.open(io.BytesIO(img))
    prediction = predict(graph_det, sess, img, model)
    class_name = "NON_Scratch" if prediction[0] < 0.5 else "Scratch"
    response = {"prediction": class_name}
    return jsonify(response)


if __name__ == '__main__':
    http_server = WSGIServer(('', 8080), app)
    http_server.serve_forever()