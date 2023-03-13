import os
import io
import pickle
import numpy as np
from flask import Flask, request, redirect, render_template,url_for, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

app = Flask(__name__, template_folder='.')

filename = 'preprocessing.pkl'
with open(filename, 'rb') as f:
    preprocess_input = pickle.load(f)

model = tf.keras.models.load_model('model.h5')
clf = pickle.load(open('svc_model.pkl', 'rb'))

list_vowels = ['𑒁','𑒂','𑒃','𑒄', '𑒅','𑒆','𑒇','𑒉','𑒋','𑒌','𑒍','𑒎','𑓀','𑓁']

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['image']
    img = load_img(io.BytesIO(file.read()), target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array = model.predict(img_array)
    flat_arr = img_array.flatten()
    y_clf = clf.predict(flat_arr.reshape(1,-1))
    # save the image to the server
    img.save(os.path.join("static", "image.jpg"))
    # get the image url
    image_url = url_for("static", filename="image.jpg")

    # Return the prediction as a JSON response
    return jsonify(prediction =  str(list_vowels[y_clf[0]]), image_url=image_url)

'''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        img = load_img(io.BytesIO(file.read()), target_size=(224,224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_array = model.predict(img_array)
        flat_arr = img_array.flatten()
        y_clf = clf.predict(flat_arr.reshape(1,-1))
        # save the image to the server
        img.save(os.path.join("static", "image.jpg"))
        # get the image url
        image_url = url_for("static", filename="image.jpg")

        return str(list_vowels[y_clf[0]]),image_url
    return render_template('index.html')
'''
if __name__ == '__main__':
    app.run()
