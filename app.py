import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cnn_model(resim):
    path = './static/uploads/'+resim
    model = load_model('cnn_cifar10.h5')
    image = cv.imread(path)
    image = cv.resize(image,(32,32))
    img = np.array([image])
    img = img / 255.0
    y_pred = model.predict(img)
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    y_classes = [np.argmax(element) for element in y_pred]
    os.remove(path)
    return classes[y_classes[0]]

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('sonuc.html',cnn_model = cnn_model(filename))