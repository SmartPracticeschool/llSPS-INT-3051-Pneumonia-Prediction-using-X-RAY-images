import sys
import glob
import numpy as np
from tensorflow.keras import backend

import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()
tf.get_default_graph=tf.compat.v1.get_default_graph()
from skimage.transform import resize
from flask import Flask , request, render_template,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model =load_model(r"./models/cnn.h5")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST'or request.method =='GET':
        print("done")
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        print("done2")
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        #with graph.as_default():
        #    preds = (model.predict(x)>0.5).astype("int32")
        preds=model.predict_classes(x)
        preds=list(map(int,preds))
        print("prediction",preds)
        index = ['negative','positive']
        text = "the prediction is : "+ str(index[preds[0]])
        return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    
