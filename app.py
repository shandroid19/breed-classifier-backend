from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.backend import clear_session
import numpy as np
import cv2
from numpy import array
from numpy import argmax
import urllib
import json
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

f = open('./data.json')
data = json.load(f)

@app.route('/',methods = ['GET','POST'])
@cross_origin()
def home():
    if request.method == 'POST':
        try:
            payload = json.loads(request.data)
            url = payload["img_link"]
            urllib.request.urlretrieve(url,'sample.jpg')
            clear_session()
            model = load_model('./finmodel.h5')
            inp = np.expand_dims(cv2.resize(cv2.imread('./sample.jpg'),(224,224)),axis=0)
            out= model.predict(inp)
            output = data[str(np.argmax(out))]
            print(output)
            response = json.dumps([{'breed':output}])
            print(response)
            return response,200
        except:
            return (jsonify({"data":"err"}),500)
    
    # return (jsonify({"data":"err"}),500)

if __name__ == '__main__':
        app.run(debug = True)



