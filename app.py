from flask import Flask, request, url_for, render_template,jsonify
import utils
import os
import cv2
import base64



app = Flask(__name__,template_folder="templates",static_folder='static')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        image = request.files['file']  
        image_string = base64.b64encode(image.read())
        # Make prediction
        result = utils.prediction(image_string,None) 
        return result
    return None


if __name__ == "__main__":
    
    app.run(debug=True)
