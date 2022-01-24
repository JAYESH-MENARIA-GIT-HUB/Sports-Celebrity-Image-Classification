import cv2
from tensorflow.keras.models import load_model
import numpy as np
import json
import base64

def result_name(result_index):
    with open("label_name.json")  as f:
        players_name = json.load(f)
    name = players_name[str(result_index)]
    
    return name


def decode_base64(path):
    # decoded = path.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(path), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img
    
def get_face(image):
    detect = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
    eye_detect = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")
    crop_faces = []
    
    image = image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(image)
    
    for x1,y1,w,h in faces:
        face_crop = image[y1:y1+h, x1:x1+w]
        face_crop_gray = gray[y1:y1+h, x1:x1+w]
        eyes = eye_detect.detectMultiScale(face_crop_gray)
        
        if len(eyes) >= 2:
            crop_faces.append(cv2.resize(face_crop,(150,150)))
            
            
    return crop_faces 



def prediction(basecode,path=None):
    if path is not None:
        image = cv2.imread(path)
    else:
        image = decode_base64(basecode)
        
    faces = get_face(image)
    if faces == []:
        return "No face detected"
    
    else:
        results = []
        for face in faces:
            image = face.reshape(1,150,150,3)
            model = load_model("model/model.h5")
            prediction = model.predict(image)*10
            predictions = np.round(prediction,2)
            index = predictions.argmax()
            plyer_name = result_name(index)
            result = {"prediction":list(predictions[0]), "index":index, "name":plyer_name}
            results.append(plyer_name)
        return plyer_name


if __name__ == "__main__":
    print(prediction(basecode=None,path='test_images/massi.jpeg'))