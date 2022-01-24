#import important lib
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os
import shutil

# cascade for face and eye detect
detect = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

# function that return face_img if face have two eye
def get_face(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(image)
    for x1,y1,w,h in faces:
        face_crop = image[y1:y1+h, x1:x1+w]
        face_crop_gray = gray[y1:y1+h, x1:x1+w]
        eyes = eye_detect.detectMultiScale(face_crop)
        if len(eyes) >= 2:
            return face_crop

# directory paths
current_path = "./images_dataset/" # path where image are
destination = "./images_dataset/croped_images" # path where to save image 

# save folder name and images_path 
current_dir_lis = []
dirs_name = []
for folder in os.scandir(current_path):
    if folder.is_dir() == True:
        current_dir_lis.append(folder.path)
        dirs_name.append(folder.name)

# make croped folder if it dose not exist
if os.path.exists(destination) == True:
    shutil.rmtree(destination)
os.mkdir(destination)

# save croped images to its corresponding folder in other directory
for dirs,name in zip(current_dir_lis,dirs_name):
    save_to = destination +"/"+ name # saving location
    if os.path.exists(save_to) == True:
        shutil.rmtree(destination) 
    os.mkdir(save_to) # make destination dir
    i = 0
    for folder in os.scandir(dirs):
        image = get_face(folder.path) # get croped image
        if image is not None:
            extension = save_to+"/"+name+"_"+str(i)+".jpg" # saving extension with name
            cv2.imwrite(extension,image) # save image
            i += 1
