#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:43:04 2021

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import time
from datetime import datetime
import pymysql as py

    #training_data.class_indices  , valid_data.class_indices



    #images = [training_data[0][0][0] for i in range(5)]
    #plotImages(images)
def detect_student():
    
    train_data_path="/home/user/Documents/TUTORIALS/project/students/train"
    val_data_path="/home/user/Documents/TUTORIALS/project/students/test"



    # this is the augmentation configuration we will use for training
    # It generate more images using below parameters
    training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

    # this is a generator that will read pictures found in
    # at train_data_path, and indefinitely generate
    # batches of augmented image data
    training_data = training_datagen.flow_from_directory(train_data_path, # this is the target directory
                                      target_size=(150, 150), # all images will be resized to 150x150
                                      batch_size=32,
                                      class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels


    # this is the augmentation configuration we will use for validation:
        # only rescaling
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # this is a similar generator, for validation data
    valid_data = valid_datagen.flow_from_directory(val_data_path,
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')

    now = datetime.now()
    date=now.strftime("%d-%m-%Y %H-%M-%S")
    facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)
    sampleNum=0
    path="detected_students"
    
    while(True):
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            sampleNum+=1
            cv2.imwrite(path+"/"+date+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("face",img)
        cv2.waitKey(1)
        if sampleNum>2:
            break;
    cam.release()
    cv2.destroyAllWindows()

    #print(img.shape)
    file_path=path+"/"+date+".jpg"
    file_path="/home/user/Documents/TUTORIALS/project/detected_students/"+date+".jpg"
    #print(file_path)
    model=load_model('attendence_model.h5')
    test_image = load_img(file_path, target_size = (150, 150)) # load image 
    
  
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
    result = model.predict(test_image).round(3) # predict student
    print('@@ Raw result = ', result)
  
    pred = np.argmax(result) # get the index of max value
    
    key_list = list(training_data.class_indices.keys())
    val_list = list(training_data.class_indices.values())
 
    # print key with val 100
    position = val_list.index(pred)
    print("label-",pred,"Student name-",key_list[position]," is present")

    #connection
    db=py.connect(host='localhost',user='root',password='welcome123',database='PROJECT')

    #cursor
    cur=db.cursor()
    name=key_list[position]
    present='Present'
    '''
    print(name)
    
    query="INSERT INTO Student_Attendence VALUES(CURRENT_TIMESTAMP,name,'Present');"
    cur.execute(query)
    db.commit()
    '''
    #ts = time.time()
    #timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    date=now.strftime("%Y-%m-%d %H-%M-%S")
    print(date)
    try:
        cur.execute("""INSERT into Student_Attendence (Reporting_Time,Name,Status) values(%s,%s,%s)""",(date,name,present))
        db.commit()
    except:
            db.rollback()

def waste_function():
    print("This function is defined to avoid direct call ")
    