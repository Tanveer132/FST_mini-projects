


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

from datetime import datetime



#Define function to register new student
def new_registration():

    facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)
    id=input("Enter user Registration No:")
    sampleNum=0
    train_dir="/home/user/Documents/TUTORIALS/project/students/train/"
    test_dir="/home/user/Documents/TUTORIALS/project/students/test/"
    directory=id
    path1=os.path.join(train_dir, directory)
    os.mkdir(path1)
    path2=os.path.join(test_dir, directory)
    os.mkdir(path2)
    print(path1)
    print(path2)

    while(True):
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            sampleNum+=1
            if sampleNum<=100:
                cv2.imwrite(path1+"/"+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.waitKey(100)
            if 100<sampleNum:
                cv2.imwrite(path2+"/"+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.waitKey(100)
        cv2.imshow("face",img)
        cv2.waitKey(1)
        if sampleNum>120:
                break;     
        
    cam.release()
    cv2.destroyAllWindows()
            








# show augmented images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


    



def train_model():
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



    #training_data.class_indices  , valid_data.class_indices



    #images = [training_data[0][0][0] for i in range(5)]
    #plotImages(images)
    # save best model using vall accuracy
    model_path = '/home/user/Documents/TUTORIALS/project/attendence_model.h5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]




    #Building cnn model
    cnn_model = keras.models.Sequential([
                                    keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150, 150, 3]),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(filters=64, kernel_size=3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(filters=128, kernel_size=3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),                                    
                                    keras.layers.Conv2D(filters=256, kernel_size=3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),

                                    keras.layers.Dropout(0.5),                                                                        
                                    keras.layers.Flatten(), # neural network beulding
                                    keras.layers.Dense(units=128, activation='relu'), # input layers
                                    keras.layers.Dropout(0.1),                                    
                                    keras.layers.Dense(units=256, activation='relu'),                                    
                                    keras.layers.Dropout(0.25),                                    
                                    keras.layers.Dense(units=4, activation='softmax') # output layer
                                    ])


    # compile cnn model
    cnn_model.compile(optimizer = Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])




    history = cnn_model.fit(training_data, 
                          epochs=50, 
                          verbose=1,
                          validation_data=valid_data,
                          callbacks=callbacks_list) 



# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()



#valid_data.classes



#cnn_model.predict_classes(valid_data)


#student=detect_student()







