'''Testing the data''' 
import tflearn 
import argparse
import imutils
import time
import cv2
import os
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression   
import numpy as np 
from tqdm import tqdm 
from imutils.video import VideoStream
import pickle

IMG_SIZE = 224
LR = 1e-3

faceList = []

with open('facelist.data', 'rb') as filehandle:  
    # read the data as binary data stream
    faceList = pickle.load(filehandle)

# if you need to create the data: 
# test_data = process_test_data() 
# if you already have some saved:

convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input')   
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 2, activation ='softmax') 
convnet = regression(convnet, optimizer ='rmsprop', learning_rate = LR, to_one_hot = True, 
    n_classes = 2,
    loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log') 

model.load('./face.model')

test_data = np.load('test_data.npy') 
  

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(h, w) = (None, None)
zeros = None


time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream, clone it, (just
    # in case we want to write it to disk), and then resize the frame
    # so we can apply face detection faster
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
 
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30))
 
    # loop over the face detections and draw them on the frame
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped = frame[y:y+h,x:x+w]
        grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(grey, (IMG_SIZE, IMG_SIZE)) 
        data = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

        model_out = model.predict(data)

        str_label = faceList[np.argmax(model_out)]

        top = y - 15 if y - 15 > 15 else y + 15

        cv2.putText(frame, ""+str_label, (x, top), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)

    # show the output frame
    if writer is None:
        # store the image dimensions, initialzie the video writer,
        # and construct the zeros array
        (he, wi) = frame.shape[:2]
        writer = cv2.VideoWriter('output.avi', fourcc, 20,
            (wi, he), True)
        zeros = np.zeros((he, wi), dtype="uint8")
 
    # construct the final output frame, storing the original frame
    # at the top-left, the red channel in the top-right, the green
    # channel in the bottom-right, and the blue channel in the
    # bottom-left
    output = np.zeros((he, wi, 3), dtype="uint8")
    output[0:he, 0:wi] = frame
    # write the output frame to file
    writer.write(output)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

writer.release()
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()