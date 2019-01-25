import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm
import pickle
  
'''Setting up the env'''
  
TRAIN_DIR = 'C:/Users/asus/facedetect/trial/training'
TEST_DIR = 'C:/Users/asus/facedetect/trial/testing'
IMG_SIZE = 224
LR = 1e-3

'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'face.model'.format(LR, '6conv-basic') 

face_list = []

'''Labelling the dataset'''
def label_img(img): 
    word_label = img.split('.')[0] 
    
    if word_label in face_list:
        return face_list.index(word_label)
    else:
        face_list.append(word_label)
        return face_list.index(word_label)
  
'''Creating the training data'''
def create_train_data(): 
    # Creating an empty list where we should the store the training data
    # after a little preprocessing of the data
    training_data = [] 
  
    # tqdm is only used for interactive loading 
    # loading the training data 
    for img in tqdm(os.listdir(TRAIN_DIR)): 
  
        # labeling the images 
        label = label_img(img)
  
        path = os.path.join(TRAIN_DIR, img) 
  
        # loading the image from the path and then converting them into 
        # greyscale for easier covnet prob 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
  
        # resizing the image for processing them in the covnet 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
  
        # final step-forming the training data list with numpy array of the images 
        training_data.append([np.array(img), label]) 
  
    # shuffling of the training data to preserve the random state of our data 
    shuffle(training_data) 
  
    # saving our trained data for further uses if required 
    np.save('train_data.npy', training_data) 
    return training_data 
  
'''Processing the given test data'''
# Almost same as processing the traning data but 
# we dont have to label it. 
def process_test_data(): 
    testing_data = [] 
    for img in tqdm(os.listdir(TEST_DIR)): 
        path = os.path.join(TEST_DIR, img) 
        img_num = label_img(img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
        testing_data.append([np.array(img), img_num]) 
          
    shuffle(testing_data) 
    np.save('test_data.npy', testing_data) 
    return testing_data 
  
'''Running the training and the testing in the dataset for our model'''
train_data = create_train_data() 
test_data = process_test_data() 


with open('facelist.data', 'wb') as filehandle:  
    # store the data as binary data stream
    pickle.dump(face_list, filehandle)