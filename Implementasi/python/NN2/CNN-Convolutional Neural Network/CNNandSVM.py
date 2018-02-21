from sklearn import svm
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import six
import pandas as pd
import time

TRAIN_DIR = 'train'
TEST_DIR = 'test1'
IMG_SIZE = 32

NUMBER_OF_FEATURES = 1024
converter = np.array([0,1])

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    label = []
    if word_label == 'cat':
        label = np.array([1,0])
    elif word_label == 'dog':
        label =  np.array([0,1])

    return np.sum(np.multiply(converter, label))

def create_traintest_data(TRAINTEST_DIR):
    traintest_data = []
    traintest_label = []
    for img in tqdm(os.listdir(TRAINTEST_DIR)):
        path = os.path.join(TRAINTEST_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img_data)
        img_arr_flatten = img_arr.flatten()
        traintest_data.append(img_arr_flatten)
        traintest_label.append(create_label(img))
    return traintest_data,traintest_label


def getCNN_features(train_data):
    df = pd.read_csv('weights.txt')
    weights = df['weight'].as_matrix()
    arr_features = []

    for x in range(0, len(train_data)):
        arr_features.append(train_data[x]*weights[x])
    return  arr_features

train_data, train_label = create_traintest_data(TRAIN_DIR)
train_features_cnn = getCNN_features(train_data)

test_data, test_label = create_traintest_data(TEST_DIR)
test_features_cnn = getCNN_features(test_data)

#Train SVM
initial_time = time.time()
clf = svm.SVC()
clf.fit(train_features_cnn, train_label)
training_time = time.time()-initial_time
print("\nTraining Time = ", training_time)

accuracy = clf.score(test_features_cnn, test_label)
print("\nConvNetSVM accuracy =", accuracy)
