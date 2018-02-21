from sklearn import svm
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import time

TRAIN_DIR = 'training_data/mix'
TEST_DIR = 'testing_data/mix'
IMG_SIZE = 32

NUMBER_OF_FEATURES = 1024
converter = np.array([0,1])

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[0]
    label = []
    if word_label == 'cat':
        label = 0 #np.array([1,0])
    elif word_label == 'dog':
        label = 1# np.array([0,1])

    return label #np.sum(np.multiply(converter, label))

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
    return traintest_data,np.asarray(traintest_label)


def getCNN_features(train_data):
    df = pd.read_csv('weights.txt')
    weights = df['weight'].as_matrix()
    arr_features = []

    for x in range(0, len(train_data)):
        new_train_features = []
        train_features = train_data[x]
        for y in range(0, len(train_features)):
            new_train_features.append(train_features[y]*weights[y])
        arr_features.append(new_train_features)
    return arr_features

train_data, train_label = create_traintest_data(TRAIN_DIR)
train_features_cnn = getCNN_features(train_data)

test_data, test_label = create_traintest_data(TEST_DIR)
test_features_cnn = getCNN_features(test_data)

#Train SVM
initial_time = time.time()
clf = svm.SVC()
clf.fit(train_data, train_label)
training_time = time.time()-initial_time
print("\nTraining Time = ", training_time)

accuracy = clf.score(test_data, test_label)
print("\nConvNetSVM accuracy =", accuracy)
