from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
import numpy as np
from keras_vggface import utils
import scipy.io
from PIL import Image
import glob

# Layer Features
model = 'vgg16'
layer_name = 'fc7'
vgg_model = VGGFace(model=model)
out = vgg_model.get_layer(layer_name).output
vgg_model_new = Model(vgg_model.input, out)

trainingfemale_directory = "C:/Users/Newbie/Desktop/MasterThesis/masterthesis/Implementasi/matlab/GENDER-FERET/female/training_set/*.jpg"
trainingmale_directory = "C:/Users/Newbie/Desktop/MasterThesis/masterthesis/Implementasi/matlab/GENDER-FERET/male/training_set/*.jpg"
testfemale_directory = "C:/Users/Newbie/Desktop/MasterThesis/masterthesis/Implementasi/matlab/GENDER-FERET/female/test_set/*.jpg"
testmale_directory = "C:/Users/Newbie/Desktop/MasterThesis/masterthesis/Implementasi/matlab/GENDER-FERET/male/test_set/*.jpg"

#Load image
trainingfeaturelist = []
testfeaturelist = []

def extractFeatures(featurelist,directory):
    for filename in sorted(glob.glob(directory)):
        img = image.load_img(filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        preds = vgg_model_new.predict(x)
        featurelist.append(preds[0])
    return featurelist;

# Extract training features
trainingfeaturelist = extractFeatures(trainingfeaturelist,trainingfemale_directory);
trainingfeaturelist = extractFeatures(trainingfeaturelist,trainingmale_directory);

# Extract testing features
testfeaturelist = extractFeatures(testfeaturelist,testfemale_directory);
testfeaturelist = extractFeatures(testfeaturelist,testmale_directory);

scipy.io.savemat('vggfacefeatures16.mat', dict(trainingFeatures=trainingfeaturelist,testFeatures=testfeaturelist))
print('Extracting features done')