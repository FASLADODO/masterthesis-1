import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import operator
import dataset


# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
# file_object = "mix/1.jpg"
# image_path= 'testing_data/'+file_object
# filename = dir_path +'/' +image_path
# image_size=128
# num_channels=3
# images = []
# # Reading the image using OpenCV
# image = cv2.imread(filename)
# # Resizing the image to our desired size and preprocessing will be done exactly as done during training
# image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
# images.append(image)
# images = np.array(images, dtype=np.uint8)
# images = images.astype('float32')
# images = np.multiply(images, 1.0/255.0)
# #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
# x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('model/cnn_model.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, 'model/cnn_model.ckpt') #tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))



### Creating the feed_dict that is required to be fed to calculate y_pred
# feed_dict_testing = {x: x_batch, y_true: y_test_images}
# result=sess.run(y_pred, feed_dict=feed_dict_testing)
#
# classes = ['dog','cat']
# index, value = max(enumerate(result[0]), key=operator.itemgetter(1))
# label = classes[index]
# cv2.putText(image,label, (6,100), cv2.FONT_HERSHEY_DUPLEX, 2, 255)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #get weight
# trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# weights_last_layer = sess.run(trainable_variables[7])
# np.savetxt('weights.txt', weights_last_layer, delimiter=',')
# print("Training done")

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[0]
    label = []
    if word_label == 'cat':
        label = 0
    elif word_label == 'dog':
        label = 1

    return label

train_path='training_data'
validation_size = 0.2
img_size = 128
num_channels = 3
BATCH_SIZE = 32
classes = ['dogs','cats']
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
train_data = data.train.img_names
train_labels = data.train.labels
TRAIN_SIZE = len(train_data)
num_iter = TRAIN_SIZE/BATCH_SIZE
NUMBER_OF_FEATURES = 1024
converter = np.array([0,1])

train_features_cnn = np.zeros((TRAIN_SIZE, NUMBER_OF_FEATURES), dtype=float)
train_labels_cnn = np.zeros(TRAIN_SIZE, dtype=int)
# test_labels_cnn = np.zeros(TEST_SIZE, dtype=int)

trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

for i in range(0,num_iter):
    train_batch, y_true_batch, _, cls_batch = data.train.next_batch(BATCH_SIZE)
    features_batch = sess.run(trainable_variables[7], feed_dict={x: train_batch})
    labels_batch = y_true_batch
    for j in range(BATCH_SIZE):
        for k in range(NUMBER_OF_FEATURES):
            train_features_cnn[BATCH_SIZE * i + j, k] = features_batch[j, k]
            train_labels_cnn[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

print("test")

