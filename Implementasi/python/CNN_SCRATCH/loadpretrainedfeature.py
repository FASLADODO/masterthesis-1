import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import operator

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('model/cnn_model.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, 'model/cnn_model.ckpt') #tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("y_true:0")

print(w1)
#
# all_weights = []
# for layer in saver.layers:
#    w = layer.get_weights()
#    all_weights.append(w)
