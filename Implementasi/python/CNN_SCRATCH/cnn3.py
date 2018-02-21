import dataset2
import tensorflow as tf

validation_size = 0.2
img_size = 128
n_classes = 2
batch_size = 32
train_path='training_data'
classes = ['dogs','cats']

#Load dataset and resize them to 28*28
data = dataset2.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
training = data.train
validation = data.valid

x = tf.placeholder('float', shape=[None, img_size, img_size])
y = tf.placeholder('float', )


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               # 'W_conv2':tf.Variable(tf.random_normal([5,5,32,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([32*32*64,128])),
               'out':tf.Variable(tf.random_normal([128, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              # 'b_conv2':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([128])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, img_size, img_size, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    # conv3 = maxpool2d(conv3)

    fc = tf.reshape(conv2,[-1, 32*32*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(x):

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 30
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        saver = tf.train.Saver()

        for epoch in range(hm_epochs):
            epoch_loss = 0
            init_idx = 0
            acc = 0
            for _ in range(int(len(training.images)/batch_size)):
                #Load training data perbatch
                epoch_x = training.images[init_idx:init_idx+batch_size]
                epoch_y = training.labels[init_idx:init_idx+batch_size]
                feed_dict_train = {x: epoch_x, y: epoch_y}

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                #Compute validation accuracy and loss
                acc = sess.run(accuracy, feed_dict=feed_dict_train)

                epoch_loss += c
                init_idx += batch_size

            #Load validation data
            valid_x = validation.images[0:batch_size]
            valid_y = validation.labels[0:batch_size]
            feed_dict_validate = {x: valid_x,y: valid_y}
            #Compute validation accuracy and loss
            val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
            val_loss = sess.run(cost, feed_dict=feed_dict_validate)

            saver.save(sess, "model/cnn_model_bg.ckpt")
            print("Training Epoch %s --- Training Accuracy: %s, Validation Accuracy: %s, Validation Loss: %s"%(str(epoch),str(acc*100),str(val_acc*100),str(val_loss)))

            # print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        # print('Accuracy:',accuracy.eval({x:validation.images, y:validation.labels}))

train_neural_network(x)


