import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images.reshape([-1,28,28,1])
y_train = mnist.train.labels
x_test = mnist.test.images.reshape([-1,28,28,1])
y_test = mnist.test.labels

w_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(0.1)

init_params = {
    "kernel_initializer": w_init,
    "bias_initializer": b_init,
}

conv_params = {
    "strides": 1,
    "padding": 'same',
    "activation": tf.nn.relu,
}

pool_params = {
    "pool_size": 2,
    "strides": 2,
    "padding": 'valid'
}

def Residual(x):
    h_conv1 = tf.layers.conv2d(x, kernel_size=3, filters=32, **conv_params, **init_params)
    h_conv2 = tf.layers.conv2d(h_conv1, kernel_size=3, filters=32, **conv_params, **init_params)
    h_conv3 = tf.layers.conv2d(h_conv2, kernel_size=3, filters=32, **conv_params, **init_params)
    h_conv3 = h_conv3 + h_conv1
    h_pool3 = tf.layers.max_pooling2d(h_conv3, **pool_params)
    return h_pool3

# Model
def ResidualNet(x):
    h_res1 = Residual(x)
    h_res2 = Residual(h_res1)
    h_reshape2 = tf.reshape(h_res2, [-1, 7*7*32])
    y_logit = tf.layers.dense(h_reshape2, 10, **init_params)
    y_prob = tf.nn.softmax(y_logit)
    return y_prob, y_logit

# Placeholder
x_ = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y')

# Loss and Solver
y_prob, y_logit = ResidualNet(x_)
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y_logit))
solver = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_prob,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
batch_size = 64
for i in range(5001):
    batch_id = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_id]
    y_batch = y_train[batch_id]
    _ = sess.run(solver, feed_dict={x_: x_batch, y_: y_batch})

    if i%100 == 0:
        batch_id = np.random.choice(x_test.shape[0], 512)
        x_test_batch = x_test[batch_id]
        y_test_batch = y_test[batch_id]
        loss_, acc_ = sess.run([loss, acc], feed_dict={x_: x_test_batch, y_: y_test_batch})
        print('Iter', i, 'Loss:', loss_, 'Acc:', acc_)