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
    "activation": None,
}

pool_params = {
    "pool_size": 2,
    "strides": 2,
    "padding": 'valid'
}

# Model
def CNN(x, training, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        h_conv1 = tf.layers.conv2d(x, kernel_size=5, filters=16, **conv_params, **init_params)
        h_conv1 = tf.layers.batch_normalization(h_conv1, training=training)
        h_conv1 = tf.nn.relu(h_conv1)
        h_pool1 = tf.layers.max_pooling2d(h_conv1, **pool_params)

        h_conv2 = tf.layers.conv2d(h_pool1, kernel_size=3, filters=32, **conv_params, **init_params)
        h_conv2 = tf.layers.batch_normalization(h_conv2, training=training)
        h_conv2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.layers.max_pooling2d(h_conv2, **pool_params)
        
        h_flat2 = tf.reshape(h_pool2, [-1,7*7*32])
        h_fc3 = tf.layers.dense(h_flat2, 256, activation=tf.nn.relu, **init_params)
        h_fc3 = tf.layers.dropout(h_fc3, rate=0.4, training=training)
        
        y_logit = tf.layers.dense(h_fc3, 10, activation=None, **init_params)
        y_prob = tf.nn.softmax(y_logit)

        return y_prob, y_logit

# Placeholder
x_ = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y')

# Loss and Solver
y_prob, y_logit = CNN(x_, training=True, name="MNIST_Model", reuse=False) # tf.AUTO_REUSE
y_prob_eval, y_logit_eval = CNN(x_, training=False, name="MNIST_Model", reuse=True)
var_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MNIST_Model")
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

print("Model Parameters:")
for var in var_model:
    print(var)

print("Update Operations:")
for op in update_ops:
    print(op)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y_logit))
with tf.control_dependencies(update_ops):
    solver = tf.train.AdamOptimizer(2e-4).minimize(loss, var_list=var_model)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_prob_eval,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
batch_size = 64
for i in range(20000):
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