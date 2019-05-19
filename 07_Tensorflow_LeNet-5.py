import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# Visualize Dataset
vid = 100
x_samp = x_train[vid].reshape([28,28])
y_samp = np.argmax(y_train[vid])
print(y_samp)
plt.imshow(x_samp, cmap='gray')
plt.show()

# Variables
w_conv1 = tf.Variable(tf.random_normal(shape=[9,9,1,6]), name='w_conv1')
b_conv1 = tf.Variable(tf.random_normal(shape=[6]), name='b_conv1')
w_conv2 = tf.Variable(tf.random_normal(shape=[5,5,6,16]), name='w_conv2')
b_conv2 = tf.Variable(tf.random_normal(shape=[16]), name='b_conv2')
w_fc3 = tf.Variable(tf.random_normal(shape=[5*5*16,120]), name='w_fc3')
b_fc3 = tf.Variable(tf.random_normal(shape=[120]), name='b_fc3')
w_fc4 = tf.Variable(tf.random_normal(shape=[120,84]), name='w_fc4')
b_fc4 = tf.Variable(tf.random_normal(shape=[84]), name='b_fc4')
w_fc5 = tf.Variable(tf.random_normal(shape=[84,10]), name='w_fc5')
b_fc5 = tf.Variable(tf.random_normal(shape=[10]), name='b_fc5')

# Model
def LaNet5(x):
    h_conv1 = tf.nn.conv2d(x ,w_conv1 ,strides=[1,1,1,1], padding='SAME') + b_conv1
    h_conv1 = tf.nn.relu(h_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')  

    h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2
    h_conv2 = tf.nn.relu(h_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')  
    
    h_flat2 = tf.reshape(h_pool2, [-1,5*5*16])
    h_fc3 = tf.matmul(h_flat2, w_fc3) + b_fc3
    h_fc3 = tf.nn.relu(h_fc3)

    h_fc4 = tf.matmul(h_fc3, w_fc4) + b_fc4
    h_fc4 = tf.nn.relu(h_fc4)

    y_logit = tf.matmul(h_fc4, w_fc5) + b_fc5
    y_prob = tf.nn.softmax(y_logit)

    return y_prob, y_logit

# Placeholder
x_ = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y')

# Loss and Solver
y_prob, y_logit = LaNet5(x_)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_logit))
solver = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_prob,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
batch_size = 64
for i in range(10000):
    batch_id = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_id].reshape([-1,28,28,1])
    y_batch = y_train[batch_id]
    _, loss_, acc_ = sess.run([solver, loss, acc], feed_dict={x_: x_batch, y_: y_batch})

    if i%100 == 0:
        print('Iter', i, 'Loss:', loss_, 'Acc:', acc_)

# Result
train_acc = sess.run(acc, feed_dict={x_:x_train.reshape([-1,28,28,1]), y_:y_train})
test_acc = sess.run(acc, feed_dict={x_:x_test.reshape([-1,28,28,1]), y_:y_test})
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)