import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Hyper Parameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 1e-3
batch_size = 256

# Data Points
data_size = 1000
x_train = np.linspace(-5, 5, data_size).reshape(-1,1)
y_gt = np.power(x_train, 2)
y_train = y_gt + np.random.randn(y_gt.shape[0], y_gt.shape[1])

# Model Parameters
W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden_size]), name='W1')
b1 = tf.Variable(tf.random_normal(shape=[1, hidden_size]), name='b1')
W2 = tf.Variable(tf.random_normal(shape=[hidden_size, output_size]), name='W2')
b2 = tf.Variable(tf.random_normal(shape=[1, output_size]), name='b2')

# Placeholder
x_ = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, output_size])

# Model
def Predict(x_, W1, b1, W2, b2):
    a1 = tf.matmul(x_, W1) + b1
    h1 = tf.nn.relu(a1)
    y_pred = tf.matmul(h1, W2) + b2
    return y_pred

y_pred = Predict(x_, W1, b1, W2, b2)
loss = 0.5 * tf.reduce_mean(tf.square(y_pred - y_))
solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training the Model
loss_rec = []
for i in range(20001):
    batch_id = np.random.choice(data_size, batch_size)
    x_batch = x_train[batch_id]
    y_batch = y_train[batch_id]

    _, loss_np = sess.run([solver, loss], feed_dict={x_: x_batch, y_: y_batch})
    loss_rec.append(loss_np)

    if i%100 == 0:
        print("Iter:", i, ",Loss:", loss_np)

y_pred_np, total_loss = sess.run([y_pred, loss], feed_dict={x_: x_train, y_:y_train})
print(y_pred_np.shape)
print("Total Loss:", total_loss)

print("[Show Fitting Curve]")
plt.plot(x_train, y_train,'b.')
plt.plot(x_train, y_gt, 'g.')
plt.plot(x_train, y_pred_np,'r.')
plt.show()

print("[Show Loss Curve]")
plt.plot(loss_rec[100:])
plt.show()

# Write to file
writer = tf.summary.FileWriter('./log', sess.graph)