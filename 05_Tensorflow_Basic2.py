import tensorflow as tf
import numpy as np

t1 = tf.Variable(tf.ones([2,3], tf.float32, name='t1_init'), name='t1_var')
t2 = tf.Variable(tf.random_normal([3,4], name='t2_init'), name='t2_var')
t3 = tf.matmul(t1, t2, name='t3')
t4 = tf.placeholder(tf.float32, shape=[2, 4], name='t4_ph')
t5 = tf.add(t3, t4, name='t5')

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Result
t1_np, t2_np, t3_np = sess.run([t1, t2, t3])
t4_np = sess.run(t4, feed_dict={t4: np.ones([2, 4])})
t5_np = sess.run(t5, feed_dict={t4: np.ones([2, 4])})

print(t1_np)
print(t2_np)
print(t3_np)
print(t4_np)
print(t5_np)

# Write to file
writer = tf.summary.FileWriter('./log', sess.graph)
