import tensorflow as tf
import numpy as np

# tf.zeros(shape, dtype, name)
t1 = tf.zeros([2,3], tf.float32, 't1')
t2 = tf.ones([3,4], tf.float32, 't2')
t3 = tf.matmul(t1, t2, name='t3')

# tf.constant(value, dtype, shape, name)
t4 = tf.constant([[1, 2, 3], [4, 5, 6]], tf.float32, name='t4')
t5 = tf.constant(5, tf.float32, [3,4], name='t5')
t6 = tf.matmul(t4, t5, name='t6')
t7 = tf.add(t3, t6, name='t7')

# Session
sess = tf.Session()

# Result
t1_np = sess.run(t1)
t2_np = sess.run(t2)
t3_np = sess.run(t3)
t4_np, t5_np, t6_np, t7_np = sess.run([t4, t5, t6, t7])

print(t1_np)
print(t2_np)
print(t3_np)
print(t4_np)
print(t5_np)
print(t6_np)
print(t7_np)

# Write to file
writer = tf.summary.FileWriter('./log', sess.graph)
