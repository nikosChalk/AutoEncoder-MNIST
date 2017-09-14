
import tensorflow as tf

a = tf.Variable(tf.random_normal([2, 100]))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

slice = a[:, 0:4]
slice = sess.run(slice)

print(slice)