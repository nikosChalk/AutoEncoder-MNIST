
import tensorflow as tf
import numpy
import random

a = tf.Variable(tf.random_normal([2, 100]))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

slice = a[:, 0:4]
slice = sess.run(slice)

a = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(a[:, 0:2])