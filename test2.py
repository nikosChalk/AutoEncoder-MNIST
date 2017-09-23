
import tensorflow as tf
import numpy
import random

a = tf.Variable(tf.random_normal([2, 100]))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

slice = a[:, 0:4]
slice = sess.run(slice)

a = numpy.array([])
print(a)