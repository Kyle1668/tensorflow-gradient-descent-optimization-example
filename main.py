import tensorflow as tf
import numpy as np

x = tf.Variable([1., 10., 100])
y = tf.abs(x)

session = tf.Session()

opt = tf.train.GradientDescentOptimizer(0.1).minimize(y)
init = tf.global_variables_initializer()
session.run(init)

for i in range(1, 101):
	s = session.run([opt, x, y])	
	print("-------------------------------")
	print("Iteration: ", i)
	print("Optimizer: ",  s[0])	
	print("X: ", s[1])
	print("Y: ", s[2])

