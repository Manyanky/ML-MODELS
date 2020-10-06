#numpy recap
print("Using numpy")
import numpy as np
a = np.zeros((2,2))
b = np.ones((2,2))
print(a)
print(b)
print (np.sum(b, axis=1)) 		# array([ 2., 2.])
print (a.shape) 			#(2, 2)
print (np.reshape(a, (1,4)))    	#array([[ 0., 0., 0., 0.]])

print("Using TensorFlow")
#Repeat in TensorFlow
import tensorflow as tf
tf.InteractiveSession()
a = tf.zeros((2,2)); b = tf.ones((2,2))
print(a)
print(b)

print (tf.reduce_sum(b, reduction_indices=1).eval()) 	#array([ 2., 2.], dtype=float32)
print (a.get_shape()) 					#TensorShape([Dimension(2), Dimension(2)])
print (tf.reshape(a, (1, 4)).eval()) 			#array([[ 0., 0., 0., 0.]], dtype=float32)