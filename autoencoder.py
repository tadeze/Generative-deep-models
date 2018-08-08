import tensorflow as tf 


class AutoEncoder(object):
	def __init__(self,):
		pass
	def fit(self,n_dimensions):
		graph = tf.Graph()
		with graph.as_default():
			# Input variable 
			X = tf.placeholder(self.dtype, shape=(None,
				self.features.shape[1]))
			# Network variables 
			encoder_weights = tf.Variable(tf.random_normal())