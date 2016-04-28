'''
A logistic Regression layer
'''

__author__ = 'Ayan Das'

import theano.tensor as T
from theano import config as cfg
from theano import shared
from numpy import zeros, sqrt
from numpy.random import uniform, random

class LogisticRegression (object):
	
	'''
	A class that represents LogisticRegression
	layer symbolically
	'''
	
	def __init__ (self, X, n_in, n_out, param_init = ( 'zero', 'zero' ), activation_func = T.nnet.softmax):
		
		'''
		X: the input matrix (design matrix) should be of (m X n_in)
		Y: the output matrix should be of (m X n_out)
		W will be weight matrix of (n_in X n_out)
		B will be bias vector of (n_out, 1)
		
		param_init: a tuple containing a pair of strings
		indicating the initial values of W and B
		'''	

		self.W_shape = (n_in, n_out)
		
		if param_init[0] == 'glorot' :
			W_default = uniform (
                    low=-sqrt(6. / (n_in + n_out)),
                    high=sqrt(6. / (n_in + n_out)),
                    size = self.W_shape )
		elif param_init[0] == 'zero' :
			W_default = zeros (
				self.W_shape,
				dtype = cfg.floatX
			)
		else:
			W_default = random (
				self.W_shape,
			)
		
		self.W = shared (
			value = W_default,
			borrow = True,
			name = 'W'
		)
		
		self.B_shape = (1, n_out)
		self.B = shared (
			value = zeros (
				self.B_shape,
				dtype = cfg.floatX
			),
			borrow = True,
			name = 'B',
			broadcastable = (True, False)
		)
		
		self.Y_hat = activation_func ( T.dot ( X, self.W ) + self.B )
		
		self.Y_pred = T.argmax ( self.Y_hat, axis = 1 )
		
	def cost ( self, Y ):
		
		''' The cost function symbolically '''
		return ((self.Y_hat-Y)**2).mean()
		
	def regularizer ( self, lamb ):
		
		''' Regularizer function 
		In case different Lambda for W & B
		use a dict like {'weights':w_lamb, 'bias':b_lamb}
		'''
		assert isinstance(lamb, int) or isinstance(lamb, float) or isinstance(lamb, dict)
		
		if isinstance(lamb, dict):
			return lamb['weights'] * (self.W**2).mean() + lamb['bias'] * (self.B**2).mean()
		else:
			return lamb * ( (self.W**2).mean() + (self.B**2).mean() )
		
	def predict (self):
		
		''' The forward pass; i.e. the prediction '''
		
		return self.Y_pred
	
	def weightShapes (self) :
		
		''' The shape of the weight and bias matrix '''
		
		return [self.W_shape, self.B_shape]