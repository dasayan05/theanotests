'''
A Hidden layer
'''

__author__ = 'Ayan Das'

import theano.tensor as T
from theano import config as cfg
from theano import shared
from numpy import zeros, sqrt
from numpy.random import uniform, random

class Hidden (object):
	
	'''
	A class that represents Hidden
	layer symbolically
	'''
	
	__COUNT = 0
	
	def __init__ (self, X, n_in, n_out, activation_func = T.nnet.sigmoid, param_init = ( 'glorot', 'zero' )):
		
		'''
		X: the input matrix (design matrix) should be of (m X n_in)
		Y: the output matrix should be of (m X n_out)
		W will be weight matrix of (n_in X n_out)
		B will be bias vector of (n_out, 1)
		
		activation_func: the activation function of this layer
		should be a theano symbolic function, like T.nnet.sigmoid
		'''
		
		Hidden.__COUNT += 1 # to keep track of number of
							# hidden layers so that the
							# symbolic names do not mess up

		self.W_shape = (n_in, n_out)
		
		if param_init[0] == 'glorot' :
			W_default = uniform (
                    low=-4*sqrt(6. / (n_in + n_out)),
                    high=4*sqrt(6. / (n_in + n_out)),
                    size = self.W_shape )
		elif param_init[0] == 'zeros' :
			W_default = zeros (
				self.W_shape,
				dtype = cfg.floatX
			)
		else:
			W_default = random (
				self.W_shape,
			)
		
		if activation_func is T.tanh:
			W_default = W_default / +4.0
		
		self.W = shared (
			value = W_default,
			borrow = True,
			name = 'WH' + str(Hidden.__COUNT)
		)
		
		self.B_shape = (1, n_out)
		self.B = shared (
			value = zeros (
				self.B_shape,
				dtype = cfg.floatX
			),
			borrow = True,
			name = 'BH' + str(Hidden.__COUNT),
			broadcastable = (True, False)
		)
		
		self.Y_hat = activation_func ( T.dot ( X, self.W ) + self.B )
	
	def output ( self ):
		
		''' Output activation of this layer;
		i.e. forward pass prediction '''
		
		return self.Y_hat
	
	def weightShapes (self) :
		
		''' The shape of the weight and bias matrix '''
		
		return [self.W_shape, self.B_shape]
		
	def regularizer_L2 ( self ) :
		
		''' The L2 regularizer term assosiated with the weights
		of this layer '''
		
		return ( ( self.W**2 ).mean( ) )
	
	def regularizer_L1 ( self ) :
		
		''' The L1 regularizer term assosiated with the weights
		of this layer '''
		
		return ( abs( self.W ).mean( ) )