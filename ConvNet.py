'''
A Convolution + Pooling layer
'''

__author__ = 'Ayan Das'

import theano.tensor as T
from theano import config as cfg
from theano import shared
from numpy import zeros, sqrt
from numpy.random import uniform
from theano.tensor.nnet import conv
from theano.tensor.signal.downsample import max_pool_2d
from hiddenlayer import Hidden
from LogisticRegression import LogisticRegression

class ConvPoolLayer (object):
	
	'''
	A class that represents a Convolution +
	pooling layer symbolically
	'''
	
	__COUNT = 0
	
	def __init__ ( self, x, inp_feamaps, num_filters, filter_shape = (7,7), poolsize = (2,2) ) :
		
		'''
		x: a 4D tensor of shape (m_sample, inp_feas, image_row, image_col)
		inp_feas: number of feature maps from previous layer
		num_filters: number of feature maps to the output
		poolsize: the size of max-pooling window
		'''
		ConvPoolLayer.__COUNT += 1
		
		self.x = x
		# keep track of input tensor
		
		self.fan_in = inp_feamaps * filter_shape[0] * filter_shape[1]
		# fan-in of a neuron
		self.fan_out = (num_filters * filter_shape[0] * filter_shape[1]) / (poolsize[0] * poolsize[1])
		# fan-out of a neuron
		
		W_bound = sqrt(6. / (self.fan_in + self.fan_out)) # bounds on initial weights
		
		lowBound, highBound = -W_bound, +W_bound
		
		self.W_shape = ( num_filters, inp_feamaps, filter_shape[0], filter_shape[1] )
		# keep track of the shape of W
		
		self.W = shared (
			value = uniform( low = lowBound, high = highBound,
                size = self.W_shape),
				borrow = True,
				name = 'CF' + str(ConvPoolLayer.__COUNT)
		) # filter weights initialized as uniform distribution
		# between lowBound and highBound
		
		self.B_shape = (num_filters,)
		# keep track of the shape of B
		
		self.B = shared (
			zeros( self.B_shape, dtype = cfg.floatX ),
			borrow = True,
			name = 'CB' + str(ConvPoolLayer.__COUNT)
		) # same for the bias
		# initialized to zeros as usual for biases
		
		self.conv_out = conv.conv2d(
			input = x,
			filters = self.W,
			border_mode = 'full'
		) # Get convolved output
		
		self.pool_out = max_pool_2d(
			input = self.conv_out,
			ds = poolsize,
			ignore_border = True
		) # Get pooled output
		
		self.Y = T.tanh(self.pool_out + self.B.dimshuffle( 'x', 0, 'x', 'x' ))
		
	def predict ( self ) :
		
		''' The output after convolved and pooled '''
		
		return self.Y
		
	def weightshapes (self):
		
		''' The shapes of parameters '''
		
		return [self.W_shape, self.B_shape]

class LeConvNet( object ):
	
	'''
	A Convolutional NN module consists of
	ConvPoolLayer, LogReg and Hidden modules
	specifically, CP->CP->HID->LOGREG (the LeNet)
	'''
	
	def __init__ ( self, x, arch = { 'CP1':[1,4,(7,7),(2,2)], 'CP2':[4,6,(5,5),(2,2)], 'HID':[600,100], 'LogReg':[100,10] }):
		''' Some hard-coded architechture definition; will generalize later
		x: input 4D tensor
		'''
		
		self.Layers = []
		
		self.Layers.append( ConvPoolLayer(x, *arch['CP1']) )
		self.Layers.append( ConvPoolLayer(self.Layers[0].predict( ), *arch['CP2']) )
		self.Layers.append( self.Layers[1].predict().flatten(ndim = 2) )
		self.Layers.append( Hidden( self.Layers[2], *arch['HID'], activation_func = T.tanh ) )
		self.Layers.append( LogisticRegression( self.Layers[3].output(), *arch['LogReg']) )
		
		self.Y_final = self.Layers[4].Y_hat
		self.Y_pred = self.Layers[4].predict()
		
	def predict (self):
		
		''' Predict output of CNN '''
		
		return self.Y_pred
		
	def cost ( self, Y ):
		
		''' Cost function '''
		
		return ((self.Y_final - Y)**2).mean()
		
	def params ( self ):
		
		''' all the parameters '''
		
		return [
			self.Layers[0].W,self.Layers[0].B,
			self.Layers[1].W,self.Layers[1].B,
			# Layer[2] is flatten layer
			# hence doesn't have params
			self.Layers[3].W,self.Layers[3].B,
			self.Layers[4].W,self.Layers[4].B
		]
		
	def paramshapes ( self ):
		
		''' Shapes of all parameters '''
		
		return [
			self.Layers[0].W_shape,self.Layers[0].B_shape,
			self.Layers[1].W_shape,self.Layers[1].B_shape,
            # Layer[2] is flatten layer
            # hence doesn't have params
            self.Layers[3].W_shape,self.Layers[3].B_shape,
            self.Layers[4].W_shape,self.Layers[4].B_shape
		]
		
	def regularization ( self ):
		
		return (self.Layers[3].W**2).mean() + (self.Layers[4].W**2).mean()