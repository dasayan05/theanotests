'''
A bunch of classes for
different Autoencoders
'''

from numpy.random import uniform
from numpy import sqrt, zeros
from theano import pp, shared, scan, tensor as T
from theano.gradient import jacobian

class cA ( object ):
	
	''' The Contractive autoencoder '''
	
	def __init__ ( self, x, n_in, n_hid ):
		
		'''
		x: input (symbolic)
		n_in: input length
		n_hid: hidden size
		'''
		
		self.x = x
		
		self.W_shape = (n_in, n_hid)
		self.W_init = uniform(
			low = -4 * sqrt(6. / (n_in + n_hid)),
			high = 4 * sqrt(6. / (n_in + n_hid)),
			size = self.W_shape
		)
		
		self.W = shared(
			value = self.W_init,
			borrow = True,
			name = 'W'
		)
		
		self.B1_shape = (1, n_hid)
		self.B2_shape = (1, n_in)
		
		self.B1 = shared(
			value = zeros( self.B1_shape ),
			borrow = True,
			name = 'B1',
			broadcastable = (True, False)
		)
		self.B2 = shared(
			value = zeros( self.B2_shape ),
			borrow = True,
			name = 'B2',
			broadcastable = (True, False)
		)
		
		self.encoded = T.nnet.sigmoid( T.dot( self.x, self.W ) + self.B1 )
		
		self.decoded = T.nnet.sigmoid( T.dot( self.encoded, self.W.T ) + self.B2)
		
	def cost ( self, contraction_level = 0.5 ):
		# DEFAULT cA_level should be changed # <-----
		
		# Calculate the jacobian tensor
		
		# def jacobMean ( hv, xv ):
		# 	# calculate the mean of jacobian
		# 	# of h w.r.t. x
		# 	j = jacobian( hv, wrt = xv )
		# 	return (j**2).mean()
		# 
		# jacob, _ = scan (
		# 	fn = jacobMean,
		# 	sequences = [ self.encoded, self.x ],
		# 	outputs_info = None
		# )
		# 
		# self.jacob_loss = jacob.mean()
		
		logloss = - ( self.x * T.log( self.decoded ) + ( 1-self.x ) * T.log ( 1-self.decoded ) )
		
		t = ( self.encoded * ( T.ones_like(self.encoded) - self.encoded ) )**2
		Wts = ((self.W.T**2).mean(axis=1,keepdims=True)).T
		self.jacob_loss = (t * Wts).mean() # <--- should be t * Wts
		
		return logloss.mean() + self.jacob_loss * contraction_level
		
	def regularizer ( self, lam ):
		
		return ( self.W**2 ).mean() * lam
		
	def paramshapes ( self ):
		
		return [ self.W_shape, self.B1_shape, self.B2_shape ]