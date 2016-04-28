'''
Weights visualization tools
'''

def visualize ( W, sz = (28,28), maprange = (-1.0, +1.0) ) :
	
	'''
	'W' will be a nxl matrix where 'n' is the number of visual objects
	and 'l' is the length of flattened weight
	
	'sz' is the size we want the 'n'th visual objects will be
	and should be resizable from 'l'
	
	maprange is a tuple containing min-max of the desired pixel
	intensity range; better be default.
	'''
	
	from numpy import hstack, empty, uint8
	
	W_sz = W.shape
	assert W_sz[1] == sz[0]*sz[1], 'Sizes not compatible'
	
	line = empty( ( sz[0], 0 ) )
	
	for i in range(W_sz[0]):
		t = W[i,:].reshape(sz)
		t = 0.0 + (t-t.min())/(t.max()-t.min())*255.0
		line = hstack( (line, t.astype(uint8)) )
	
	return line