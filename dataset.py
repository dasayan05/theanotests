'''
The module for manipulating Dataset
efficiently.
'''

__author__ = 'Ayan Das'

class DatasetError( Exception ):
	
	''' The dataset related error class '''
	
	def __init__ ( self, arg, extra = '' ):
		
		# arg = 1 means dataset file not found
		if int(arg) == 1:
			self.msg = 'Dataset File not found'
		
		if int(arg) == 2:
			self.msg = 'Problem with UnPickling'

def loadData ( path, gzipped = False ):
	
	''' Load data from a pickled file '''
	# path: a path to the pickle file
	# storeformat: 'ooo' for a list of 3 objects
	
	from os.path import isabs, join as pathjoin	# to check absolute path
	from os.path import isfile
	from os import getcwd
	
	if gzipped: #if gzipped then use gzip.open()
		try:	#instead of open()
			from gzip import open as fileOpen
		except ImportError:
			fileOpen = open
	else:
		fileOpen = open
	
	if not isabs(path):
		path = pathjoin( getcwd(), path )
		
	if not isfile(path):
		raise DatasetError('1') # 1 for File-not-found type error
		return
	
	# Get a 'load' function from Pickle or cPickle
	global ld
	try:
		from cPickle import load as ld
	except ImportError:
		from pickle import load as ld
	
	global Q
	try:
		with fileOpen(path, 'rb') as file:
			Q = ld(file)
	except:
		raise DatasetError('2') # for unpickling failure
	
	return Q
	
def OneToMany ( trg, noclass = 10, zeroOrNegOne = True ):
	
	''' Make the target matrix in SoftMax
	compatible form; trg should be of (m,) '''
	# trg: a linear array with m target labels
	
	from numpy import ndarray, zeros, ones
	
	assert ((isinstance(trg, ndarray) and len(trg.shape)==1) \
	or isinstance(trg, list))
	
	if zeroOrNegOne:
		T = zeros((len(trg), noclass))
	else:
		T = -1.0 * ones((len(trg), noclass))
	
	C = 0
	for l in trg :
		T[C, l] = 1.
		C += 1
	
	return T

from numpy import sum, arange

def ManyToOne( tsty ):
	
	''' Softmax decoder '''
	ret = []
	
	for i in range( tsty.shape[0] ):
		ret.append( sum( tsty[i, :] * arange(10) ) )
	
	return ret