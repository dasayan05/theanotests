from numpy import empty, inf, zeros, array, abs, count_nonzero
from matplotlib.pyplot import ion, draw, plot, savefig
from cv2 import imwrite, waitKey
from LogisticRegression import LogisticRegression as LogReg
from theano import function, pp, config as cfg
from time import sleep
import theano.tensor as T
from dataset import loadData, OneToMany
from visual import visualize
from hiddenlayer import Hidden

# Dataset
Tr, Ts, _ = loadData('mnist.pkl.gz', True)
m_sample = Tr[0].shape[0]
m_test_sample = Ts[1].shape[0]

x, y = T.dmatrices('x', 'y')
H = Hidden( x, 784, 50, param_init = ( 'glorot', 'zero' ) , activation_func = T.tanh )
h = H.output( )
L = LogReg( h, 50, 10, param_init = ( 'zero', 'zero' ) )

lam = 0.02 # regularizer
alpha = 0.8 # learning rate/ weight decay
zeta = 0.995 # nestrov momentum

global cost
if lam is None:
	cost = L.cost( y )
else:
	cost = L.cost( y ) + L.regularizer ( {'weights':lam, 'bias':0.0} ) \
	+ lam * H.regularizer_L2( ) + 0.02 * H.regularizer_L1( )
pred = L.predict ( )
gw = T.grad (cost, wrt=L.W)
gb = T.grad (cost, wrt=L.B)
gwh1 = T.grad ( cost, wrt=H.W )
gbh1 = T.grad ( cost, wrt=H.B )

W_shape, B_shape = L.weightShapes()
WH1_shape, BH1_shape = H.weightShapes()
VW = zeros(W_shape)
VB = zeros(B_shape)
VWH1 = zeros(WH1_shape)
VBH1 = zeros(BH1_shape)

train = function( [x,y], [cost, gw, gb, gwh1, gbh1] )
predict = function ( [x], pred )
print 'Function Compiled'

# ---------------------------------------------------------------------------

EPOCHS = 200 # total epochs
verbose = True

ion()
errAcc = []
TOTAL_BATCHES = 1000
MINI_BATCH = m_sample/TOTAL_BATCHES

global vis

for i in range(EPOCHS):
	n_batch = 0
	GW = zeros(W_shape)
	GB = zeros(B_shape)
	GWH1 = zeros( WH1_shape )
	GBH1 = zeros( BH1_shape )
	Err = inf
	for x in range(0, m_sample-MINI_BATCH, MINI_BATCH):
		err, gW, gB, gWH1, gBH1 = train(Tr[0][x:x+MINI_BATCH,:], OneToMany(Tr[1][x:x+MINI_BATCH], zeroOrNegOne = True))
		n_batch += 1
		if n_batch==TOTAL_BATCHES/2-1 or n_batch==TOTAL_BATCHES-1: # to make logs shorter
			if verbose:
				print '%f Error in Training batch %d/%d/%d/%d' % (err, n_batch+1, TOTAL_BATCHES, i, EPOCHS)
		GW = (GW*(n_batch-1) + gW)/n_batch # Accumulate average gradient of W
		GB = (GB*(n_batch-1) + gB)/n_batch # Accumulate average gradient of B
		GWH1 = (GWH1*(n_batch-1) + gWH1)/n_batch # Accumulate average gradient of WH1
		GBH1 = (GBH1*(n_batch-1) + gBH1)/n_batch # Accumulate average gradient of BH1
		if err < Err:
			Err = err # Accumulate minimum mini-batch error
	if verbose:
		print '			Min error in epoch %d/%d is %f' % (i, EPOCHS, Err)
	VW = VW*zeta - alpha * GW
	VB = VB*zeta - alpha * GB
	VWH1 = VWH1*zeta - alpha * GWH1
	VBH1 = VBH1*zeta - alpha * GBH1
	L.W.set_value( L.W.get_value() + VW )
	L.B.set_value( L.B.get_value() + VB )
	H.W.set_value( H.W.get_value() + VWH1 )
	H.B.set_value( H.B.get_value() + VBH1 )
		
	Q = abs( array( predict ( Ts[0] ) ) - Ts[1] )
	if verbose:
		print m_test_sample - count_nonzero(Q)
	errAcc.append(Err)

	plot(errAcc)
	savefig('plot_'+str(EPOCHS)+'_'+str(TOTAL_BATCHES)+'_'+str(alpha)+'_'+str(zeta)+'_'+str(lam)+'best'+'.png')


	vis = visualize ( H.W.get_value().transpose()[:10,:] )
	imwrite('weights_mlp'+str(i)+'.png', vis)
# savefig('plot_'+str(EPOCHS)+'_'+str(TOTAL_BATCHES)+'_'+str(alpha)+'_'+str(zeta)+'_'+str(lam)+'best'+'.png')
# fl = open('LR_weights', 'wb')
# import pickle
# pickle.dump([L.W.get_value(), L.B.get_value()], fl)
# fl.close() '''