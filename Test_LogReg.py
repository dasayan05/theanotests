from numpy import empty, inf, zeros, array, abs, count_nonzero
from matplotlib.pyplot import ion, draw, plot, savefig
from cv2 import imwrite, waitKey
from LogisticRegression import LogisticRegression as LogReg
from theano import function, pp, config as cfg
from time import sleep
# cfg.openmp = True
import theano.tensor as T
from dataset import loadData, OneToMany
from visual import visualize

Tr, Ts, _ = loadData('mnist.pkl.gz', True)
m_sample = Tr[0].shape[0]
m_test_sample = Ts[1].shape[0]

x, y = T.dmatrices('x', 'y')
L = LogReg(x, 784, 10)
lam = 0.04

p = L.predict()
l = L.cost(y) + L.regularizer(lam)
gw = T.grad (l, wrt=L.W)
gb = T.grad (l, wrt=L.B)
alpha = 0.05

W_shape = L.weightShapes()[0]
B_shape = L.weightShapes()[1]
VW = zeros(W_shape)
VB = zeros(B_shape)

train = function ([x,y], [l, gw, gb])
test = function ([x], p)
print 'Function Compiled'

EPOCHS = 350
verbose = True

ion()

errAcc = []
TOTAL_BATCHES = 100
MINI_BATCH = m_sample/TOTAL_BATCHES
zeta = 0.98
global vis
for i in range(EPOCHS):
	n_batch = 0
	GW = zeros(W_shape)
	GB = zeros(B_shape)
	Err = inf
	for x in range(0, m_sample-MINI_BATCH, MINI_BATCH):
		err, gW, gB = train(Tr[0][x:x+MINI_BATCH,:], OneToMany(Tr[1][x:x+MINI_BATCH]))
		n_batch += 1
		if n_batch==TOTAL_BATCHES/2-1 or n_batch==TOTAL_BATCHES-1: # to make logs shorter
			if verbose:
				print 'Error in Training batch %d/%d/%d/%d' % (n_batch+1, TOTAL_BATCHES, i, EPOCHS)
		GW = (GW*(n_batch-1) + gW)/n_batch # Accumulate average gradient of W
		GB = (GB*(n_batch-1) + gB)/n_batch # Accumulate average gradient of B
		if err < Err:
			Err = err # Accumulate minimum mini-batch error
	if verbose:
		print 'Min error in epoch %d/%d is %f' % (i, EPOCHS, Err)
	VW = VW*zeta - alpha * GW
	VB = VB*zeta - alpha * GB
	L.W.set_value( L.W.get_value() + VW )
	L.B.set_value( L.B.get_value() + VB )
	
	Q = abs(array(test( Ts[0] ))-Ts[1])
	if verbose:
		print m_test_sample - count_nonzero(Q)
	errAcc.append(Err)

plot(errAcc)
vis = visualize ( L.W.get_value().transpose() )
imwrite('weights'+'.png', vis)
savefig('plot_'+str(EPOCHS)+'_'+str(TOTAL_BATCHES)+'_'+str(alpha)+'_'+str(zeta)+'_'+str(lam)+'best'+'.png')
fl = open('LR_weights', 'wb')
import pickle
pickle.dump([L.W.get_value(), L.B.get_value()], fl)
fl.close()