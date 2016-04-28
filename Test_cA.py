from theano import pp, function, tensor as T
from Autoencoder import cA
from dataset import loadData, OneToMany
from numpy import zeros
from matplotlib.pyplot import ion, draw, plot, savefig
from visual import visualize
from cv2 import resize, imwrite
from theano import config as cfg
cfg.openmp = True

Tr, Ts, _ = loadData('mnist.pkl.gz', True)
m_sample = Tr[0].shape[0]
m_test_sample = Ts[1].shape[0]

# Hyper-Parameters
alpha = 0.1
zeta = 0.995

x = T.dmatrix('x')
CA = cA(x, 784, 100)
xr = CA.decoded
xh = CA.encoded
cost = CA.cost( 0.1 )
gw = T.grad(cost, wrt=CA.W)
gb1 = T.grad(cost, wrt=CA.B1)
gb2 = T.grad(cost, wrt=CA.B2)

W_shape = CA.paramshapes()[0]
B1_shape = CA.paramshapes()[1]
B2_shape = CA.paramshapes()[2]
VW = zeros(W_shape)
VB1 = zeros(B1_shape)
VB2 = zeros(B2_shape)

train = function([x], [cost, gw, gb1, gb2])
reconstruct = function([x], CA.decoded)
print 'Compiled'

# ------------------------------------------------------

EPOCHS = 2
verbose = True
# ion()
errAcc = [1.0]
TOTAL_BATCHES = 100
MINI_BATCH = m_sample/TOTAL_BATCHES

if False:
	import pickle
	f = open('ca_weights', 'rb')
	WI, B1I, B2I = pickle.load(f)
	CA.W.set_value(WI)
	CA.B1.set_value(B1I)
	CA.B2.set_value(B2I)
	f.close()
	del f

global vis
for i in range(EPOCHS):
	n_batch = 0
	GW = zeros(W_shape)
	GB1 = zeros(B1_shape)
	GB2 = zeros(B2_shape)
	Err = 0
	for x in range(0, m_sample-MINI_BATCH, MINI_BATCH):
		err, gW, gB1, gB2 = train(Tr[0][x:x+MINI_BATCH,:])
		n_batch += 1
		GW = (GW*(n_batch-1) + gW)/n_batch # Accumulate average gradient of W
		GB1 = (GB1*(n_batch-1) + gB1)/n_batch # Accumulate average gradient of B1
		GB2 = (GB2*(n_batch-1) + gB2)/n_batch # Accumulate average gradient of B2
		Err = (Err * (n_batch-1) + err)/n_batch
	
	print 'Avg error in epoch %d/%d is %f' % (i, EPOCHS, Err)
	errAcc.append(Err)
	VW = VW*zeta - alpha * GW
	VB1 = VB1*zeta - alpha * GB1
	VB2 = VB2*zeta - alpha * GB2
	CA.W.set_value( CA.W.get_value() + VW )
	CA.B1.set_value( CA.B1.get_value() + VB1 )
	CA.B2.set_value( CA.B2.get_value() + VB2 )

	if i%100 == 0 or i==EPOCHS-1 or i==EPOCHS-2:
		fl = open('ca_weights', 'wb')
		pickle.dump([CA.W.get_value(), CA.B1.get_value(), CA.B2.get_value()], fl)
		fl.close()
		print 'Weights written to file'
		del fl
		hEnc = reconstruct( Ts[0][:20,:] )
		line = visualize( hEnc, (28, 28) )
		imwrite( 'Epoch'+str(i)+'.png', line )
		alpha = alpha*0.95
		plot(errAcc)
		savefig('errplot_cAE'+str(i)+'.png')
