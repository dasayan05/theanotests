from matplotlib.pyplot import ion, draw, plot, savefig
from ConvNet import LeConvNet
from theano import function, pp, tensor as T
from dataset import loadData, OneToMany
from numpy import zeros, abs, array, count_nonzero
Tr, Ts, _ = loadData('mnist.pkl.gz', True)
m_sample = 10000
m_test_sample = 500
X = Tr[0].reshape((50000,1,28,28))[:m_sample,:,:,:];Y = Tr[1][:m_sample]
Xt = Ts[0].reshape((10000,1,28,28))[:m_test_sample,:,:,:];Yt = Ts[1][:m_test_sample]
del Tr; del Ts

# Hyper-parameters
alpha = 0.5
zeta = 0.99
lam = 0.01

x = T.tensor4('x')
y = T.dmatrix('y')
LCV = LeConvNet(x)
cost = LCV.cost(y) + LCV.regularization() * lam
params = LCV.params()

grads = [ T.grad(cost,wrt=p) for p in params ]
V = [zeros(g) for g in LCV.paramshapes()]
train = function( [x, y], [cost] + grads )

p = LCV.predict()
predict = function ([x],p)
print 'Functions compiled'

EPOCHS = 300
TOTAL_BATCHES = 500
MINI_BATCH = m_sample/TOTAL_BATCHES #1000

ion();errs = []

for i in range(EPOCHS):
	n_batch = 0
	G = [zeros(g) for g in LCV.paramshapes()]
	avgloss = 0
	for x in range(0, m_sample-MINI_BATCH, MINI_BATCH):
		n_batch += 1
		op = train(X[x:x+MINI_BATCH,:,:,:],OneToMany(Y[x:x+MINI_BATCH], zeroOrNegOne=True))
		if n_batch%100 == 0:
			print 'Error %f in mini-batch %d/%d'%( op[0], n_batch, TOTAL_BATCHES )
		avgloss += op[0]
		op = op[1:]
		g_c = 0
		for _ in G:
			G[g_c] = ( G[g_c]*(n_batch-1) + op[g_c] ) / n_batch; g_c += 1
	
	avgloss /= TOTAL_BATCHES # avg loss on this epoch
	# errs.append(avgloss);plot(errs); draw() # <-- Accumulate errors
	print '			Avg Error %f at Epoch %d'%(avgloss, i)
	cnt = 0
	for _ in V:
		V[cnt] = V[cnt] - alpha*G[cnt];cnt += 1
	cnt = 0
	for _ in params:
		params[cnt].set_value( params[cnt].get_value() + zeta * V[cnt] );cnt += 1
	print 'Parameters updated ..'
	Q = abs( array( predict ( Xt ) ) - Yt )
	print '			Correct test samples %d/%d'%( m_test_sample-count_nonzero(Q), m_test_sample)
	print '------- moving to new epoch ----------'