from theano import*
import theano.tensor as T 
import numpy as np 
import cPickle as cp
#import matplotlib.pyplot as plt 

class hidden_layer():

	def __init__(self,rng,inputs,n_in,n_out,w = None,b = None,activation = T.tanh):
		self.inputs = inputs
		W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )			
		if activation == T.nnet.sigmoid:
			W_values *= 4 #large w if acti is sigmoid!

		W = shared(value = W_values,name = 'W',borrow = True)
		b = shared(value = np.zeros((n_out)), name = 'b',borrow = True)
		self.W = W
		self.b = b 

		lin_out = T.dot(inputs,self.W) + self.b 
		self.output = activation(lin_out) #Activation shud be tanh,sig,elliot_sig ...

		self.params = [self.W,self.b]		

class Logi():

	def __init__(self,inputs,n_in,n_out):
		self.W = shared(value = np.zeros((n_in,n_out),dtype = config.floatX ),name = 'W',borrow = True)
		self.b = shared(value = np.zeros( (n_out,),dtype = config.floatX),name = 'b',borrow = True)
		self.p = T.nnet.softmax(T.dot(inputs,self.W) + self.b)	
		self.y_pred = T.argmax(self.p,axis = 1)
		self.params = [self.W,self.b]
		self.inputs = inputs



	def cost_function(self,y): #y is a target
		return -T.mean(T.log(self.p)[T.arange(y.shape[0]), y])

	def errors(self,y):
		#if y.ndim != self.y.ndim:
		#	raise TypeError

		#if y.dtype.startswith('int'):
		return T.mean(T.neq(self.y_pred,y))

		#else:
		#	raise "boo!"
	

	 

class mlp():
	
	def __init__ (self,rng,inputs,n_in,n_hidden,n_out):
		self.hidden = hidden_layer(
			rng = rng,
			inputs = inputs,
			n_in = n_in,
			n_out = n_hidden,
			activation = T.tanh
			)

		self.final_layer = Logi(
			inputs = self.hidden.output,
			n_in = n_hidden,
			n_out = n_out
			) 	

		self.L1 = (								#Regularization parameters
			abs(self.hidden.W).sum() + 
			abs(self.final_layer.W).sum()
			)

		self.L2_sqr = (
			(self.hidden.W **2).sum() + 
			 (self.final_layer.W **2).sum()
			 )
			 

		self.errors = self.final_layer.errors
		self.params = self.hidden.params + self.final_layer.params

		#self.cost = self.final_layer.cost

		self.inputs = inputs

def test(alpha = 0.01,L1 = 0.00,L2 = 0.001,n_epoch = 1000,data = 'mnist.pkl',batch_size = 20,n_hidden = 500):
	with open('mnist.pkl','rb') as f:
		a = cp.load(f)
	train,test,cv = a 
	train_x,train_y = train 
	test_x,test_y = test
	cv_x ,cv_y = cv

	#No of training batches
	n_train_batches = train_x.shape[0] // batch_size
	n_valid_batches = cv_x.shape[0] // batch_size
	n_test_batches = test_x.shape[0] // batch_size


	index = T.lscalar()

	x = T.matrix('x')
	y = T.ivector('y')


	test_x = shared(test_x)
	test_y = shared(test_y)
	train_y = shared(train_y)
	train_x = shared(train_x)

	rng = np.random.RandomState(1235)

	network = mlp( #MLP object
    rng = rng,
    inputs = x,
    n_in = 28*28,
    n_hidden = n_hidden,
    n_out = 10
    	)

	cost = (
    	network.final_layer.cost_function(y) +
    	L1 * network.L1 + 
    	L2 * network.L2_sqr
    	)

	test_model = function( 
    	inputs = [index],
    	outputs = network.errors(y),
    	givens = {
			x: test_x[index * batch_size:(index + 1) * batch_size],
            y: test_y[index * batch_size:(index + 1) * batch_size]
        }
    	)

	validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: cv_x[index * batch_size:(index + 1) * batch_size],
            y: cv_y[index * batch_size:(index + 1) * batch_size]
        }
    	)

	gparams = [T.grad(cost, param) for param in network.params]

	updates = [ (param,param - alpha * gparam) for param,gparam in zip(network.params,gparams) ]

	train_model = function(
		inputs = [index],
		outputs = cost,
		updates = updates,
		givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )

	best_valid_loss = np.inf


	n_epoch = 200

	for i in range(n_epoch):
		for j in range(n_train_batches):
			mini_batch_avg_cost = train_model(j)
			test_loss = [test_model(i) for i in range(n_test_batches)]
			test_score = np.mean(test_loss)

			print 'epoch %i, minibatch %i/%i, test error %f ' %(i,j+1,n_train_batches,test_score *100)
		

if __name__ == '__main__':
	test()
