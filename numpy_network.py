import numpy as np
import math
import random
import matplotlib.pyplot as mp
import scipy.io as sio
import cPickle as cp

def sigmoid(z):
	return 1.0/(1 + math.e*-z)
	#return 0.5*z/(1+abs(z) + 0.5

def derivative(z):
	return sigmoid(z)*(1-sigmoid(z))

def plot(inputs,outputs,actual):
	fig = mp.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(inputs,actual,'b-')
	ax1.plot(inputs, outputs, 'r.')
	mp.draw()
	mp.show()
	
class network:
	def __init__ (self,ni,nh,no,input_rows):
		
		self.ni = ni + 1
		self.nh = nh + 1 # Bias unit 1
		self.no = no

		x = np.array(range(1,2))

		#self.ai = np.ones((input_rows,401))
		#self.ah = np.ones((input_rows,26))
		self.ao = np.array( x for i in range(self.no))

		self.syn1 = 2*np.random.random((self.ni,nh)) - 1
		self.syn2 = 2*np.random.random((self.nh,self.no)) - 1

	def update(self,inputs): # inputs = x
		self.ai = np.ones((inputs[:,0].size,self.ni))
		self.ai[:,:-1] = inputs
		
		self.ah = np.ones((inputs[:,0].size,self.nh))
		k = sigmoid(self.ai.dot(self.syn1))
		for i in range(k[0].size):
			self.ah[:,i] = k[:,i]

		self.ao = sigmoid(self.ah.dot(self.syn2))
		return self.ao

	def back_prop(self,target,N=0.1): #target = y
		
		delta_3 = self.ao - target
		delta_2 = delta_3.dot(self.syn2.T) * derivative(self.ah)
		
		self.syn2 = self.syn2 + self.ah.T.dot(delta_3)*N
	
		z = np.zeros((self.ni,self.nh))
		for i in range(z[0].size-1):
			z[:,i] = self.syn1[:,i]
		self.syn1 = z
		self.syn1 = self.syn1  + self.ai.T.dot(delta_2)*N
		op = np.argmax(self.ao)
		#TotalError = 0.5*((target - self.ao)**2)
		TotalError = 0.5*((target - op)**2)
		return TotalError

	def train(self,x,y,epoch = 1000,N = 0.01):
		error = 0.0
		for i in range(epoch):
			self.update(x)		
			delta = self.back_prop(y)
			error = error + delta
			#if i%100==0:
			#	print'error ',error
		return error

	def test(self,x,y):
		k = self.update(x)
		c = 0
		pr = 0
		prediction = np.array([])
		print k
		for i in range(k[:,0].size):
			pr = np.argmax(k[i])
			prediction = np.append(prediction,pr)
		
		print prediction,y
		return k
	

mat = sio.loadmat('data.mat')
x = mat['X']
y = mat['y']
y[y == 10] = 0
n = network(400,100,10,3000)
k = n.train(x[:3000],y[:3000])
t = n.test(x[2000:],y[2000:])

with open('model.pkl','wb') as f:
	cp.dump(n,f)


