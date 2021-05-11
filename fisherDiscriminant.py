import numpy as np
from matplotlib import pyplot as plt

class FisherDiscriminat:
	"""
	Calculates the fishers discriminant for two class problem and projects it onto a line 

	"""
	def __init__(self,X,y):
		self.w = None
		self.b = None
		self.X = X
		self.y = y
		self.optimum_w()

	def optimum_w(self):

		#the mean for both the class variables m1 and m2
		self.mean_1 = self.mean(self.X[self.y==0])
		self.mean_2 = self.mean(self.X[self.y==1])

		#assume the same covarience matrix for both 
		self.cov = self.cov(self.X)

		cov_inv = np.linalg.inv(self.cov)

		#for a two class case the optimum solution can be inv(cov)*(m2-m1)
		self.w = np.linalg.inv(self.cov).dot(self.mean_2 - self.mean_1)

		#the decision boundry  
		self.b = np.log(len(self.y==0)/len(self.y==1)) + 0.5*self.mean_1.dot(np.dot(cov_inv,self.mean_1)) + 0.5*self.mean_2.dot(np.dot(cov_inv,self.mean_2))

	def mean(self,X):
		return np.mean(X,axis = 0)

	def cov(self,X):
		return np.cov(X.T)

	@classmethod
	def from_csv(cls, csv_file):
		X = np.genfromtxt(csv_file,delimiters=",")
		return cls(X)

def create_dataset(sample_size = 50, m=[[0,1],[1,0]], sigma = [[0.5,0.3],[0.3,0.5]]):
	"""
	create two multivarient normal functions 
	"""
	X1 = np.random.multivariate_normal(m[0], sigma, sample_size)
	X2 = np.random.multivariate_normal(m[1], sigma, sample_size)
	X  = np.concatenate([X1,X2])
	y  = np.concatenate([np.zeros(sample_size),np.ones(sample_size)])

	return X, y

def plot(self):
	"""
	Plot funtion for the decision boundry for two class problem
	"""
	plt.scatter(self.X[y==0],'r')
	plt.scatter(self.X[y==1],"b")
	plt.show()



if __name__=="__main__":

	X ,y = create_dataset(50)
	discriminant  = FisherDiscriminat(X,y)
	print(discriminant.w)