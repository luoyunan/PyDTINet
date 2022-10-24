import numpy as np
import __train_mf as c_train_mf
from scipy.sparse import csc_matrix, csr_matrix, issparse

def to_csc(A, dtype=np.double):
	csc = A if type(A) is csc_matrix else csc_matrix(A)

	if not dtype is None:
		if csc.dtype != dtype:
			print("changing type from {0} to {1}".format(csc.dtype, dtype))
			csc.data = csc.data.astype(dtype)

	return csc

def check_array(A, nr, nc, name, dtype=np.double):
	if A.shape != (nr, nc):
		s = "Shape error. {0} is {1} expected {2} by {3}"\
			.format(name, A.shape, nr, nc)
		raise Exception(s)


	if (not dtype is None) and (A.dtype != dtype):
		s = "Type error for {0}. Type is {1} but expected {2}"\
			.format(name, A.dtype, dtype)
		raise Exception(s)

class train_mf_prob:
	def __init__(self, Y, X1=None, X2=None, W=None, H=None, k=10, lamb=0.1, solver_type=0, maxiter=10, threads=1, seed=None):
		self.n1, self.n2 = Y.shape # number of users (rows) and items (columns)

		self.Yin = Y
		self.X1in = X1
		self.X2in = X2
		self.Win = W
		self.Hin = H
		self.k = k
		self.lamb = lamb
		self.solver_type = solver_type
		self.maxiter = maxiter
		self.threads = threads

		self.Y = to_csc(Y)
		self.X1 = identity(self.n1, format="csc") if X1 is None else to_csc(X1)
		self.X2 = identity(self.n2, format="csc") if X2 is None else to_csc(X2)
		self.d1 = self.X1.shape[1] # number of row features
		self.d2 = self.X2.shape[1] # number of column features

		if not seed is None:
			np.random.seed(seed)
		self.W = np.random.rand(self.d1, k) if W is None else W
		self.H = np.random.rand(self.d2, k) if H is None else H
		self.Y_hat = None


		check_array(self.Y, self.n1, self.n2, "Y")
		check_array(self.X1, self.n1, self.d1, "X1")
		check_array(self.X2, self.n2, self.d2, "X2")
		check_array(self.W, self.d1, self.k, "W")
		check_array(self.H, self.d2, self.k, "H")

	def train_mf(self):
		print("module is", c_train_mf)
		c_train_mf.train_mf(Y=self.Y, X1=self.X1, X2=self.X2, W=self.W, H=self.H,
							k=self.k, lamb=self.lamb, solver_type=self.solver_type,
							maxiter=self.maxiter, threads=self.threads)

	def predict_mf(self):
		if issparse(self.X1):
			X1_W = self.X1.dot(self.W) # sparse x dense
		else:
			X1_W = np.dot(X1, W) # dense x dense

		X1_W_Ht = np.dot(X1_W, self.H.transpose()) # dense x dense

		if issparse(self.X2):
			X1_W_Ht_X2t = self.X2.dot(X1_W_Ht.transpose()).transpose()#sp x dns
		else:
			X1_W_Ht_X2t = np.dot(X1WH, X2.transpose()) # dense x dense

		self.Y_hat = X1_W_Ht_X2t

		return self.Y_hat

	def rmse(self, ignore_non_zeros):
		if ignore_non_zeros:
			assert not self.Y_hat is None
			Y_hat = self.Y_hat
			Y = self.Y

			[r, c] = Y.nonzero()
			Y_hat_values = Y_hat[r, c]

			if False:
				# slow
				Y_values = [Y[i, j] for i, j in zip(r, c)]
			else:
				# fast. But counts on Y.nonzero() returning in row major order!
				Y_csr = csr_matrix(Y)
				Y_values = Y_csr.data

			rmse = np.sqrt(np.power(Y_hat_values - Y_values, 2).mean())
		else:
			Y = self.Y.todense() if issparse(self.Y) else self.Y
			rmse = np.sqrt(np.power(self.Y_hat - Y, 2).mean())
		return rmse

def IMC(Y, X1, X2, k, lamb, solver_type=10, maxiter=10, threads=4, seed=None):
	#pass
	# Y: user-item sparse matrix (n1-by-n2)
	# X1: user features (n1-by-d1)
	# X2: item features (n2-by-d2)
	# W: initial W (d1-by-k)
	# H: initial H (d2-by-k)
	#
	# k: rank (default 10)
	# lamb: regularization parameter lambda (default 0.1)
	# solver_type: type of solver (default 0)
	#       0 -- L2R_LS (Squared Loss)
	#       1 -- L2R_LR (Logistic Regression)
	#       2 -- L2R_SVC (Squared Hinge Loss)
	#       10 -- L2R_LS (Squared Loss) Fully observation
	# maxiter: number of iterations (default 10)
	# threads: number of threads (default 4)
	#
	Y = to_csc(Y)
	X1 = to_csc(X1)
	X2 = to_csc(X2)
	n1, n2 = Y.shape # number of users (rows) and items (columns)
	d1 = X1.shape[1] # number of row features
	d2 = X2.shape[1] # number of column features
	W = np.random.rand(X1.shape[1], k)
	H = np.random.rand(X2.shape[1], k)
	check_array(Y, n1, n2, "Y")
	check_array(X1, n1, d1, "X1")
	check_array(X2, n2, d2, "X2")
	check_array(W, d1, k, "W")
	check_array(H, d2, k, "H")
	print("calling __train_mf()")
	# print("NOW!", X2)
	c_train_mf.train_mf(Y=Y, X1=X1, X2=X2, W=W, H=H, k=k, lamb=lamb,\
				solver_type=solver_type, maxiter=maxiter, threads=threads)
	print("__train_mf done")
	return train_mf_prob(Y, X1, X2, W, H, k, lamb, seed=seed)