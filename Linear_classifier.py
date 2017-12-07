import numpy as np
import math

#use SVM as loss function
class SVM:

	#lam: lambda(regularization)
	def __init__(self, W, XT, Y, lam, train_num, step):
		self.XT = XT
		self.Y = Y
		self.lam = lam

		self.N = XT.shape[0]
		self.D = XT.shape[1]
		self.M = W.shape[1]

		#init the W
		self.W = W

		#number of trains
		self.train_num = train_num
		self.step = step

	def forward(self):
		self.YT = self.XT.dot(self.W)
		self.LT = self.YT
		#SVM
		self.grad_YT = np.zeros_like(self.YT)

		""" the naive way
		for i in range(self.N):
			yt = self.YT[i]
			lyj = yt[self.Y[i]]
			for j in range(self.M):
				if j == self.Y[i] or yt[j] - lyj + 1.0 <= 0.0:
					self.LT[i][j] = 0.0
					continue
				self.LT[i][j] = yt[j] - lyj + 1.0
				self.grad_YT[i][j] = 1
				self.grad_YT[i][self.Y[i]] = self.grad_YT[i][self.Y[i]] - 1
		"""
		# use the vector feature in numpy
		right_scores = self.YT[np.arange(self.N), self.Y].reshape(self.N, 1)
		self.LT = self.YT - right_scores + 1.0
		self.LT[np.arange(self.N), self.Y] = 0.0
		self.LT[self.LT <= 0.0] = 0.0
		self.grad_YT = self.LT
		self.grad_YT[self.grad_YT > 0.0] = 1
		row_sum = np.sum(self.grad_YT, axis=1)
		self.grad_YT[np.arange(self.N), self.Y] = -row_sum

		self.LT = (1 / self.N) * self.LT
		self.grad_YT = (1 / self.N) * self.grad_YT

		self.L = self.LT.sum()

		#regularization
		self.RW = self.W ** 2
		self.L = self.L + self.lam * self.RW.sum()

		#L is the final Loss
		return self.L

	def backward(self):
		#grad_y_x means: partial y / partial x
		self.grad_RW_W = 2 * self.lam * self.W
		self.grad_L_W = self.XT.T.dot(self.grad_YT) + self.grad_RW_W

		return self.grad_L_W

	def train(self):
		for i in range(self.train_num):
			L = self.forward()
			grad = self.backward()
			self.W = self.W - self.step * grad
			# print(self.W)

		return self.W


#use softmax as loss function
class Softmax(SVM):
	def forward(self):
		self.YT = self.XT.dot(self.W)
		# SVM
		self.grad_YT = np.zeros_like(self.YT)

		score_max = np.max(self.YT, axis=1).reshape(self.N, 1)
		prob = np.exp(self.YT - score_max) / np.sum(np.exp(self.YT - score_max), axis=1).reshape(self.N, 1)
		true_class = np.zeros_like(prob)
		true_class[np.arange(self.N), self.Y] = 1
		self.LT = -np.log(prob) * true_class
		self.grad_YT = -(true_class - prob)

		self.LT = (1 / self.N) * self.LT
		self.grad_YT = (1 / self.N) * self.grad_YT

		self.L = self.LT.sum()

		# regularization
		self.RW = self.W ** 2
		self.L = self.L + self.lam * self.RW.sum()

		# L is the final Loss
		return self.L


class Linear_classifier:
	def __init__(self, type, XT, Y, M, lam = 0.1, train_num = 100000, step = 0.1):
		D = XT.shape[1]
		W = np.full((D, M), 1.0)
		self.XT = XT
		self.Y = Y
		self.M = M
		self.lam = lam
		self.train_num = train_num
		self.step = step
		self.type = type
		if type == 'svm':
			self.model = SVM(W, XT, Y, lam, train_num, step)
		else:
			self.model = Softmax(W, XT, Y, lam, train_num, step)

	def train(self):
		return self.model.train()

	def predict(self, XT):
		return XT.dot(self.model.W)

	def set_type(self, type):
		if type == self.type:
			return
		self.type = type
		if type == 'svm':
			self.model = SVM(self.W, self.XT, self.Y, self.lam, self.train_num, self.step)
		else:
			self.model = Softmax(self.W, self.XT, self.Y, self.lam, self.train_num, self.step)


def main():
	#get XT, Y, N, D, M
	N, D, M = 3, 2, 3
	XT = np.array([[1, 4], [2, -1], [-3, 9]])
	Y = [0, 2, 2]
	lc = Linear_classifier('softmax', XT, Y, M)
	W = lc.train()
	print(W)
	print(XT.dot(W))
	# print(W)

if __name__ == '__main__':
	main()