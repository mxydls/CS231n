import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of num_train examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (num_train, D) containing a minibatch of data.
	- y: A numpy array of shape (num_train,) containing training labels; y[i] = c means
		that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	dW = np.zeros(W.shape)
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in xrange(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		for j in xrange(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + 1 # note delta = 1
			if margin > 0:
				loss += margin
				dW[:, j] += X[i]
				dW[:, y[i]] -= X[i]

	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by num_train.
	loss /= num_train
	dW /= num_train
	
	# Add regularization to the loss.
	loss += reg * np.sum(W * W)
	dW += 2 * reg * W

	#############################################################################
	# TODO:                                                                     #
	# Compute the gradient of the loss function and store it dW.                #
	# Rather that first computing the loss and then computing the derivative,   #
	# it may be simpler to compute the derivative at the same time that the     #
	# loss is being computed. As a result you may need to modify some of the    #
	# code above to compute the gradient.                                       #
	#############################################################################


	return loss, dW


def svm_loss_vectorized(W, XT, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	loss = 0.0
	# dW = np.zeros(W.shape) # initialize the gradient as zero

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	#############################################################################
	YT = XT.dot(W)
	LT = YT
	num_train = XT.shape[0]
	#SVM
	grad_YT = np.zeros_like(YT)

	# use the vector feature in numpy
	right_scores = YT[np.arange(num_train), y].reshape(-1, 1)
	LT = YT - right_scores + 1.0
	LT[np.arange(num_train), y] = 0.0
	LT = LT * (LT > 0.0)

	LT = (1 / num_train) * LT

	#add the regularization
	RW = W ** 2
	loss = LT.sum() + reg * RW.sum()

	grad_YT = LT
	grad_YT[grad_YT > 0.0] = 1
	row_sum = np.sum(grad_YT, axis=1)
	grad_YT[np.arange(num_train), y] = -row_sum
	grad_YT = (1 / num_train) * grad_YT
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################


	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	#############################################################################
	dW = XT.T.dot(grad_YT) + 2 * reg * W
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return loss, dW
