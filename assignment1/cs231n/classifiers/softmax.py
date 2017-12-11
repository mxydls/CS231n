import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, XT, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
		that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	pass
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW


def softmax_loss_vectorized(W, XT, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	# dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	YT = XT.dot(W)

	grad_YT = np.zeros_like(YT)

	score_max = np.max(YT, axis=1).reshape(N, 1)
	prob = np.exp(YT - score_max) / np.sum(np.exp(YT - score_max), axis=1).reshape(N, 1)
	true_class = np.zeros_like(prob)
	true_class[np.arange(N), y] = 1
	LT = -np.log(prob) * true_class
	grad_YT = -(true_class - prob)

	LT = (1 / N) * LT
	grad_YT = (1 / N) * grad_YT

	# add the regularization
	RW = W ** 2
	loss = LT.sum() + reg * RW.sum()

	dW = XT.T.dot(grad_YT) + 2 * reg * W
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW

