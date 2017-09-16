import numpy as np
from random import shuffle
from past.builtins import xrange
from math import log,e
def softmax_loss_naive(W, X, y, reg):
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
  
  for i in range(X.shape[0]):
    score = X[i].dot(W)
    escore = e**score
    loss += - log(escore[y[i]]/np.sum(escore))
    dW[:,y[i]] -= X[i]
    sumES = escore.sum()
    for j in range(W.shape[1]):
        dW[:,j] += X[i] * escore[j] / sumES
  
  dW /= X.shape[0]
  dW += reg *2 *W
  loss /= X.shape[0]
  loss += reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  loss = 0.0
  dW = np.zeros_like(W)
  escore = e**X.dot(W)
  sumEs = escore.sum(axis = 1)
  #print escore.shape,sumEs.shape
  loss = -(np.log(escore[np.arange(X.shape[0]),y]/sumEs)).sum()/X.shape[0]
  escore /= sumEs.reshape(sumEs.shape[0],1)
  escore[np.arange(X.shape[0]),y] -= 1
  dW += X.T.dot(escore)/X.shape[0]
  
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

