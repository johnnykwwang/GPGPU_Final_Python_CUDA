"""Utilities for the neural network modules
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# License: BSD 3 clause

import time
import numpy as np
import cupy as cp
import pdb

from scipy.special import expit as logistic_sigmoid


def identity(X):
    """Simply return the input array.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return X


def logistic(X):
    """Compute the logistic function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return logistic_sigmoid(X, out=X)


def tanh(X):
    """Compute the hyperbolic tan function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return np.tanh(X, out=X)


def relu(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

def relu_cuda(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    cp.clip(X, 0, cp.finfo(X.dtype).max, out=X)
    return X



def softmax(X):
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X

kernel_softmax = cp.ElementwiseKernel(
    'float64 max_x',
    'float64 x',
    'x = exp(x - max_x)',
    'kernel_softmax'
)

def softmax_cuda(X):
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    
    kernel_softmax(X.max(axis=1)[:, cp.newaxis], X)
    x_sum = X.sum(axis=1)[:, cp.newaxis] 
    cp.divide(X, x_sum, out=X)

    return X


ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu, 'softmax': softmax, 'relu_cuda': relu_cuda, 'softmax_cuda': softmax_cuda }


kernel_inplace_relu_derivative = cp.ElementwiseKernel(
    'float64 Z',
    'float64 delta',
    'delta = Z == 0 ? 0 : delta',
    'kernel_inplace_relu_derivative')

def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do


def inplace_logistic_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0


def inplace_relu_derivative_cuda(Z, delta):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    kernel_inplace_relu_derivative(Z, delta)
    #delta[Z == 0] = 0


DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': inplace_tanh_derivative,
               'logistic': inplace_logistic_derivative,
               'relu': inplace_relu_derivative,
               'relu_cuda': inplace_relu_derivative_cuda
               }


def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return ((y_true - y_pred) ** 2).mean() / 2


def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob)) / y_prob.shape[0]


kernel_log_loss = cp.ReductionKernel(
    'float64 y_true, float64 y_prob, float64 size',
    'float64 val',
    'y_true * log( min(max(y_prob,1e-10),1 - 1e-10) ) * -1.0',
    'a + b',
    'val = a / size',
    '0',
    'kernel_log_loss'
)

def log_loss_cuda(y_true, y_prob):
    """Compute Logistic loss for classification.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
 
    #y_prob = cp.clip(y_prob, 1e-10, 1 - 1e-10) # Optimized

    if y_prob.shape[1] == 1:
        y_prob = cp.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = cp.append(1 - y_true, y_true, axis=1)

    #val = -cp.sum(y_true * cp.log(y_prob)) / y_prob.shape[0] # Optimized

    val = kernel_log_loss(y_true, y_prob, y_prob.shape[0]) # Optimization Kernel
    return val


def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    return -np.sum(y_true * np.log(y_prob) +
                   (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]


LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss, 'log_loss_cuda': log_loss_cuda}
