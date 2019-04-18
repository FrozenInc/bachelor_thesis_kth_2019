#TODO: tensorflow
#import theano as th
#import theano.tensor as tt
#import theano.tensor.slinalg as ts
import scipy.optimize
import numpy as np
import time

# TODO: find a way to optimize all the functions as they are very slow. expecially grad, jacobian and hessian
# this might be possible to achive by switching the vecotr and matrix from a thenao style to tensorflow style
def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()

def shape(var):
    return extract(var.shape)

def vector(n):
    return th.shared(np.zeros(n))

def matrix(n, m):
    return tt.shared(np.zeros((n, m)))

def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret

def jacobian(f, x, constants=[]):
    sz = shape(f)
    return tt.stacklists([grad(f[i], x) for i in range(sz)])
    ret = th.gradient.jacobian(f, x, consider_constant=constants)
    if isinstance(ret, list):
        ret = tt.concatenate(ret, axis=1)
    return ret

def hessian(f, x, constants=[]):
    return jacobian(grad(f, x, constants=constants), x, constants=constants)
