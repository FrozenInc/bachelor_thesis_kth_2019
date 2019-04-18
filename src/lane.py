import numpy as np
#TODO: tensorflow
#import theano as th
#import theano.tensor as tt
import feature

class Lane(object): pass

# TODO: 
# 1. rewrtie with better names
# 2. make the lanes to be able to stop at a point
class StraightLane(Lane):
    def __init__(self, p, q, w): # skapar en lane
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
    def shifted(self, m): # roterar en lane
        return StraightLane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)
    def dist2(self, x): #hittar avstand mellan 2 lanes (tror jag)
        r = (x[0]-self.p[0])*self.n[0]+(x[1]-self.p[1])*self.n[1]
        return r*r
    def gaussian(self, width=0.5): # gor gausisk reward over lanet
        @feature.feature
        def f(t, x, u):
            return tt.exp(-0.5*self.dist2(x)/(width**2*self.w*self.w/4.))
        return f