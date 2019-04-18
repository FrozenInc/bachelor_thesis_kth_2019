#TODO: tensorflow
#import theano as th
#import theano.tensor as tt
#import theano.tensor.slinalg as ts
import scipy.optimize
import numpy as np
import time
import utils


# TODO:
# 1. rewrite with snesible names
# 2. make it so that it can accept several other cars be it their state or already calculated maximizer (this is for optimization purposes)
# 3. find what cn be made to multicore and/or gpu acceleration (most stuff are matrix operations so it should be possible)
# 4. make it possible to calculate several control steps per time step (for example if we want the car to be able to check 5 steps forward, but actually go 25 smaller steps)
class NestedMaximizer(object):
    # kors bara for nested optimizer i car.py, hoppa over sa lange. 
    # f1, vs1 are other car
    # f2, vs2 are own car
    def __init__(self, f1, vs1, f2, vs2):
        # creates a isolated var of the vars
        self.f1 = f1
        self.f2 = f2
        self.vs1 = vs1
        self.vs2 = vs2
        #-----

        # 
        self.sz1 = [shape(v)[0] for v in self.vs1] # converts from tensor variable to normal array (uses also some math magic)
        self.sz2 = [shape(v)[0] for v in self.vs2]
        for i in range(1, len(self.sz1)): # adds together all future sz1 with old sz1
            self.sz1[i] += self.sz1[i-1]
        self.sz1 = [(0 if i==0 else self.sz1[i-1], self.sz1[i]) for i in range(len(self.sz1))] #makes the array into a 2d-array with (prevoius var, var)
        for i in range(1, len(self.sz2)): # same thing az sz1
            self.sz2[i] += self.sz2[i-1]
        self.sz2 = [(0 if i==0 else self.sz2[i-1], self.sz2[i]) for i in range(len(self.sz2))] # samme thing as sz1
        self.df1 = grad(self.f1, vs1) # IMPORTANT: VERY SLOW
        self.new_vs1 = [tt.vector() for v in self.vs1] # back from normal array to tensorVector
        self.func1 = th.function(self.new_vs1, [-self.f1, -self.df1], givens=zip(self.vs1, self.new_vs1)) # IMPORTANT: VERY VERY VERY SLOW
        def f1_and_df1(x0):
            return self.func1(*[x0[a:b] for a, b in self.sz1])
        self.f1_and_df1 = f1_and_df1
        J = jacobian(grad(f1, vs2), vs1) # IMPORTANT: VERY VERY VERY VERY SLOW
        H = hessian(f1, vs1) # IMPORTANT: VERY VERY VERY VERY SLOW
        g = grad(f2, vs1 )# IMPORTANT: SLOW
        self.df2 = -tt.dot(J, ts.solve(H, g))+grad(f2, vs2) # IMPORTANT: SLOW
        self.func2 = th.function([], [-self.f2, -self.df2]) # IMPORTANT: EXREMELY SLOW
        def f2_and_df2(x0):
            for v, (a, b) in zip(self.vs2, self.sz2):
                v.set_value(x0[a:b])
            self.maximize1()
            return self.func2()
        self.f2_and_df2 = f2_and_df2
    def maximize1(self):
        x0 = np.hstack([v.get_value() for v in self.vs1])
        opt = scipy.optimize.fmin_l_bfgs_b(self.f1_and_df1, x0=x0)[0]
        for v, (a, b) in zip(self.vs1, self.sz1):
            v.set_value(opt[a:b])
    def maximize(self, bounds={}):
        t0 = time.time()
        if not isinstance(bounds, dict):
            bounds = {v: bounds for v in self.vs2}
        B = []
        for v, (a, b) in zip(self.vs2, self.sz2):
            if v in bounds:
                B += bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([v.get_value() for v in self.vs2])
        def f(x0):
            #if time.time()-t0>60:
             #   raise Exception('Too long')
            return self.f2_and_df2(x0)
        opt = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=B) # IPORTANT: slow
        diag = opt[2]['task']
        opt = opt[0]
        for v, (a, b) in zip(self.vs2, self.sz2):
            v.set_value(opt[a:b])
        self.maximize1()


# TODO: the same things as the other class except not going forward more than 1 step (after all this is a 1 step maximizer)
class Maximizer(object):
    # maximerar over en reward och trajectory
    def __init__(self, f, vs, g={}, pre=None, gen=None, method='bfgs', eps=1, iters=100000, debug=False, inf_ignore=np.inf):
        # __init__ bygger bara upp ratt struktur pa maximeraren, medans argmax ar det som optimerar over reward
        self.inf_ignore = inf_ignore # har en infinite variable sa att den inte krashar nar den forsoker dela med 0 eller nat
        self.debug = debug # om den ska debuga 
        self.iters = iters # hur manga interationer den ska maximera over
        self.eps = eps
        self.method = method # vilken metod den ska anvanda for maximering
        def one_gen(): # skapar en ny gen
            yield # skicka vidare utrycket och inte bara svaret
            # kanske skapar en generator som kan skapa nya gen?
        self.gen = gen # skapar en ny gen, gen kanske generator?
        if self.gen is None:
            self.gen = one_gen # ingen parantes; referrar till one_gen(), men kor ej funktionet
        self.pre = pre # ingen anning
        self.f = f # ingen anning
        self.vs = vs # ingen anning
        self.sz = [shape(v)[0] for v in self.vs] # ingen anning
        for i in range(1,len(self.sz)):
            self.sz[i] += self.sz[i-1]
        self.sz = [(0 if i==0 else self.sz[i-1], self.sz[i]) for i in range(len(self.sz))]
        if isinstance(g, dict):
            self.df = tt.concatenate([g[v] if v in g else grad(f, v) for v in self.vs])
        else:
            self.df = g
        self.new_vs = [tt.vector() for v in self.vs]
        self.func = th.function(self.new_vs, [-self.f, -self.df], givens=zip(self.vs, self.new_vs))
        def f_and_df(x0): #ASK Elis
            if self.debug:
                print x0
            s = None
            N = 0
            for _ in self.gen():
                if self.pre:
                    for v, (a, b) in zip(self.vs, self.sz):
                        v.set_value(x0[a:b])
                    self.pre()
                res = self.func(*[x0[a:b] for a, b in self.sz])
                if np.isnan(res[0]).any() or np.isnan(res[1]).any() or (np.abs(res[0])>self.inf_ignore).any() or (np.abs(res[1])>self.inf_ignore).any():
                    continue
                if s is None:
                    s = res
                    N = 1
                else:
                    s[0] += res[0]
                    s[1] += res[1]
                    N += 1
            s[0]/=N
            s[1]/=N
            return s
        self.f_and_df = f_and_df
    def argmax(self, vals={}, bounds={}):
        if not isinstance(bounds, dict):
            bounds = {v: bounds for v in self.vs}
        B = []
        for v, (a, b) in zip(self.vs, self.sz):
            if v in bounds:
                B += bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([np.asarray(vals[v]) if v in vals else v.get_value() for v in self.vs])
        # bygger upp alla variabler som behovs
        if self.method=='bfgs':
            # hittar optimala argumentet med hjalp av scypy
            opt = scipy.optimize.fmin_l_bfgs_b(self.f_and_df, x0=x0, bounds=B)[0]
        elif self.method=='gd':
            opt = x0
            for _ in range(self.iters):
                opt -= self.f_and_df(opt)[1]*self.eps
        else:
            opt = scipy.optimize.minimize(self.f_and_df, x0=x0, method=self.method, jac=True).x
        return {v: opt[a:b] for v, (a, b) in zip(self.vs, self.sz)}
    def maximize(self, *args, **vargs):
        # tar in alla argument och maximerar over de med hjalv av argmax
        result = self.argmax(*args, **vargs)
        for v, res in result.iteritems():
            v.set_value(res)
