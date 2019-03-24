import theano as th
import theano.tensor as tt
import utils
import numpy as np
import feature
import lane

class Trajectory(object):
    def __init__(self, T, dyn):
        self.dyn = dyn # dynamiken for systemet
        self.T = T # hur manga steg fram den ska kolla 
        self.x0 = utils.vector(dyn.nx) # state vektorn
        self.u = [utils.vector(dyn.nu) for t in range(self.T)] # matris for de nastkommande T stegen
        self.x = [] # vektor for alla states
        z = self.x0 
        for t in range(T): # hitta dynamiken for alla tidsteg den ska plannera for
            z = dyn(z, self.u[t])
            self.x.append(z)
        self.next_x = th.function([], self.x[0]) # konverterar grafen av states till en callable object
    def tick(self):
        self.x0.set_value(self.next_x()) # lagg nasta state som x_0 state
        for t in range(self.T-1):
            self.u[t].set_value(self.u[t+1].get_value()) # lagg nasta kontroll variabel som nuvarande kontroll variabel
        self.u[self.T-1].set_value(np.zeros(self.dyn.nu)) # gor en ny tid T som ar langst bak till nollor
    def gaussian(self, height=.07, width=.03): # gor gausisk fordelning
        @feature.feature # gor en alias for att implementera f
        def f(t, x, u):
            d = (self.x[t][0]-x[0], self.x[t][1]-x[1])
            theta = self.x[t][2]
            dh = tt.cos(theta)*d[0]+tt.sin(theta)*d[1]
            dw = -tt.sin(theta)*d[0]+tt.cos(theta)*d[1]
            return tt.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))
        return f
    def reward(self, reward):
        # gor en lista
        # listan bestar av summan av alla rewards
        # kallas pa rekursivt
        # raknar ut varje sma reward som finns
        # summerar ihop alla sma reward till stora REWARD
        r = [reward(t, self.x[t], self.u[t]) for t in range(self.T)]
        return sum(r)
        """
        g = [utils.grad(r[t], self.x[t]) for t in range(self.T)]
        for t in reversed(range(self.T-1)):
            g[t] = g[t]+tt.dot(g[t+1], utils.jacobian(self.x[t+1], self.x[t]))
        for t in range(self.T):
            g[t] = tt.dot(g[t], utils.jacobian(self.x[t], self.u[t]))+utils.grad(r[t], self.u[t], constants=[self.x[t]])
        return sum(r), {self.u[t]: g[t] for t in range(self.T)}
        """

if __name__ == '__main__':
    from dynamics import CarDynamics
    import math
    dyn = CarDynamics(0.1)
    traj = Trajectory(5, dyn)
    l = lane.StraightLane([0., -1.], [0., 1.], .1)
    reward = feature.speed()+l.feature()#+feature.speed()
    r = traj.reward(reward)
    #traj.x0.value = np.asarray([0., 0., math.pi/2, 1.])
    traj.x0.set_value([0.1, 0., math.pi/2, 1.])
    optimizer = utils.Maximizer(r, traj.u)
    import time
    t = time.time()
    for i in range(1):
        optimizer.maximize(bounds=[(-1., 1.), (-2, 2.)])
    print (time.time()-t)/1.
    print [u.get_value() for u in traj.u]
