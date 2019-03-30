import numpy as np
import utils
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
from trajectory import Trajectory
import feature

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        self.data0 = {'x0': x0} # init state av bilen
        self.bounds = [(-1., 1.), (-1., 1.)] # kollisions boxen for bilen
        self.T = T # hur manga tidsteg fram den ska berakna
        self.dyn = dyn # dynamiken for bilen
        self.traj = Trajectory(T, dyn) # trajectory for bilen som ar utraknad med hjalp av reward, ar i framtiden
        self.traj.x0.set_value(x0) # satter start vardet for trajectory
        self.linear = Trajectory(T, dyn) # samma sak som traj, men i tiden nu 
        self.linear.x0.set_value(x0)
        self.color = color # byter farg pa bilen
        self.default_u = np.zeros(self.dyn.nu) # gor en matris av storleken av kontroll variabeln med bara nollor for att ha ne referens
    def reset(self): # resetar alla variabler till deras start varden som finns i __init__
        self.traj.x0.set_value(self.data0['x0']) 
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):
            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def move(self): # flyttar fram bilen
        pass
        #self.traj.tick()
        #self.linear.x0.set_value(self.traj.x0.get_value())
    @property # tar value av x0
    def x(self):
        return self.traj.x0.get_value()
    @property # tar value av u[0]
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter # satter en ny value for u[0]
    def u(self, value):
        pass
        self.traj.u[0].set_value(value)
    def control(self, steer, gas): # gor literally ingenting
        pass


class SimpleOptimizerCar(Car): # expanderar Car klassen
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.cache = [] # minns var den har varit forut
        self.index = 0 # minns pa vilken plats den ar
        self.sync = lambda cache: None # lamba kopierar i detta fall cache och satter den lika med None
    def reset(self): # resetar bilen 
        Car.reset(self)
        self.index = 0
    @property
    def reward(self): # returnerar rewarden for bilen
        return self._reward
    @reward.setter
    def reward(self, reward): 
        # tar fram reward med hjalp av input och reward fran bounded_control
        # bounded_control anvander sig av kollisions boxarna i varlden
        self._reward = reward+100.*feature.bounded_control(self.bounds)
        self.optimizer = None # skapar en tom optimizer
    def control(self, steer, gas):
        print len(self.cache) # VIKTIGT: printar ut vilken tidsteg ar nu
        if self.index<len(self.cache):
            self.u = self.cache[self.index]
        else:
            if self.optimizer is None:
                # skickar self.reward functionet till traj.reward
                # det som faktiskt blir skickad ar self._reward
                # detta behandlas i traj.reward for att far var reward for bilen
                r = self.traj.reward(self.reward)
                # skapar en instans av Maximizer for foljande reward och trajectory
                self.optimizer = utils.Maximizer(r, self.traj.u) #IMPORTANT: slow
            # maximerar rewarden med hjalp av maximizer
            self.optimizer.maximize()
            # cachar vad som har hand
            self.cache.append(self.u)
            # uppdaterar tiden nu
            self.sync(self.cache)
        # gar fram en tidsteg
        self.index += 1