import numpy as np
import tensorflow as tf

import trajectory
import utils

class Car(object):
    def __init__(self, dynamic, x_now, color='yellow', time_steps=5):
        self.data_now = {'x_now': x_now}
        self.bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        self.time_steps = time_steps
        self.dynamic = dynamic
        self.trajectory = trajectory.Trajectory(time_steps, dynamic)
        self.trajectory.x_now = x_now
        self.linear = trajectory.Trajectory(time_steps, dynamic)
        self.linear.x_now = x_now
        self.color = color
        self.default_control = np.zeros(self.dynamic.now)
        self.movable = True
    
    def reset(self):
        self.trajectory.x_now = self.data_now['x_now']
        self.linear.x_now = self.data_now['x_now']
        for t in range(self.time_steps):
            self.trajectory.control[t] = np.zeros(self.dynamic.now)
            self.linear.control[t] = self.default_control
    
    def move(self):
        self.trajectory.tick()
        self.linear.x_now = self.trajectory.x_now

class SimpleOptimizerCar(Car): # expanderar Car klassen
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.cache = [] # minns var den har varit forut
        self.index = 0 # minns pa vilken plats den ar
        self.sync = lambda cache: None # lamba kopierar i detta fall cache och satter den lika med None
        #self.r_temp = 0
    def reset(self): # resetar bilen 
        Car.reset(self)
        self.index = 0
    # TODO: find a better solution for this
#    @property
#    def reward(self): # returnerar rewarden for bilen
#        return self._reward
#    @reward.setter
#    def reward(self, reward): 
#        # tar fram reward med hjalp av input och reward fran bounded_control
#        # bounded_control anvander sig av kollisions boxarna i varlden
#        self._reward = reward+100.*feature.bounded_control(self.bounds)
#        self.optimizer = None # skapar en tom optimizer

    def control(self, steer, gas):
        print (len(self.cache)) # VIKTIGT: printar ut vilken tidsteg ar nu
        if self.movable == False:
            self.index += 1
            return
        
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


#TODO: update the nested optimizer to be able to do the following:
# 1. have N-number of time steps forward
# 2. be able to see "between" timesteps
# 3. be able to take care of several different cars/obstacles (use a array of car objects for this)
# 3a. be able to not need to recalculate the other cars if they are already calculated
class NestedOptimizerCar(Car):
    # skippa sa lange, dubbelkolla med elis om vi ska ha med den. 
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
    # TODO: better solution than get/set
    # TODO: make it modular to be able to take care of N-amount of cars/objects
    #@property
    #def human(self):
    #    return self._human
    #@human.setter
    #def human(self, value):
    #    self._human = value
    #    self.traj_h = Trajectory(self.T, self.human.dyn)
    #def move(self):
    #    Car.move(self)
    #    self.traj_h.tick()
    #@property
    #def rewards(self):
    #    return self._rewards
    #@rewards.setter
    #def rewards(self, vals):
    #    self._rewards = vals
    #    self.optimizer = None
    # see the todo of the class
    def control(self, steer, gas):
        if self.optimizer is None:
            reward_h, reward_r = self.rewards
            reward_h = self.traj_h.reward(reward_h)
            reward_r = self.traj.reward(reward_r)
            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
        self.traj_h.x0.set_value(self.human.x)
        self.optimizer.maximize(bounds = self.bounds)
        
k = Car(1,2)
print(k.bounds)