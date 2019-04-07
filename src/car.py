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
        self.movable = True
    def reset(self): # resetar alla variabler till deras start varden som finns i __init__
        self.traj.x0.set_value(self.data0['x0']) 
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):
            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def move(self): # flyttar fram bilen
        if self.movable:
            self.traj.tick()
            self.linear.x0.set_value(self.traj.x0.get_value())
        else:
            self.traj.tick()            
            self.linear.x0.set_value(self.traj.x0.get_value())
            pass
    @property # tar value av x0
    def x(self):
        return self.traj.x0.get_value()
    @property # tar value av u[0]
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter # satter en ny value for u[0]
    def u(self, value):
        if self.movable:
            self.traj.u[0].set_value(value)
        else:
            pass
    def control(self, steer, gas): # gor literally ingenting
        pass

class UserControlledCar(Car): # klassen for en bil som kan koras av en riktig person
    # expanderar pa klassen Car
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.follow = None
        self.fixed_control = None
        self._fixed_control = None
    def fix_control(self, ctrl): # kan kora med ctrl, aka input fran user
        self.fixed_control = ctrl
        self._fixed_control = ctrl
    def control(self, steer, gas): # tar in steer och gas fran user och anvander de for att kora
        if self.fixed_control is not None:
            self.u = self.fixed_control[0]
            print self.fixed_control[0]
            if len(self.fixed_control)>1:
                self.fixed_control = self.fixed_control[1:]
        elif self.follow is None:
            self.u = [steer, gas]
        else:
            u = self.follow.u[0].get_value()
            if u[1]>=1.:
                u[1] = 1.
            if u[1]<=-1.:
                u[1] = -1.
            self.u = u
    def reset(self): # resetar bilen
        Car.reset(self)
        self.fixed_control = self._fixed_control

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
        if self.movable == False:
            self.index += 1

            # just to test, but the simple optimizer seem to be able to find the other cars and get a reward depending on that
            #print self.traj.reward(self.reward).eval()
            #IMPORTANT
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

class NestedOptimizerCar(Car):
    # skippa sa lange, dubbelkolla med elis om vi ska ha med den. 
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
    @property
    def human(self):
        return self._human
    @human.setter
    def human(self, value):
        self._human = value
        self.traj_h = Trajectory(self.T, self.human.dyn)
    def move(self):
        Car.move(self)
        self.traj_h.tick()
    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, vals):
        self._rewards = vals
        self.optimizer = None
    def control(self, steer, gas):
        if self.optimizer is None:
            reward_h, reward_r = self.rewards
            reward_h = self.traj_h.reward(reward_h)
            reward_r = self.traj.reward(reward_r)
            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
        self.traj_h.x0.set_value(self.human.x)
        self.optimizer.maximize(bounds = self.bounds)


# DONE: Fix collision with object
# TODO: Fix it to actuallly behave as a follower
class NestedOptimizerCarFollower(Car):
    # skippa sa lange, dubbelkolla med elis om vi ska ha med den. 
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]

    # Obstacle-----
    @property
    def obstacle(self):
        return self._obstacle
    @obstacle.setter
    def obstacle(self, value):
        self._obstacle = value
        self.traj_o = Trajectory(self.T, self.obstacle.dyn)
        self.r_temp = 0
    # -------------

    # Leader --------
    @property
    def leader(self):
        return self._leader
    @leader.setter
    def leader(self, value):
        self._leader = value
        self.traj_h = Trajectory(self.T, self.leader.dyn)
    #----------------

    # Move and update traj for leader and obstacle---
    def move(self):
        Car.move(self)
        self.traj_h.tick()
        self.traj_o.tick()
    # -----------------------------------------------
    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, vals):
        self._rewards = vals
        self.optimizer = None
    def control(self, steer, gas):
        if self.optimizer is None:
        #if True:
            reward_h, reward_r, reward_o = self.rewards
            self.t_temp = reward_o
            #reward_h = reward_h + reward_o
            

            reward_h = self.traj_h.reward(reward_h)
            reward_r = self.traj.reward(reward_r)
            reward_o = self.traj_o.reward(reward_o)
            #print reward_h.eval()
            #print reward_o.eval()
            #exit()
            # TEST:
            #reward_h = reward_h + reward_o
            #print self.traj_h.u[0].eval()
            #print self.traj_h.u
            #exit()
            
            for i in range(0, len(self.traj_h.u)):
                print self.traj_h.u[i].eval()
                #self.traj_h.u[i] = tt.TensorType('float64', (0,)*2)
                k = np.transpose(np.array([0., 0.]))
                self.traj_h.u[i].set_value(k)
            for i in range(0, len(self.traj_h.u)):
                print self.traj_h.u[i].eval()
                print self.traj_h.u[i]
            #exit()

            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
            
            # TODO: fixa optimizer to care about obstacles too
        for i in range(0, len(self.traj_h.u)):
            print self.traj_h.u[i].eval()
            #self.traj_h.u[i] = tt.TensorType('float64', (0,)*2)
            k = np.transpose(np.array([0., 0.]))
            self.traj_h.u[i].set_value(k)
        for i in range(0, len(self.traj_h.u)):
            print self.traj_h.u[i].eval()
            
        self.traj_h.x0.set_value(self.leader.x)
        self.traj_o.x0.set_value(self.obstacle.x)
        self.optimizer.maximize(bounds = self.bounds)
        #print self.obstacle.x
        #print self.traj_o.reward(self.t_temp).eval()

# DONE: Fix collision with object
# TODO: Fix it to actuallly behave as a leader
class NestedOptimizerCarLeader(Car):
    # skippa sa lange, dubbelkolla med elis om vi ska ha med den. 
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]

    # Obstacle-----
    @property
    def obstacle(self):
        return self._obstacle
    @obstacle.setter
    def obstacle(self, value):
        self._obstacle = value
        self.traj_o = Trajectory(self.T, self.obstacle.dyn)
    # -------------

    # Follower --------
    @property
    def follower(self):
        return self._follower
    @follower.setter
    def follower(self, value):
        self._follower = value
        self.traj_h = Trajectory(self.T, self.follower.dyn)
    # -----------------

    # Move and update traj for follower and obstacle---
    def move(self):
        Car.move(self)
        self.traj_h.tick()
        self.traj_o.tick()
    # -----------------------------------------------

    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, vals):
        self._rewards = vals
        self.optimizer = None
    def control(self, steer, gas):
        if self.optimizer is None:
            reward_h, reward_r, reward_o = self.rewards
            reward_h = reward_h + reward_o

            reward_h = self.traj_h.reward(reward_h)
            reward_r = self.traj.reward(reward_r)
            reward_o = self.traj_o.reward(reward_o)
            # TEST:
            #reward_h = reward_h + reward_o
        
            # reward_r ar for leader
            # reward_h ar for follower
            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
            # TODO: fixa optimizer to care about obstacles too
        self.traj_h.x0.set_value(self.follower.x)
        self.traj_o.x0.set_value(self.obstacle.x)
        self.optimizer.maximize(bounds = self.bounds)

