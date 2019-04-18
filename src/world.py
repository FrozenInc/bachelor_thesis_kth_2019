import lane
import car
import math
import feature
import dynamics
import visualize
import utils
import sys
#TODO: tensorflow
#import theano as th
#import theano.tensor as tt
import numpy as np
import shelve

#import static_obj # a copy of the car class that s unable to move

th.config.optimizer_verbose = True
th.config.allow_gc = False
th.config.optimizer = 'fast_compile'

# TODO: new functions that can read a world from a JSON object, this is to make it easier to build a world
# TODO: make it easier to actually build a world, just specify the shape of the world, all the object and the objects optimizers+start states

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)

class World(object):
    def __init__(self):
        # alla objekt som existerar i en world
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
    def simple_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        # skapar simple reward for en bil
        # TODO: make this more modular so that it doesnt need to be changed if new object (other types) are added, aka have a default thingy for optimizing non-programmed objects
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        elif isinstance(trajs, static_obj.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        theta = [1., -50., 10., 10., -60.] # Simple model
        # theta = [.959, -46.271, 9.015, 8.531, -57.604]
        # skapar alla lanes, fences, roads, speed och trajectory for alla bilar
        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            # increase the negative reward for the fences so that the cars dont go outside of the road
            #r = r+theta[1]*fence.gaussian()*1000000
            r = r+theta[1]*fence.gaussian()
        if roads == None:
            pass
        else:
            for road in roads:
                r = r+theta[2]*road.gaussian(10.)
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        try:#quick fix, if there is just 1 car it will not be a list
            for traj in trajs:
                r = r+theta[4]*traj.gaussian()
        except:
            r = r+theta[4]*trajs.gaussian()
        return r


# TODO: basically find a way to not have to do this much shit for every scenario, and instead have a more modular system that can easaly read from a json and build a world easaly, for example we dont want to have to specify which obejcts every car should take into considerations instead have it as all and just change if other code is writen
def world_kex(know_model=True):
    dyn = dynamics.CarDynamics(0.1)
    dyn.dt = 1.0
    dyn.fiction = 0.0
    world = World()
    # clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    # world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    # world.roads += [clane]
    # world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]

    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1)]
    #world.roads += [clane, clane.shifted(1)]
    world.fences += [clane.shifted(2), clane.shifted(-1)]

    human_is_follower = True

    # CAR 0 = Human
    # CAR 1 = Robot
    # CAR 2 = Obstacle

    # depending on what our human is, follower or leader we create the cars differently
    if human_is_follower:
        # Create the cars-----
        # Human Car
        world.cars.append(car.NestedOptimizerCarFollower(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
        # Robot Car
        world.cars.append(car.NestedOptimizerCarLeader(dyn, [-0., 0., math.pi/2., 0.5], color='yellow'))
        # --------------------
    else:
        # Create the cars-----
        # Human Car
        world.cars.append(car.NestedOptimizerCarLeader(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
        # Robot Car
        world.cars.append(car.NestedOptimizerCarFollower(dyn, [0., 0., math.pi/2., 0.5], color='yellow'))
        # --------------------
            
    
    # Obstacle Car
    #world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.5, math.pi/2., 0.5], color='blue'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 3, math.pi/4., 0.], color='blue'))
    # --------------------

    # Reward and default for the Human ---
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    # ------------------------------------

    # Reward and default for the Robot ---
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.6)
    world.cars[1].default_u = np.asarray([0., 1.])
    # ------------------------------------

    # Reward and default for the Obstacle ---
    world.cars[2].reward = world.simple_reward(world.cars[2], speed=0.)
    world.cars[2].default_u = np.asarray([0., 0.])
    world.cars[2].movable = False
    # ------------------------------------

    # CAR 0 = Human
    # CAR 1 = Robot
    # CAR 2 = Obstacle

    if human_is_follower:
        world.cars[0].leader = world.cars[1]
        world.cars[0].obstacle = world.cars[2]
        world.cars[1].follower = world.cars[0]
        world.cars[1].obstacle = world.cars[2]
    else:
        world.cars[0].follower = world.cars[1]
        world.cars[0].obstacle = world.cars[2]
        world.cars[1].leader = world.cars[0]
        world.cars[1].obstacle = world.cars[2]

    @feature.feature
    def goal(t, x, u):
        return -(10.*(x[0])**2+0.5*(x[1]-2.)**2)
        #return  ((x[1] + 0.13)**2)/(0.13)**2 +((x[0]-0.5)**2)/(0.13)**2 

    # CAR 0 = Human
    # CAR 1 = Robot
    # CAR 2 = Obstacle

    # TODO: Fix this part, unsure how to make the world.simplereward
    # calculates the dynamic(chaning) rewards for the cars depending on their speed and collision with other cars and obstacles

    # TODO: cars dont want to slow down, find a solution that works
    if human_is_follower:        
        # HUMAN
        #r_h = world.simple_reward([world.cars[1].traj], speed=0.6)+100.*feature.bounded_control(world.cars[0].bounds)+world.simple_reward(world.cars[0].traj_o, speed=0.) # Reward for the human
        r_h = world.simple_reward([world.cars[1].traj], speed=0.8)+2*world.simple_reward(world.cars[0].traj_o, speed=0.8) # Reward for the human

        # ROBOT
        r_r = 0.5*world.simple_reward([world.cars[1].traj_h], speed=0.8)+100.*feature.bounded_control(world.cars[1].bounds)+1*world.simple_reward(world.cars[0].traj_o, speed=0.8) # Reward for the robot
    else:
        # HUMAN
        r_h = world.simple_reward([world.cars[0].traj_h], speed=0.8)+100.*feature.bounded_control(world.cars[0].bounds)+5*world.simple_reward(world.cars[0].traj_o, speed=0.8)# Reward for the human

        # ROBOT
        r_r = world.simple_reward([world.cars[0].traj], speed=0.8)+100.*feature.bounded_control(world.cars[1].bounds)+5*world.simple_reward(world.cars[0].traj_o, speed=0.8)# Reward for the robot
     
    r_o = 1.*feature.bounded_control(world.cars[2].bounds)
    #r_o = world.simple_reward([world.cars[0].traj_o], speed=0.)

    # TODO: fix this too, world.cars[1].rewards = (r_h, r_r) is correct, need to fix it also for cars[0]
    #world.cars[0].rewards = (r_r, r_h)
    #world.cars[1].rewards = (r_h, r_r) # Tells the robot what cars to take care of
    world.cars[0].rewards = (r_r, r_h, r_o)
    world.cars[1].rewards = (r_h, r_r, r_o)
    # ------------------------------------

    return world