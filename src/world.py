import lane
import car
import math
import feature
import dynamics
import visualize
import utils
import sys
import theano as th
import theano.tensor as tt
import numpy as np
import shelve

import static_obj # a copy of the car class that s unable to move

th.config.optimizer_verbose = True
th.config.allow_gc = False
th.config.optimizer = 'fast_compile'

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
            r = r+theta[1]*fence.gaussian()
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

def world_kex(know_model=True):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    # Create the cars-----
    # Human Car
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
    # Robot Car
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0., math.pi/2., 0.5], color='yellow'))
    # --------------------
    # Obstacle Car
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.5, math.pi/2., 0.5], color='blue'))
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

    # TODO: fix 2 new nestedoptimizer, leader and follower
    world.cars[0].robot = world.cars[1] # TODO: add a robot var in the nestedotpimzier car class and fix the control to take care of it
    world.cars[0].obstacle = world.cars[2] # TODO: add an obstacle var in the nestedoptimizer car class and fix the control
    world.cars[1].human = world.cars[0] # Tells the robot who the human is
    world.cars[1].obstacle = world.cars[2]

    # calculates the dynamic(chaning) rewards for the cars depending on their speed and collision with other cars and obstacles
    # ROBOT
    r_h = world.simple_reward([world.cars[1].traj], speed=0.6)+100.*feature.bounded_control(world.cars[0].bounds)+100.*feature.bounded_control(world.cars[2].bounds) # Reward for the human

    # HUMAN
    r_r = 10*goal+world.simple_reward([world.cars[1].traj_h], speed=0.5)+100.*feature.bounded_control(world.cars[2].bounds) # Reward for the robot
    world.cars[1].rewards = (r_h, r_r) # Tells the robot what cars to take care of
    # ------------------------------------

    return world


def world_kex1(know_model=True):
    start_human= -0.13
    start_robot= -0.00
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    #world.cars.append(car.SimpleOptimizerCar(dyn, [start_human, 0., math.pi/2., 0.5], color='red')) # red car is human
    world.cars.append(car.NestedOptimizerCar(dyn, [start_human, 0., math.pi/2., 0.5], color='red')) # red car is human
    if know_model: # yellow car is the robot that uses nested optimizer to find the way
        world.cars.append(car.NestedOptimizerCar(dyn, [start_robot, 0.0, math.pi/2., 0.5], color='yellow'))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [start_robot, 0.0, math.pi/2., 0.5], color='yellow')) 
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    @feature.feature
    def goal(t, x, u): # doesnt need this
        k = -(10.*(x[0]+0.13)**2+0.5*(x[1]-2.)**2) #ASK Elis
        #print("--------", x[0].auto_name)
        #print("--------", x[1].auto_name)
        #exit()
        return k

    # object--------------
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.5, math.pi/2., 0.0], color='blue')) # blue car is obstacle
    #world.cars.append(car.NestedOptimizerCar(dyn, [-0.13, 0.5, math.pi/2., 0.0], color='blue')) # blue car is obstacle
    #print(world.cars)
    #exit()
    world.cars[2].reward = world.simple_reward(world.cars[2], speed=0.0)
    #world.cars[2].reward = 1
    world.cars[2].default_u = np.asarray([0., 0.])
    world.cars[2].movable = False

    #------------------


    if know_model:
        world.cars[1].human = world.cars[0] # [1] is robot, asigns that the robot knows who is the human
        world.cars[1].obstacle = world.cars[2]
        world.cars[0].obstacle = world.cars[2]
        world.cars[0].human = world.cars[1]
        

        # reward with respect to the robot trajectory: world.cars[1].traj
        r_h = world.simple_reward([world.cars[1].traj], speed=0.5)+100.*feature.bounded_control(world.cars[0].bounds)+100.*feature.bounded_control(world.cars[2].bounds)

        #r_r = 10*goal+world.simple_reward([world.cars[1].traj_h], speed=0.5
        r_r = world.simple_reward([world.cars[1].traj_h], speed=0.5)+100.*feature.bounded_control(world.cars[2].bounds)

        r_h2 = world.simple_reward([world.cars[1].traj_h], speed=0.5)+100.*feature.bounded_control(world.cars[0].bounds)
        +100.*feature.bounded_control(world.cars[2].bounds)
        #r_r = 10*goal+world.simple_reward([world.cars[1].traj_h], speed=0.5
        r_r2 = world.simple_reward([world.cars[1].traj], speed=0.5)+100.*feature.bounded_control(world.cars[2].bounds)

        
        #r_obj = world.simple_reward([world.cars[1].traj_h], speed=0.0)
        world.cars[1].rewards = (r_h, r_r)#ADD: r_object
        world.cars[0].rewards = (r_h2, r_r2) #(optimize on, the car)
        #print(r_h)
        #print(r_r)
        #print(world.cars[1].rewards)
        #exit()
    else:
        r = 10*goal+world.simple_reward([world.cars[0].linear], speed=0.5)
        world.cars[1].reward = r
    
    #world.cars.append(static_obj.SimpleOptimizerCar(dyn, [-0.13, 0.5, math.pi/2., 0.0], color='blue')) # blue car is obstacle)


    return world


def playground():
    # detta ar en playground varld, den ar tom forutom en person bil
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    #world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.], color='orange'))
    world.cars.append(car.UserControlledCar(dyn, [-0.17, -0.17, math.pi/2., 0.], color='white'))
    return world

def irl_ground():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    d = shelve.open('cache', writeback=True)
    cars = [(-.13, .1, .5, 0.13),
            (.02, .4, .8, 0.5),
            (.13, .1, .6, .13),
            (-.09, .8, .5, 0.),
            (0., 1., 0.5, 0.),
            (-.13, -0.5, 0.9, 0.13),
            (.13, -.8, 1., -0.13),
           ]
    def goal(g):
        @feature.feature
        def r(t, x, u):
            return -(x[0]-g)**2
        return r
    for i, (x, y, s, gx) in enumerate(cars):
        if str(i) not in d:
            d[str(i)] = []
        world.cars.append(car.SimpleOptimizerCar(dyn, [x, y, math.pi/2., s], color='yellow'))
        world.cars[-1].cache = d[str(i)]
        def f(j):
            def sync(cache):
                d[str(j)] = cache
                d.sync()
            return sync
        world.cars[-1].sync = f(i)
    for c, (x, y, s, gx) in zip(world.cars, cars):
        c.reward = world.simple_reward(c, speed=s)+10.*goal(gx)
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.7], color='red'))
    world.cars = world.cars[-1:]+world.cars[:-1]
    return world



def world_test():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.5)
    return world

def world0():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human_speed(t, x, u):
        return -world.cars[1].traj_h.x[t][3]**2
    r_r = world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world1(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj], speed_import=.2 if flag else 1., speed=0.8 if flag else 1.)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human_speed(t, x, u):
        return -world.cars[1].traj_h.x[t][3]**2
    r_r = 300.*human_speed+world.simple_reward(world.cars[1], speed=0.5)
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('cone', [0., 1.8]))
    return world

def world2(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-1., 1.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -(world.cars[1].traj_h.x[t][0])*10
    r_r = 300.*human+world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('firetruck', [0., 0.7]))
    return world

def world3(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-1., 1.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return (world.cars[1].traj_h.x[t][0])*10
    r_r = 300.*human+world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('firetruck', [0., 0.7]))
    return world

def world4(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes += [vlane, hlane]
    world.fences += [hlane.shifted(-1), hlane.shifted(1)]
    world.cars.append(car.UserControlledCar(dyn, [0., -.3, math.pi/2., 0.0], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.3, 0., 0., 0.], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-2., 2.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    world.cars[1].bounds = [(-3., 3.), (-2., 2.)]
    @feature.feature
    def horizontal(t, x, u):
        return -x[2]**2
    r_h = world.simple_reward([world.cars[1].traj], lanes=[vlane], fences=[vlane.shifted(-1), vlane.shifted(1)]*2)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -tt.exp(-10*(world.cars[1].traj_h.x[t][1]-0.13)/0.1)
    r_r = human*10.+horizontal*30.+world.simple_reward(world.cars[1], lanes=[hlane]*3, fences=[hlane.shifted(-1), hlane.shifted(1)]*3+[hlane.shifted(-1.5), hlane.shifted(1.5)]*2, speed=0.9)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world5():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes += [vlane, hlane]
    world.fences += [hlane.shifted(-1), hlane.shifted(1)]
    world.cars.append(car.UserControlledCar(dyn, [0., -.3, math.pi/2., 0.0], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.3, 0., 0., 0.0], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[1].bounds = [(-3., 3.), (-2., 2.)]
    @feature.feature
    def horizontal(t, x, u):
        return -x[2]**2
    r_h = world.simple_reward([world.cars[1].traj], lanes=[vlane], fences=[vlane.shifted(-1), vlane.shifted(1)]*2)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -tt.exp(10*(world.cars[1].traj_h.x[t][1]-0.13)/0.1)
    r_r = human*10.+horizontal*2.+world.simple_reward(world.cars[1], lanes=[hlane]*3, fences=[hlane.shifted(-1), hlane.shifted(1)]*3+[hlane.shifted(-1.5), hlane.shifted(1.5)]*2, speed=0.9)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world6(know_model=True):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
    if know_model:
        world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='yellow'))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='yellow'))
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    @feature.feature
    def goal(t, x, u):
        return -(10.*(x[0]+0.13)**2+0.5*(x[1]-2.)**2)
    if know_model:
        world.cars[1].human = world.cars[0]
        r_h = world.simple_reward([world.cars[1].traj], speed=0.6)+100.*feature.bounded_control(world.cars[0].bounds)
        r_r = 10*goal+world.simple_reward([world.cars[1].traj_h], speed=0.5)
        world.cars[1].rewards = (r_h, r_r)
    else:
        r = 10*goal+world.simple_reward([world.cars[0].linear], speed=0.5)
        world.cars[1].reward = r
    return world

def world_features(num=0):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.Car(dyn, [0., 0.1, math.pi/2.+math.pi/5, 0.], color='yellow'))
    world.cars.append(car.Car(dyn, [-0.13, 0.2, math.pi/2.-math.pi/5, 0.], color='yellow'))
    world.cars.append(car.Car(dyn, [0.13, -0.2, math.pi/2., 0.], color='yellow'))
    #world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    return world

if __name__ == '__main__':
    world = playground()
    #world.cars = world.cars[:0]
    vis = visualize.Visualizer(0.1, magnify=1.2)
    vis.main_car = None
    vis.use_world(world)
    vis.paused = True
    @feature.feature
    def zero(t, x, u):
        return 0.
    r = zero
    #for lane in world.lanes:
    #    r = r+lane.gaussian()
    #for fence in world.fences:
    #    r = r-3.*fence.gaussian()
    r = r - world.cars[0].linear.gaussian()
    #vis.visible_cars = [world.cars[0]]
    vis.set_heat(r)
    vis.run()
