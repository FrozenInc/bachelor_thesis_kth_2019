#!/usr/bin/env python
import sys
import visualize
import world
import theano as th
from car import UserControlledCar

th.config.optimizer_verbose = True
th.config.allow_gc = False


if __name__ == '__main__':
    name = "world6"
    name = "world_kex"
    #name = "irl_ground"
    #name = sys.argv[1]
    # if len(sys.argv)>2 and sys.argv[2]=='fast':
    #   th.config.optimizer = 'fast_compile'
    # if len(sys.argv)>2 and sys.argv[2]=='FAST':
    #   th.config.mode = 'FAST_COMPILE'
    world = getattr(world, name)()
    # if len(sys.argv)>3 or (len(sys.argv)>2 and sys.argv[2] not in ['fast', 'FAST']):
    #     ctrl = eval(sys.argv[-1])
    #     for car in world.cars:
    #         if isinstance(car, UserControlledCar):
    #             print 'User Car'
    #             car.fix_control(ctrl)
    vis = visualize.Visualizer(1.0, name=name)
    vis.use_world(world)
    vis.main_car = world.cars[0]
    vis.run()

if __name__ == '__main__1':
    name = sys.argv[1]
    if len(sys.argv)>2 and sys.argv[2]=='fast':
        th.config.optimizer = 'fast_compile'
    if len(sys.argv)>2 and sys.argv[2]=='FAST':
        th.config.mode = 'FAST_COMPILE'
    world = getattr(world, name)()
    if len(sys.argv)>3 or (len(sys.argv)>2 and sys.argv[2] not in ['fast', 'FAST']):
        ctrl = eval(sys.argv[-1])
        for car in world.cars:
            if isinstance(car, UserControlledCar):
                print 'User Car'
                car.fix_control(ctrl)
    vis = visualize.Visualizer(0.5, name=name)
    vis.use_world(world)
    vis.main_car = world.cars[0]
    vis.run()
