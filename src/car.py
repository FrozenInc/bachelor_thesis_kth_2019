import numpy as np
import tensorflow as tf

#import trajectory
#import utils

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
        
k = Car(1,2)
print(k.bounds)