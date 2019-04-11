import theano as th
import theano.tensor as tt

# Det har ar for bilarnas states
class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        self.nx = nx # state
        self.nu = nu # kontroll
        self.dt = dt # tid
        if dt is None:
            self.f = f
        else:
            self.f = lambda x, u: x+dt*f(x, u)
    def __call__(self, x, u):
        return self.f(x, u)

class CarDynamics(Dynamics):
    def __init__(self, dt=0.1, ub=[(-3., 3.), (-1., 1.)], friction=1.):
        def f(x, u):
            return tt.stacklists([
                x[3]*tt.cos(x[2]),
                x[3]*tt.sin(x[2]),
                x[3]*u[0],
                u[1]-x[3]*friction
            ])
        Dynamics.__init__(self, 4, 2, f, dt)

class CarDynamics2(Dynamics):
    def __init__(self, dt=0.5, ub=[(-0.104, 0.104), (-2*0.0878, 0.0878)], friction=0.007943232248521):
        def f(x,u):
            return tt.stacklists([
                ((u[1]-friction*x[3]**2)*dt**2/2+x[3]*dt)*tt.cos(x[2])+x[0],
                ((u[1]-friction*x[3]**2)*dt**2/2+x[3]*dt)*tt.sin(x[2])+x[1],
                ((u[1]-friction*x[3]**2)*dt**2/2+x[3]*dt)*u[0]+x[2],
                (u[1]-friction*x[3]**2)*dt+x[3]
            ])
        Dynamics.__init__(self, 4, 2, f, dt=None)
        self.dt = dt # haxy solution for setting dt back to value after specifying f.

if __name__ == '__main__':
    dyn = CarDynamics(0.1)
    x = tt.vector()
    u = tt.vector()
    dyn(x, u)
