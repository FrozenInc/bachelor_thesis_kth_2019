import theano as th
import theano.tensor as tt

class Feature(object): #ASK Elis
    def __init__(self, f):
        self.f = f
    def __call__(self, *args): # *args kan vara av vilken storlek som helst
        return self.f(*args) # gor att self.f blir lika med alla argument som har blivit inskickad
    def __add__(self, r):
        return Feature(lambda *args: self(*args)+r(*args)) # addera ihop self arhumenten och other argumenten
    def __radd__(self, r):
        return Feature(lambda *args: r(*args)+self(*args)) # samma sak som __add__ fast a andra hallet
    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r) # multiplicerar self arg med en siffra
    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args)) # samma sak som __mul__ men a andra hallet
    def __pos__(self, r): 
        return self # returnerar minnes platsen av instansen
    def __neg__(self):
        return Feature(lambda *args: -self(*args)) # byter tecken for argumenten
    def __sub__(self, r):
        return Feature(lambda *args: self(*args)-r(*args)) # self-r
    def __rsub__(self, r):
        return Feature(lambda *args: r(*args)-self(*args)) # r-self

def feature(f):
    return Feature(f) # returnerar en instans av Feature

def speed(s=1.):
    @feature
    def f(t, x, u): # bygger up argumenten for hastighet
        return -(x[3]-s)*(x[3]-s)
    return f

def control():
    @feature
    def f(t, x, u): # bygger upp argumenten for kontrol av bilen
        return -u[0]**2-u[1]**2 
    return f

def bounded_control(bounds, width=0.05):
    @feature
    def f(t, x, u): # bygger upp argumenten for "kollisionen" for alla objekt
        ret = 0.
        for i, (a, b) in enumerate(bounds):
            return -tt.exp((u[i]-b)/width)-tt.exp((a-u[i])/width)
    return f

if __name__ == '__main__':
    pass
