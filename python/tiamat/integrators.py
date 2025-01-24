from .numerical_methods_core import integrator
from .interpolators import cubic_spline_1d

import numpy as np

class runge_kutta(integrator):
    A1 = 0
    A2 = 2/9
    A3 = 1/3
    A4 = 3/4
    A5 = 1
    A6 = 5/6
    
    B21 = 2/9
    B31 = 1/12
    B32 = 1/4
    B41 = 69/128
    B42 = -243/128
    B43 = 135/64
    B51 = -17/12
    B52 = 27/4
    B53 = -27/5
    B54 = 16/15
    B61 = 65/432
    B62 = -5/16
    B63 = 13/16
    B64 = 4/27
    B65 = 5/144
    
    CH1 = 47/450
    CH2 = 0
    CH3 = 12/25
    CH4 = 32/225
    CH5 = 1/30
    CH6 = 6/25
    
    CT1 = 1/150
    CT2 = 0
    CT3 = -3/100
    CT4 = 16/75
    CT5 = 1/20
    CT6 = -6/25
    
    def __init__(self, f: callable = None, x0: np.ndarray = None, t0: np.ndarray = None)->None:
        super().__init__(f=f, x0=x0, t0=t0)
        self.spline = cubic_spline_1d()
        return
    
    def integrate(self, tf: float, x0: np.ndarray = None, t0: float = None, dt: float = 1, eps: float = 1e-6)->tuple:
        if x0 is None and self.x0 is None:
            raise Exception("No starting state")
        elif x0 is None:
            x0 = self.x0.copy()
        
        if t0 is None and self.t0 is None:
            raise Exception("No starting time")
        elif t0 is None:
            t0 = self.t0
        
        if tf < t0:
            raise Exception("Final time is less than initial time")
        
        if self.f is None:
            raise Exception("No function to integrate")
        
        t = [t0]
        x = [x0]
        k = [self.f(t0,x0)]
        h = dt
        while t[-1] < tf:
            x_next, err = self.__step(x[-1], t[-1], h)
            while np.max(err) > eps:
                h = 0.9*h*(eps/np.max(err))**(0.2)
                x_next, err = self.__step(x[-1], t[-1], h)
                
            t.append(t[-1]+h)
            x.append(x_next)
            k.append(self.f(t[-1], x[-1]))
            h = dt
            
        return (np.array(x), np.array(t), np.array(k))
    
    def __step(self, x: np.ndarray, t: float, h: float) -> tuple:
        k1 = h * self.f(t + self.A1*h, x)
        k2 = h * self.f(t + self.A2*h, x + self.B21*k1)
        k3 = h * self.f(t + self.A3*h, x + self.B31*k1 + self.B32*k2)
        k4 = h * self.f(t + self.A4*h, x + self.B41*k1 + self.B42*k2 + self.B43*k3)
        k5 = h * self.f(t + self.A5*h, x + self.B51*k1 + self.B52*k2 + self.B53*k3 + self.B54*k4)
        k6 = h * self.f(t + self.A6*h, x + self.B61*k1 + self.B62*k2 + self.B63*k3 + self.B64*k4 + self.B65*k5)
        return (x + self.CH1*k1 + self.CH2*k2 + self.CH3*k3 + self.CH4*k4 + self.CH5*k5 + self.CH6*k6, np.abs(self.CT1*k1 + self.CT2*k2 + self.CT3*k3 + self.CT4*k4 + self.CT5*k5 + self.CT6*k6))
        
    def solve(self, t: np.ndarray, dt: float = 1, eps: float = 1e-6)->np.ndarray:
        x_rk, t_rk, k_rk = self.integrate(t[-1], dt=dt, eps=eps)
        
        x = []
        
        i = 0
        j = 0
        update_model = True
        while i < len(t):
            while j < len(t_rk)-1:
                if t_rk[j] == t[i]:
                    x.append(x_rk[j])
                    break
                elif t_rk[j] < t[i] and t_rk[j+1] > t[i]:
                    if update_model:
                        self.spline.set_model((t_rk[j], x_rk[j], self.f(t_rk[j], x_rk[j])),(t_rk[j+1], x_rk[j+1], self.f(t_rk[j+1], x_rk[j+1])))
                        update_model = False
                        
                    x_next = self.spline.evaluate(t[i])
                    x.append(x_next)
                    break
                j += 1
                update_model = True
            i += 1
        
        return np.array(x)
