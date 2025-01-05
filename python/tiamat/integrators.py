from .numerical_methods_core import integrator

import numpy as np

class runge_kutta(integrator):
	
    def __init__(self, f: callable = None, x0: np.ndarray = None, t0: np.ndarray = None)->None:
        super().__init__(f=f, x0=x0, t0=t0)
        return
    
    def integrate(self, tf: float, x0: np.ndarray = None, t0: float = None, dt: float = 1e-3)->tuple:
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
        while t[-1] != tf:
            print(t[-1])
            dt = tf - t[-1] if dt > tf - t[-1] else dt
            
            step = self.__step(x[-1], t[-1], dt)
            half_step = self.__step(x[-1], t[-1], dt/2)
            double_step = self.__step(x[-1], t[-1], 2*dt) if t[-1] + 2*dt <= tf else np.inf*np.ones_like(x[-1])
            
            if np.any(np.abs(step) < 1e-8):
                x.append(step)
                t.append(t[-1] + dt)
            else:
                if np.any(np.abs(step - half_step)/np.abs(step) > 1e-3):
                    dt /= 2
                    x.append(half_step)
                    t.append(t[-1] + dt)
                elif np.all(np.abs(step - double_step)/np.abs(step) < 1e-3):
                    dt *= 2
                    x.append(double_step)
                    t.append(t[-1] + dt)
                else:
                    x.append(step)
                    t.append(t[-1] + dt)
            
        return (np.array(x), np.array(t))
    
    def __step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self.f(t, x)
        k2 = self.f(t+dt/2, x+dt/2*k1)
        k3 = self.f(t+dt/2, x+dt/2*k2)
        k4 = self.f(t+dt, x+dt*k3)
        return x + dt*(k1/6 + k2/3 + k3/3 + k4/6)
        
    def solve(self, t: np.ndarray):
        raise NotImplementedError()