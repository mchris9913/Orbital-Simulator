import tiamat
import tiamat.integrators
import tiamat.dynamics_core

import unittest
import numpy as np
import matplotlib.pyplot as plt

class translational_sat(object):
    def __init__(self)->None:
        self.pos_idx = np.s_[:3]
        self.vel_idx = np.s_[3:6]
        self.mu = 1
        return
        
    def dynamics(self, t: float, x: np.ndarray)->np.ndarray:
        params = (self.pos_idx, self.vel_idx, self.mu)
        xdot = tiamat.dynamics_core.translational_dynamics(x, params)
        return xdot

class TestTranslationalSat(unittest.TestCase):
    
    def test_circular_orbit(self):
        """
            Test Circular Orbit (in plane)
        """
        
        sat = translational_sat()
        func = sat.dynamics
        
        T = 2*np.pi #Orbital Period
        t = np.linspace(0,T,1000)
        x0 = np.array([1.0,0.0,0.0,0.0,1.0,0.0])
        solver = tiamat.integrators.runge_kutta(f=func,x0=x0.copy(),t0 = t[0])
        x1 = solver.solve(t, eps = 1e-8)
        
        propagator = tiamat.dynamics_core.ClassicalElementsPropagator(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        x2 = propagator.solve(t, ε = 1e-8)
        
        self.assertTrue(np.all(np.abs(x1 - x2) <= 1e-6))
        return
        
    def test_elliptical_orbit(self):
        """
            Test Elliptical Orbit (out of plane)
        """
        
        propagator = tiamat.dynamics_core.ClassicalElementsPropagator(5.0, 0.5, np.pi/3.0, np.pi/4.0, np.pi/6.0, 3.0*np.pi/4.0)
        
        sat = translational_sat()
        func = sat.dynamics
        
        T = 2*np.pi * 1/np.sqrt(1/(5**3)) #Orbital Period
        t = np.linspace(0,T,10000)
        x0 = propagator.get_state(propagator.ν0)
        solver = tiamat.integrators.runge_kutta(f=func,x0=x0.copy(),t0 = t[0])
        
        x1 = solver.solve(t, eps = 1e-14)
        x2 = propagator.solve(t, ε = 1e-14)
        
        self.assertTrue(np.all(np.abs(x1 - x2) <= 1e-6))
        return

if __name__ == '__main__':
    unittest.main()