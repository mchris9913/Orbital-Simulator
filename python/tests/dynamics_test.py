import tiamat
import tiamat.integrators
import tiamat.dynamics_core

import unittest
import numpy as np

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
        
class rotational_sat(object):
    def __init__(self)->None:
        self.rot_idx = np.s_[:3]
        self.quat_idx = np.s_[3:7]
        self.I = np.diag([1.0,2.0,3.0])
        return
        
    def dynamics(self, t: float, x: np.ndarray)->np.ndarray:
        xdot = np.zeros_like(x)
        params = (self.I, np.zeros(3))
        xdot[self.rot_idx] = tiamat.dynamics_core.rotational_dynamics(x[self.rot_idx], params)
        
        params = (self.rot_idx, self.quat_idx)
        xdot[self.quat_idx] = tiamat.dynamics_core.quaternion_kinematics(x, params)
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
        
class TestRotationalSat(unittest.TestCase):
    
    def test_three_axis_rotation(self):
        
        sat = rotational_sat()
        func = sat.dynamics
        
        T = 2*np.pi
        t = np.linspace(0, T, 1000)
        x0 = np.array([1,1,1,1,0,0,0])
        solver = tiamat.integrators.runge_kutta(f=func, x0=x0.copy(),t0=t[0])
        x1 = solver.solve(t, eps=1e-10)
        
        Ix = sat.I[0,0]
        Iy = sat.I[1,1]
        Iz = sat.I[2,2]
        
        L = np.zeros((len(t),3))
        T = np.zeros_like(t)
        for i in range(len(t)):
            L[i,:] = sat.I @ x1[i,0:3]
            T[i] = 1/2 * L[i,:] @ x1[i, 0:3]
            
        L = np.array(L)
        T = np.array(T)
        
        
        self.assertTrue(np.all(np.abs(x1[:,0]**2/(2*T/Ix) + x1[:,1]**2/(2*T/Iy) + x1[:,2]**2/(2*T/Iz) - np.ones_like(t)) <= 1e-8))
        return
        
    def test_axial_symmetric_rotation(self):
        
        sat = rotational_sat()
        sat.I = np.array([[1,0,0],
                          [0,1,0],
                          [0,0,5]])
        func = sat.dynamics
        
        T = 2*np.pi
        t = np.linspace(0, T, 1000)
        x0 = np.array([1,1,1,1,0,0,0])
        solver = tiamat.integrators.runge_kutta(f=func, x0=x0.copy(),t0=t[0])
        x1 = solver.solve(t, eps=1e-10)
        
        Ix = sat.I[0,0]
        Iy = sat.I[1,1]
        Iz = sat.I[2,2]
        
        lam = (Iz - Ix)/Ix * x0[2]
        x2 = x0[2] * np.ones_like(x1[:,0:3])
        x2[:,0] = x0[0] * np.cos(lam * t) - x0[1] * np.sin(lam * t)
        x2[:,1] = x0[1] * np.cos(lam * t) + x0[0] * np.sin(lam * t)
        
        self.assertTrue(np.all(np.abs(x1[:,0:3]-x2) <= 1e-8))
        return

if __name__ == '__main__':
    unittest.main()