import tiamat
import tiamat.integrators

import unittest
import numpy as np
from scipy.linalg import expm

class TestRungeKutta(unittest.TestCase):
    
    def test_fode(self):
        """
        x' + x = 0

        solution x = x(0)*e^(-t)
        """
        
        def func(t: float, x: np.ndarray)->np.ndarray:
            return -x
        
        x0 = np.array([1])
        t = np.linspace(0,10,100).reshape(-1,1)
        solver = tiamat.integrators.runge_kutta(f=func,x0=x0.copy(),t0=0)
        x1 = solver.solve(t)
        
        x2 = x0[0]*np.exp(-t)
        self.assertTrue(np.all(np.abs(x1 - x2) <= 1e-5))
        return
        
    def test_sode(self):
        """
        x'' + x' + x = 0

        solution x(t) = C1*e^(-1/2*t)*cos(sqrt(3)/2*t) + C2*e^(-1/2*t)*sin(sqrt(3)/2*t)
        C1 = x(0)
        C2 = 2/sqrt(3)*x'(0) + 1/sqrt(3)*x(0)
        """
        
        def func(t: float, x: np.ndarray)->np.ndarray:
            return np.array([x[1],-x[0]-x[1]])

        x0 = np.array([1,1])
        t = np.linspace(0,10,100).reshape(-1,1)
        solver = tiamat.integrators.runge_kutta(f=func,x0=x0.copy(),t0=0)
        x1 = solver.solve(t)
        
        x1 = x1[:,0]
        C1 = x0[0]
        C2 = 2/np.sqrt(3)*x0[1] + 1/np.sqrt(3)*x0[0]
        x2 = C1*np.exp(-1/2*t)*np.cos(np.sqrt(3)/2*t) + C2*np.exp(-1/2*t)*np.sin(np.sqrt(3)/2*t)
        self.assertTrue(np.all(np.abs(x1 - x2.reshape(-1)) <= 1e-5))
        return
        
    def test_random_dynamics(self):
        """
        A = rand(3,3)
        D = -(A'*A)

        f(x) = D*x
        solution x(t) = x(0)*expm(D*t)
        """
        
        A = np.random.randn(9).reshape(3,3)
        D = -(A.T @ A)
        
        def func(t: float, x: np.ndarray)->np.ndarray:
            return D @ x
            
        x0 = np.array([1,1,1])
        t = np.linspace(0,10,100).reshape(-1,1)
        solver = tiamat.integrators.runge_kutta(f=func,x0=x0.copy(),t0=0)
        x1 = solver.solve(t)
        
        x2 = []
        for ti in t:
            x2.append(expm(D*ti) @ x1[0,:])
        x2 = np.array(x2)
        
        self.assertTrue(np.all(np.abs(x1 - x2) <= 1e-5))
        
if __name__ == '__main__':
    unittest.main()