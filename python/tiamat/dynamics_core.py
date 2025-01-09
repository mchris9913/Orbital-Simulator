from abc import ABC, abstractmethod
import numpy as np

def translational_dynamics(x: np.ndarray, params: tuple) -> np.ndarray:
    """
    r' = v
    v' = -mu/r^3 * r
    """
    
    POS_IDX, VEL_IDX, mu = params
    
    xdot = np.zeros_like(x)
    xdot[POS_IDX] = x[VEL_IDX]
    xdot[VEL_IDX] = -mu/(np.linalg.norm(x[POS_IDX])**3) * x[POS_IDX]
    return xdot

def rotational_dynamics(x: np.ndarray, params: tuple) -> np.ndarray:
    """
    Iw' + w x Iw = M
    """
    I, M = params
    xdot = np.linalg.inv(I) @ (M - np.outer(x, I @ x))
    return x
