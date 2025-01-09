from abc import ABC, abstractmethod
import numpy as np

def translational_dynamics(x: np.ndarray, param: tuple) -> np.ndarray:
    POS_IDX, VEL_IDX, mu = param
    
    xdot = np.zeros_like(x)
    xdot[POS_IDX] = x[VEL_IDX]
    xdot[VEL_IDX] = -mu/(np.linalg.norm(x[POS_IDX])**3) * x[POS_IDX]
    return xdot

def rotational_dynamics(x: np.ndarray, param: tuple) -> np.ndarray:
    return
