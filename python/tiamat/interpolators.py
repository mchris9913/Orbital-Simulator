from tiamat.tiamat import cubic_spline_1d_interpolate
from .numerical_methods_core import interpolator

import numpy as np

class cubic_spline_1d(interpolator):
    
    def __init__(self)->None:
        super().__init__()
        return
        
    def interpolate(self, t: np.ndarray, data: tuple) -> np.ndarray:
        return np.array(cubic_spline_1d_interpolate(t, data))