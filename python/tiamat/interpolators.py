from .numerical_methods_core import interpolator

import numpy as np

class cubic_spline_1d(interpolator):
    
    def __init__(self)->None:
        super().__init__()
        
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.k1 = None
        self.k2 = None
        self.a = None
        self.b = None
        return
        
    def set_model(self, p1: tuple, p2: tuple)->None:
        self.x1 = p1[0]
        self.y1 = p1[1]
        self.k1 = p1[2]
        
        self.x2 = p2[0]
        self.y2 = p2[1]
        self.k2 = p2[2]
        
        self.a = self.k1*(self.x2 - self.x1) - (self.y2 - self.y1)
        self.b = -self.k2*(self.x2 - self.x1) + (self.y2 - self.y1)
        return
        
    def evaluate(self, x: float)->np.ndarray:
        p = self.t(x)
        return (1-p)*self.y1 + p*self.y2 + p*(1-p)*((1-p)*self.a + p*self.b)
        
    def t(self, x: float)->float:
        return (x - self.x1)/(self.x2 - self.x1)
    