from abc import ABC, abstractmethod
import numpy as np

class integrator(ABC):
	
    def __init__(self, f: callable = None, x0: np.ndarray = None, t0: float = None) -> None:
        self.f = f
        self.x0 = x0
        self.t0 = t0
        self.x = []
        self.t = []
        return
		
    @abstractmethod
    def integrate(self, x0: np.ndarray, t0: float, tf: float, dt: float = 1e-3):
        ...
		
    @abstractmethod
    def solve(self, t: np.ndarray):
        ...
    
    def reset(self):
        self.f = None
        self.x0 = None
        self.t0 = None
        self.x = []
        self.t = []
        return
        
    def set_func(self, f: callable) -> None:
        self.f = f
        return
    
    def set_ic(self, x0: np.ndarray, t0: np.ndarray) -> None:
        self.x0 = x0
        self.t0 = t0
        return
        
    def get_func(self) -> callable:
        return self.f
        
    def get_ic(self) -> tuple:
        return (self.x0, self.t0)
    
    def get_output(self) -> tuple:
        return (np.array(self.x), np.array(self.t))
    
    
class interpolator(ABC):
    
    def __init__(self)->None:
        return
        
    @abstractmethod
    def evaluate(self, x: float)->np.ndarray:
        pass
        
    @abstractmethod
    def set_model(self)->None:
        pass