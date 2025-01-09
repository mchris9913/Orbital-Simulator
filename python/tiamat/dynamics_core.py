from abc import ABC, abstractmethod
import numpy as np

class vehicle(ABC):
    def __init__(self, x0: np.ndarray)->None:
        self.x = [x0]
        return
