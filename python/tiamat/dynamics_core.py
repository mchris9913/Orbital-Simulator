from abc import ABC, abstractmethod
import numpy as np

class ClassicalElementsPropagator(object):
    def __init__(self, a: float, e: float, i: float, Ω: float, ω: float, ν0: float, μ: float = 1.0)->None:
        self.e = e
        self.a = a
        self.i = i
        self.Ω = Ω
        self.ω = ω
        self.ν0 = ν0
        self.μ = μ
        self.n = np.sqrt(self.μ/self.a**3)
        self.E0 = self.E(ν0)
        return

    def E(self, ν: float)->float:
        return 2*np.arctan(np.sqrt((1-self.e)/(1+self.e))*np.tan(ν/2))
        
    def Rx(self, θ: float)->np.ndarray:
        return np.array([[1,0,0],[0,np.cos(θ),np.sin(θ)],[0,-np.sin(θ),np.cos(θ)]])
        
    def Rz(self, θ: float)->np.ndarray:
        return np.array([[np.cos(θ),np.sin(θ),0],[-np.sin(θ), np.cos(θ),0],[0,0,1]])
  
    def get_state(self, θ: float, anomaly_type: str = 'true')->np.ndarray:
        R = self.Rz(-self.Ω) @ self.Rx(-self.i) @ self.Rz(-self.ω)
        if anomaly_type == 'true':
            E = self.E(θ)
        elif anomaly_type == 'mean':
            E = self.NewtonRaphson(θ)
        elif anomaly_type == 'eccentric':
            E = θ
        else:
            raise Exception("Given anomaly type is not accounted for")
        
        r = np.array([self.a*(np.cos(E) - self.e), self.a*np.sqrt(1-self.e**2)*np.sin(E), 0.0])
        r = R @ r
        
        v = self.a*self.n/(1-self.e*np.cos(E))*np.array([-np.sin(E),np.sqrt(1-self.e**2)*np.cos(E),0.0])
        v = R @ v
        return np.concatenate((r,v))
    
    def NewtonRaphson(self, M0: float, ε: float = 1e-6)->float:
        E = M0
        δ = -(E - self.e*np.sin(E) - M0)/(1 - self.e*np.cos(E))
        
        while np.abs(δ) >= ε:
            E += δ
            δ = -(E - self.e*np.sin(E) - M0)/(1 - self.e*np.cos(E))
            
        return E
    
    def solve(self, t: np.ndarray, ε: float = 1e-6)->np.ndarray:
        x = [self.get_state(self.ν0)]
        M = self.E0 - self.e * np.sin(self.E0)
        
        for i in range(1,len(t)):
            M += self.n*(t[i] - t[i-1])
            E = self.NewtonRaphson(M, ε)
            x.append(self.get_state(E, anomaly_type='eccentric'))
            
        return np.array(x)
        

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
    xdot = np.linalg.inv(I) @ (M - np.cross(x, I @ x))
    return xdot
    
def quaternion_kinematics(x: np.ndarray, params: tuple) -> np.ndarray:
    def W(q: np.ndarray)->np.ndarray:
        return np.array([[-q[1],q[0],q[3],q[2]],[-q[2],-q[3],q[0],q[1]],[-q[3],q[2],-q[1],q[0]]])
    
    ROT_IDX, QUAT_IDX = params
    
    q = x[QUAT_IDX] / np.linalg.norm(x[QUAT_IDX])
    qdot = 1/2 * W(q).T @ x[ROT_IDX]   
    return qdot
