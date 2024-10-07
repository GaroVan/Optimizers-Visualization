import numpy as np
import matplotlib.pyplot as plt
from .vector import vector

class Surface:
    def __init__(self, function, x_min:float, x_max:float, y_min:float, y_max:float, granularity:int=50):
        self.function = function
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.X, self.Y = np.meshgrid(np.linspace(x_min, x_max, granularity), np.linspace(y_min, y_max, granularity))
        self.Z = self.function(self.X, self.Y, self.x_min, self.x_max, self.y_min, self.y_max)
    
    def get_z(self, x:float, y:float)-> float:
        return self.function(x, y, self.x_min, self.x_max, self.y_min, self.y_max)
    
    def derivative(self, x:float, y:float)-> tuple[float, float]:
        d = 1e-6
        dz_dx = (self.get_z(x + d, y) - self.get_z(x, y)) / d
        dz_dy = (self.get_z(x, y + d) - self.get_z(x, y)) / d
        return dz_dx, dz_dy
