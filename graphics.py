import matplotlib.pyplot as plt
import numpy as np
import vpython as vp

from classes.Surface import Surface
from classes.Optimizer import Optimizer

def normalize(x, x_min, x_range):
    return (x - x_min) / x_range


class Graphics:
    def __init__(self, surface:Surface):
        self.surface:Surface = surface
        self.optimizers:list[Optimizer] = []
        self.spheres:list[vp.sphere] = []
    
    def add_optimizer(self, optimizer:Optimizer)-> None:
        self.optimizers.append(optimizer)
        self.spheres.append(vp.sphere(pos=vp.vector(optimizer.position.x, optimizer.position.y, optimizer.position.z), 
                                      radius=0.5, 
                                      color=optimizer.color, 
                                      make_trail=True,
                                      trail_color=optimizer.color,
                                      pps=10,
                                      retain=40)
                            )
    
    def labeling(self)-> None:
        y_min, y_max = self.surface.y_min, self.surface.y_max
        x_min, x_max = self.surface.x_min, self.surface.x_max
        n = len(self.optimizers)
        gap = (x_max - x_min) / n
        for i, optim in enumerate(self.optimizers):
            vp.sphere(pos=vp.vector(i * gap + x_min, y_min - 5, 0), radius=0.5, color=optim.color)
            vp.text(text=f'{optim}', pos=vp.vector(i * gap + x_min + 1, y_min - 5, 0), height=0.5, color=optim.color)

    def plot_surface(self, colormap:str='viridis')-> None:
        """
        Plots a 3D surface using vpython.

        Parameters:
        - x, y, z: np.ndarray of shape (n, n) representing the coordinates of the surface.
                These arrays must be square matrices of the same shape.
        - colormap: A string specifying the colormap to be used (default is 'viridis').
        """
        x, y, z = self.surface.X, self.surface.Y, self.surface.Z
        assert x.shape == y.shape == z.shape, 'x, y, z must have the same shape'
        assert x.shape[0] == x.shape[1], 'x, y, z must be square matrices'

        n = x.shape[0]
        cmap = plt.get_cmap(colormap)
        z_min, z_max = np.min(z), np.max(z)
        z_range = z_max - z_min
        
        for i in range(n-1):
            for j in range(n-1):
                z_avg = (z[i,j] + z[i,j+1] + z[i+1,j+1] + z[i+1,j]) / 4
                z_norm = normalize(z_avg, z_min, z_range)
                color = cmap(z_norm)
                color = vp.vector(color[0], color[1], color[2])

                v1 = vp.vertex(pos=vp.vector(x[i,j], y[i,j], z[i,j]), color=color)
                v2 = vp.vertex(pos=vp.vector(x[i,j+1], y[i,j+1], z[i,j+1]), color=color)
                v3 = vp.vertex(pos=vp.vector(x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]), color=color)
                v4 = vp.vertex(pos=vp.vector(x[i+1,j], y[i+1,j], z[i+1,j]), color=color)

                vp.quad(vs=[v1, v2, v3, v4])

    def render_optimizers(self)-> None:
        for optim, sphere in zip(self.optimizers, self.spheres):
            sphere.pos = vp.vector(optim.position.x, optim.position.y, optim.position.z)

    
