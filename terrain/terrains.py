import numpy as np
from .perlin import generate_terrain, terrain_built, p_terrain
from .utils import grid_to_coordinate, bilinear_interpolation
from .gaussian import gaussian

def square(x, y, x_min, x_max, y_min, y_max, scale=15):
    return (x**2 + y**2) / scale

def saddle(x, y, x_min, x_max, y_min, y_max, scale=15):
    return (x**2 - y**2) / scale

def ripple(x, y, x_min, x_max, y_min, y_max, scale=1.5):
    d = np.sqrt(x**2 + y**2)
    return d - np.cos(d)


# def perlin_terrain(x, y, x_min, x_max, y_min, y_max):
#     global terrain_built, p_terrain
#     if not terrain_built:
#         terrain_built = True
#         p_terrain = generate_terrain()
#     # terrain is a (100, 100) shaped numpy array
#     # we need to be able to get values from the terrain even from floating x, y
#     n = p_terrain.shape[0]
#     div = n -1
#     x_norm = (x - x_min) / (x_max - x_min)
#     y_norm = (y - y_min) / (y_max - y_min)
#     del_x = (x_max - x_min) / div
#     del_y = (y_max - y_min) / div
#     if type(x) == np.ndarray:
#         # will only be used for plotting the surface
#         x_idx = (x_norm * div).astype(int)
#         y_idx = (y_norm * div).astype(int)
#         return p_terrain[x_idx, y_idx] * 50
#     else:
#         # fine grained calculation needed
#         i, j = int(x_norm * div), int(y_norm * div)
#         x1, y1, z1 = grid_to_coordinate(j/div, x_min, x_max), grid_to_coordinate(i/div, y_min, y_max), p_terrain[i, j]
#         x2, y2, z2 = x1 + del_x, y1, p_terrain[i, j+1]
#         x3, y3, z3 = x1 + del_x, y1 + del_y, p_terrain[i+1, j+1]
#         x4, y4, z4 = x1, y1 + del_y, p_terrain[i+1, j]
#         return bilinear_interpolation(x, y, [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]) * 50
        
def gaussian_terrain(x, y, x_min, x_max, y_min, y_max):
    """
    Function to generate the z-value of a terrain consisting of 2 or 3 Gaussian functions.
    Two of the Gaussian functions are multiplied by -1 to create valleys.
    
    Parameters:
    - x, y: Coordinates where to evaluate the terrain.
    
    Returns:
    - z: Value of the terrain at point (x, y).
    """
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Parameters for Gaussian 1 (a peak)
    x0_1, y0_1 = x_min, y_min  # Center of the first Gaussian
    sigma_x1, sigma_y1 = 5, 5  # Spread along x and y
    amplitude_1 = 10.0   # Positive amplitude for the peak

    # Parameters for Gaussian 2 (a valley)
    x0_2, y0_2 = x_range * 0.75 + x_min, y_range * 0.25 + y_min  # Center of the second Gaussian
    sigma_x2, sigma_y2 = 7, 7  # Spread along x and y
    amplitude_2 = -8  # Negative amplitude for the valley

    # Parameters for Gaussian 3 (another valley)
    x0_3, y0_3 = x_range * 0.5 + x_min, y_range * 0.75 + y_min  # Center of the third Gaussian
    sigma_x3, sigma_y3 = 10, 7  # Spread along x and y
    amplitude_3 = -10  # Negative amplitude for the valley

    # Calculate the terrain height z as a sum of the Gaussian functions
    z = (gaussian(x, y, x0_1, y0_1, sigma_x1, sigma_y1, amplitude_1) +
         gaussian(x, y, x0_2, y0_2, sigma_x2, sigma_y2, amplitude_2) +
         gaussian(x, y, x0_3, y0_3, sigma_x3, sigma_y3, amplitude_3))
    
    return z




