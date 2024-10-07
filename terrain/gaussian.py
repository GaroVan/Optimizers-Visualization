import numpy as np

def gaussian(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    """
    Optimized 2D Gaussian function.
    
    Parameters:
    - x, y: Coordinates where to evaluate the function (can be scalar or numpy arrays).
    - x0, y0: Center of the Gaussian.
    - sigma_x, sigma_y: Spread (standard deviation) along the x and y axes.
    - amplitude: The amplitude (height) of the Gaussian.
    
    Returns:
    - z: Value of the Gaussian at point (x, y).
    """
    # Precompute constants
    inv_sigma_x = 1 / (2 * sigma_x ** 2)
    inv_sigma_y = 1 / (2 * sigma_y ** 2)
    
    # Compute squared distances
    dx2 = (x - x0) ** 2
    dy2 = (y - y0) ** 2
    
    # Calculate Gaussian function
    z = amplitude * np.exp(-(dx2 * inv_sigma_x + dy2 * inv_sigma_y))
    
    return z
