from terrain.terrains import *
import numpy as np

X_MIN = -15
X_MAX = 15
Y_MIN = -15
Y_MAX = 15

CHOSEN_FUNCTION = gaussian_terrain

LEARNING_RATE = 0.1
ADAGRAD_LEARNING_RATE = 1
RMSPROP_LEARNING_RATE = 0.5
ADAM_LEARNING_RATE = 0.5
dt = 0.005

T = 20