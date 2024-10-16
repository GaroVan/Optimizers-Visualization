from terrain.terrains import *
import numpy as np

X_MIN = -15
X_MAX = 15
Y_MIN = -15
Y_MAX = 15

START_X = -11
START_Y = -13

assert X_MIN < START_X < X_MAX, f'starting x value {START_X} is not within the bounds of the function {X_MIN} to {X_MAX}'
assert Y_MIN < START_Y < Y_MAX, f'starting y value {START_Y} is not within the bounds of the function {Y_MIN} to {Y_MAX}'

CHOSEN_FUNCTION = gaussian_terrain

LEARNING_RATE = 0.1
ADAGRAD_LEARNING_RATE = 1
RMSPROP_LEARNING_RATE = 0.5
ADAM_LEARNING_RATE = 0.5

dt = 0.03
T = 20
