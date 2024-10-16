import time
import matplotlib.pyplot as plt
import numpy as np
import vpython as vp
import params
import graphics
from classes.vector import vector
from classes import Optimizer
from classes import Surface




vp.scene.caption = f'''
    Press the Play button to start the optimization process.
    Right click/ctrl+click to rotate the camera
    shift+left click to drag
    scroll to zoom
    '''

is_paused = True
def toggle_pause():
    """Toggles the pause state."""
    global is_paused
    if is_paused:
        is_paused = False
        pause_button.text = "Pause"
    else:
        is_paused = True
        pause_button.text = "Play"



if __name__ == '__main__':
    vp.canvas(background=vp.vector(0.9, 0.9, 0.9), width=800, height=800)

    global pause_button
    pause_button = vp.button(text="Play", bind=lambda: toggle_pause())
    
    surface = Surface.Surface(params.CHOSEN_FUNCTION, params.X_MIN, params.X_MAX, params.Y_MIN, params.Y_MAX)
    rendering = graphics.Graphics(surface)
    rendering.plot_surface()
    
    start_x = params.START_X
    start_y = params.START_Y
    
    graddesc = Optimizer.GradientDescent(start_x, start_y, surface=surface, lr=params.LEARNING_RATE, color=vp.color.red)
    momentum = Optimizer.Momentum(start_x, start_y, surface=surface, lr=params.LEARNING_RATE, color=vp.color.blue, gamma=0.95)
    nesterov = Optimizer.Nesterov(start_x, start_y, surface=surface, lr=params.LEARNING_RATE, color=vp.color.orange, gamma=0.95)
    adagrad = Optimizer.AdaGrad(start_x, start_y, surface=surface, lr=params.ADAGRAD_LEARNING_RATE, color=vp.color.green)
    rmsprop = Optimizer.RMSProp(start_x, start_y, surface=surface, lr=params.RMSPROP_LEARNING_RATE, color=vp.color.yellow, gamma=0.9)
    adam = Optimizer.Adam(start_x, start_y, surface=surface, lr=params.ADAM_LEARNING_RATE, color=vp.color.purple, beta_1=0.7, beta_2=0.999)
    
    rendering.add_optimizer(graddesc)
    rendering.add_optimizer(nesterov)
    rendering.add_optimizer(momentum)
    rendering.add_optimizer(adagrad)
    rendering.add_optimizer(rmsprop)
    rendering.add_optimizer(adam)

    rendering.show_labels()
    
    t = 0
    winner_optimizers = 0   # for leaderboard positions

    while True:
        vp.rate(30)
        while is_paused:
            vp.rate(30)

        del_v = []
        t += params.dt
        
        for optim in rendering.optimizers:
            del_v.append(optim.step())

        rendering.render_optimizers()

        # code for the leaderboard
        if any([v < 1e-4 for v in del_v]):
            # if an optimizer makes sufficiently small steps, it is considered to have converged
            # get information of the converged optimizer
            idx = np.argmin(del_v)
            optim_str = repr(rendering.optimizers[idx])
            optim_color = rendering.optimizers[idx].color
            # add the converged optimizer to the leaderboard
            winner_optimizers += 1
            rendering.add_to_leaderboard(optim_str, place=winner_optimizers, color=optim_color)
            rendering.optimizers.pop(idx)
            

        # stopping the simulation
        if t > params.T:
            vp.text(text="Time's up!", pos=vp.vector(0, 0, 0), height=2, color=vp.color.black)
            time.sleep(5)
            break