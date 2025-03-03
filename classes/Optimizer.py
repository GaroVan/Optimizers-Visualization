# import vector
import numpy as np
import vpython as vp
from .vector import vector
from .Surface import Surface

class Optimizer:
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color)->None:
        self.position = vector(position_x, position_y, surface.get_z(position_x, position_y))
        self.velocity = vector()
        self.surface = surface
        self.lr = lr
        self.color = color
        self.losses = []

    def step(self)->float:
        '''
        updates the x and y position of the optimizer according to the optimization algorithm.
        returns the length of the velocity vector
        '''
        raise NotImplementedError
    
    def __repr__(self) -> str:
        pass


class GradientDescent(Optimizer):
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color):
        super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)

    def step(self)-> float:
        gradient_x, gradient_y = self.surface.derivative(self.position.x, self.position.y)
        self.velocity.x = - gradient_x * self.lr
        self.velocity.y = - gradient_y * self.lr
        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()
    
    def __repr__(self) -> str:
        return f'GD'


class StochGradDesc(Optmimizer):
     def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color, batch_size=1):
          super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)
          self.batch_size=batch_size #Number of batches in each iteration

    def step(self)-> float:
        batch_x, batch_y=self.surface.get_random_sample(self.batch_size)
        gradient_x, gradient_y = self.surface.derivative(batch_x, batch_y)

        self.velocity.x = - gradient_x * self.lr
        self.velocity.y = - gradient_y * self.lr
        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()
    def _repr_(self)->str:
        return f'SGD (batch_size={self.batch_size})'

class Momentum(Optimizer):
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color, gamma:float):
        super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)
        assert 0 <= gamma <= 1, 'gamma must be in the range [0, 1]'
        self.gamma = gamma

    def step(self)->float:
        gradient_x, gradient_y = self.surface.derivative(self.position.x, self.position.y)
        self.velocity.x = self.gamma * self.velocity.x - gradient_x * self.lr
        self.velocity.y = self.gamma * self.velocity.y - gradient_y * self.lr
        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()
    
    def __repr__(self) -> str:
        return f'Momentum'

class Nesterov(Optimizer):
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color, gamma:float):
        super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)
        assert 0 <= gamma <= 1, 'gamma must be in the range [0, 1]'
        self.gamma = gamma

    def step(self)-> float:
        x_ahead, y_ahead = self.position.x + self.gamma * self.velocity.x, self.position.y + self.gamma * self.velocity.y
        gradient_x, gradient_y = self.surface.derivative(x_ahead, y_ahead)
        self.velocity.x = self.gamma * self.velocity.x - gradient_x * self.lr
        self.velocity.y = self.gamma * self.velocity.y - gradient_y * self.lr

        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()

    def __repr__(self) -> str:
        return f'Nesterov'
    
class AdaGrad(Optimizer):
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color):
        super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)
        self.sum_square_grad_x = 0
        self.sum_square_grad_y = 0

    def step(self)-> float:
        gradient_x, gradient_y = self.surface.derivative(self.position.x, self.position.y)
        self.sum_square_grad_x += gradient_x ** 2
        self.sum_square_grad_y += gradient_y ** 2
        self.velocity.x = - gradient_x * self.lr / np.sqrt(self.sum_square_grad_x + 1e-8)
        self.velocity.y = - gradient_y * self.lr / np.sqrt(self.sum_square_grad_y + 1e-8)

        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()
    
    def __repr__(self) -> str:
        return f'AdaGrad'
    
class RMSProp(Optimizer):
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color, gamma:float):
        super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)
        assert 0 <= gamma <= 1, 'gamma must be in the range [0, 1]'
        self.gamma = gamma
        self.sum_square_grad_x = 0
        self.sum_square_grad_y = 0
    
    def step(self)->float:
        gradient_x, gradient_y = self.surface.derivative(self.position.x, self.position.y)
        self.sum_square_grad_x = self.gamma * self.sum_square_grad_x + (1-self.gamma) * gradient_x ** 2
        self.sum_square_grad_y = self.gamma * self.sum_square_grad_y + (1-self.gamma) * gradient_y ** 2
        self.velocity.x = - gradient_x * self.lr/ np.sqrt(self.sum_square_grad_x + 1e-8)
        self.velocity.y = - gradient_y * self.lr/ np.sqrt(self.sum_square_grad_y + 1e-8)

        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()
    
    def __repr__(self) -> str:
        return f'RMSProp'
    
class Adam(Optimizer):
    def __init__(self, position_x, position_y, *, surface:Surface, lr:float, color:vp.color, beta_1:float, beta_2:float):
        super().__init__(position_x, position_y, surface=surface, lr=lr, color=color)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 1
        self.sum_grad_x = 0
        self.sum_grad_y = 0
        self.sum_square_grad_x = 0
        self.sum_square_grad_y = 0
    
    def step(self)->float:
        self.t += 1
        grad_x, grad_y = self.surface.derivative(self.position.x, self.position.y)
        self.sum_grad_x = self.beta_1 * self.sum_grad_x + (1 - self.beta_1) * grad_x
        self.sum_grad_y = self.beta_1 * self.sum_grad_y + (1 - self.beta_1) * grad_y
        self.sum_square_grad_x = self.beta_2 * self.sum_square_grad_x + ( 1 - self.beta_2) * grad_x ** 2
        self.sum_square_grad_y = self.beta_2 * self.sum_square_grad_y + ( 1 - self.beta_2) * grad_y ** 2

        denominator_m = 1 - self.beta_1 ** self.t
        denominator_v = 1 - self.beta_2 ** self.t

        m_t_x, m_t_y = self.sum_grad_x / denominator_m, self.sum_grad_y / denominator_m
        v_t_x, v_t_y = self.sum_square_grad_x / denominator_v, self.sum_square_grad_y / denominator_v

        self.velocity.x = - m_t_x * self.lr / np.sqrt(v_t_x + 1e-8)
        self.velocity.y = - m_t_y * self.lr / np.sqrt(v_t_y + 1e-8)

        self.position.x = self.position.x + self.velocity.x
        self.position.y = self.position.y + self.velocity.y
        self.position.z = self.surface.get_z(self.position.x, self.position.y)

        return self.velocity.length()

    def __repr__(self) -> str:
        return f'Adam'

