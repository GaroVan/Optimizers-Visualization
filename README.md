# Machine Learning Optmizers Visualization
## About the project
An interactive visualization of the most popular optimization algorithms used in machine learning.
The goal of this project was to visually learn the differences between the optimization algorithms and which one might be suited for a particular problem.

![Demo](./assets/gifs/all%20demo%20places.gif)

## Demo Video
[![Project Demo](https://img.youtube.com/vi/wnicogJJn1g/0.jpg)](https://www.youtube.com/watch?v=wnicogJJn1g)



## How to run
- Clone the repository
- Install the requirements from the `requirements.txt` file
    - `pip install -r requirements.txt`
- Run the main.py file
    - You can change the terrain by changing the `CHOSEN_FUNCTION` variable in the `params.py` file
    - To define a new terrain, write a new function that will take in (x, y, xmin, xmax, ymin, ymax) and return the z value. You can assign the new function in the `CHOSEN_FUNCTION` variable in the `params.py` file to see it in action.
    - The `params.py` file also contains the bounds for the terrain, the initial point for the optimization algorithm, and the learning rate for the optimization algorithm. You can change them as you wish and re run the code to see the changes.
    - You can also change the parameters of the optimization algorithms by changing the lines in the `main.py` file where the optimizer objects are created. 

## Optimizers
Currently supports the following optimizers:
- Gradient Descent
- Momentum
- Nesterov Accelerated Gradient
- Adagrad
- RMSprop
- Adam