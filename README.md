 # Machine Learning Optmizers Visualization
 ## About the project
 This project is a visualization of the most popular optimization algorithms used in machine learning.
 The goal of this project was to visually learn the differences between the optimization algorithms and which one might be suited for a particular problem.


## Demo Video
[![Project Demo](https://youtu.be/wnicogJJn1g)](https://youtu.be/wnicogJJn1g)



## How to run
- Clone the repository
- Install the requirements from the `requirements.txt` file
    - `pip install -r requirements.txt`
- Run the main.py file
    - You can change the terrain by changing the `CHOSEN_FUNCTION` variable in the `params.py` file
    - the starting point of the optimizers can also be changed in the `main` function in the `main.py` file

## Optimizers
This project currently supports the following optimizers:
- Gradient Descent
- Momentum
- Nesterov Accelerated Gradient
- Adagrad
- RMSprop
- Adam