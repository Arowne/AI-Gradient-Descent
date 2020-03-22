from numpy import *

# Datasets => 100 points
# Prediction model => ax+b
# Loss function => mse
# Minimization function => gradient_descent
class Gradient_descent():

    def __init__(self):
        
        # Init intersection
        self.b = 0
        # Init slope
        self.m = 0
        
        # Set number of epochs
        self.iterations = 1000

        # Set learning rate
        self.learning_rate = 0.0001

        # Set mean squarred error value
        self.mse_value = 0

        # Init datasets
        self.points = array(genfromtxt('data.csv', delimiter=","))

    def run(self):
        self.mse()
        print(self.m)
        print(self.b)
        print(self.mse_value)
        self.gradient_descent()
        self.mse()
        print(self.m)
        print(self.b)
        print(self.mse_value)
        print(self.m * len(self.points+10) + self.b)

    # Gradient descent
    def gradient_descent(self):
        iteration = self.iterations
        for x in range(iteration):
            self.gradient_step()

    # Update slope (m) and intersect (b) with them derivative time learning_rate
    def gradient_step(self):
        
        points = self.points
        m = self.m
        b = self.b
        N = float(len(points))
        m_derivative = 0
        b_derivative = 0

        for i in range(len(self.points)):
            x = points[i, 0]
            y = points[i, 1]
            b_derivative += -(2/N) * (y - ((m * x) + b))
            m_derivative += -(2/N) * x * (y - ((m*x) + b) )
            
        self.b = b - (self.learning_rate * b_derivative)
        self.m = m - (self.learning_rate * m_derivative)

    # Calculate mean square error (Error average)
    def mse(self):

        points = self.points
        m = self.m
        b = self.b
        get_sum_abs_delta = 0

        for i in range(len(self.points)):
            x = points[i, 0]
            y = points[i, 1]
            get_sum_abs_delta += (y - (m*x+b))**2

        self.mse_value = get_sum_abs_delta/len(self.points)

if __name__ == "__main__":
    gradient_descent = Gradient_descent()
    gradient_descent.run()