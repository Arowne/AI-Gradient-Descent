from numpy import *
import numpy as np
import csv
import matplotlib.pyplot as plt
    
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
        self.iterations = 10000**2

        # Set learning rate
        self.learning_rate = 0.0001

        # Set mean squarred error value
        self.mse_value = None

        # Init datasets
        self.points = array(genfromtxt('data.csv', delimiter=","))
        self.computed_rows = []
        self.end = False

    def run(self):
        self.mse()
        print(self.m)
        print(self.b)
        print(self.mse_value)
        self.gradient_descent()
        self.mse()
        self.get_slope()
        print(self.m)
        print(self.b)
        print(self.mse_value)

    # Gradient descent
    def gradient_descent(self):
        iteration = self.iterations
        while self.end != True:
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
            b_derivative += -(2) * (y - ((m * x) + b))
            m_derivative += -(2) * x * (y - ((m*x) + b) )
            
        b_derivative = b_derivative/N
        m_derivative = m_derivative/N
        
        self.b = b - (self.learning_rate * b_derivative)
        self.m = m - (self.learning_rate * m_derivative)
        self.mse()

        if self.mse_value <= 8.31:
            self.end = True

    # Calculate mean absolute square error (Error average)
    def mse(self):

        points = self.points
        m = self.m
        b = self.b
        get_sum_abs_delta = 0

        for i in range(len(self.points)):
            x = points[i, 0]
            y = points[i, 1]
            get_sum_abs_delta += abs(y - (m*x+b))


        self.mse_value = get_sum_abs_delta/len(self.points)
        print(self.mse_value)

    def get_slope(self):
        for i in range(len(self.points)):
            x = self.points[i, 0]
            y = (self.m * x + self.b)
            print("Prediction => " + str(y))
            print("Expected => " + str(self.points[i, 1]))
            self.computed_rows.append([x, y])


if __name__ == "__main__":
    gradient_descent = Gradient_descent()
    gradient_descent.run()


    x = []
    y = []
    linear_x = []
    linear_y = []

    with open('data.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(row[0])
            y.append(row[1])
        
        for row in gradient_descent.computed_rows:
            linear_x.append(row[0])
            linear_y.append(row[1])

    
    plt.plot(linear_x, linear_y, 'r', x, y, label='Linear overall')
    plt.show()

    plt.scatter(x,y, label='Loaded from file!')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()
