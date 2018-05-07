import numpy as np
import pandas as pd

class LinearModel():

    def __init__(self):
        pass

    def compute_cost(self, X, y, theta):
        # TODO: implemet try: except statement
        # data size
        m = y.count()
        # add x0's row
        #X['x0'] = 1
        error = np.dot(X, theta) - np.array(y)
        return  (error ** 2).sum() / (2 * m)

    def gradient_descent(self, X, y, theta, iterations, alpha):
        """
        args:
          alpha: Step size/Learning rate
          iterations: No. of iterations(Number of iterations)
        """
        # data size
        m = int(y.count())
        # add x0's row
        X['x0'] = 1
        # need initial
        past_costs = [compute_cost(X, y, theta)]
        past_thetas = [theta]
        for i in range(iterations):
            # update theta parameter
            error = np.dot(X, theta) - np.array(y)
            theta = theta - alpha/m * np.dot(X.T, error)
            cost = compute_cost(X, y, theta)
            # store now cost and theta
            past_costs.append(cost)
            past_thetas.append(theta)
        return (past_costs, past_thetas)

    def plot_learning_curve(self, X, y, iterations, alpha):
        # parameter size
        # theta_size = X.shape[1]

        # set initial theta
        theta = pd.DataFrame(np.random.rand(X.shape[1]))

        past_costs = gradient_descent(X, y, theta, iterations, alpha)[0]
        plt.plot(np.array(list(range(iterations+1))), np.array(past_costs))

        plt.title('Learning Curve', fontsize=20)

        plt.xlabel("iteration", fontsize=15)
        plt.ylabel("cost", fontsize=15)
        print("last cost is {}".format(past_costs[-1]))

