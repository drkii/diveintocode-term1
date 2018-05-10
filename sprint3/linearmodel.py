import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearModel():

    def __init__(self, theta):
        self.theta = theta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self, X, y, lam):
        z = np.dot(X, self.theta)
        sig = self.sigmoid(z)
        m = len(X)
        # シグモイド関数、ラムダ、目的変数を目的関数に組み込む
        return ((-1 * y * np.log(sig) - (1 - y) * np.log(1 - sig)).sum() / m)[0] + (lam / (2 * m)) * \
               ((self.theta ** 2) * pd.DataFrame([0, 1, 1, 1, 1])).sum()

    def gradient_descent(self, X, y, iterations, alpha, lam, reset_theta=False):
        """
        args:
          alpha: Step size/Learning rate
          iterations: No. of iterations(Number of iterations)
        """
        if reset_theta:
            self.theta = pd.DataFrame(np.random.rand(5))

        past_costs = []
        past_thetas = [self.theta]
        correct_or_not = pd.DataFrame(np.ones(self.theta.count()))
        correct_or_not[0][0] = 0

        m = len(X)

        for a in range(iterations):
            reg_term = lam * self.theta * correct_or_not
            z = np.dot(X, self.theta)
            sig = self.sigmoid(z)
            self.theta = self.theta - alpha / m * (np.dot(X.T, (sig - y)) + reg_term)
            past_costs.append(self.compute_cost(X, y, lam))
            past_thetas.append(self.theta)

        return past_costs, past_thetas

    def plot_learning_curve(self, X, y, iterations, alpha, lam, reset_theta=False):
        if reset_theta:
            self.theta = pd.DataFrame(np.random.rand(5))

        past_costs, past_thetas = self.gradient_descent(X, y, iterations, alpha, lam)

        plt.figure(figsize=(12, 8))
        plt.plot(range(iterations), past_costs, label='Iris-setosa')

        plt.legend()
        plt.title('Cost Fusion J')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

    def predict_probs(self, X):
        z = np.dot(X, self.theta)
        return self.sigmoid(z)

    def predict(self, X):
        probs = self.predict_probs(X)
        species = []
        for i in range(len(X)):
            if probs[i, -1] >= 0.5:
                species.append(1)
            else:
                species.append(0)
        return species


if __name__ == '__main__':
    iris_df = pd.read_csv('../input/Iris.csv')
    # 2クラスに絞る
    iris_df = iris_df[iris_df["Species"] != "Iris-setosa"]

    y_df = iris_df[["Species"]]
    x_df = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

    x_df.insert(0, 'x0', 1)
    # 種類を0,1に変換する
    class_mapping = {x: i for i, x in enumerate(np.unique(y_df["Species"]))}
    y_df["Species"] = y_df["Species"].map(class_mapping)

    theta = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5])
    model = LinearModel(theta)

    print(model.compute_cost(x_df, y_df, 0.01))
    (past_costs,past_thetas)=model.gradient_descent(x_df, y_df, 100, 0.01, 0.01)
    print('last cost is {}'.format(past_costs[-1]))

    model.plot_learning_curve(x_df, y_df, 2000, 0.01, 0.01, reset_theta=True)

    predict_list = model.predict(x_df)

    compare_df = pd.concat([pd.DataFrame(predict_list), y_df.reset_index(drop=True)], axis=1)
    compare_df['same'] = compare_df[0] == compare_df['Species']
    print(compare_df['same'].value_counts())