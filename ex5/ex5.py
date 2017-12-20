import pickle
import numpy as np
import copy
import random as rand
import matplotlib.pyplot as plt
from robo.fmin import bayesian_optimization

rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))


def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x by the prediction of a surrogate regression model,
        which was trained on the validation error of randomly sampled hyperparameter configurations.
        The original surrogate predicts the validation error after a given epoch. Since all hyperparameter configurations were trained for a total amount of
        40 epochs, we will query the performance after epoch 40.
    """

    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)

    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]

    return y


def runtime(x, epoch=40):
    """
        Function wrapper to approximate the runtime of the hyperparameter configurations x.
    """

    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y


def optimize_random(bounds):
    D = len(bounds)
    best_y_performance = 1
    best_y_runtime = 999999.0
    list_y_performance = np.zeros((10, 50))
    list_y_runtime = np.zeros((10, 50))
    for k in range(10):
        for i in range(50):
            new_configuration = [rand.randint(bounds[d][0], bounds[d][1]) for d in range(D)]
            new_output = objective_function(new_configuration)
            if new_output < best_y_performance:
                best_y_performance = new_output
            new_runtime = runtime(new_configuration)
            if new_runtime < best_y_runtime:
                best_y_runtime = new_runtime
            list_y_performance[k][i] = best_y_performance
            list_y_runtime[k][i] = best_y_runtime
    mean_performance = np.mean(list_y_performance, axis=0)
    mean_runtimes = np.cumsum(np.mean(list_y_runtime, axis=0))
    return mean_performance, mean_runtimes


def optimizate_bayesian():
    lower = np.array([-6, 32, 4, 4, 4])
    upper = np.array([0, 512, 10, 10, 10])
    list_y_performance = np.zeros((10, 50))
    list_y_runtime = np.zeros((10, 50))
    for i in range(10):
        result = bayesian_optimization(objective_function, lower, upper, num_iterations=50)
        list_y_performance[i] = result.get('incumbent_values')
        list_y_runtime[i] = result.get('runtime')
        mean_performance = np.mean(list_y_performance, axis=0)
        mean_runtime = np.cumsum(np.mean(list_y_runtime, axis=0))
    return mean_performance, mean_runtime


if __name__ == '__main__':
    bounds = [[-6, 1], [32, 512], [16, 1024], [16, 1024], [16, 1024]]
    performance_random, runtimes_random = optimize_random(bounds)
    performance_bayesian, runtimes_bayesian = optimizate_bayesian()
    iterations = range(50)
    plt.plot(iterations, performance_random, label='random search')
    plt.plot(iterations, performance_bayesian, label='bayesian search')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.legend(loc='lower right')
    plt.show()
    plt.plot(iterations, runtimes_random, label='random search')
    plt.plot(iterations, runtimes_bayesian, label='bayesian search')
    plt.xlabel('iterations')
    plt.ylabel('runtimes')
    plt.legend(loc='lower right')
    plt.show()