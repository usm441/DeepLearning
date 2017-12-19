import pickle
import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))


def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x by the prediction of
        a surrogate regression model,which was trained on the validation error of randomly sampled hyperparameter
        configurations.The original surrogate predicts the validation error after a given epoch. Since all
        hyperparameter configurations were trained for a total amount of 40 epochs, we will query the performance after
        epoch 40.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = deepcopy(x)
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
    x_norm = deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y


def random_optimize(bounds, times=10, iterations=50):
    x_plot = range(1, 51)
    D = len(bounds)
    best_y = [[1]]*times
    best_x = [None]*times
    for j in range(times):
        sum_y = 0
        for i in range(iterations):
            new_x = [random.randint(bounds[d][0], bounds[d][1]) for d in range(D)]
            new_y = objective_function(new_x)
            if new_y < best_y[j][len(best_y[j])-1]:
                best_y_temp = new_y
                best_x[j] = new_x
            best_y[j].append(best_y_temp)
    avg_y = []
    for i in range(1,51):
        sum = 0
        for j in range(10):
            sum = sum + best_y[j][i]
        avg_y.append(sum/10)

    plt.plot(x_plot,avg_y , label="abc")
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()



bounds = [[-6, 1], [32, 512], [16, 1024], [16, 1024], [16, 1024]]
random_optimize(bounds)