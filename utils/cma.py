import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


def clip_sol(solution, max_val):
    # Make the solution to be within the range of 0-1
    solution = solution + max_val

    # x, y in the [-0.5, 1] range
    solution[:, 0:2] = solution[:, 0:2] * 1.5 - 0.5

    # scale in the [0.2, 1.2] range
    solution[:, 2] = solution[:, 2] * 1.2 + 0.2

    return solution