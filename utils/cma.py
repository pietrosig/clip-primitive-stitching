import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


def clip_sol(solution, max_val):
    solution = solution + max_val

    # Scale
    solution[:, 2] = np.clip(solution[:, 2], 0.01, 1)

    return solution