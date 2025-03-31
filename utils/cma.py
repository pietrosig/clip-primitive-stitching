import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


def clip_sol(solution, max_val):
    # Make the solution to be within the range of 0-1
    solution = solution + max_val

    return solution