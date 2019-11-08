import numpy as np

def linreg(xs, ys, debug=False):
    A = np.vstack([xs, np.ones(len(xs))]).T
    res = np.linalg.lstsq(A, ys)
    
    a, b = res[0]
    residuals = res[1]

    if debug:
        print("slope = {}, y-cut = {}".format(a,b))

    return a, b, residuals

def approx(first, second):
    if np.fabs(first - second) <= 0.01*second:
        return True
    return False

