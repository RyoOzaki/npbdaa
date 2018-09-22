import numpy as np

def np2list(d):
    new_d = {}
    for key, value in d.items():
        if type(value) is np.ndarray:
            new_d[key] = value.tolist()
        elif type(value) is dict:
            new_d[key] = np2list(value)
        else:
            new_d[key] = value
    return new_d
