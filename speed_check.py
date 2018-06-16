import numpy as np
from scipy.misc import logsumexp
import time

T = 100
TT = 10
a = np.linspace(0, 0.99999, 10000)

times = [0] * T


for t in range(T):
    st = time.time()
    for tt in range(TT):
        r = np.logaddexp.reduce(a)
    times[t] = time.time() - st

print(r)
print(sum(times))


for t in range(T):
    st = time.time()
    for tt in range(TT):
        r = logsumexp(a)
    times[t] = time.time() - st

print(r)
print(sum(times))
