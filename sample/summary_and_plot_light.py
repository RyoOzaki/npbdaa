#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import trange, tqdm
from sklearn.metrics import adjusted_rand_score

#%%
def get_names():
    return np.loadtxt("files.txt", dtype=str)

def get_letter_labels(names):
    return _get_labels(names, "lab")

def get_word_labels(names):
    return _get_labels(names, "lab2")

def _get_labels(names, ext):
    return [np.loadtxt("LABEL/" + name + "." + ext) for name in names]


def get_datas_and_length(names):
    datas = [np.loadtxt("DATA/" + name + ".txt") for name in names]
    length = [len(d) for d in datas]
    return datas, length

def get_results_of_word(names, length):
    return _joblib_get_results(names, length, "s")

def get_results_of_letter(names, length):
    return _joblib_get_results(names, length, "l")

def get_results_of_duration(names, length):
    return _joblib_get_results(names, length, "d")

def _get_results(names, lengths, c):
    return [np.loadtxt("results/" + name + "_" + c + ".txt").reshape((-1, l)) for name, l in zip(names, lengths)]

def _joblib_get_results(names, lengths, c):
    from joblib import Parallel, delayed
    def _component(name, length, c):
        return np.loadtxt("results/" + name + "_" + c + ".txt").reshape((-1, length))
    return Parallel(n_jobs=-1)([delayed(_component)(n, l, c) for n, l in zip(names, lengths)])

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

if not os.path.exists("summary_files"):
    os.mkdir("summary_files")

#%%
print("Loading results....")
names = get_names()
datas, length = get_datas_and_length(names)
l_labels = get_letter_labels(names)
w_labels = get_word_labels(names)
concat_l_l = np.concatenate(l_labels, axis=0)
concat_w_l = np.concatenate(w_labels, axis=0)

l_results = get_results_of_letter(names, length)
w_results = get_results_of_word(names, length)
d_results = get_results_of_duration(names, length)

concat_l_r = np.concatenate(l_results, axis=1)
concat_w_r = np.concatenate(w_results, axis=1)

log_likelihood = np.loadtxt("summary_files/log_likelihood.txt")
resample_times = np.loadtxt("summary_files/resample_times.txt")
print("Done!")

L = 10
S = 10
T = l_results[0].shape[0]

#%%

letter_ARI = np.zeros(T)
word_ARI = np.zeros(T)

#%% calculate ARI
print("Calculating ARI...")
for t in trange(T):
    letter_ARI[t] = adjusted_rand_score(concat_l_l, concat_l_r[t])
    word_ARI[t] = adjusted_rand_score(concat_w_l, concat_w_r[t])
print("Done!")

#%% plot ARIs.
plt.clf()
plt.title("Letter ARI")
plt.plot(range(T), letter_ARI, ".-")
plt.savefig("figures/Letter_ARI.png")

#%%
plt.clf()
plt.title("Word ARI")
plt.plot(range(T), word_ARI, ".-")
plt.savefig("figures/Word_ARI.png")

#%%
plt.clf()
plt.title("Log likelihood")
plt.plot(range(T+1), log_likelihood, ".-")
plt.savefig("figures/Log_likelihood.png")

#%%
plt.clf()
plt.title("Resample times")
plt.plot(range(T), resample_times, ".-")
plt.savefig("figures/Resample_times.png")

#%%
np.savetxt("summary_files/Letter_ARI.txt", letter_ARI)
np.savetxt("summary_files/Word_ARI.txt", word_ARI)
with open("summary_files/Sum_of_resample_times.txt", "w") as f:
    f.write(str(np.sum(resample_times)))
