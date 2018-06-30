#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

#%%
def get_names():
    return np.loadtxt("files.txt", dtype=str)

def get_labels(names):
    letter_labels = [np.loadtxt("LABEL/" + name + ".lab") for name in names]
    word_labels = [np.loadtxt("LABEL/" + name + ".lab2") for name in names]
    return letter_labels, word_labels

def get_datas_and_length(names):
    datas = [np.loadtxt("DATA/" + name + ".txt") for name in names]
    length = [len(d) for d in datas]
    return datas, length

def get_results(names, length):
    letter_results = [np.loadtxt("results/" + name + "_l.txt").reshape((-1, l)) for name, l in zip(names, length)]
    word_results = [np.loadtxt("results/" + name + "_s.txt").reshape((-1, l)) for name, l in zip(names, length)]
    dur_results = [np.loadtxt("results/" + name + "_d.txt").reshape((-1, l)) for name, l in zip(names, length)]
    return letter_results, word_results, dur_results

def save_results(names, letter, word, dur):
    for idx, name in enumerate(names):
        np.loadtxt("results/" + name + "_l.txt", letter[idx])
        np.loadtxt("results/" + name + "_s.txt", word[idx])
        np.loadtxt("results/" + name + "_d.txt", dur[idx])

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

#%%
names = get_names()
datas, length = get_datas_and_length(names)
l_labels, w_labels = get_labels(names)
concat_l_l = np.concatenate(l_labels, axis=0)
concat_w_l = np.concatenate(w_labels, axis=0)

l_results, w_results, d_results = get_results(names, length)
concat_l_r = np.concatenate(l_results, axis=1)
concat_w_r = np.concatenate(w_results, axis=1)

L = 10
S = 10
T = l_results[0].shape[0]

#%%

letter_ARI = np.zeros(T)
word_ARI = np.zeros(T)

#%% calculate ARI
for t in range(T):
    letter_ARI[t] = adjusted_rand_score(concat_l_l, concat_l_r[t])
    word_ARI[t] = adjusted_rand_score(concat_w_l, concat_w_r[t])
log_likelihood = np.loadtxt("results/log_likelihood.txt")

#%% plot ARIs.
plt.title("Letter ARI")
plt.plot(range(T), letter_ARI, ".-")
plt.savefig("figures/Letter_ARI.png")

#%%
plt.title("Word ARI")
plt.plot(range(T), word_ARI, ".-")
plt.savefig("figures/Word_ARI.png")

#%%
plt.title("Log likelihood")
plt.plot(range(T), log_likelihood, ".-")
plt.savefig("figures/Log_likelihood.png")

#%%
np.savetxt("Letter_ARI.txt", letter_ARI)
np.savetxt("Word_ARI.txt", word_ARI)
