import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

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
    # return letter_results, word_results

def _plot_discreate_sequence(true_data, title, sample_data, plotopts = {}, cmap = None, cmap2 = None):
        ax = plt.subplot2grid((10, 1), (1, 0))
        plt.sca(ax)
        ax.matshow([true_data], aspect = 'auto', cmap=cmap)
        plt.ylabel('Truth Label')
        #label matrix
        ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
        plt.suptitle(title)
        plt.sca(ax)
        if cmap2 is not None:
            cmap = cmap2
        ax.matshow(sample_data, aspect = 'auto', **plotopts, cmap=cmap)
        #write x&y label
        plt.xlabel('Frame')
        plt.ylabel('Iteration')
        plt.xticks(())

#%%
names = get_names()
datas, length = get_datas_and_length(names)
l_labels, w_labels = get_labels(names)
concat_l_l = np.concatenate(l_labels, axis=0)
concat_w_l = np.concatenate(w_labels, axis=0)

l_results, w_results, d_results = get_results(names, length)
# l_results, w_results = get_results(names, length)
concat_l_r = np.concatenate(l_results, axis=1)
concat_w_r = np.concatenate(w_results, axis=1)

L = 10
S = 10
T = l_results[0].shape[0]

#%%

letter_ARI = np.zeros(T)
word_ARI = np.zeros(T)

#%%
lcolors = ListedColormap([cm.tab20(float(i)/L) for i in range(L)])
wcolors = ListedColormap([cm.tab20(float(i)/S) for i in range(S)])

#%%
for i, name in enumerate(tqdm(names)):
    _plot_discreate_sequence(l_labels[i], name + "_l", l_results[i], cmap=lcolors)
    plt.savefig("figures/" + name + "_l.png")
    _plot_discreate_sequence(w_labels[i], name + "_s", w_results[i], cmap=wcolors)
    plt.savefig("figures/" + name + "_s.png")
    _plot_discreate_sequence(w_labels[i], name + "_d", d_results[i], cmap=wcolors, cmap2=cm.binary)
    plt.savefig("figures/" + name + "_d.png")

#%% calculate ARI
for t in range(T):
    letter_ARI[t] = adjusted_rand_score(concat_l_l, concat_l_r[t])
    word_ARI[t] = adjusted_rand_score(concat_w_l, concat_w_r[t])

#%% plot ARIs.
plt.plot(range(T), letter_ARI, ".-")

#%%
plt.plot(range(T), word_ARI, ".-")
