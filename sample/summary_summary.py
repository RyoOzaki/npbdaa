#%%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
label = sys.argv[1]
dirs = sys.argv[2:]
dirs.sort()

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

if not os.path.exists("summary_files"):
    os.mkdir("summary_files")

#%%
print("Initialize variables....")
N = len(dirs)
tmp = np.loadtxt(label + "/" +  dirs[0] + "/summary_files/resample_times.txt")
T = tmp.shape[0]

resample_times  = np.empty((N, T))
log_likelihoods = np.empty((N, T+1))
word_ARIs       = np.empty((N, T))
letter_ARIs     = np.empty((N, T))
print("Done!")

#%%
print("Loading results....")
for i, path in enumerate(dirs):
    resample_times[i] = np.loadtxt(label + "/" + path + "/summary_files/resample_times.txt")
    log_likelihoods[i] = np.loadtxt(label + "/" + path + "/summary_files/log_likelihood.txt")
    word_ARIs[i] = np.loadtxt(label + "/" + path + "/summary_files/Word_ARI.txt")
    letter_ARIs[i] = np.loadtxt(label + "/" + path + "/summary_files/Letter_ARI.txt")
print("Done!")

#%%
print("Ploting...")
plt.clf()
plt.errorbar(range(T), resample_times.mean(axis=0), yerr=resample_times.std(axis=0))
plt.xlabel("Iteration")
plt.ylabel("Execution time [sec]")
plt.title("Transitions of the execution time")
plt.savefig("figures/summary_of_execution_time.png")

plt.clf()
plt.errorbar(range(T+1), log_likelihoods.mean(axis=0), yerr=log_likelihoods.std(axis=0))
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title("Transitions of the log likelihood")
plt.savefig("figures/summary_of_log_likelihood.png")

plt.clf()
plt.errorbar(range(T), word_ARIs.mean(axis=0), yerr=word_ARIs.std(axis=0), label="Word ARI")
plt.errorbar(range(T), letter_ARIs.mean(axis=0), yerr=letter_ARIs.std(axis=0), label="Letter ARI")
plt.xlabel("Iteration")
plt.ylabel("ARI")
plt.title("Transitions of the ARI")
plt.legend()
plt.savefig("figures/summary_of_ARI.png")
print("Done!")

#%%
print("Save npy files...")
np.save("summary_files/resample_times.npy", resample_times)
np.save("summary_files/log_likelihoods.npy", log_likelihoods)
np.save("summary_files/word_ARI.npy", word_ARIs)
np.save("summary_files/letter_ARI.npy", letter_ARIs)
print("Done!")
