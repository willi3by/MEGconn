from scipy.io import loadmat
import numpy as np

def load_mat_trials_data(file_path):
    ds = loadmat(file_path)
    ds_trials = ds['vs_verb'][0][0][2].T
    trials = np.zeros((ds_trials.shape[0], ds_trials[0][0].shape[0], ds_trials[0][0].shape[1]))
    for i in range(trials.shape[0]):
        trials[i] = ds_trials[i][0]
    
    return trials

def demean_trials(trials):
    mean_per_trial = np.mean(trials, axis=(1, 2), keepdims=True)
    return trials - mean_per_trial