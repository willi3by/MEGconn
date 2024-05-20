import numpy as np
import cupy as cp
import multiprocessing as mp

def compute_wpli(freq_data, l2norm=False):
    n_trials, n_channels, n_freqs = freq_data.shape
    outsum = np.zeros((n_channels, n_channels, n_freqs), dtype=np.complex128)
    outsumW = np.zeros((n_channels, n_channels, n_freqs), dtype=np.complex128)
    # Reshape test_fft to prepare for broadcasting
    # New shape: (n, m, 1, p)
    freq_trials = freq_data.transpose(0, 2, 1).reshape(n_trials, n_freqs, n_channels, 1)

    # Compute the cross-spectral density matrices in a vectorized manner
    csdimag = np.imag(freq_trials @ freq_trials.conj().transpose(0, 1, 3, 2))  # Shape (n, p, m, m)

    # Sum over all trials (axis 0)
    outsum = np.sum(csdimag, axis=0)
    outsumW = np.sum(np.abs(csdimag), axis=0)

    # Compute wPLI
    wpli = np.divide(outsum, outsumW, where=outsumW != 0)

    if l2norm:
        wpli = np.linalg.norm(wpli, axis=0)
    
    return wpli

def compute_wpli_surrogates(freq_data, l2norm=False):

    n_surrogates, n_trials, n_channels, n_freqs = freq_data.shape

    with mp.Pool(processes=mp.cpu_count() - 2) as p:
        results = p.map(compute_wpli, [freq_data[i] for i in range(n_surrogates)])
    
    wpli = np.stack(results)
    
    if l2norm:
        wpli = np.linalg.norm(wpli, axis = 1)
    
    return wpli
    


