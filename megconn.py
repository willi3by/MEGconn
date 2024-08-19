from conn_fns import *
from freq_fns import *
from surrogate_fns import *
from utils import load_mat_trials_data

def get_wpli_z(filename, mat_field='vs_verb', tapsmofrq=2, fsample=1200, freq_range=[1,100], 
               n_surr=100, n_blocks=10, max_iterations = 200, tolerance=4, 
               l2norm=True, use_gpu=True):
    trials = load_mat_trials_data(filename, mat_field=mat_field)
    freq_data, freqs = mtmfft(trials, tapsmofrq=tapsmofrq, fsample=fsample, freq_range=freq_range)
    wpli = compute_wpli(freq_data, l2norm=l2norm)
    n_surr_per_iter = n_surr//n_blocks
    all_surr_wpli = []
    for i in range(n_blocks):
        surr = refined_AAFT_surrogates(trials, max_iterations = max_iterations, n_surr = n_surr_per_iter, tolerance = tolerance, use_gpu=use_gpu)
        freq_surr, freqs = mtmfft_surrogates(surr, tapsmofrq = 2, fsample = 1200, freq_range = [1,100], use_gpu=use_gpu)
        del surr
        wpli_surr = compute_wpli_surrogates(freq_surr.get(), l2norm=l2norm)
        del freq_surr
        all_surr_wpli.append(wpli_surr)
        del wpli_surr

    all_surr_wpli_arr = np.vstack(all_surr_wpli)
    surr_mean = all_surr_wpli_arr.mean(axis=0)
    surr_std = all_surr_wpli_arr.std(axis=0)
    wpli_z = (wpli - surr_mean)/surr_std
    wpli_z[np.isnan(wpli_z)] = 0
    
    return wpli_z