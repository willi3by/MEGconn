from scipy.signal.windows import dpss
import numpy as np

def dpss_filters(n_time, tapsmofrq, fsample):
    #Currently only works for single dpss taper
    taps = dpss(n_time, n_time*(tapsmofrq/fsample), Kmax=2)
    taps = taps[:-1,:]
    return taps

def mtmfft(data, tapsmofrq, fsample, freq_range):
    n_trials, n_channels, n_time = data.shape
    taps = dpss_filters(n_time, tapsmofrq, fsample)
    rep_taps = np.tile(taps, (n_trials, n_channels, 1))
    data *= rep_taps
    #Get frequencies and filter data to a certain frequency range
    freqs = np.fft.fftfreq(n_time, d=1/fsample)
    freq_idxs = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    data = np.fft.fft(data, axis=2)[:,:,freq_idxs]
    
    return data, freqs[freq_idxs]