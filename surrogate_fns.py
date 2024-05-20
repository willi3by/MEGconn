import numpy as np

#Original code assumes 2d array with (channels, time) dimensions
#New data will be 3d array with (trials, channels, time) dimensions

def correlated_noise_surrogates(original_data, use_gpu=False):
    if use_gpu:
        import cupy as cp
        np = cp
    
        #  Get shapes
    (n_surr, n_trials, n_channels, n_time) = original_data.shape
    surrogates = np.fft.rfft(original_data, axis=3)
    len_phase = surrogates.shape[3]
    #  Generate random phases uniformly distributed in the
    #  interval [0, 2*Pi]
    phases = np.random.uniform(low=0, high=2 * np.pi, size=(n_surr, n_trials, n_channels, len_phase))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates *= np.exp(1j * phases)

    #  Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors.
    return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                        axis=3)))

def AAFT_surrogates(original_data, n_surr=10, use_gpu=False):
    if use_gpu:
        import cupy as cp
        np = cp
    n_surr, n_trials, n_channels, n_time = original_data.shape
    #  Create sorted Gaussian reference series
    gaussian = np.random.randn(n_surr, n_trials, n_channels, n_time)
    gaussian.sort(axis=3)

    #  Rescale data to Gaussian distribution
    ranks = original_data.argsort(axis=3).argsort(axis=3)
    rescaled_data = np.zeros(original_data.shape)

    rescaled_data = gaussian[np.arange(n_surr)[:, None, None, None], 
                             np.arange(n_trials)[:, None, None], 
                             np.arange(n_channels)[:, None], 
                             ranks]
    
    #  Phase randomize rescaled data
    phase_randomized_data = \
        correlated_noise_surrogates(rescaled_data, use_gpu=use_gpu)

    #  Rescale back to amplitude distribution of original data
    sorted_original = original_data.copy()
    sorted_original.sort(axis=3)

    ranks = phase_randomized_data.argsort(axis=3).argsort(axis=3)

    rescaled_data = sorted_original[np.arange(n_surr)[:, None, None, None], 
                                    np.arange(n_trials)[:, None, None], 
                                    np.arange(n_channels)[:, None], 
                                    ranks]

    return rescaled_data, ranks

def refined_AAFT_surrogates(original_data, max_iterations=200, n_surr=10, tolerance=4, use_gpu=False):
    if use_gpu:
        import cupy as cp
        np = cp
        original_data = cp.array(original_data)
    extended_data = np.repeat(original_data[None,...], n_surr, axis=0)
    #  Get size of dimensions
    n_surr, n_trials, n_channels, n_time = extended_data.shape

    fourier_transform = np.fft.rfft(extended_data, axis=3)

    #  Get Fourier amplitudes
    original_fourier_amps = np.abs(fourier_transform)

    #  Get sorted copy of original data
    sorted_original = extended_data.copy()
    sorted_original.sort(axis=3)

    #  Get starting point / initial conditions for R surrogates
    # (see [Schreiber2000]_)
    R, indold = AAFT_surrogates(extended_data, use_gpu=use_gpu)
    counter = 0
    convergence = False

    while counter <= max_iterations and convergence == False:
        #  Get Fourier phases of R surrogate
        r_fft = np.fft.rfft(R, axis=3)
        r_phases = r_fft / np.abs(r_fft)

        #  Transform back, replacing the actual amplitudes by the desired
        #  ones, but keeping the phases exp(iψ(i)
        s = np.fft.irfft(original_fourier_amps * r_phases, n=n_time,
                            axis=3)

        #  Rescale to desired amplitude distribution
        T_ranks = s.argsort(axis=3)
        indnew = T_ranks.argsort(axis=3)

        R = sorted_original[np.arange(n_surr)[:, None, None, None], 
                            np.arange(n_trials)[:, None, None], 
                            np.arange(n_channels)[:, None], 
                            indnew]
        
        if np.allclose(indold, indnew, atol=tolerance):
            convergence = True
            print('Convergence reached at iteration: ', counter)
        else:    
            indold = indnew
            counter += 1
        

    return R