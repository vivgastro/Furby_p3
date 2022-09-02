import numpy as np
ssp = 0.5 

def cff(f1, f2, fmin, fmax):

    return (f1**-2 - f2**-2) / (fmin**-2 - fmax**-2)


def FDMT_initialize(din, f_min, f_max, maxDT):
    '''
    Initialises the FDMT iterations
    '''
    [n_f, n_t] = din.shape
    df = ( f_max - f_min ) / n_f

    max_dm_init = int(np.ceil(cff(f_min, f_min + df, f_min, f_max) * maxDT))

    dout = np.zeros((nf, max_dm_init, n_t))

    dout[:, 0, :] = din
    
    freqs_l = np.arange(f_min, f_max, df)
    freqs_u = freqs_l + df
    all_dms = np.arange(maxDT+1)

    subchannel_delays = np.empty((n_f, all_dms))


    for i_dm in range(1, max_dm_init+1, 1):
        #subchannel_delays = cff(freqs_l, freqs_u, f_min, f_max) * i_dm
        #start_delays = cff(freqs_l, f_min, f_min, f_max) * i_dm
        #end_delays = cff(freqs_u, f_min, f_min, f_max)

        #samp_starts = int(np.floor(start_delays + ssp))
        #samp_ends = int(np.ceil(end_delays + ssp))
        
        dout[:, i_dm, i_dm:] = din[:, -i_dm] + dout[:, i_dm -1, i_dm:]

    return dout

def FDMT_iteration(prev_dout, max_dm, n_f, f_min, f_max, i_iter):
    '''
    Does one FDMT iteration
    i_iter needs to start from 1 for 1st iteration. 0th iteration signifies the initialization step
    n_f is the total no of channels i.e. 256 (stays fixed for every fx call)
    '''
    [nf_in, n_dm_in, n_t] = prev_dout.shape
    nf_out = nf_in / 2

    df_in = (f_max - f_min) / n_f * 2**(i_iter-1)
    df_out = df_in * 2

    max_dm_iter = int(np.ceil(cff(f_min, f_min + df_out, f_min, f_max)))
    n_dm_out = max_dm_iter
    
    dout = np.zeros((df_out, n_dm_out, n_t ))

    dout[:, 0, :] = din.reshape(nf_in / 2, 2, 0, n_t).sum(axis=1)

    freqs_l_in = np.arange(f_min, f_max, df_in)
    freqs_u_in = freqs_l_in + df_in
    
    freqs_l_out = freqs_l_in[::2]
    freqs_u_out = freqs_u_in[::2]

    cff_start_delays = cff(freqs_l_in, f_min, f_min, f_max)     #-ve
    cff_end_delays = cff(freqs_u_in, f_min, f_min, f_max)       #-ve

    for i_dm in range(max_dm_iter):
        start_delays_in = cff_start_delays * i_dm + ssp         #will be a -ve number
        end_delays_in = cff_end_delays * i_dm + ssp           #will be a -ve number

        start_delays_in -= int(np.min(end_delays_in))           #made it positive
        end_delays_in -= int(np.min(end_delays_in))             #made it positive

        for i_outchan in range(nf_out):
            start_delay_out = start_delays_in[i_outchan]        #will be a positive number
            mid_delay_out = start_delays_in[i_outchan+1]        #will be a positive number
            end_delay_out = end_delays_in[i_outchan+1]          #will be a positive number

            start_delay_out_samp = int(np.round(start_delay_out))   #will be a positive number
            mid_delay_out_samp = int(np.round(mid_delay_out))       #will be a positive number

            dm_lower_band = mid_delay_out - start_delay_out         #Will be a -ve number
            dm_upper_band = end_delay_out - mid_delay_out           #Will be a -ve number

            dm_lower_band_samps = int(np.round(np.abs(dm_lower_band)))      #+ve
            dm_upper_band_samps = int(np.round(np.abs(dm_upper_band)))      #+ve

            tstart_lower_band = start_delay_out_samp
            tstart_upper_band = mid_delay_out_samp
            tstart_diff = tstart_lower_band - tstart_upper_band


            dout[i_outchan, i_dm, start_delay_out_samp : n_t+1] = din[i_outchan*2, dm_lower_band_samps, tstart_lower_band : n_t + 1] + din[i_outchan*2 +1, dm_upper_band_samps, tstart_upper_band : n_t + 1 - tstart_diff]


    return dout

def FDMT(din, f_min, f_max, max_dm):
    '''
    Performs FDMT on din

    Input:
    din: np.ndarray
        A 2-d numpy array with shape (freq, time)
        The first channel should have the lowest frequency
    
    f_min: float
        Freq of the lower edge of the band (Units don't matter)
    f_max: float
        Freq of the upper edge of the band (Units don't matter)
    max_dm: int
        Maximum dispersion delay to correct for (in samples)
    
    Returns:
    dout: numpy.ndarray
        The FDMT transform of the din with shape (1, maxDT, time)
    '''

    [nf, nt] = din.shape
    print(din.shape, nf, nt)
    n_iter = nf//2

    init_din = FDMT_initialize(din, f_min, f_max, max_dm)
    current_dout = init_din
    for i_iter in range(n_iter+1):
        cuurent_dout = FDMT_iteration(current_dout, max_dm, nf, f_min, f_max, i_iter)

