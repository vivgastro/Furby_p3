import numpy as np
import matplotlib.pyplot as plt

def get_dm_mask(dm_samps, chan_cfreqs, subsample_offset = 0.5, return_edges = False):
    '''
    Returns a De-dispersion mask for a given dm_samps
    and chan_cfreqs (should work in all units for chan_cfreqs)

    Needs the highest freq channel to be the zeroth channel
    '''

    chw = np.abs(chan_cfreqs[0] - chan_cfreqs[1])
    chan_top_freqs = chan_cfreqs + chw/2.
    chan_bottom_freqs = chan_cfreqs - chw/2.

    #start_samps = dm_samps * (chan_top_freqs**-2 - chan_top_freqs[0]**-2) / (chan_cfreqs[-1]**-2 - chan_cfreqs[0]**-2) + 1e-16 + subsample_offset
    #end_samps = dm_samps * (chan_bottom_freqs**-2 - chan_top_freqs[0]**-2) / (chan_cfreqs[-1]**-2 - chan_cfreqs[0]**-2) + 1e-16 + subsample_offset
    #mid_samps = dm_samps * (chan_cfreqs**-2 - chan_top_freqs[0]**-2) / (chan_cfreqs[-1]**-2 - chan_cfreqs[0]**-2) + 1e-16 + subsample_offset

    start_samps = dm_samps * (chan_top_freqs**-2 - chan_bottom_freqs[-1]**-2) / (chan_cfreqs[-1]**-2 - chan_cfreqs[0]**-2) + 1e-16 + subsample_offset
    end_samps = dm_samps * (chan_bottom_freqs**-2 - chan_bottom_freqs[-1]**-2) / (chan_cfreqs[-1]**-2 - chan_cfreqs[0]**-2) + 1e-16 + subsample_offset
    mid_samps = dm_samps * (chan_cfreqs**-2 - chan_bottom_freqs[-1]**-2) / (chan_cfreqs[-1]**-2 - chan_cfreqs[0]**-2) + 1e-16 + subsample_offset

    end_samps -= np.floor(start_samps[0])
    mid_samps -= np.floor(start_samps[0])
    start_samps -= np.floor(start_samps[0])
    full_samps = np.floor(end_samps) - np.ceil(start_samps)
    max_samps = np.ceil(end_samps) - np.floor(start_samps)
    total_samps = end_samps - start_samps

    start_fractions = start_samps % 1
    end_fractions = end_samps % 1

    start_ints = np.floor(start_samps)
    end_ints = np.ceil(end_samps)
    ms = max_samps>1
    end_ints[ms] = np.where(1./ np.sqrt(max_samps[ms]) > (1 - end_fractions[ms] / total_samps[ms]) / np.sqrt(max_samps[ms] - np.ceil(end_fractions[ms]) ), np.ceil(end_samps[ms]), np.floor(end_samps[ms]))
    start_ints[ms] = np.where(1. / np.sqrt(max_samps[ms]) > (1 - (1 - start_fractions[ms]) / total_samps[ms]) / np.sqrt(max_samps[ms] - np.ceil(1-start_fractions[ms])), np.floor(start_samps[ms]), np.ceil(start_samps[ms])  )

    mask = np.zeros((chan_cfreqs.size, int(np.ceil(end_samps[-1]))))

    for ii in range(chan_cfreqs.size):
        mask[ii, int(start_ints[ii]):int(end_ints[ii])] = 1
    if return_edges:
        return mask, start_samps, end_samps, mid_samps
    else:
        return mask


def bfdmt(block, Ndm, Sdm, Ddm, Nt, St, chan_freqs):
    '''
    Does a brute force DMT-transform of the block of data for
    Ndm DM trials starting at Sdm DM with Ddm DM steps,
    and for Nt samples starting at St

    block: np.ndarray
        A block of data of the shape (freq, time) to run
        FDMT on
    Ndm: int
        Number of DM trials to run
    Sdm: int
        Starting DM trial in samples
    Ddm: int
        DM step in samples
    Nt: int
        No of time samples to run FDMT on
    St: int
        Starting sample to run the FDMT on
    chan_freqs: np.ndarray or list
        A list or numpy array containing a list of chan center freqs
    '''

    #HERE I AM MAKING THE ASSUMPTION THAT THE DM (in samples) IS 
    #CALCULATED USING THE CENTER FREQUENCIES OF THE HIGHEST AND 
    #THE LOWEST CHANNEL, AND THE FIRST CHANNEL HAS THE HIGHEST 
    #FREQUENCY

    dm_trials = np.arange(Ndm) * Ddm + Sdm
    samp_trials = np.arange(Nt) + St
    assert St - (Sdm + Ndm) > 0, "Not enough samples at the start to\
        dedipserse out to the maximum DM trial"
    assert (St + Nt) < block.shape[1], "Not enough samples at the end to\
        accomodate all the requested time samples trials"

    b = np.abs(block).sum(axis=0)
    chan_freqs = chan_freqs
    time_serieses = []
    for idm in dm_trials:
        dm_mask = get_dm_mask(idm, chan_freqs)[::-1]
        time_series = []
        for isamp in samp_trials:
            #print(idm, dm_mask.shape, isamp)
            frb_ex = b[:, isamp - dm_mask.shape[1]+1: isamp+1]
            time_series.append( np.sum(b[:, isamp - dm_mask.shape[1]+1: isamp+1]  *   dm_mask ) / np.sum(b) )

        time_serieses.append(time_series)
    return np.array(time_serieses)


def overplot_block_mask(block, idm, chan_freqs, subsample_offset = 0.5):
    mask = get_dm_mask(idm, chan_freqs, subsample_offset)[::-1]
    iblock = np.abs(block).sum(axis=0)
    peak_loc = np.argmax(iblock[0])
    frb_ex = iblock[:, peak_loc - mask.shape[1]+1:peak_loc+1]
    plt.figure()
    plt.imshow(frb_ex, aspect='auto', interpolation='None', alpha=0.3)
    plt.figure()
    plt.imshow(mask, aspect='auto', interpolation='None', alpha=0.3)
    plt.show()
