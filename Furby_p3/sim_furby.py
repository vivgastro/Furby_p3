import numpy as np
from Furby_p3.Signal import Pulse, SUPPORTED_FREQ_STRUCTURES
from Furby_p3.Telescope import Telescope
from Furby_p3.utility import tscrunch, get_matched_filter_snr, get_boxcar_width_and_snr, get_FWHM
import os

def parse_spectrum_argument(spectrum):
    '''
    Parses the spectrum argument

    Can accept a string for any of the supported spectrum types, or a 
    1-D array containing the gain values for each channel, or a path 
    to a file which contains the gain values for each channel.

    Params
    ------
    spectrum : str or numpy.ndarray
        The desired spectrum type. Supported values are [flat, slope, 
        gaussian, pathcy, random]. Alternatively, you can provide
        a 1-D numpy array or a path to the file on disk which contains
        the gain values of each channel

    Returns
    -------
    spectrum_type : str or numpy.ndarray
        The parsed spectrum_type / gain values

    Raises
    ------
    ValueError
        If the spectrum type string is not supported, or if 
        numpy.loadtxt fails on the provided file.
    '''
    if type(spectrum) == np.ndarray:
        spectrum_type = spectrum.copy()

    elif type(spectrum) == str:
        if os.path.exists(spectrum):
            try:
                spectrum_type = np.loadtxt(spectrum)
            except Exception as E:
                raise ValueError("The file provided as input for spectrum type failed to load with the following error: \n{0}".format(E))
    
        elif spectrum.lower() == "random":
            randint = np.random.randint(
                0, len(SUPPORTED_FREQ_STRUCTURES), 1)[0]
            spectrum_type = SUPPORTED_FREQ_STRUCTURES[randint]
        elif spectrum.lower() in SUPPORTED_FREQ_STRUCTURES:
            spectrum_type = spectrum.lower()
        else:
            raise ValueError("Unknown spectrum requested: {0}",format(spectrum))
    else:
        raise ValueError(
            "The requested spectrum type: {0} is unknown".format(spectrum))
    return spectrum_type


def get_furby(dm, snr, width, tau0, telescope_params, spectrum_type, 
            noise_per_sample=1, tfactor=10, tot_nsamps=None, 
            scattering_index = 4.4):
    '''
    Generates a noise-free mock FRB template based on the given params

    Calls the relevant functions and creates a mock FRB template.

    Params
    ------
    dm : float
        DM value in pc/cc
    snr : float 
        SNR value
    width : float
        Width value in seconds
    tau0 : float
        The scattering timescale value in seconds
    telescope_params : dict
        A dictionary containing the properties of the telescope for
        which the furby has to be simulated. Required properties
        include - [ftop, fbottom, nch, tsamp, name]
    spectrum_type : str or numpy.ndarray
        The spectrum type that needs to be simulated
        Can be any of the following:
        1) a string from one of the supported spectrum types
        2) a 1-D numpy array containing the gain values of channel
        3) a path to a file containing the gain values of channel
    noise_per_sample : float, optional
        rms of the noise along the time axis onto which this frb would
        be added/injected. This is used to normalise the height of the
        simulated furby such that when it is added to the real-time 
        data, it attains the desired snr
    tfactor : int, optional
        Oversampling factor. This is used to simulate the mock frb at 
        a higher resolution than the tsamp, so that even for extremely
        narrow furbies (~ 1 samp wide) the snr and width calculations
        can be accurately carried out. 10 is a reasonable number to use
    tot_nsamps : int, optional
        Total number of samples in the output furby block/template. If
        None is provided, this value is automatically computed based on
        the provided DM. Default is `None`
    scattering_index : float, optional
        The power-law index of the scattering function which will be
        used to scale the scattering timescale in each channel.
        Default is `4.4`

    Returns
    -------
    final_frb : numpy.ndarray
        A 2-D numpy array containing the time-freq data of the 
        simulated noise-free mock frb template.
    undispersed_time_series : numpy.ndarray
        A 1-D numpy array containing the time-series data of the 
        frequency averaged noise-free mock FRB after it has been
        dm-smeared. This is useful for calculating the actual
        snr and width of the simulated frb.
    '''

    spectrum_type = parse_spectrum_argument(spectrum_type)

    telescope = Telescope(telescope_params['ftop'],
                          telescope_params['fbottom'],
                          telescope_params['nch'],
                          telescope_params['tsamp'],
                          telescope_params['name'])
    pulse = Pulse(telescope, tfactor, scattering_index, tot_nsamps)

    frb_hires = pulse.get_pure_frb(width)
    frb_hires = pulse.create_freq_structure(frb_hires, spectrum_type)
    frb_hires = pulse.scatter(frb_hires, tau0)
    frb_hires, undispersed_time_series_hires = pulse.disperse(
        frb_hires, dm)

    frb = tscrunch(frb_hires, tfactor)
    undispersed_time_series = tscrunch(
        undispersed_time_series_hires, tfactor)

    top_hat_width = np.sum(undispersed_time_series_hires) / \
        np.max(undispersed_time_series) * telescope.tsamp / tfactor
    FWHM = get_FWHM(undispersed_time_series_hires) * telescope.tsamp / tfactor
    boxcar_width, boxcar_snr = get_boxcar_width_and_snr(undispersed_time_series_hires, noise_per_sample)

    noise_after_averaging_channels = noise_per_sample * np.sqrt(telescope.nch)
    mf_snr = get_matched_filter_snr(undispersed_time_series, noise_after_averaging_channels)
    normalizing_factor = snr / mf_snr

    frb *= normalizing_factor
    final_frb = frb.astype('float32')

    return final_frb, top_hat_width, FWHM, pulse.tot_nsamps

