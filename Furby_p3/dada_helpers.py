
def make_psrdada_header_string(params):
    '''
    Converts a dictionary into a psrdada format header string

    Params
    ------
    params : dict
        Dictionary containing header params

    Returns
    -------
    header : str
        Header string in the psrdada format
    '''
    hdr_len = params["HDR_SIZE"]
    header=""
    for i in params:
        header += i
        tabs = 3 - int(len(i)/8)
        header += "\t"*tabs
        header += str(params[i])+"\n"
    leftover_space = hdr_len - len(header)
    header += '\0' * leftover_space
    return header


def make_psrdada_header(telescope, pulse, ID, furby_name, matched_filter_snr, FWHM, top_hat_width, dm, tau0):
    '''
    Creates a header string that is compatible with the psrdada format
    
    Sets the values of the Furby in a dictionary and converts it into
    a properly formatted psrdada header.
    
    Params
    ------
    telescope : object
        An instance of Telescope() class which contains the relevant
        information about the furby's telescope
    pulse : object
        An instance of the Pulse() class which contains the relevant
        information about the simulated furby
    ID : str
        ID of the furby
    furby_name : str
        Name of the furby file
    matched_filter_snr : float
        Matched filter SNR of the furby
    FWHM : float
        FWHM of the furby (in seconds)
    top_hat_width : float
        Top hat width of the furby (in seconds)
    dm : float
        DM of the furby (in pc/cc)
    tau0 : float
        Scattering timescale tau0 of the furby (in seconds)

    Returns
    -------
    hdr_string : str
        Properly formatted string containing the furby header in the 
        psrdada format
    '''

    header_params = {
        "HDR_VERSION": 1.0,
        "HDR_SIZE": 16384,
        "TELESCOPE": telescope.name,
        "ID": ID,
        "SOURCE": furby_name,
        "FREQ": (telescope.ftop + telescope.fbottom)/2.,
        "BW":   telescope.bw,
        "NPOL": 1,
        "NBIT": 32,
        "NCHAN": telescope.nch,
        "TSAMP": telescope.tsamp * 1e-6,
        "NSAMPS": pulse.tot_nsamps,
        "UTC_START": "2022-01-01-00:00:00",
        "STATE": "Intensity",
        "OBS_OFFSET": 0,
        "ORDER": pulse.order,
        "FTOP": telescope.ftop,
        "FBOTTOM": telescope.fbottom,
        "INSTRUMENT": "FAKE",
        "SNR": matched_filter_snr,
        "FWHM": FWHM * 1e3,  # ms
        "WIDTH": top_hat_width * 1e3,  # ms
        "DM": dm,
        "TAU0": tau0 * 1e3,  # ms
    }
    hdr_string = make_psrdada_header_string(header_params)
    return hdr_string