
def format_psrdada_header_string(params):
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


def make_psrdada_header(furby_header, order, ID, furby_name, extra_params = {}):
    '''
    Creates a header string that is compatible with the psrdada format
    
    Sets the values of the Furby in a dictionary and converts it into
    a properly formatted psrdada header.
    
    Params
    ------
    furby_header : dict
        A dictionary containing the params like SNR, DM etc for the frb
    
    order : str
        A sting indicating the order in which the data has to be saved
        on disk. Options : [TF, FT]

    ID : str
        The ID associated with this furby
    
    furby_name : str
        The filename to which this furby has to be saved

    extra_params : dict
        Any extra params you may want to include in the header

    Returns
    -------
    hdr_string : str
        Properly formatted string containing the furby header in the 
        psrdada format
    '''

    dada_header_params = {
        "HDR_VERSION": 1.0,
        "HDR_SIZE": 16384,
        "ID": ID,
        "SOURCE": furby_name,
        "UTC_START": "2022-01-01-00:00:00",
        "STATE": "Intensity",
        "OBS_OFFSET": 0,
        "ORDER": order,
        "INSTRUMENT": "FAKE",
    }
    full_header = {**dada_header_params, **furby_header, **extra_params}
    hdr_string = format_psrdada_header_string(full_header)
    return hdr_string
