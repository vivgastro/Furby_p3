import numpy as N
import os


def tscrunch(data, tx):
    '''
    Averages the time-freq data along the time axis by a factor `tx`
    
    Assumes that the data is arranged in the FT order, that is freq
    on the y-axis and time on the x-axis.
    If the number of samples is not an integer multiple of `tx`, then
    last few samples are deleted and samples upto the nearest multiple
    are averaged.

    Accepts 1-D and 2-D numpy arrays.

    Params
    ------
    data : numpy.ndarray 
        1-D or 2-D numpy array containing the time-series or time-freq
        data
    tx : int
        Factor by which to average
    
    Returns
    -------
    tsdata : numpy.ndarray
        1-D or 2-D numpy array containing the averaged time-series or
        time-freq data

    Raises
    ------
    TypeError if the data is not in the desired shape or format
    
    '''
    if type(data) != N.ndarray:
        raise TypeError("Only numpy 1-D or 2-D arrays are accepted by tscrunch()")

    if tx == 1:
        return data

    if len(data.shape) == 1:
        endpoint = int(len(data) / tx) * tx
        return data[:endpoint].reshape(-1, tx).sum(axis=-1)

    if len(data.shape) == 2:
        nr = data.shape[0]
        nc = data.shape[1]

        endpoint = int(nc/tx) * tx
        tmp = data[:, :endpoint].reshape(nr, int(nc/tx), tx)
        tsdata = tmp.sum(axis=-1)

        return tsdata
    else:
        raise TypeError("Can only scrunch 1D/2D arrays")



def gauss(x, a, x0, sigma):
    '''
    Simulates a Gaussian function on the axis `x`. It is defined
    by Amplitude `a`, center `x0` and sigma `sigma`
    '''
    return a/N.sqrt(2*N.pi*sigma**2) * N.exp(-(x-x0*1.)**2 / (2.*sigma**2))


def gauss2(x, a, x0, FWHM):
    '''
    Simulates a Gaussian function on the axis `x`. It is defined by
    Amplitude `a`, center `x0`, and FWHM `FWHM`
    '''
    sigma = FWHM/2. / (2*N.log(2))**0.5  # FWHM = 2 * sqrt( 2 * ln(2) ) * sigma
    return a/N.sqrt(2*N.pi*sigma**2) * N.exp(-(x-x0*1.)**2 / (2.*sigma**2))


def pad_along_axis(array, target_length, axis=0):
    '''
    Pads an array with 0s uptill the `target_length` along a given
    `axis`

    Params
    ------
    array : numpy.ndarray
        N-D numpy array to pad
    target_length : int
        Target length along the requested axis
    axis : int
        Axis along with the array has to be padded
    
    Returns
    -------
    padded_array : numpy.ndarray
        N-D numpy array after padding

    Raises
    ------
    RuntimeError 
        If the padded array's shape does not match the desired shape
    '''
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    if pad_size % 2 != 0:
        npad[axis] = ((int(pad_size / 2) + 1), int(pad_size / 2))
    else:
        npad[axis] = (int(pad_size / 2), int(pad_size/2))

    padded_array = N.pad(array, pad_width=npad,
                         mode='constant', constant_values=0)

    if padded_array.shape[axis] != target_length:
        raise RuntimeError("Padded array shape is wrong")
    else:
        return padded_array


def check_for_permissions(db_d):
    '''
    Checks for relevant permissions to create a Furby database
    in a given directory
    
    If the directory exists, checks for write and execution permissions
    If it doesn't exists, attempts to create it.
    
    Params
    ------
    db_d : str
        Path to the database directory

    Raises
    ------
    OSError
        If desired permissions are not available
    '''
    if not os.path.exists(db_d):
        try:
            print(
                "The database directory: {0} does not exist. Attempting to create it now.".format(db_d))
            os.makedirs(db_d)
        except OSError as E:
            print("Attempt to create the database directory failed because:\n{0}".format(
                E.strerror))
            raise
    if os.access(db_d, os.W_OK) and os.access(db_d, os.X_OK):
        return
    else:
        print(
            "Do not have permissions to write/create in the database directory: {0}".format(db_d))
        print("Exiting...")
        raise OSError("Do not have permissions to write/create files in the database directory")
