import numpy as N
import os, sys


def tscrunch(data, tx):
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
        raise RuntimeError("Can only scrunch 1D/2D arrays")


def fscrunch(data, fx):
    if fx == 1:
        return data
    if fx == 0:
        raise ValueError("Cannot fscrunch by a factor of 0")
    nr = data.shape[0]
    nc = data.shape[1]

    if nr % fx != 0:
        raise RuntimeError(
            "Cannot scrunch at factors which do not exactly divide the no. of channels")
    fsdata = N.mean(data.reshape(int(nr/fx), -1, nc), axis=1)
    return fsdata


def gauss(x, a, x0, sigma):
    return a/N.sqrt(2*N.pi*sigma**2) * N.exp(-(x-x0*1.)**2 / (2.*sigma**2))


def gauss2(x, a, x0, FWHM):
    sigma = FWHM/2. / (2*N.log(2))**0.5  # FWHM = 2 * sqrt( 2 * ln(2) ) * sigma
    return a/N.sqrt(2*N.pi*sigma**2) * N.exp(-(x-x0*1.)**2 / (2.*sigma**2))


def get_bandpass(nch):
    bp = N.loadtxt("/home/vgupta/resources/BANDPASS_normalized_320chan.cfg")
    if nch == 320:
        pass
    elif nch == 40:
        bp = tscrunch(bp, 8) / 8.
    else:
        raise ValueError(
            "NCHAN expected: [40 or 320]. Got: {0}".format(str(nch)))
    return bp*1./bp.max()


def pad_along_axis(array, target_length, axis=0):
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
    if not os.path.exists(db_d):
        try:
            print(
                "The database directory: {0} does not exist. Attempting to create it now.".format(db_d))
            os.makedirs(db_d)
        except OSError as E:
            print("Attempt to create the database directory failed because:\n{0}".format(
                E.strerror))
            print("Exiting....")
            sys.exit(1)

    if os.access(db_d, os.W_OK) and os.access(db_d, os.X_OK):
        return
    else:
        print(
            "Do not have permissions to write/create in the database directory: {0}".format(db_d))
        print("Exiting...")
        sys.exit(1)
