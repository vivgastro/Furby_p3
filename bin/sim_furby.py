from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from Furby_p3.Signal import Telescope, Pulse, SUPPORTED_FREQ_STRUCTURES
from Furby_p3.dada_helpers import make_psrdada_header, make_psrdada_header_string
from Furby_p3.utility import check_for_permissions, tscrunch
import os
import time
import glob
import yaml
import argparse


def start_logging(ctl, db_d, args):
    '''
    Creates the furbies.cat catalog file. Designed to log the name and
    other properties of simulated FRBs in run-time.
    
    Opens/creates the `furbies.cat` file in the database directory.
    Writes a list of all pre-existing furbies in the database directory
    at the start of the catalog.
    Writes the header for the new furby entries.
    
    Returns the file handler `logger` of the catalog file.
    '''
    if os.path.exists(ctl):
        logger = open(ctl, 'a')
    else:
        logger = open(ctl, 'a')
        logger.write(
            "#This is the furby catalogue for {0} directory\n".format(db_d))
        logger.write("#Created on : {0}\n\n".format(time.asctime()))
        existing_furbies = glob.glob(db_d+"furby_*")
        if len(existing_furbies) > 0:
            logger.write(
                "#The following furbies were found to be present in the directory\
                 before the creation of this catalogue:\n")
            for furby in existing_furbies:
                logger.write("#"+furby+"\n")
        logger.write("\n")
        logger.write(
            "#The arguments that were passed to create this catalog are:{0}".format(args))
        logger.write("#FURBY_ID\tDM(pc/cc)\tFWHM(ms)\tTAU0(ms)\tSNR\n")
    return logger


def read_params_file(pfile):
    '''
    Reads the telescope parameter yaml file
    '''
    with open(pfile) as f:
        params = yaml.safe_load(f)

    return params

def parse_spectrum_argument(spectrum):
    '''
    Parses the spectrum argument
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
            spectrum_type = np.random.randint(
                0, len(SUPPORTED_FREQ_STRUCTURES, 1)[0])
        elif spectrum.lower() in SUPPORTED_FREQ_STRUCTURES:
            spectrum_type = spectrum.lower()
        else:
            raise ValueError("Unknown spectrum requested: {0}",format(spectrum))
    else:
        raise ValueError(
            "The requested spectrum type: {0} is unknown".format(spectrum))
    return spectrum_type

    
def parse_cmd_line_params(args):

    # Parsing the cmd-line input parameters
    if args.Num < 1:
        raise ValueError(
            "Invalid number of furbies requested: {0}".format(args.Num))

    if args.order not in ["TF", "FT"]:
        raise ValueError(
            "Invalid value specified for the 'order' option: {0}".format(args.order))

    if args.scattering_index > 100 or args.scattering_index < -100:
        raise ValueError(
            "Scattering Index has an insane value:{0}".format(args.scattering_index))

    if args.noise_per_channel < 0:
        raise ValueError("Noise_per_channel is invalid:{0}".format(
            args.noise_per_channel))

    if args.tfactor < 0:
        raise ValueError(
            "Invalid value of tfactor specified: {0}".format(args.tfactor))
    elif args.tfactor > 1000:
        raise ValueError("tfactor is too high: {0}".format(args.tfactor))

    if args.tot_nsamps is not None and args.tot_nsamps < 1:
        raise ValueError(
            "Invalid value of tot_nsamps specified: {0}".format(args.tot_nsamps))

    check_for_permissions(args.D)

    if isinstance(args.snr, float):
        snrs = args.snr * np.ones(args.Num)
    elif isinstance(args.snr, list) and len(args.snr) == 1:
        snrs = args.snr[0] * np.ones(args.Num)
    elif isinstance(args.snr, list) and len(args.snr) == 2:
        snrs = np.random.uniform(args.snr[0], args.snr[1], args.Num)
    else:
        raise ValueError("Invalid input for SNR:{0}".format(args.snr))

    if isinstance(args.width, float):
        widths = args.width * 1e-3 * np.ones(args.Num)
    elif isinstance(args.width, list) and len(args.width) == 1:
        widths = args.width[0]*1e-3 * np.ones(args.Num)
    elif isinstance(args.width, list) and len(args.width) == 2:
        widths = np.random.uniform(
            args.width[0]*1e-3, args.width[1]*1e-3, args.Num)
    else:
        raise IOError("Invalid input for Width: {0}".format(args.width))

    if isinstance(args.dm, float):
        dms = args.dm * np.ones(args.Num)
    elif isinstance(args.dm, list) and len(args.dm) == 1:
        dms = args.dm[0] * np.ones(args.Num)
    elif isinstance(args.dm, list) and len(args.dm) == 2:
        dms = np.random.uniform(args.dm[0], args.dm[1], args.Num)
    else:
        raise ValueError("Invalid input for dm: {0}".format(args.dm))

    if isinstance(args.tau, float):
        tau0s = args.tau * 1e-3 * np.ones(args.Num)
    elif isinstance(args.tau, list) and len(args.tau) == 1:
        tau0s = args.tau[0] * 1e-3 *  np.ones(args.Num)
    elif isinstance(args.tau, list) and len(args.tau) == 2:
        tau0s = 10**np.random.uniform(
            np.log10(args.tau[0]*1e-3), np.log10(args.tau[1]*1e-3), args.Num)
    else:
        raise ValueError("Invalid input for tau - {0}".format(args.tau))

    return snrs, dms, widths, tau0s


def get_furby_ID(db_d):
    max_ID = 100000
    while(True):
        ID = np.random.randint(0, max_ID, 1)[0]
        ID = str(ID).zfill(int(np.log10(max_ID)))
        furby_name = "furby_"+ID

        if os.path.exists(os.path.join(args.D, furby_name)):
            continue
        else:
            break

    return ID, furby_name

def get_furby(dm, snr, width, tau0, telescope_params, spectrum_type, 
            noise_per_channel=1, tfactor=10, tot_nsamps=None, 
            scattering_index = 4.4):

    spectrum_type = parse_spectrum_argument(spectrum_type)

    telescope = Telescope(telescope_params['ftop'],
                          telescope_params['fbottom'],
                          telescope_params['nch'],
                          telescope_params['tsamp'],
                          telescope_params['name'])
    pulse = Pulse(telescope, noise_per_channel, tfactor, tot_nsamps, scattering_index)

    frb_hires = pulse.get_pure_frb(snr, width)
    frb_hires = pulse.create_freq_structure(frb_hires, spectrum_type)
    frb_hires = pulse.scatter(frb_hires, tau0, snr)
    frb_hires, undispersed_time_series_hires = pulse.disperse(
        frb_hires, dm)

    frb = tscrunch(frb_hires, tfactor)

    undispersed_time_series = tscrunch(
        undispersed_time_series_hires, tfactor)

    final_frb = frb.astype('float32')

    return final_frb, undispersed_time_series


def main(args):
    P = read_params_file(args.params)
    snrs, dms, widths, tau0s = parse_cmd_line_params(args)

    catalog_file = os.path.join(args.D, "furbies.cat")
    logger = start_logging(catalog_file, args.D, args)

    for num in range(args.Num):
        dm = dms[num]
        snr = snrs[num]
        width = widths[num]
        tau0 = tau0s[num]

        ID, furby_name = get_furby_ID(args.D)

        telescope = Telescope(P["ftop"], P["fbottom"], P["nch"], P["tsamp"], P["name"])
        pulse = Pulse(telescope, args.noise_per_channel,
                      args.tfactor, args.tot_nsamps, args.scattering_index)

        final_frb, undispersed_time_series = get_furby(dm, snr, width, tau0, P, args.spectrum,
            noise_per_channel=args.noise_per_channel, tfactor = args.tfactor, tot_nsamps=args.tot_nsamps,
            scattering_index=args.scattering_index)

        matched_filter_snr = pulse.get_matched_filter_snr(
            undispersed_time_series)
        top_hat_width = np.sum(undispersed_time_series) / \
            np.max(undispersed_time_series) * telescope.tsamp
        FWHM = pulse.get_FWHM(undispersed_time_series)
        FWHM = FWHM * telescope.tsamp

        hdr_string = make_psrdada_header(
        telescope, args, ID, furby_name, 
        matched_filter_snr, FWHM, top_hat_width, dm, tau0)

        outfile = open(os.path.join(args.D, furby_name), 'wb')
        outfile.write(hdr_string.encode('ascii'))

        if args.order == "TF":
            O = 'F'  # column-major
        if args.order == "FT":
            O = 'c'  # row-major

        final_frb.flatten(order=O).tofile(outfile)
        outfile.close()
        logger.write(
            ID+"\t" +
            str(dm)+"\t" +
            str(FWHM*1e3) + "\t" +
            str(tau0*1e3) + "\t" +
            str(snr) + "\n"
        )
    logger.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser()

    a.add_argument("Num", type=int, help="Number of FRBs to simulate")
    a.add_argument("params", type=str, help="Path to the params file")
    a.add_argument("-dm", nargs='+', type=float, help="DM value or DM range endpoints\
     in pc/cm3 (e.g. 1000 or 100, 2000)", default=None)
    a.add_argument("-snr", nargs='+', type=float, help="SNR value or SNR range endpoints\
         (e.g. 20 or 10, 50)", default=None)
    a.add_argument("-width", nargs='+', type=float, help="Width value or Width range endpoints\
     in ms (e.g. 2 or 0.1, 10]", default=None)
    a.add_argument("-tau", nargs='+', type=float, help="Tau value or range endpoints\
        in ms (e.g. 0.2 or 0.1, 1). Specify 0 if you don't want scattering.", default=None)
    a.add_argument("-spectrum", type=str, help="Type of frequency structure in the spectrum\
        to simulate. Options:[slope, smooth_envelope,\
             two_peaks, three_peaks, patchy, random]", default="patchy")
    a.add_argument("-D", type=str, help="Path to the database directory (existing or new).\
         Default=cwd", default="./")
    a.add_argument("-order", type=str, help="Order in which data has to be written - TF/FT\
         Default = TF", default="TF")
    a.add_argument("-scattering_index", type=float,
                   help="Scattering index. Def=-4.4", default=-4.4)
    a.add_argument("-tfactor", type=int, help="Oversampling factor in time to simulate\
        smooth FRBs at low widths. Def = 10", default=10)
    a.add_argument("-noise_per_channel", type=float, help="Noise per channel in the real\
        data this furby will be injected into. Def = 1.0", default=1.0)
    a.add_argument("-tot_nsamps", type=int,
                   help="Total no. of samps in the output block.Leave unspecified\
                        for auto-calculation", default=None)

    args = a.parse_args()

    main(args)
