
def make_psrdada_header_string(params):
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


def make_psrdada_header(telescope, args, ID, furby_name, matched_filter_snr, FWHM, top_hat_width, dm, tau0, spectrum_type):

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
            "NSAMPS": args.tot_nsamps,
            "UTC_START": "2022-01-01-00:00:00",
            "STATE": "Intensity",
            "OBS_OFFSET": 0,
            "ORDER": args.order,
            "FTOP": telescope.ftop,
            "FBOTTOM": telescope.fbottom,
            "INSTRUMENT": "FAKE",
            "SNR": matched_filter_snr,
            "FWHM": FWHM * 1e3,  # ms
            "WIDTH": top_hat_width * 1e3,  # ms
            "DM": dm,
            "TAU0": tau0 * 1e3,  # ms
            "KIND": spectrum_type,
        }

        hdr_string = make_psrdada_header_string(header_params)
        return hdr_string