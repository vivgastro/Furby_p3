
import numpy as N
from Furby_p3.utility import gauss2, gauss, pad_along_axis

from scipy import signal
SUPPORTED_FREQ_STRUCTURES = [
    "flat",
    "slope",
    "smooth_envolope",
    "two_peaks",
    "three_peaks",
    "patchy"
]
class Telescope(object):

    def __init__(self, ftop, fbottom, nch, tsamp, name):
        self.name = name
        self.ftop = ftop
        self.fbottom = fbottom
        self.fcenter = (ftop + fbottom) /2
        self.nch = nch
        self.tsamp = tsamp
        self._set_bw()

    def _set_bw(self):
        self.bw = self.ftop - self.fbottom
        self.chw = self.bw / self.nch
        self.f_ch = N.linspace(self.ftop - self.chw/2,
                               self.fbottom + self.chw/2, self.nch)



class Pulse(object):

    def __init__(self, tel_obj, noise_per_channel, tfactor, tot_nsamps, scattering_index):
        self.noise_per_channel = noise_per_channel
        self.scattering_index = scattering_index
        self.tfactor = tfactor
        self.tel = tel_obj
        self.tot_nsamps = tot_nsamps
        if self.tot_nsamps is None:
            self._set_tot_nsamps(max_dm = 3000, buffer = 500)
        print("Total nsamps = ", self.tot_nsamps)
        

    def _set_tot_nsamps(self, max_dm, buffer):
        max_dm_sweep = self.dm_smear_delay(max_dm, self.tel.fcenter, self.tel.tsamp, chw=self.tel.bw)
        self.tot_nsamps = max_dm_sweep + buffer


    def get_pure_frb(self, snr, width):
        print(f"Requested params - {snr}, {width}")
        
        width_samps = width / self.tel.tsamp
        width_samps = width_samps * self.tfactor
        self._nsamps_for_gaussian = int(max([1, 5 * width_samps]))
        x = N.arange(self._nsamps_for_gaussian)

        clean_noise_rms = self.noise_per_channel 
        # Dividing snr equally among all channels for the pure case
        snr_per_channel = snr*1./N.sqrt(self.tel.nch)

        # width is supposed to be FWHM
        tmp_sigma = width_samps/2. / (2*N.log(2))**0.5
        W_tophat_gauss_samps = N.sqrt(2*N.pi) * tmp_sigma

        #TODO: Fix this calculation to use matched_filter_snr
        desired_signal = snr_per_channel * \
            clean_noise_rms * N.sqrt(W_tophat_gauss_samps)

        pure_frb_single_channel = gauss2(
            x, desired_signal, int(len(x)/2), width_samps)

        if N.abs(N.sum(pure_frb_single_channel) - desired_signal) > desired_signal/50.:
            raise RuntimeError("The generated signal is off by more than 2% of the desired value, desired_signal = {0}, generated_signal = {1}. Diff: {2}% \nThis is often the case when requested width is << t_samp. Try increasing the width to >= tsamp and see if it works.".format(
                desired_signal, N.sum(pure_frb_single_channel), ((N.sum(pure_frb_single_channel) - desired_signal)/desired_signal * 100)))

        # Copying single channel nch times as a 2D array
        pure_frb = N.array([pure_frb_single_channel] * self.tel.nch)

        assert pure_frb.shape[0] == self.tel.nch, "Could not copy 1D array {0} times".format(
            self.tel.nch)

        return pure_frb

    def create_freq_structure(self, frb, kind):
        nch = frb.shape[0]
        x = N.arange(nch)
        # kind of scintillation

        if kind == 'flat':
            f = N.ones(nch)
        if kind == 'slope':
            # Slope will be a random number between -0.5 and 0.5
            slope = N.random.uniform(-0.5*nch, 0.5*nch, 1)
            f = x * slope
        if kind == 'smooth_envelope':
            # Location of Maxima of the smooth envelope can be on any channel
            center = N.random.uniform(0, nch, 1)
            z1 = center - nch/2
            z2 = center + nch/2
            f = -1 * (x - z1) * (x - z2)
        if kind == 'two_peaks':
            z1 = 0
            z2 = N.random.uniform(0 + 1, nch/2, 1)
            z3 = N.random.uniform(nch/2, nch-1, 1)
            z4 = nch
            f = -1 * (x-z1) * (x-z2) * (x-z3) * (x-z4)
        if kind == 'three_peaks':
            z1 = 0
            z2 = N.random.uniform(0 + 1, nch/4, 1)
            z3 = N.random.uniform(1*nch/4, 2*nch/4, 1)
            z4 = N.random.uniform(2*nch/4, 3*nch/4, 1)
            z5 = N.random.uniform(3*nch/4, nch-1, 1)
            z6 = nch
            f = -1 * (x-z1) * (x-z2) * (x-z3) * (x-z4) * (x-z5) * (x-z6)
        if kind == 'patchy':
            n_blobs = int(N.abs(N.random.normal(loc=self.tel.bw / 50, scale=3)))
            n_blobs = int(N.floor(N.random.exponential(scale=5, size=1))) + 1
            f = N.zeros(nch)
            for i in range(n_blobs):
                center_of_blob = N.random.uniform(0, nch, 1)
                width_of_blob = N.abs(N.random.normal(
                    loc=20 / n_blobs, scale=10, size=1))
                # We want roughly 10 +- 5 MHz blobs. 10 MHz = 10/chw chans = 10./((P.ftop - P.bottom)/nch) chans
                NCHAN_PER_MHz = N.abs(1./((self.tel.ftop-self.tel.fbottom)/nch))
                width_of_blob = N.random.normal(
                    loc=width_of_blob*NCHAN_PER_MHz, scale=NCHAN_PER_MHz, size=1)
                # For just one blob (n_blobs=1), this does not matter because we rescale the maxima to 1 evetually. For more than one blobs, this random amp will set the relative power in different blobs. So, the power in weakest blob can be as low as 1/3rd of the strongest blob)
                amp_of_blob = N.random.uniform(1, 3, 1)
                f += gauss(x, amp_of_blob, center_of_blob, width_of_blob)

        if kind != 'flat':
            f = f - f.min()  # Bringing the minima to 0
            f = f * 1./f.max()  # Bringing the maxima to 1
            f = f - f.mean() + 1  # Shifting the mean to 1

        frb = frb * f.reshape(-1, 1)
        return frb

    def scatter(self, frb, tau0, desired_snr):
        f_ch = self.tel.f_ch
        tau0_samps = int(tau0/self.tel.tsamp) * self.tfactor
        self._nsamps_for_exponential = int( 6 * tau0_samps * ((self.tel.ftop + self.tel.fbottom)/2 / self.tel.fbottom)**self.scattering_index )

        nsamps = self._nsamps_for_exponential

        k = tau0_samps * (f_ch[0])**self.scattering_index  # proportionality constant
        taus = k / f_ch**self.scattering_index  # Calculating tau for each channel
        exps = []
        scattered_frb = []
        for i, t in enumerate(taus):
            # making the exponential with which to convolve each channel
            exps.append(N.exp(-1 * N.arange(nsamps) / t))
            # convolving each channel with the corresponding exponential ( N.convolve gives the output with length = len(frb) + len(exp) )
            result = N.convolve(frb[i], exps[-1])
            #result *= 1./result.max() * frb[i].max()
            result *= 1./result.sum() * frb[i].sum()
            scattered_frb.append(result)

        scattered_frb = N.array(scattered_frb)
        scattered_tseries = scattered_frb.sum(axis=0)
        scattered_width = scattered_tseries.sum() / N.max(scattered_tseries) / self.tfactor
        new_snr = scattered_tseries.sum() / (N.sqrt(self.tel.nch) * self.noise_per_channel) /  N.sqrt(scattered_width)
        normalizing_factor = new_snr / desired_snr
        scattered_frb /= normalizing_factor
        return scattered_frb

    def dm_smear_delay(self, dm, cfreq, tres, chw=None):
        if chw is None:
            chw = self.tel.chw
        dm_width = 8.3 * chw * dm * (cfreq / 1e3)**(-3) * 1e-6 #s
        dms_width_samples = max([1, int(dm_width / tres)])
        return dms_width_samples

    def dm_smear_channel(self, d_ch, dm, cfreq, tres):
        '''
        d_ch: 1-D array -- data of a single channel
        dm: DM to dmsmear at (pc/cc)
        cfreq:  Center freq of the channel (in MHz)
        '''
        dms_width_samps = self.dm_smear_delay(dm, cfreq, tres)
        dms_kernel = N.ones(dms_width_samps)

        dmsmeared_data = signal.fftconvolve(d_ch, dms_kernel, mode='full')
        return dmsmeared_data / dms_width_samps

    def get_FWHM(self, frb_tseries):
        maxx = N.argmax(frb_tseries)
        hp = frb_tseries[maxx] / 2.
        # Finding the half-power points
        hpp1 = (N.abs(frb_tseries[:maxx] - hp)).argmin()
        hpp2 = (N.abs(frb_tseries[maxx:] - hp)).argmin() + maxx

        FWHM = hpp2-hpp1
        assert FWHM > 0, "FWHM calculation went wrong somewhere. HPP points, maxx point and FWHM are {0} {1} {2} {3}".format(
        hpp1, hpp2, maxx, FWHM)
        return FWHM

    def get_matched_filter_snr(self, tseries):
        '''
        Computes the matched filter snr for a given time series.
        This method only works if we assume that noise is absolutely white and has no covariance.
        '''
        #First, we have to normalise the time series such that the rms of the noise is 1
        normalised_tseries = tseries / (self.tel.nch**0.5 * self.noise_per_channel)

        #Now the matched filter SNR is simply the quadrature sum of the SNRs of individual samples
        snr = N.sqrt(N.sum(normalised_tseries**2))
        return snr

    def disperse(self, frb, dm):

        nch = self.tel.nch
        tres = self.tel.tsamp / self.tfactor *1e3  #ms
        chw = self.tel.chw
        f_ch = self.tel.f_ch

        D = 4.14881e6       #ms; from Pulsar Handbook 4.1.1
        delays = D * f_ch**(-2) * dm    #Only works if freq in MHz and D in ms. Output delays in ms
        delays -= delays[int(nch/2)]
        delays_in_samples = N.rint(delays / tres).astype('int') #here we will have slight approximations due to quantization, but at 10.24 usec resolution they should be minimal

        #nsamps = delays_in_samples[-1] - delays_in_samples[0] + 2*frb.shape[1]
        #nsamps = delays_in_samples[-1]*2 + 2*frb.shape[1]

        nsamps = self.tot_nsamps * self.tfactor
        pre_shift = self._nsamps_for_gaussian
        start = int(nsamps/2) - int(pre_shift*self.tfactor)
        #end = start + frb.shape[1]

        dm_smear_max = self.dm_smear_delay(dm, chw, f_ch[-1], tres)
        dm_smear_max_nsamps = frb[1].shape[0] + dm_smear_max -1

        end = start + dm_smear_max_nsamps

        dispersed_frb = N.zeros(nch * nsamps).reshape(nch, nsamps)
        undispersed_time_series = N.zeros(dm_smear_max_nsamps)

        for i in range(nch):
            delay = delays_in_samples[i]
            if (end + delay > nsamps) or (start + delay < 0):
                raise RuntimeError("nsamps (={0}) is too small to accomodate an FRB with DM = {1}".format(self.tot_nsamples, dm))
            
            dmsmeared_channel = self.dm_smear_channel(frb[i], dm, cfreq = f_ch[i], tres=tres)

            padded_chan = pad_along_axis(dmsmeared_channel, dm_smear_max_nsamps, axis=0)
            dispersed_frb[i, start+delay : end+delay] += padded_chan
            undispersed_time_series += padded_chan

        return dispersed_frb, undispersed_time_series
