import numpy as N
from Furby_p3.utility import gauss2, gauss, pad_along_axis
from Furby_p3.Telescope import Telescope

from scipy import signal
SUPPORTED_FREQ_STRUCTURES = [
    "flat",
    "slope",
    "gaussian",
    "power_law",
    "patchy"
]

class Pulse(object):
    '''
    A class that encapsulates all the functions needed to simulate 
    a mock FRB.
    Needs to be initialised with an object of the Telescope() class.
    Simulates noise-free templates of mock FRB according to the
    requested parameters.
    '''

    def __init__(self, tel_obj, tfactor, scattering_index, tot_nsamps = None):
        '''
        Parameters
        ----------
        tel_obj : object
            An instance of the Telescope() class
        tfactor : float
            The oversampling factor in time. Allows for more accurate
            representation of the pulse with requested parameters,
            specially in those cases where the requested width is of
            the order ~1 sample. 10 is usuallt a good number. 
        scattering_index : float
            The scattering index to be used when scattering the frb.
            Scattering from a medium exhibiting Kolmogorov turbulence
            results in an index of -4.4.
        tot_nsamps : int, optional
            Total number of samples desired in the output mock frb
            template. If this option is None or left unspecified, then
            it is automatically calculated based on the requested DM in
            the disperse() function.
            '''
        self.scattering_index = scattering_index
        self.tfactor = tfactor
        self.tel = tel_obj
        self.tot_nsamps = tot_nsamps
        

    def _set_tot_nsamps(self, dm, buffer = 100):
        max_dm_sweep = self.dm_smear_delay(dm, self.tel.fcenter, self.tel.tsamp, chw=self.tel.bw)
        self.tot_nsamps = int(int( (max_dm_sweep + 2*buffer) / 100  + 1) * 100)        #Rounding up to the nearest 100 samps


    def get_pure_frb(self, width):
        '''
        Creates a simple Gaussian FRB profile with frequency and time axis
        
        Parameters
        ----------
        width : float
            Width expected (in seconds)

        Returns
        -------
        frb : numpy.ndarray
            A 2-D numpy array with freq and time axis containing the
            Gaussian FRB profile.
        '''
        width_samps = width / self.tel.tsamp
        width_samps = width_samps * self.tfactor
        self._nsamps_for_gaussian = 2*int(max([self.tfactor, 5 * width_samps]))
        x = N.arange(self._nsamps_for_gaussian)

        pure_frb_single_channel = gauss2(
            x, 1.0, int(len(x)/2), width_samps)

        # Copying single channel nch times as a 2D array
        pure_frb = N.array([pure_frb_single_channel] * self.tel.nch)

        assert pure_frb.shape[0] == self.tel.nch, "Could not copy 1D array {0} times".format(
            self.tel.nch)

        return pure_frb

    def create_freq_structure(self, frb, kind):
        '''
        Applies frequency structure to the frb template.

        This function simulates and applies various frequency
        structures to the 2D template. It can be provided a 
        frequency structure to apply in the form of a numpy array
        containing values of the gain in each frequency channel, or
        can be provided a freq structure type (as a string) using which
        this function will simulate the desired pattern (with some
        degree of randomness built-in). The supported freq structure 
        types are: [flat, gaussian, patchy, power_law, random].
        
        flat : Equal gain in each channel, no evolution with frequency
        slope : Channel gains evolve linearly with freq. The slope of
                the line is chosed randomly to be b/w -5 and +5
        power_law : Gains follow a power-law. The power-law index is chosen
                    randomly b/w -10 and +10
        gaussian : Gains evolve smoothly as a Gaussian. The center is a
                   Gaussian random variable with a mean of nch/2 and 
                   sigma of nch/2. The width of the Gaussian envelope
                   is a uniform random variable b/w 5 and nch/2
        pathcy :   N Gaussian blobs are combined to determine the gain
                   for each channel. The number of blobs, center, width
                   and relative height of each blob is randomly chosen
        random - Any one of the above types is randomly chosen.

        The simulated spectrum is normalised to have a minima of 0, 
        maxima of 1. Then the spectrum is shifted to have a mean of 1.
        This conserves the total amount of the signal after averaging
        along the frequency axis.

        Parameters
        ----------
        frb : numpy.ndarray
            2-D numpy array containing the FRB time-freq profile
        kind : str or numpy.ndarray
            Kind of the desired frequency options. The valid options
            are - [flat, slope, power_law, gaussian, patchy].
            Alternatively, you can provide a 1-D numpy array containing
            the values of the relative gain of each channel as floats.
            The array should have `nch` elements.

        Returns
        -------
        frb : numpy.ndarray
            2-D numpy array containing the FRB time-freq profile
            after it has been modulated according to the desired
            spectrum.
        
        '''
        nch = frb.shape[0]
        x = N.arange(nch)
        # kind of scintillation
        if type(kind) == N.ndarray:
            if kind.size != nch or kind.dtype != N.float64:
                raise ValueError("The spectrum array does not match the required format")
            f = kind.copy()
            if N.all(f == f[0]):
                kind = 'flat'
        if kind == 'flat':
            f = N.ones(nch)
        elif kind == 'slope':
            # Slope will be a random number between -0.5 and 0.5
            slope = N.random.uniform(-5, 5, 1)
            f = x * slope
        elif kind == 'gaussian':
            center = N.random.normal(nch/2, nch/2, 1)
            sig_gauss = N.random.uniform(5, self.tel.nch/2, 1)   
            ff = N.arange(nch)
            f = N.exp(-1*(ff - center)**2 / (2 * sig_gauss**2))
        elif kind == 'power_law':
            pli = N.random.uniform(-10, 10, 1)
            f_ch = self.tel.f_ch
            f = f_ch**pli
        elif kind == 'patchy':
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
        else:
            raise ValueError("Invalid value of 'kind': {0}".format(kind))

        if kind != 'flat':
            f = f - f.min()  # Bringing the minima to 0
            f = f * 1./f.max()  # Bringing the maxima to 1
            f = f - f.mean() + 1  # Shifting the mean to 1


        frb = frb * f.reshape(-1, 1)
        return frb

    def scatter(self, frb, tau0):
        '''
        Scatters the frb profile

        Convolves the frb profile in each channel with an exponential 
        function. The decay timescale of the exponential kernel is 
        determined using the value of tau0. The decay timescale is 
        scaled with frequency according to the self.scattering index.

        Convolution results in modification to the shape of the pulse
        resulting in a change in the area under the curve as well as
        the width of the pulse. This function re-normalises the 
        convolved frb profile trying to keep the same area as the area
        of the original profile, but the width does increase regardless.
        
        Parameters
        ----------
        frb : numpy.ndarray
            2-D numpy array containing the FRB time-freq profile
        tau0 : float
            The value of the decay timescale of the exponential kernel
            at the frequency of the highest channel in sec.

        Returns
        -------
        frb : numpy.ndarray
            2-D numpy array containing the FRB time-freq profile after
            convolution with the exponential
        '''
        f_ch = self.tel.f_ch
        tau0_samps = int(tau0/self.tel.tsamp) * self.tfactor
        nsamps = int( 6 * tau0_samps * ((self.tel.ftop + self.tel.fbottom)/2 / self.tel.fbottom)**self.scattering_index )
        nsamps = max([1, nsamps])
        self._nsamps_for_exponential = nsamps

        if tau0_samps == 0:
            return frb

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
        return scattered_frb

    def dm_smear_delay(self, dm, cfreq, tres, chw=None):
        '''
        Calculates the delay in samples due to dm-smearing within a 
        channel.
        
        Parameters
        ----------
        dm : float
            DM in pc/cc
        cfreq : float
            Center freq of the channel in MHz
        tres : float
            Time resolution in seconds
        chw : float
            Width of the channel in frequency (in MHz)
        
        Returns
        -------
        dms_width_samples : float
            The dm-smearing width in samples
        '''

        if chw is None:
            chw = self.tel.chw
        dm_width = 8.3 * chw * dm * (cfreq / 1e3)**(-3) * 1e-6 #s
        dms_width_samples = max([1, int(dm_width / tres)])
        return dms_width_samples

    def dm_smear_channel(self, d_ch, dm, cfreq, tres):
        '''
        Applies dm-smearing to the data in a channel.
        The method used is convolution of the data in a channel
        with a boxcar of height 1 and width = the dm-smearing width
        in that channel. It uses scipy.signal.fftconvolve for the
        convolution.

        Parameters
        ----------
        d_ch: numpy.ndarray
            1-D array containing data of a single channel
        dm : float
            DM to dmsmear at (in pc/cc)
        cfreq : float
            Center freq of the channel (in MHz)
        tres : float
            Time resolution of the data (in seconds)

        Returns
        -------
        dmsmeared_data : numpy.ndarray
            1-D array containing dm-smeared data of the input channel
        '''
        dms_width_samps = self.dm_smear_delay(dm, cfreq, tres)
        dms_kernel = N.ones(dms_width_samps)

        dmsmeared_data = signal.fftconvolve(d_ch, dms_kernel, mode='full')
        return dmsmeared_data / dms_width_samps

    def get_FWHM(self, frb_tseries):
        '''
        Computes the FWHM of a pulse
        Only works if there is a single peak. Will fail if the signal
        contains more than one peaks.

        Parameters
        ----------
        frb_tseries : numpy.ndarray
            1-D numpy array containing the signal
        
        Returns
        -------
        FWHM : float
            The full-width at half maximum in sample units
        '''
        maxx = N.argmax(frb_tseries)
        hp = frb_tseries[maxx] / 2.
        # Finding the half-power points
        hpp1 = (N.abs(frb_tseries[:maxx] - hp)).argmin()
        hpp2 = (N.abs(frb_tseries[maxx:] - hp)).argmin() + maxx

        FWHM = hpp2-hpp1
        assert FWHM > 0, "FWHM calculation went wrong somewhere. HPP points, maxx point and FWHM are {0} {1} {2} {3}".format(
        hpp1, hpp2, maxx, FWHM)
        return FWHM


    def disperse(self, frb, dm):
        '''
        Disperses the signal at a given DM. Also applied dm-smearing to
        individual channels
        
        dm-smearing is implemented as a convolution of the data in each
        channel with a box-car of width = dm-smearing width in that 
        channel. 
        Note, the output number of samples is larger than the input,
        and is calculated automatically based on the dm.
        
        Parameters
        ----------
        frb : numpy.ndarray
            2-D numpy array containing the frb time-freq profile
            
        dm : float
            DM at which the data needs to be dispersed (in pc/cc)
            
        Returns
        -------
        dispersed_frb : numpy.ndarray
            2-D array containing the disperse time-freq profile
            of the frb
        undispersed_time_series : numpy.ndarray
            1-D array containing the frequency averaged time series
            data of the frb profile. The averaging is done after
            dm-smearing has been applied, so this profile will be fatter
            and has a lower peak than the input frb.
            '''

        nch = self.tel.nch
        tres = self.tel.tsamp / self.tfactor  #s
        chw = self.tel.chw
        f_ch = self.tel.f_ch

        D = 4.14881e6       #ms; from Pulsar Handbook 4.1.1
        delays = D * f_ch**(-2) * dm    *1e-3  #Only works if freq in MHz and D in ms. Output delays in ms, multiply by 1e-3 to conver to 's'
        delays -= delays[int(nch/2)]
        delays_in_samples = N.rint(delays / tres).astype('int') #here we will have slight approximations due to quantization, but at 10.24 usec resolution they should be minimal

        #nsamps = delays_in_samples[-1] - delays_in_samples[0] + 2*frb.shape[1]
        #nsamps = delays_in_samples[-1]*2 + 2*frb.shape[1]
        
        if self.tot_nsamps is None:
            self._set_tot_nsamps(dm, buffer = frb.shape[1]//self.tfactor)
        nsamps = self.tot_nsamps * self.tfactor
        pre_shift = self._nsamps_for_gaussian // 2
        start = int(nsamps/2) - pre_shift
        #end = start + frb.shape[1]
        dm_smear_max = self.dm_smear_delay(dm, f_ch[-1], tres)
        dm_smear_max_nsamps = frb[1].shape[0] + dm_smear_max

        print(f"frb.shape = {frb.shape}, dm_smear_max = {dm_smear_max}")

        end = start + dm_smear_max_nsamps

        dispersed_frb = N.zeros(nch * nsamps).reshape(nch, nsamps)
        undispersed_time_series = N.zeros(dm_smear_max_nsamps)

        for i in range(nch):
            delay = delays_in_samples[i]
            if (end + delay > nsamps) or (start + delay < 0):
                raise RuntimeError("nsamps (={nsamps}) is too small to accomodate an FRB with DM = {dm}. Values: start = {start}, end={end}, delay={delay}".format(nsamps=self.tot_nsamps, dm=dm, start=start, end=end, delay=delay))
            dmsmeared_channel = self.dm_smear_channel(frb[i], dm, cfreq = f_ch[i], tres=tres)
            padded_chan = pad_along_axis(dmsmeared_channel, dm_smear_max_nsamps, axis=0)
            dispersed_frb[i, start+delay : end+delay] += padded_chan
            undispersed_time_series += padded_chan

        return dispersed_frb, undispersed_time_series
