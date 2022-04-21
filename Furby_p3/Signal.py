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

    def __init__(self, tel_obj, tfactor = 10, tot_nsamps = None, subsample_phase = 0.5):
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
        tot_nsamps : int, optional
            Total number of samples desired in the output mock frb
            template. If this option is None or left unspecified, then
            it is automatically calculated based on the requested DM in
            the disperse() function.
            '''
        self.tfactor = tfactor
        self.tel = tel_obj
        self.tot_nsamps = tot_nsamps
        self._set_sub_sample_phase(subsample_phase)

    def _set_sub_sample_phase(self, subsample_phase):
        if 0<=subsample_phase<=1:
            valid_options = N.linspace(0, 1, self.tfactor+1, endpoint=True)
            closest_idx = N.argmin(N.abs(valid_options - subsample_phase))
            closest_option = valid_options[closest_idx]
            self.subsample_phase = closest_option
            if self.subsample_phase == 1.0:
                self.subsample_phase = 0.0
        else:
            ValueError("Invalid value of subsample_phase: {0}".format(subsample_phase))
        

    def _set_tot_nsamps(self, delays_in_samples, buffer = 100, tfactor = 1):
        max_delay_right = delays_in_samples.max()
        self.tot_nsamps = (2 * (max_delay_right + buffer) //100 + 2 ) * 100 // tfactor
        #max_dm_sweep = self.dm_smear_delay(dm, self.tel.fcenter, self.tel.tsamp, chw=self.tel.bw)
        #self.tot_nsamps = int(int( (max_dm_sweep + 2*buffer) / 100  + 1) * 100)        #Rounding up to the nearest 100 samps


    def get_pure_frb(self, width, shape='gaussian'):
        '''
        Creates a simple Gaussian FRB profile with frequency and time axis
        
        Parameters
        ----------
        width : float
            Width expected (in seconds)
            box-car width incase shape is tophat
            FWHM width incase shape is gaussian
        shape : str
            Shape of the desired pulse
            Options - ['tophat', 'gaussian']

        Returns
        -------
        frb : numpy.ndarray
            A 2-D numpy array with freq and time axis containing the
            Gaussian FRB profile.
        '''

        width_samps = width / self.tel.tsamp
        width_samps = width_samps * self.tfactor
        self._nsamps_for_pure_frb = 2 * int(max([self.tfactor, 5 * width_samps/2])) + 1


        if shape.lower() == 'gaussian':
            x = N.arange(self._nsamps_for_pure_frb)
            pure_frb_single_channel = gauss2(
                x, a=1.0, x0=int(len(x)/2), FWHM=width_samps)
        elif shape.lower() == 'tophat':
            tophat_width_samps = max([1, int(width_samps)])
            pure_frb_single_channel = pad_along_axis(N.ones(tophat_width_samps), self._nsamps_for_pure_frb)
        else:
            raise ValueError("Requested pulse shape : {0} is not supported yet".format(shape))

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
            #n_blobs = int(N.abs(N.random.normal(loc=self.tel.bw / 50, scale=3)))
            n_blobs = int(N.floor(N.random.exponential(scale=5, size=1))) + 1
            f = N.zeros(nch)
            for i in range(n_blobs):
                center_of_blob = N.random.uniform(0, nch, 1)
                width_of_blob = N.abs(N.random.normal(
                    loc=20 / n_blobs, scale=10, size=1))
                # We want roughly 10 +- 5 MHz blobs. 10 MHz = 10/chw chans = 10./((P.ftop - P.bottom)/nch) chans
                NCHAN_PER_MHz = N.abs(1./((self.tel.ftop-self.tel.fbottom)/nch)) * 1e6
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

    def scatter(self, frb, tau0, scattering_index = 4.4):
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
        scattering_index : float
            The value of the power-law index to scale the tau0 in each
            channel with. For Kolmogorov like spectrum the value is 4.4

        Returns
        -------
        frb : numpy.ndarray
            2-D numpy array containing the FRB time-freq profile after
            convolution with the exponential
        '''
        f_ch = self.tel.f_ch
        tau0_samps = int(tau0/self.tel.tsamp) * self.tfactor
        nsamps = int( 6 * tau0_samps * ((self.tel.ftop + self.tel.fbottom)/2 / self.tel.fbottom)**scattering_index )
        nsamps = max([self.tfactor, nsamps])
        self._nsamps_for_exponential = nsamps

        if tau0_samps == 0:
            return frb

        k = tau0_samps * (f_ch[0])**scattering_index  # proportionality constant
        taus = k / f_ch**scattering_index  # Calculating tau for each channel
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

        dmsmeared_data = N.convolve(d_ch, dms_kernel, mode='full')
        return dmsmeared_data / dms_width_samps


    def find_subsample_offset(self, undispersed_time_series):
        '''
        Finds the offset required to acheive the requested 
        subsample_phase
        '''
        nsamps_hires = undispersed_time_series.shape[0]
        mean_pos_of_pulse = int(N.round(N.sum(N.arange(nsamps_hires) * undispersed_time_series) / N.sum(undispersed_time_series)))
        
        current_subsample_offset = mean_pos_of_pulse % self.tfactor
        desired_subsample_offset = int(self.subsample_phase * self.tfactor)

        return desired_subsample_offset - current_subsample_offset
        


    def disperse(self, frb, dm, dmsmear = True):
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
            
        dmsmear : bool
            Whether to enable dmsmearing or not. Def = True
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
        
        current_nsamps = frb.shape[1]
        dm_smear_max = self.dm_smear_delay(dm, f_ch[-1], tres)
        dm_smear_max_nsamps = current_nsamps + dm_smear_max
        if self.tot_nsamps is None:
            self._set_tot_nsamps(delays_in_samples, buffer = dm_smear_max_nsamps, tfactor=self.tfactor)
    
        nsamps = self.tot_nsamps * self.tfactor
        pre_shift = self._nsamps_for_pure_frb // 2
        start = int(nsamps/2) - pre_shift
        #end = start + frb.shape[1]
        #print(f"frb.shape = {frb.shape}, dm_smear_max = {dm_smear_max}, dm_smear_max_nsamps = {dm_smear_max_nsamps}")

        end = start + dm_smear_max_nsamps

        dispersed_frb = N.zeros(nch * nsamps).reshape(nch, nsamps)
        undispersed_time_series = N.zeros(nsamps)

        for i in range(nch):
            delay = delays_in_samples[i]
            if (end + delay > nsamps) or (start + delay < 0):
                raise RuntimeError("nsamps (={nsamps}) is too small to accomodate an FRB with DM = {dm}. Values: start = {start}, end={end}, delay={delay}".format(nsamps=self.tot_nsamps, dm=dm, start=start, end=end, delay=delay))
            if dmsmear:
                dmsmeared_channel = self.dm_smear_channel(frb[i], dm, cfreq = f_ch[i], tres=tres)
            else:
                dmsmeared_channel = frb[i]

            padded_chan = pad_along_axis(dmsmeared_channel, dm_smear_max_nsamps, axis=0)
            dispersed_frb[i, start+delay : end+delay] += padded_chan
            undispersed_time_series[start : end] += padded_chan
        
        required_subsample_offset = self.find_subsample_offset(undispersed_time_series)
        dispersed_frb = N.roll(dispersed_frb, required_subsample_offset, axis=1)
        undispersed_time_series = N.roll(undispersed_time_series, required_subsample_offset)

        return dispersed_frb, undispersed_time_series
