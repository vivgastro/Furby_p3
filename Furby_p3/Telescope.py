import numpy as N
class Telescope(object):
    '''
    A class that encapsulates the properties of a telescope for which
    mock FRBs have to be simulated.
    '''

    def __init__(self, ftop, fbottom, nch, tsamp, name):
        '''
        Params
        ------
        name : str
            Name of the instrument
        ftop : float
            Frequency of the upper edge of highest channel (in MHz)
        fbottom : float
            Frequency of the lower edge of the lowest channel (in MHz)
        nch : int
            Number of frequency channels
        tsamp : float
            Sampling time of the instrument (in sec)
         '''
        self.name = name
        self.ftop = ftop
        self.fbottom = fbottom
        self.fcenter = (ftop + fbottom) /2
        self.nch = nch
        self.tsamp = tsamp
        self._set_bw()

    def _set_bw(self):
        '''
        Sets the total bandwidth (bw), channel bandwidth (chw)
        and the center freq (f_ch) of each channel.
        '''
        self.bw = self.ftop - self.fbottom
        self.chw = self.bw / self.nch
        self.f_ch = N.linspace(self.ftop - self.chw/2,
                               self.fbottom + self.chw/2, self.nch)

