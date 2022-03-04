import numpy as N
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as M
import argparse
import os
import sys

from Furby_p3.Furby_reader import Furby_reader as F
from Furby_p3.Furby_reader import Furby_Error
from Furby_p3.utility import tscrunch, fscrunch, get_matched_filter_snr

def main(args):
    print("Plotting {0} furby(s)".format(len(args.furbies)))
    for k,fil in enumerate(args.furbies):
        try:
          f=F(fil)
        except Furby_Error as fe:
          if fe.id == 0:
            print(fe.message)
            continue
          
        tres=f.header.TSAMP/1e6		#dada format requires tsamp to be in usec
        print("Filename: {0}\nID:{1}, SNR:{2}, DM:{3}, Width(top hat):{4} ms, Width(FWHM):{5} ms".format(f.filename, f.header.ID, f.header.SNR, f.header.DM, f.header.WIDTH, f.header.FWHM))

        data=f.read_data()
        dm = 0
        if args.dedisp:
            ddata=f.read_data(dd=True)
            dm = f.header.DM
        else:
            ddata=data

        fsdata=fscrunch(ddata,args.freq_sc)
        tfsdata=tscrunch(fsdata, args.t_sc)
        
        tseries=tfsdata.sum(axis=0)*1.0
        fseries = tfsdata.sum(axis=1)*1.0

        if args.print_header:
            print("The header of the furby is:\n{0}\n".format(f.header))
        

        if not args.dedisp:
            dedisp_furby = f.dedisperse(data, dm = f.header.DM)
            dedisp_tseries = dedisp_furby.sum(axis=0)*1.0
        else:
            dedisp_tseries = tseries

        mf_snr = get_matched_filter_snr(dedisp_tseries, f.header.NOISE_PER_SAMPLE * N.sqrt(f.header.NCHAN))
        print("The matched filter snr is: {0}".format(mf_snr))


        toff=0.5*tres*args.t_sc
        x=N.arange(0,len(tseries))*tres*args.t_sc + toff
        
        f0 = f.header.FBOTTOM
        fn = f.header.FTOP
        chw = f.header.BW/f.header.NCHAN
        
        if f.header.BW<0:
          (f0, fn) = (fn, f0)
          
        fa = f0 + chw/2*args.freq_sc
        fb = fn - chw/2*args.freq_sc
        y = N.arange(fa, fb+chw*args.freq_sc, chw*args.freq_sc)
        
        extent=[x[0]-toff, x[-1]+toff, fn, f0]

        fig=M.figure(k, figsize=(6.5,5))
        
        ax1=M.subplot2grid((6,8), (0,0), rowspan=5, colspan=6)
        ax1.imshow(tfsdata, interpolation='none', aspect='auto', cmap='afmhot', extent=extent)
        ax1.set_title(fil+" De-DM: "+str(dm), fontsize=8)
        ax1.set_xlim(0,tfsdata.shape[1])
        ax1.set_ylabel("Freq (MHz)")
        
        ax2=M.subplot2grid((6,8), (5,0), rowspan=1, colspan = 6, sharex=ax1)
        ax2.plot(x, tseries)
        ax2.set_xlim(x[0]-toff, x[-1]+toff)
        ax2.set_xlabel("Time (s)")
        
        ax3=M.subplot2grid((6,8), (0,6),rowspan = 5, colspan = 2, sharey=ax1)
        ax3.plot(fseries, y)
        ax3.set_xlabel("Power")
        ax3.set_ylim(fn, f0)
        
        M.subplots_adjust(hspace=0,wspace=0, bottom=0.1)
        M.setp(ax1.get_xticklabels(), visible=False)
        M.setp(ax3.get_yticklabels(), visible=False)
        #M.setp(ax2.get_yticklabels(), visible=False)
        #if not args.one:     
        #    mgr=M.get_current_fig_manager()
        #    mgr.window.move((k%3)*640, int(k/3)*600)
        
        if args.pngs:
            print("saving",fil)
            M.savefig((str(fil))+".png", dpi=200)
            M.close('all')
            continue
        
        if(k<len(args.furbies)-1):
            M.show(block=False)
            input("<Press Enter to see next plot>\n")
            M.close('all')
    if not args.pngs:
        M.show()

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("furbies", type=str, nargs='+', help="Furby files to plot")
    a.add_argument("-dd", "--dedisp", action='store_true', help="Dedisperse the furby? (def=False)", default=False)
    a.add_argument("-fs","--freq_sc", type=int, help="Freq scrunch factor (def=1)", default=1)
    a.add_argument("-ts","--t_sc", type=int, help="Time scrunch factor (def=1)", default=1)
    a.add_argument("-ph", "--print-header", action='store_true', help="Print out the header information (def = False)", default=False)

    a.add_argument("-pngs", action='store_true', help="Save pngs instead of plotting")
    args=a.parse_args()
    main(args)




