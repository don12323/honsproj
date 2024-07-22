#!/usr/bin/env python3

import argparse
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

import sys
import os

def create_cutout(fitsfile, ra, dec, size):
    with fits.open(fitsfile) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header, naxis=2)

        position = SkyCoord(ra, dec, unit='deg', frame='fk5')
        size = u.Quantity((size, size), u.arcsec)
        print("number of dimensions",hdu.data.ndim)
        cutout = Cutout2D(hdu.data, position, size, wcs=wcs)
        
        print("new data dim", cutout.data.ndim)
        #update cutout header with new wcs 
        new_header = cutout.wcs.to_header()
        new_header['BMAJ'] = hdu.header['BMAJ']
        new_header['BMIN'] = hdu.header['BMIN']
        new_header['BPA'] = hdu.header['BPA']

        cutout_hdu = fits.PrimaryHDU(cutout.data, header=new_header)
        new_hdul = fits.HDUList([cutout_hdu])
        base, ext = os.path.splitext(fitsfile)
        out_fitsfile = f"{base}_cutout{ext}"

        new_hdul.writeto(out_fitsfile, overwrite=True)
        print(f"Cutout saved to {out_fitsfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infits", type=str, required=True, 
            help="Path to the input Fits file to be cutout")
    parser.add_argument("--ra", type=float, required=True,
            help="RA in deg")
    parser.add_argument("--dec", type=float, required=True,
            help="DEC in deg")
    parser.add_argument("--size", type=float, required=True,
            help="size for square cutout in arcsecs")
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    create_cutout(args.infits, args.ra, args.dec, args.size)


