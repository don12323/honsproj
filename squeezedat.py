#!/usr/bin/env python3

import argparse
from astropy.io import fits
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Process a FITS file and save with a fixed suffix.")
    parser.add_argument("infits", help="Input FITS file")
    args = parser.parse_args()
    
    infits = args.infits
    base_name = os.path.splitext(infits)[0]
    outfits = f"{base_name}fixed.fits"
    
    hdu = fits.open(infits)
    new_data = np.squeeze(np.squeeze(hdu[0].data))
    
    fits.writeto(outfits, new_data, hdu[0].header, overwrite=True)

if __name__ == "__main__":
    main()



