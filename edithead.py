from astropy.io import fits

original_file = '../fits/J01445_TGSS_corrected.fits'
new_file = '../fits/J01445_TGSS_corrected2.fits'

with fits.open(original_file, mode='update') as hdul:
    header = hdul[0].header
    
    pixel_scale_deg = 0.0008
    header['CDELT1'] = -pixel_scale_deg
    header['CDELT2'] = pixel_scale_deg
    
    rest_frequency_hz = 150.e6
    header['RESTFRQ'] = rest_frequency_hz
    
    hdul.writeto(new_file, overwrite=True)

print(f"Updated FITS file saved as {new_file}")

