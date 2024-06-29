from astropy.io import fits

original_file = '../J01445/J01445_NVSS.fits'
new_file = 'fits/J01445_NVSS_corrected.fits'

with fits.open(original_file, mode='update') as hdul:
    header = hdul[0].header
    
    # Correct the pixel scale values
    pixel_scale_deg = 15.0 / 3600.0
    header['CDELT1'] = -pixel_scale_deg
    header['CDELT2'] = pixel_scale_deg
    
    # Add the RESTFRQ header attribute
    rest_frequency_hz = 1.4e9
    header['RESTFRQ'] = rest_frequency_hz
    
    hdul.writeto(new_file, overwrite=True)

print(f"Updated FITS file saved as {new_file}")

