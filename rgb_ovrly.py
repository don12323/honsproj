#!/usr/bin/env python3

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import wcsaxes
import argparse
from scipy.ndimage import gaussian_filter

from astropy.visualization import PercentileInterval
from astropy.visualization import AsinhStretch
from astropy.visualization import PowerDistStretch

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

# Got the normalize function from code that Natasha gave
def normalize(arr, vmin, vmax):
    nor = (arr - vmin) / (vmax - vmin)
    nor[np.where(nor<0.0)] = 0.0
    nor[np.where(nor>1.0)] = 1.0
    return nor

def main(rgb_fits, radio_fits, rms_r, rms_c, contour_fits, coords_file):
    # RGB image
    with fits.open(rgb_fits) as ir_hdul:
        ir_data = ir_hdul[0].data
        ir_wcs = WCS(ir_hdul[0].header,naxis=2)
    
    
    print(f'rgb fits data shape: {ir_data.shape}')
    # Radio image
    with fits.open(radio_fits) as radio_hdul:
        radio_data = radio_hdul[0].data
        radio_wcs = WCS(radio_hdul[0].header, naxis=2)
        radio_header = radio_hdul[0].header  # Store the header for later use
    # Contour image
    with fits.open(contour_fits) as contour_hdul:
        contour_data = contour_hdul[0].data
        contour_wcs = WCS(contour_hdul[0].header, naxis=2)
        contour_header = contour_hdul[0].header

    # Load coordinates from text file
    coords_list = []
    with open(coords_file, 'r') as file:
        for line in file:
            ra, dec = map(float, line.strip().split(','))
            coords_list.append((ra, dec))

    host_coords = SkyCoord(ra=[c[0] for c in coords_list] * u.deg, 
                           dec=[c[1] for c in coords_list] * u.deg, frame='fk5')

    # Create mask for values greater than 3*rms
    
    # normalize
    pct = 99.6
    interval = PercentileInterval(pct)
    stretch =AsinhStretch(a=0.04) + PowerDistStretch(a=1000)  #makes bkg noise worse

    i = interval.get_limits(ir_data[0])
    r = stretch(normalize(ir_data[0], *i))
    i = interval.get_limits(ir_data[1])
    g = stretch(normalize(ir_data[1], *i))
    i = interval.get_limits(ir_data[2])
    b = stretch(normalize(ir_data[2], *i))
    # Create rgb stack
    rgb_im = np.dstack([r, g, b])


    # Reproject radio im to match the rgb WCS and shape
    radio_interp,_ = reproject_interp((radio_data, radio_wcs),ir_wcs, shape_out=r.shape)
    mask = radio_interp > 3 * rms_r
    radio_masked = np.where(mask, radio_interp, np.min(radio_data)) #np.min(radio_data)) 
    
    print(f'slice shape: {r.shape}')
    print('rbg wcs:', ir_wcs)
    print('radio wcs:', radio_wcs)
    

    #plotting----------------------------------------
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection=ir_wcs)
    # Plot the rgb im #TODO Interpolation messes up the infrared image in some cases
    ax.imshow(rgb_im, origin='lower')


    ax.set_xlabel('R.A. (J2000)')
    ax.set_ylabel('Dec. (J2000)')
    #instead of gaussian blending normalise radio_masked array and use that as alpha array
    alpha = normalize(radio_masked, 3*rms_r, np.max(radio_masked))
    alpha_stretch = AsinhStretch(a=0.5)
    alpha = alpha_stretch(alpha)
    # Overlay masked radio im in mJy 
    gaus_bl = gaussian_filter(mask.astype(float), sigma=3)*0.99
    ax.imshow(radio_interp * 1e3, cmap='magma', origin='lower', #gist_heat
            alpha = alpha)
            #alpha=gaussian_filter(mask.astype(float), sigma=150)*1) # Blend from 0.9 alpha

    ax.tick_params(direction='in', colors='white')
    ax.tick_params(axis='x', which='both', labelcolor='black')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    # Plot radio contours 'YlOrd'
    contour_lvls = np.logspace(np.log10(3), np.log10(22), num=int((np.log10(22) - np.log10(3)) / 0.15 +1)) * rms_c
    print(contour_lvls/rms_c)
    ax.contour(contour_data, levels=contour_lvls, cmap='YlOrRd', linewidths=0.5, alpha=0.4,transform=ax.get_transform(contour_wcs))
    
    # Mark possible hg positions
    for i, c in enumerate(host_coords,start=1):
        ax.plot(c.ra.deg, c.dec.deg,marker='x', color='cyan',transform=ax.get_transform('fk5'), markersize=6)
        if len(host_coords) > 1:
            ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='cyan', 
                    transform=ax.get_transform('fk5'), fontsize=6, ha='left', va='top')

    # Add synthesized beam
    wcsaxes.add_beam(ax, header=contour_header,alpha=0.9,pad=0.65,frame=False)
    wcsaxes.add_beam(ax,header=radio_header,color='orange',alpha=0.9,pad=0.9) 
    # Radio image color bar
    cbar = fig.colorbar(ax.images[-1], ax=ax, shrink=1, pad=0.01,aspect=40) #pad=0.04
    cbar.set_label('Brightness (mJy/beam)')
    # Scale
    kpc_per_arcsec = 1.54 * u.kpc / u.arcsec

    # Scale bar length in kpc
    scale_length_kpc = 50 * u.kpc
    scale_length_arcsec = scale_length_kpc / kpc_per_arcsec

    # Convert to deg
    scale_length_deg = scale_length_arcsec.to(u.deg)
    
    # Add the scale bar
    wcsaxes.add_scalebar(ax, length=scale_length_deg, label=f'{scale_length_kpc.value:.0f} kpc',
            corner='bottom right', frame=False, color='white')
    
    # Plot and save
    fig.tight_layout()
    plt.show()
    fig.savefig('ovrly_magma_test.pdf')
    fig.savefig('rgb_rad_ovrly.jpg',dpi=800)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay radio image on its optical counterpart and mark possible host galaxies.')
    parser.add_argument('--rgb', required=True, help='Path to the rbg FITS image')
    parser.add_argument('--radio', required=True, help='Path to the radio FITS image')
    parser.add_argument('--rms_r', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--contour',required=True,help="Path to the contour FITS image")
    parser.add_argument('--rms_c', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--coords', required=True, help='Path to the text file containing host galaxy coordinates')

    args = parser.parse_args()

    main(args.rgb, args.radio, args.rms_r, args.rms_c, args.contour, args.coords)

