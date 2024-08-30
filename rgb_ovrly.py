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

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

# got the normalize function from Natasha
def normalize(arr, vmin, vmax):
    nor = (arr - vmin) / (vmax - vmin)
    nor[np.where(nor<0.0)] = 0.0
    nor[np.where(nor>1.0)] = 1.0
    return nor**2

def main(infrared_fits, radio_fits, rms_r, rms_c, contour_fits, coords_file):
    # Infrared image
    with fits.open(infrared_fits) as ir_hdul:
        ir_data = ir_hdul[0].data
        ir_wcs = WCS(ir_hdul[0].header,naxis=2)
    
    # Create rbg stack
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
    pct = 99.5
    interval = PercentileInterval(pct)
    stretch = AsinhStretch(a=0.1)

    i = interval.get_limits(ir_data[0])
    r = normalize(ir_data[0], *i)
    i = interval.get_limits(ir_data[1])
    g = normalize(ir_data[1], *i)
    i = interval.get_limits(ir_data[2])
    b = normalize(ir_data[2], *i)

    rgb_im = np.dstack([r, g, b])


    radio_interp,_ = reproject_interp((radio_data, radio_wcs),ir_wcs, shape_out=r.shape)
    mask = radio_interp > 3 * rms_r
    radio_masked = np.where(mask, radio_interp, np.min(radio_data)) #TODO np.min(radio_data)) somehow doing gaussian blending makes image dissapear completely
    # Reproject infrared im to match the radio WCS and shape
    #ir_reproj, _ = reproject_interp((rgb_im, ir_wcs), radio_wcs) #shape_out=(3,radio_data.shape[0],radio_data.shape[1]))
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

    # Overlay masked radio im in mJy  
    ax.imshow(radio_interp * 1e3, cmap='magma', origin='lower',
            alpha=gaussian_filter(mask.astype(float), sigma=3)*0.9) # Blend from 0.9 alpha
            #vmin=np.percentile(radio_masked,0.1),
            #vmax=np.percentile(radio_masked,99.9))
    print(f'{gaussian_filter(mask.astype(float), sigma=2)*0.9}')

    ax.tick_params(direction='in', colors='white')
    ax.tick_params(axis='x', which='both', labelcolor='black')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    # Plot radio contours 'YlOrd'
    #contour_lvls = np.array([3, 6, 9, 12, 15, 18, 21, 24]) * rms_c
    #contour_lvls = np.array([i for i in range(3,69,9)]) * rms_c

    contour_lvls = np.logspace(np.log10(3), np.log10(10), num=int((np.log10(10) - np.log10(3)) / 0.15 +1)) * rms_c
    print(contour_lvls/rms_c)
    ax.contour(contour_data, levels=contour_lvls, cmap='YlOrRd', linewidths=0.5, alpha=0.7,transform=ax.get_transform(contour_wcs))
    
    # Mark possible hg positions
    for i, c in enumerate(host_coords,start=1):
        ax.plot(c.ra.deg, c.dec.deg,marker='x', color='cyan',transform=ax.get_transform('fk5'), markersize=8)
        if len(host_coords) > 1:
            ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='cyan', transform=ax.get_transform('fk5'), fontsize=8, ha='left', va='top')

    # Add synthesized beam
    wcsaxes.add_beam(ax, header=contour_header,alpha=0.8,pad=0.65, frame=True)
    wcsaxes.add_beam(ax,header=radio_header,alpha=0.8, color='orange',pad=0.65) 
    # Add the radio image color bar
    cbar = fig.colorbar(ax.images[-1], ax=ax, shrink=1, pad=0.01) #pad=0.04
    cbar.set_label('Brightness (mJy/beam)')

    kpc_per_arcsec = 5.141 * u.kpc / u.arcsec

    # Scale bar length in kpc
    scale_length_kpc = 150 * u.kpc
    scale_length_arcsec = scale_length_kpc / kpc_per_arcsec

    # Convert to angular scale (arcseconds)
    scale_length_arcsec = (scale_length_kpc / kpc_per_arcsec).to(u.arcsec)
    scale_length_deg = scale_length_arcsec.to(u.deg)
    # Add the scale bar

    wcsaxes.add_scalebar(ax, length=scale_length_deg, label=f'{scale_length_kpc.value:.0f} kpc',
            corner='bottom right', frame=False, color='white')
    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()

    # Save the figure to a file
    fig.savefig('rbg_rad_ovrly.pdf')


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Overlay radio contours on its optical counterpart and mark possible host galaxies.')
    parser.add_argument('--rgb', required=True, help='Path to the rbg FITS image')
    parser.add_argument('--radio', required=True, help='Path to the radio FITS image')
    parser.add_argument('--rms_r', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--contour',required=True,help="Path to the contour FITS image")
    parser.add_argument('--rms_c', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--coords', required=True, help='Path to the text file containing host galaxy coordinates')

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.rgb, args.radio, args.rms_r, args.rms_c, args.contour, args.coords)

