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

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

def main(infrared_fits, radio_fits, rms_r, rms_c, contour_fits, coords_file):
    # Infrared image
    with fits.open(infrared_fits) as ir_hdul:
        ir_data = ir_hdul[0].data
        ir_wcs = WCS(ir_hdul[0].header)

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
    mask = radio_data > 3 * rms_r
    radio_masked = np.where(mask, radio_data, np.min(radio_data))

    # Reproject infrared im to match the radio WCS and shape
    ir_reproj, _ = reproject_interp((ir_data, ir_wcs), radio_wcs, shape_out=radio_data.shape)
    # Repoject contour im to match the radio WCS and shape 
#    cont_reproj, _ = reproject_interp((contour_data, contour_wcs), radio_wcs, shape_out=radio_data.shape)
    # Use radio im WCS projection
    #fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': radio_wcs})
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection=radio_wcs)
    # Plot the reprojected infrared im #TODO Interpolation messes up the infrared image in some cases
    ax.imshow(ir_reproj, cmap='gray', origin='lower',
              vmin=np.percentile(ir_reproj, 5),
              vmax=np.percentile(ir_reproj, 99.6))


    ax.set_xlabel('R.A. (J2000)')
    ax.set_ylabel('Dec. (J2000)')

    # Overlay masked radio im in mJy #TODO 
    ax.imshow(radio_masked * 1e3, cmap='inferno', origin='lower', alpha=0.4)
            #vmin=np.percentile(radio_masked,0),
            #vmax=np.percentile(radio_masked,99.9))

    ax.tick_params(direction='in', colors='white')
    ax.tick_params(axis='x', which='both', labelcolor='black')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    # Plot radio contours 'YlOrd'
    contour_lvls = np.array([3, 6, 9, 12, 15, 18, 21, 24]) * rms_c
    ax.contour(contour_data, levels=contour_lvls, cmap='YlOrRd', linewidths=0.5, alpha=0.7,transform=ax.get_transform(contour_wcs))
#    ax.contour(radio_data, levels=[3 * rms_r], linewidths=1,cmap='inferno', alpha=0.6,transform=ax.get_transform(radio_wcs))
    
    # Mark possible hg positions
    for i, c in enumerate(host_coords,start=1):
        ax.plot(c.ra.deg, c.dec.deg,marker='x', color='cyan',transform=ax.get_transform('fk5'), markersize=8)
        if len(host_coords) > 1:
            ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='cyan', transform=ax.get_transform('fk5'), fontsize=8, ha='left', va='top')

    # Add synthesized beam #TODO Show beams at a common center(maybe show it manually??)
    wcsaxes.add_beam(ax, header=contour_header,alpha=0.6)
    wcsaxes.add_beam(ax,header=radio_header,alpha=0.6, color='orange')

    # Add the radio image color bar
    cbar = fig.colorbar(ax.images[-1], ax=ax, shrink=1, pad=0.04)
    cbar.set_label('Brightness (mJy/beam)')

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()

    # Save the figure to a file
    fig.savefig('overlay.pdf')


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Overlay radio contours on an infrared image and mark possible host galaxies.')
    parser.add_argument('--infrared', required=True, help='Path to the infrared FITS image')
    parser.add_argument('--radio', required=True, help='Path to the radio FITS image')
    parser.add_argument('--rms_r', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--contour',required=True,help="Path to the contour FITS image")
    parser.add_argument('--rms_c', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--coords', required=True, help='Path to the text file containing host galaxy coordinates')

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.infrared, args.radio, args.rms_r, args.rms_c, args.contour, args.coords)

