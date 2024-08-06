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

def main(infrared_fits, radio_fits, rms, coords_file):
    # Load the infrared image
    with fits.open(infrared_fits) as ir_hdul:
        ir_data = ir_hdul[0].data
        ir_wcs = WCS(ir_hdul[0].header)

    # Load the radio image
    with fits.open(radio_fits) as radio_hdul:
        radio_data = radio_hdul[0].data
        radio_wcs = WCS(radio_hdul[0].header)
        radio_header = radio_hdul[0].header  # Store the header for later use

    # Load the coordinates from the text file
    coords_list = []
    with open(coords_file, 'r') as file:
        for line in file:
            ra, dec = map(float, line.strip().split(','))
            coords_list.append((ra, dec))

    host_coords = SkyCoord(ra=[c[0] for c in coords_list] * u.deg, 
                           dec=[c[1] for c in coords_list] * u.deg, frame='fk5')

    # Create a boolean mask for values greater than 6*rms
    mask = radio_data > 6 * rms
    radio_masked = np.where(mask, radio_data, np.nan)

    # Reproject the infrared image to match the radio WCS and shape
    ir_reproj, _ = reproject_interp((ir_data, ir_wcs), radio_wcs, shape_out=radio_data.shape)

    # Create the figure and axis with the radio image's WCS projection
    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': radio_wcs})

    # Plot the reprojected infrared image
    ax.imshow(ir_reproj, cmap='gray', origin='lower',
              vmin=np.percentile(ir_reproj, 5),
              vmax=np.percentile(ir_reproj, 99.5))

    # Set axis labels
    ax.set_xlabel('R.A (J2000)')
    ax.set_ylabel('Dec (J2000)')

    # Overlay the masked radio image in mJy
    ax.imshow(radio_masked * 1e3, cmap='inferno', origin='lower', alpha=0.5)

    # Customize tick parameters
    ax.tick_params(direction='in', colors='white')
    ax.tick_params(axis='x', which='both', labelcolor='black')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    # Plot radio contours
    contour_lvls = np.array([3, 6, 9, 12, 15]) * rms
    ax.contour(radio_data, levels=contour_lvls, cmap='YlOrRd', linewidths=0.8, alpha=0.95)

    # Mark possible host galaxy positions with 'X' and number them
    for c in host_coords:
        ax.plot(c.ra.deg, c.dec.deg,marker='x', color='cyan',transform=ax.get_transform('fk5'), markersize=10)



    # Add the synthesized beam ellipse
    wcsaxes.add_beam(ax, header=radio_header)

    # Add the radio image color bar
    cbar = fig.colorbar(ax.images[-1], ax=ax, shrink=1, pad=0.04)
    cbar.set_label('Brightness (mJy)')

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
    parser.add_argument('--rms', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--coords', required=True, help='Path to the text file containing host galaxy coordinates')

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.infrared, args.radio, args.rms, args.coords)

