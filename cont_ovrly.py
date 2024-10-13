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

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

def main(infrared_fits, rms_c, contour_fits, coords_file, contour2_fits, rms_c2):
    # Infrared image
    with fits.open(infrared_fits) as ir_hdul:
        ir_data = ir_hdul[0].data
        ir_wcs = WCS(ir_hdul[0].header, naxis=2)
    # Contour image
    with fits.open(contour_fits) as contour_hdul:
        contour_data = contour_hdul[0].data
        contour_wcs = WCS(contour_hdul[0].header, naxis=2)
        contour_header = contour_hdul[0].header
    
    # Optional extra contour
    if contour2_fits and rms_c2 is not None:
        with fits.open(contour2_fits) as contour_hdul:
            contour2_data = contour_hdul[0].data
            contour2_wcs = WCS(contour_hdul[0].header, naxis=2)
            contour2_header = contour_hdul[0].header


    # Reproject infrared im to match the radio WCS and shape
#    ir_reproj, _ = reproject_interp((ir_data, ir_wcs), radio_wcs, shape_out=radio_data.shape)
    # Repoject contour im to match the radio WCS and shape 
#    cont_reproj, _ = reproject_interp((contour_data, contour_wcs), radio_wcs, shape_out=radio_data.shape)
    # Use radio im WCS projection
    #fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': radio_wcs})
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection=ir_wcs)
    # Plot the reprojected infrared im #TODO Interpolation messes up the infrared image in some cases
    ax.imshow(ir_data[0], cmap='gray_r', origin='lower',
              vmin=np.percentile(ir_data, 1),
              vmax=np.percentile(ir_data, 99.8))


    ax.set_xlabel('R.A. (J2000)')
    ax.set_ylabel('Dec. (J2000)')

    ax.tick_params(direction='in', colors='black')
    ax.tick_params(axis='x', which='both', labelcolor='black')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    # Plot radio contours 'YlOrd'
    #contour_lvls = np.array([3, 6, 9, 12, 15, 18, 21, 24]) * rms_c
    #contour_lvls = np.array([i for i in range(3,69,9)]) * rms_c

    contour_lvls = np.logspace(np.log10(3), np.log10(40), num=int((np.log10(40) - np.log10(3)) / 0.15 +1)) * rms_c
    print(contour_lvls/rms_c)
    ax.contour(contour_data, levels=contour_lvls, colors='black', linewidths=0.8,transform=ax.get_transform(contour_wcs))
    
    # Plot optional contoursd
    if contour2_fits and rms_c2 is not None:
        contour2_lvls = np.logspace(np.log10(3), np.log10(10), num=int((np.log10(10) - np.log10(3)) / 0.12 +1)) * rms_c2
        ax.contour(contour2_data, levels=contour2_lvls, colors='blue', linewidths=0.8,transform=ax.get_transform(contour2_wcs))
    
    # Mark possible hg positions
    if coords_file is not None:

        coords_list = []
        with open(coords_file, 'r') as file:
            for line in file:
                ra, dec = map(float, line.strip().split(','))
                coords_list.append((ra, dec))

        host_coords = SkyCoord(ra=[c[0] for c in coords_list] * u.deg, 
                               dec=[c[1] for c in coords_list] * u.deg, frame='fk5')
        for i, c in enumerate(host_coords,start=1):
            ax.plot(c.ra.deg, c.dec.deg,marker='o', markerfacecolor='none', color='red',transform=ax.get_transform('fk5'), markersize=8)
            if len(host_coords) > 1:
                ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='red', transform=ax.get_transform('fk5'), fontsize=8, ha='left', va='top')

    # Add beam
    wcsaxes.add_beam(ax, header=contour_header,alpha=0.8,pad=0.65, facecolor='none',edgecolor='black', frame=True)
    if rms_c2 is not None:
        wcsaxes.add_beam(ax, header=contour2_header,alpha=0.8,pad=0.65, facecolor='none',edgecolor='blue')
    kpc_per_arcsec = 2.335 * u.kpc / u.arcsec

    # Scale bar in kpc
    scale_length_kpc = 50 * u.kpc
    scale_length_arcsec = scale_length_kpc / kpc_per_arcsec
    scale_length_arcsec = (scale_length_kpc / kpc_per_arcsec).to(u.arcsec)
    scale_length_deg = scale_length_arcsec.to(u.deg)
    
    # Add the scale bar
    wcsaxes.add_scalebar(ax, length=scale_length_deg, label=f'{scale_length_kpc.value:.0f} kpc',
            corner='bottom right', frame=False, color='white')
    

    fig.tight_layout()
    plt.show()
    fig.savefig('WISEplCONT.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay radio contours on an infrared image and mark possible host galaxies.')
    parser.add_argument('--infrared', required=True, help='Path to the infrared FITS image')
    parser.add_argument('--contour',required=True,help="Path to the contour FITS image")
    parser.add_argument('--rms_c', type=float, required=True, help='RMS value of the radio image')
    parser.add_argument('--coords', required=False, help='Path to the text file containing host galaxy coordinates')
    #optional
    parser.add_argument('--contour2',required=False,help="Path to the contour FITS image")
    parser.add_argument('--rms_c2', type=float, required=False, help='RMS value of the radio image')

    args = parser.parse_args()

    main(args.infrared, args.rms_c, args.contour, args.coords, args.contour2, args.rms_c2)

