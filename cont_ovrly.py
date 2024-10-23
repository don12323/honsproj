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
from astropy.nddata import Cutout2D

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

def main(infrared_fits, rms_c, contour_fits, coords_file, contour2_fits, rms_c2, name, inset_im_file):
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
    # For inset image
    if inset_im_file is not None:
        with fits.open(inset_im_file) as inset_hdul:
            inset_data = inset_hdul[0].data
            inset_wcs = WCS(inset_hdul[0].header, naxis=2)



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
              vmin=np.percentile(ir_data, 20),
              vmax=np.percentile(ir_data, 99.9))


    ax.set_xlabel('R.A. (J2000)')
    ax.set_ylabel('Dec. (J2000)')

    ax.tick_params(direction='in', colors='black')
    ax.tick_params(axis='x', which='both', labelcolor='black')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    # Plot radio contours 'YlOrd'
    #contour_lvls = np.array([3, 6, 9, 12, 15, 18, 21, 24]) * rms_c
    #contour_lvls = np.array([i for i in range(3,69,9)]) * rms_c

    contour_lvls = np.logspace(np.log10(3), np.log10(90), num=int((np.log10(90) - np.log10(3)) / 0.15 +1)) * rms_c
    print(contour_lvls/rms_c)
    ax.contour(contour_data, levels=contour_lvls, colors='black', linewidths=0.8,transform=ax.get_transform(contour_wcs))
    
    # Plot optional contoursd
    if contour2_fits and rms_c2 is not None:
        contour2_lvls = np.logspace(np.log10(3), np.log10(70), num=int((np.log10(70) - np.log10(3)) / 0.2 +1)) * rms_c2
        ax.contour(contour2_data, levels=contour2_lvls, colors='blue', linewidths=0.8,transform=ax.get_transform(contour2_wcs))
    
    # Mark possible hg positions
    if inset_im_file is None and coords_file is not None:

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

    # Add inset im if provided
    if inset_im_file is not None:
        # Cutout region pos and size
        ra =305.364159 # DEG
        dec= 12.956446  # DEG
        inset_size = 30  # Arcsec
        rms_inset = 0.0000146522

        position = SkyCoord(ra, dec, unit='deg', frame='fk5')
        size = u.Quantity((inset_size, inset_size), u.arcsec)
        cutout_ir = Cutout2D(ir_data[0], position, size, wcs=ir_wcs)
        cutout_inset = Cutout2D(inset_data, position, size, wcs=inset_wcs)

        # Add a red square  indicating the region for the zoomed-in inset
        inset_size_deg = (inset_size * u.arcsec).to(u.deg).value
        ra_min = ra - inset_size_deg / 2
        ra_max = ra + inset_size_deg / 2
        dec_min = dec - inset_size_deg / 2
        dec_max = dec + inset_size_deg / 2

        rect_x = [ra_min, ra_max, ra_max, ra_min, ra_min]
        rect_y = [dec_min, dec_min, dec_max, dec_max, dec_min]

        # Plot the square
        ax.plot(rect_x, rect_y, color='red', transform=ax.get_transform('fk5'))
        
        # Add new inset axis
        inset_ax = fig.add_axes([0.3, 0.61, 0.27, 0.27], projection=cutout_ir.wcs) # [0.65, 0.71, 0.22, 0.22]
        
        # Plot cutout data
        inset_ax.imshow(cutout_ir.data, cmap='gray', origin='lower',
                  vmin=np.percentile(cutout_ir.data, 1),
                  vmax=np.percentile(cutout_ir.data, 99.6))
        
        contour_lvls_inset = np.logspace(np.log10(3), np.log10(25), num=int((np.log10(25) - np.log10(3)) / 0.15 +1)) * rms_inset
        print(contour_lvls_inset)
        inset_ax.contour(cutout_inset.data, levels=contour_lvls_inset, colors='yellow', linewidths=0.8,transform=inset_ax.get_transform(cutout_inset.wcs))
        
        # Tick param stuff
        inset_ax.patch.set_edgecolor('red')
        inset_ax.patch.set_linewidth(4)
        inset_ax.coords[0].set_ticklabel_visible(False)
        inset_ax.coords[1].set_ticklabel_visible(False)
        inset_ax.tick_params(direction='in', color='red')
        
        #inset_ax.spines['left'].set_color('red') 
        #inset_ax.spines['top'].set_color('red') 
        #inset_ax.spines['bottom'].set_color('red') 
        #inset_ax.spines['right'].set_color('red') 
        if coords_file is not None:

            coords_list = []
            with open(coords_file, 'r') as file:
                for line in file:
                    ra, dec = map(float, line.strip().split(','))
                    coords_list.append((ra, dec))

            host_coords = SkyCoord(ra=[c[0] for c in coords_list] * u.deg, 
                                   dec=[c[1] for c in coords_list] * u.deg, frame='fk5')
            for i, c in enumerate(host_coords,start=1):
                print(i)
                inset_ax.plot(c.ra.deg, c.dec.deg,marker='o', markerfacecolor='none', color='red',transform=inset_ax.get_transform('fk5'), markersize=9)
                if len(host_coords) > 1:
                    inset_ax.text(c.ra.deg, c.dec.deg, f' {i}', color='red', transform=inset_ax.get_transform('fk5'), fontsize=10, ha='left', va='top')
    
    # Add beam
    if rms_c2 is not None:
        wcsaxes.add_beam(ax, header=contour2_header,alpha=0.65, facecolor='none',edgecolor='blue',pad=0.1, frame=True)
    wcsaxes.add_beam(ax, header=contour_header,alpha=0.9,pad=0.3, facecolor='none',edgecolor='black', frame=False)
    
    # Scale bar in kpc
    kpc_per_arcsec = 5.572 * u.kpc / u.arcsec
    scale_length_kpc = 150 * u.kpc
    scale_length_arcsec = scale_length_kpc / kpc_per_arcsec
    scale_length_arcsec = (scale_length_kpc / kpc_per_arcsec).to(u.arcsec)
    scale_length_deg = scale_length_arcsec.to(u.deg)
    
    # Add the scale bar
    #wcsaxes.add_scalebar(ax, length=scale_length_deg, label=f'{scale_length_kpc.value:.0f} kpc',
            #corner='bottom right', frame=False, color='k')
    # Name
    # Add source name text in the top left corner
    ax.text(0.02, 0.98, name, transform=ax.transAxes, fontsize=10, color='black', 
                    ha='left', va='top', bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))

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
    parser.add_argument('--name', required=True, help='Name of the source being plotted')
    parser.add_argument('--inset_im',required=False, help='Inset contour image to be plotted')

    args = parser.parse_args()

    main(args.infrared, args.rms_c, args.contour, args.coords, args.contour2, args.rms_c2, args.name,args.inset_im)

