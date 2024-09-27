import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import wcsaxes
import astropy.units as u
import glob


plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

#TODO change to using one function instead of 2 
def plot_first_four_sources(fits_files, output_file):
    fig = plt.figure(figsize=(15, 0.8*15 * np.sqrt(2)))  
    labels = ['(a)', '(b)', '(c)', '(d)']

    for i, fits_file in enumerate(fits_files[:4]):
        with fits.open(fits_file) as hdul:
            header = hdul['INDEX_MAP'].header
            wcs = WCS(header)
            spectral_index_map = hdul['INDEX_MAP'].data
            spectral_index_error_map = hdul['ERROR_MAP'].data
            chi2red_map = hdul['CHI2RED_MAP'].data
            image = hdul['IMAGE'].data
            rms = hdul[0].header['RMS']
            maxlvl = hdul[0].header['MAX_LVL']
            dex = hdul[0].header['DEX']
            
            host_coords = None
            if 'HOST_COORDS' in hdul:
                ra = hdul['HOST_COORDS'].data['RA']
                dec = hdul['HOST_COORDS'].data['DEC']
                host_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='fk5')

            #host_coords = None
            #if 'HGRA' in header and 'HGDEC' in header:
            #    hgra = header['HGRA']
            #    hgdec = header['HGDEC']
            #    host_coords = SkyCoord(ra=hgra * u.deg, dec=hgdec * u.deg, frame='fk5')

        #contour_levels = [3*rms, 6*rms, 10*rms, 15*rms, 25*rms]
        contour_levels = np.logspace(np.log10(3), np.log10(maxlvl), num=int((np.log10(maxlvl) - np.log10(3)) / dex +1)) * rms
        # Plot spectral index map
        ax1 = fig.add_subplot(4, 3, i * 3 + 1, projection=wcs)
        im1 = ax1.imshow(-1*spectral_index_map, origin='lower', cmap='turbo_r', interpolation='none', vmin=0.5, vmax=4) #gist_rainbow_r
        ax1.contour(image, levels=contour_levels, colors='black', alpha=0.8, transform=ax1.get_transform(wcs))
        ax1.tick_params(direction='in') #width=2
        if i == 0:
            ax1.set_title('Spectral Index')
        ax1.coords[0].set_axislabel('R.A. (J2000)')
        ax1.coords[1].set_axislabel('Dec. (J2000)')
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.99, pad=0.01,aspect=30)
        # Label
        ax1.text(0.02, 0.95, labels[i], transform=ax1.transAxes, fontsize=14, fontweight='bold',va='top', ha='left')
        #TODO axis width doesnt work
        for spine in ax1.spines.values():
            spine.set_linewidth(3)

        # Spectral index error map
        ax2 = fig.add_subplot(4, 3, i * 3 + 2, projection=wcs)
        im2 = ax2.imshow(spectral_index_error_map, origin='lower', cmap='turbo', interpolation='none', norm=LogNorm())
        ax2.contour(image, levels=contour_levels, colors='black', alpha=0.8, transform=ax2.get_transform(wcs))
        ax2.tick_params(direction='in')
        if i == 0:
            ax2.set_title('Spectral Index Error')
        ax2.coords[0].set_axislabel('R.A. (J2000)')
        ax2.coords[1].set_ticklabel_visible(False)
        ax2.coords[1].set_axislabel('')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.99, pad=0.01,aspect=30)
        
        # Chi2red map
        ax3 = fig.add_subplot(4, 3, i * 3 + 3, projection=wcs)
        im3 = ax3.imshow(chi2red_map, origin='lower', cmap='inferno', interpolation='none', norm=LogNorm())
        ax3.contour(image, levels=contour_levels, colors='black', alpha=0.8, transform=ax3.get_transform(wcs))
        ax3.tick_params(direction='in')
        if i == 0:
            ax3.set_title(r'$\chi^2_{red}$')
        ax3.coords[0].set_axislabel('R.A. (J2000)')
        ax3.coords[1].set_ticklabel_visible(False)
        ax3.coords[1].set_axislabel('')
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', shrink=0.99, pad=0.01,aspect=30)
        
        # Plot hg coords
        axes = [ax1, ax2, ax3]
        for ax in axes:
            wcsaxes.add_beam(ax, header=header,alpha=0.9,pad=0.65,frame=True, facecolor=None, edgecolor=None)
            if host_coords is not None:
                for i,c in enumerate(host_coords, start=1):
                    ax.plot(c.ra.deg, c.dec.deg,marker='x', color='black', transform=ax.get_transform('fk5'), markersize=10)
                    ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='black', transform=ax.get_transform('fk5'), fontsize=10, ha='left', va='top')
        
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

def plot_remaining_sources(fits_files, output_file):
    fig = plt.figure(figsize=(15, 0.8*15 * np.sqrt(2)))  # Made it match paper size #fig = plt.figure(figsize=(15, 0.8*3/4*15 * np.sqrt(2)))  # Made it match paper size
    labels = ['(e)', '(f)', '(g)','(h)']
    for i, fits_file in enumerate(fits_files[4:]):
        with fits.open(fits_file) as hdul:
            header = hdul['INDEX_MAP'].header
            wcs = WCS(header)
            spectral_index_map = hdul['INDEX_MAP'].data
            spectral_index_error_map = hdul['ERROR_MAP'].data
            chi2red_map = hdul['CHI2RED_MAP'].data
            image = hdul['IMAGE'].data
            rms = hdul[0].header['RMS']
        
            maxlvl = hdul[0].header['MAX_LVL']
            dex = hdul[0].header['DEX']
            
            host_coords = None
            if 'HOST_COORDS' in hdul:
                ra = hdul['HOST_COORDS'].data['RA']
                dec = hdul['HOST_COORDS'].data['DEC']
                host_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='fk5')

            #host_coords = None
            #if 'HGRA' in header and 'HGDEC' in header:
            #    hgra = header['HGRA']
            #    hgdec = header['HGDEC']
            #    host_coords = SkyCoord(ra=hgra * u.deg, dec=hgdec * u.deg, frame='fk5')

        #contour_levels = [3*rms, 6*rms, 10*rms, 15*rms, 25*rms]
        contour_levels = np.logspace(np.log10(3), np.log10(maxlvl), num=int((np.log10(maxlvl) - np.log10(3)) / dex +1)) * rms

        # spectral index map
        ax1 = fig.add_subplot(4, 3, i * 3 + 1, projection=wcs)
        im1 = ax1.imshow(-1*spectral_index_map, origin='lower', cmap='turbo_r', interpolation='none', vmin=0.5, vmax=3) #gist rainbow
        ax1.contour(image, levels=contour_levels, colors='black', alpha=0.8, transform=ax1.get_transform(wcs))
        ax1.tick_params(direction='in')
        if i == 0:
            ax1.set_title('Spectral Index')
        ax1.coords[0].set_axislabel('R.A. (J2000)')
        ax1.coords[1].set_axislabel('Dec. (J2000)')
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.99, pad=0.01,aspect=30)
        # Label
        ax1.text(0.02, 0.95, labels[i], transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        # spectral index error map
        ax2 = fig.add_subplot(4, 3, i * 3 + 2, projection=wcs)
        im2 = ax2.imshow(spectral_index_error_map, origin='lower', cmap='turbo', interpolation='none', norm=LogNorm())
        ax2.contour(image, levels=contour_levels, colors='black', alpha=0.8, transform=ax2.get_transform(wcs))
        ax2.tick_params(direction='in')
        if i == 0:
            ax2.set_title('Spectral Index Error')
        ax2.coords[0].set_axislabel('R.A. (J2000)')
        ax2.coords[1].set_ticklabel_visible(False)
        ax2.coords[1].set_axislabel('')
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.99, pad=0.01,aspect=30)
        
        # chi2red map
        ax3 = fig.add_subplot(4, 3, i * 3 + 3, projection=wcs)
        im3 = ax3.imshow(chi2red_map, origin='lower', cmap='inferno', interpolation='none', norm=LogNorm())
        ax3.contour(image, levels=contour_levels, colors='black', alpha=0.8, transform=ax3.get_transform(wcs))
        ax3.tick_params(direction='in')
        if i == 0:
            ax3.set_title(r'$\chi^2_{red}$')
        ax3.coords[0].set_axislabel('R.A. (J2000)')
        ax3.coords[1].set_ticklabel_visible(False)
        ax3.coords[1].set_axislabel('')
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', shrink=0.99, pad=0.01,aspect=30)
        
        # Plot hg coords
        axes = [ax1, ax2, ax3]
        for ax in axes:
            wcsaxes.add_beam(ax, header=header,alpha=0.9,pad=0.65,frame=True, facecolor=None, edgecolor=None)
            if host_coords is not None:
                for i,c in enumerate(host_coords, start=1):
                    ax.plot(c.ra.deg, c.dec.deg,marker='x', color='black', transform=ax.get_transform('fk5'), markersize=10)
                    ax.text(c.ra.deg, c.dec.deg, f'   {i}', color='black', transform=ax.get_transform('fk5'), fontsize=10, ha='left', va='top')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

pattern1 = '/data/donwijesinghe/*_res/a.fits'
pattern2 = '/data/donwijesinghe/*_res/ma.fits'
fits_files1 = glob.glob(pattern1)
fits_files2 = glob.glob(pattern2)
fits_files = fits_files1 + fits_files2

output_file_1 = '../results/spectral_index_maps_1.pdf'
output_file_2 = '../results/spectral_index_maps_2.pdf'
plot_first_four_sources(fits_files, output_file_1)
plot_remaining_sources(fits_files, output_file_2)

