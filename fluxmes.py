#!/usr/bin/env python3 

import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import random
import matplotlib.mlab as mlab

from matplotlib.patches import RegularPolygon
from matplotlib.patches import Polygon as MtPltPolygon
import pyregion
from shapely.geometry import Point, Polygon

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import wcsaxes

import warnings
import sys
import csv

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = False

warnings.simplefilter('ignore', AstropyWarning)

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def read_data(input_file):

    f = []
    S = []
    Serr = []

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        if header != ['Spectra', 'Frequency (Hz)', 'Photometry (Jy)', 'Uncertainty (Jy)']:
            print("Invalid header format. Expected ['Spectra', 'Frequency (Hz)', 'Photometry (Jy)', 'Uncertainty (Jy)']")
            return None, None, None

        for row in reader:
            f.append(float(row[1]))
            S.append(float(row[2]))
            Serr.append(float(row[3]))

    return np.array(S), np.array(f), np.array(Serr)

def extract_header_data(filename):
    print(f"{Colors.OKCYAN}>> Reading in FITS file: {filename}{Colors.ENDC} \n")
    with fits.open(filename) as hdu:
	
        data = hdu[0].data	
        header = hdu[0].header
        naxis = header["NAXIS"]
        bmaj = header["BMAJ"]
        bmin = header["BMIN"]
        bpa = header["BPA"]
        try:
            freq = header["RESTFREQ"]
        except KeyError:
            freq = header["RESTFRQ"]

        print(f"  {Colors.OKGREEN} FREQ: {freq}Hz {Colors.ENDC}")
        if naxis != 2:
            print(f"{Colors.FAIL}Error: NAXIS is not equal to 2 for file: {filename}{Colors.ENDC}")
            sys.exit(1)
        
        # Find pixel size in degrees
        try:
            pixd1, pixd2 = header["CDELT1"], header["CDELT2"]
        except KeyError:
            pixd1, pixd2 = header["CD1_1"], header["CD2_2"]

        print(f'   Pixel size: {abs(pixd1*3600):.2f}" x {pixd2*3600:.2f}"') 
        print(f"   {Colors.HEADER}Beam parameters: {bmaj*3600},{bmin*3600},{bpa} {Colors.ENDC}")
        
        barea = np.pi*bmaj*bmin/(4.0*np.log(2))
        npixpb = barea/(abs(pixd1*pixd2))
        print(f"   Beam area: {barea*3600**2:.3f} arcsec^2")        
        print(f"   Number of pixels per beam: {npixpb}\n")

        

        wcs = WCS(header, naxis= naxis)
    return data, wcs, header, pixd1, pixd2, barea, freq

    
def measure_flux(header,data, pixarea, barea, reg_file, bkg_file):
    
    # Read polygon region file and get pixel coordinates
    regions = pyregion.open(reg_file).as_imagecoord(header=header)
    poly_coords=[]
    for region in regions:
        for i in range(0, len(region.coord_list), 2):
            poly_coords.append((region.coord_list[i], region.coord_list[i+1])) 
    
    reg_polygon = Polygon(poly_coords)

    # Read in background polygon in image coordinates
    bkg_regions = pyregion.open(bkg_file).as_imagecoord(header=header)
    
    source_polygons = []
    for region in bkg_regions:
        poly_coords = []
        for i in range(0, len(region.coord_list), 2):
            poly_coords.append((region.coord_list[i], region.coord_list[i + 1]))
        source_polygons.append(Polygon(poly_coords))


    # Calculate the mean background flux
    npix_bkg=0
    bkg_flux_values = []
    bkg_squared = []
    max_x, max_y = data.shape[1], data.shape[0]
    min_x, min_y = 1, 1
    for y in range(math.ceil(min_y), math.floor(max_y)):
        for x in range(math.ceil(min_x), math.floor(max_x)):
            point = Point(x,y)
            # Corners
            if all(not poly.contains(point) for poly in source_polygons) and not np.isnan(data[y - 1, x - 1]):
             #   print(x,y) 
                bkg_flux_values.append(data[y-1,x-1])
                npix_bkg+=1
                bkg_squared.append(data[y-1,x-1]**2)

    bkg_flux = np.mean(bkg_flux_values)
    std_bkg = np.std(bkg_flux_values)
    rms_bkg = np.sqrt(np.mean(np.array(bkg_squared)))
    print(f"   {Colors.OKGREEN}Background mean flux: {bkg_flux} +\- {std_bkg} Jy/beam{Colors.ENDC}")
    print("   Number of pixels in background: ",npix_bkg) 
    print(f"   rms_bkg: {rms_bkg}")
    print("\n")
    print(f"   {Colors.UNDERLINE}CALCULATING FLUX {Colors.ENDC}")
    
    
    total_flux_in_reg = 0
    npix_reg = 0
    flux_squared = []
   
    # Find bounds for the region we are measuring fluxes
    min_x, min_y, max_x, max_y = reg_polygon.bounds
    #print("minx", min_x, "miny", min_y, "maxx",max_x, "maxy",max_y)

    for y in range(math.ceil(min_y)-1, math.floor(max_y)+1):
        for x in range(math.ceil(min_x)-1, math.floor(max_x)+1):
            if reg_polygon.contains(Point(x,y)):
                total_flux_in_reg += data[y-1,x-1]
                npix_reg += 1
                flux_squared.append(data[y-1,x-1]**2)


    #calculate the integrated flux
    print(f"   Number of pixels in region: {npix_reg} Size: {npix_reg*pixarea*3600**2:.2f} arcsec^2")
    int_flux = (total_flux_in_reg * pixarea / barea)-(bkg_flux * npix_reg *pixarea / barea) #-TODO (bkg_flux * npix_hex * pixarea / barea) #bkg_flux*(nbeams inside hex)
    # Uncertainties
    #rms = np.sqrt(np.mean(np.array(flux_squared)))
    uncertainty = np.sqrt((0.02 * int_flux)**2+ (rms_bkg * npix_reg * pixarea / barea)**2) 
    
    print(f"   bkg tot flux in reg: {bkg_flux * npix_reg * pixarea / barea}")
    print(f"   Integrated Flux = {int_flux*10**3:.4f} +\- {uncertainty*10**3:.6f} mJy, ")

    return source_polygons, reg_polygon, rms_bkg, int_flux, uncertainty
 

def plotPolygons(reg_polygon, wcs, header, data, source_polygons, rms): 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1,projection=wcs)
    im=ax.imshow(data, cmap='gray')

    # Contours
    contour_levels = rms * np.array([3,6,9]) #[-3, 3, 6, 9]
    contours = ax.contour(data, levels=contour_levels, colors='white', linewidths=0.5, linestyles='dashed')
     
    # Plot region
    reg_patch = MtPltPolygon(reg_polygon.exterior.coords, closed=True, edgecolor='red', fill=False)
    ax.add_patch(reg_patch)
    # Plot background
    for poly in source_polygons:
        bkg_patch = MtPltPolygon(poly.exterior.coords, closed=True, edgecolor='blue', fill=False)
        ax.add_patch(bkg_patch)
    
    wcsaxes.add_beam(ax, header = header)
    wcsaxes.add_scalebar(ax, 0.00396162)

    ax.grid(color='white', ls='dotted')

    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    #plot and save fig    
#    plt.show()



def plot_sed(frequencies, flux_values, flux_errors, data):
    plt.figure(figsize=(10, 6))
    
    amp, alpha, damp ,dalpha, chi2red = WLLS(flux_values, frequencies, flux_errors)
    #plt.errorbar(frequencies, flux_values, yerr=flux_errors, fmt='o', label=f'Hex {hexagon.name}', capsize=5, markersize=5, color=hexagon.color)
    #fit_freqs = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), num=100)
    fit_fluxes = (10.0 **amp) * frequencies ** alpha
    
    plt.errorbar(frequencies, flux_values, yerr=flux_errors, fmt='o', capsize=3, label=f'alph = {alpha:.2f} ± {dalpha:.2f}, chi2red = {chi2red:.3f}')
    print(f"WLLS: alpha = {alpha:.2f} ± {dalpha:.2f}, amp = {10 ** amp:.2f} ± {10 ** damp:.2f}, chi2red = {chi2red:.3f}")
    plt.plot(frequencies, fit_fluxes, '-')

	#plot lobe1/2 or total 
    if data:
        filename = data
        S, f, Serr = read_data(filename)
        plt.plot(f,S, 'o', color='Black', label='Total')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Integrated Flux (Jy)')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
  
def WLLS(S1, f1, Serr1):

    f = np.log10(np.array(f1))
    S = np.log10(np.array(S1))
    Serr = np.array(Serr1) / np.array(S1)*np.log(10)

    X = np.column_stack((np.ones_like(f), f))
    W = np.diag(1/Serr**2)
    XtWX = np.linalg.inv(X.T @ W @ X)
    beta = XtWX @ X.T @ W @ S
    r = S - X @ beta
    df = len(f) - 2
    sumR = np.sum(r**2)
    sigma = sumR / df
    beta_var = sigma * XtWX
    stderr = np.sqrt(np.diag(beta_var))
    chi2red = (r.T @ W @ r) / df
    return beta[0], beta[1] , stderr[0], stderr[1], chi2red


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hexagonal grid flux measurement")
    parser.add_argument('--infits', type=str, required=True, help="Text file containing list of FITS files")
    parser.add_argument('--data', type=str, help="File containing total flux data")
    args = parser.parse_args()

    # Read in FITS files
    with open(args.infits, 'r') as file:
        fits_files = [line.strip() for line in file.readlines()]
    
    reg_file = 'regs/ll_flux.reg' #flux_reg
    bkg_file = 'regs/bkg_flux_reg.reg'
    

    frequencies = []
    int_fluxes = []
    dint_fluxes = []

    for f in fits_files:
        # Read in data
        data, wcs, header, pixd1, pixd2, barea, freq = extract_header_data(f)
        
        # Measure flux in each hexagon
        bkg_polygon, reg_polygon, rms_bkg, int_flux, dint_flux= measure_flux(header, data, abs(pixd2*pixd1), barea, reg_file, bkg_file)
        
        frequencies.append(freq)
        int_fluxes.append(int_flux)
        dint_fluxes.append(dint_flux)

        # Plotting
        plotPolygons(reg_polygon, wcs, header, data, bkg_polygon, rms_bkg) 
        print("\n")
    # Plot sed for each hex
    print(frequencies)
    plot_sed(frequencies, int_fluxes, dint_fluxes, args.data)
    
    with open("spectra.dat", "w") as file:
        file.write("Spectra,Frequency (Hz),Photometry (Jy),Uncertainty (Jy)\n")
        for freq, int_flux, dint_flux in zip(frequencies, int_fluxes, dint_fluxes):
            file.write(f"VLA,{freq},{int_flux},{dint_flux}\n")


