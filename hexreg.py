import matplotlib
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

import warnings
import sys
import csv

#ignore warnings from wcs
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

class Hexagon:
    color_list = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#33FFF1', '#F1FF33', '#F133FF']
    def __init__(self, name):
        self.name = name
        self.vertices = None
        self.centre = None
        self.fluxes = []
        self.color = Hexagon.color_list.pop(0) if Hexagon.color_list else "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def set_attributes(self, vertices, centre):
        self.vertices = vertices
        self.centre = centre
    def add_flux(self, flux, uncertainty):
        self.fluxes.append((flux,uncertainty))

#function for calculating hexagon vertices
def calc_hex_vert(centre, width):
    vertices = []
    r = width*(1/np.sqrt(3))*0.99
    for n in range(0,6,1):
        x = centre[0] + r*np.cos(n*2*np.pi/6)
        y = centre[1] + r*np.sin(n*2*np.pi/6)
        vertices.append((x,y))
    return vertices

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
        try:
            freq = header["RESTFREQ"]
        except KeyError:
            freq = header["RESTFRQ"]

        print(f"  {Colors.OKGREEN} FREQ: {freq}Hz {Colors.ENDC}")
        if naxis != 2:
            print(f"{Colors.FAIL}Error: NAXIS is not equal to 2 for file: {filename}{Colors.ENDC}")
            sys.exit(1)
        
        #find pixel size in degrees. Pixel area in degrees = (pixd1*pixd2)
        try:
            pixd1, pixd2 = header["CDELT1"], header["CDELT2"]
        except KeyError:
            pixd1, pixd2 = header["CD1_1"], header["CD2_2"]

        print(f'   Pixel size: {abs(pixd1*3600):.2f}" x {pixd2*3600:.2f}"') 
        
        barea = np.pi*bmaj*bmin/(4.0*np.log(2))
        npixpb = barea/(abs(pixd1*pixd2))
        print(f"   Beam area: {barea*3600**2:.3f} arcsec^2")        
        print(f"   Number of pixels per beam: {npixpb}\n")

        wcs = WCS(header, naxis= naxis)
    return data, wcs, header, pixd1, pixd2, barea, freq

def polygons(reg_file, header, width, pix1, hexagons):
    
    #read polygon region file and get pixel coordinates
    regions = pyregion.open(reg_file).as_imagecoord(header=header)
    poly_coords=[]
    for region in regions:
        for i in range(0, len(region.coord_list), 2):
            poly_coords.append((region.coord_list[i], region.coord_list[i+1])) 
    
    polygon = Polygon(poly_coords)
    #create bounding rectangle on polygon
    xmin, ymin, xmax, ymax = polygon.bounds
    #convert width from arcsec to pixel units
    width = width / (abs(pixd1) * 3600)
    #find hexagon centres that fit inside rectangle
    centres = []
    for y in np.arange(ymin-width,ymax, width):
        for x in np.arange(xmin-width*np.sqrt(3), xmax, width*np.sqrt(3)):
            centres.append((x,y))
            centres.append((x+width*np.sqrt(3)/2, y + width/2))

    #filter out centres that lie inside polygon
    centres = [point for point in centres if polygon.contains(Point(point))]
    #filter out centres that have vertices outside polygon boundaries
    new_centres = []
    for centre in centres:
        hex_vertices = calc_hex_vert(centre, width)
        if all(polygon.contains(Point(v)) for v in hex_vertices):
            new_centres.append(centre)

    #form hexagons/update their attributes
    if hexagons is None:
        hexagons=[]
        i=0
        for centre in new_centres:
            hex_vertices = calc_hex_vert(centre, width)
            i+=1
            name = f"{i}"
            hexagon = Hexagon(name)
            hexagon.set_attributes(hex_vertices, centre)
            hexagons.append(hexagon)
    else:
        for hexagon, centre in zip (hexagons, new_centres):
            hex_vertices = calc_hex_vert(centre, width)
            hexagon.set_attributes(hex_vertices, centre)

    
    print(f"   {Colors.OKGREEN}Number of hexagons formed:{len(hexagons)} {Colors.ENDC}\n")
    return poly_coords, hexagons

    
def measure_flux(header, hexagons, data, pixarea, barea, bkg_file):
    #read in background polygon in image coordinates
    bkg_regions = pyregion.open(bkg_file).as_imagecoord(header=header)
    #print(bkg_regions)
    source_polygons = []
    for region in bkg_regions:
        poly_coords = []
        for i in range(0, len(region.coord_list), 2):
            poly_coords.append((region.coord_list[i], region.coord_list[i + 1]))
        source_polygons.append(Polygon(poly_coords))


    #calculate the mean background flux
    npix_bkg=0
    bkg_flux_values = []
    bkg_pixels = []
    bkg_squared = []
    max_x, max_y = data.shape[1], data.shape[0]
    min_x, min_y = 1, 1
    for y in range(math.ceil(min_y), math.floor(max_y)):
        for x in range(math.ceil(min_x), math.floor(max_x)):
            point = Point(x,y)
            #corners
            if all(not poly.contains(point) for poly in source_polygons) and not np.isnan(data[y - 1, x - 1]):
             #   print(x,y) 
                bkg_flux_values.append(data[y-1,x-1])
                npix_bkg+=1
                bkg_pixels.append((x, y))
                bkg_squared.append(data[y-1,x-1]**2)

    bkg_flux = np.mean(bkg_flux_values)
    std_bkg = np.std(bkg_flux_values)
    rms_bkg = np.sqrt(np.mean(np.array(bkg_squared)))
    print(f"   {Colors.OKGREEN}Background mean flux: {bkg_flux} +\- {std_bkg} Jy/beam{Colors.ENDC}")
    print("   Number of pixels in background: ",npix_bkg) 
    print(f"rms_bkg: {rms_bkg}")
    #calculate integrated flux for each hexagon
    print("\n")
    print(f"   {Colors.UNDERLINE}CALCULATING FLUXES FOR HEX{Colors.ENDC}")
    
    total_fluxes = []
    uncertainties = []
    for hexagon in hexagons:
        hex_polygon = Polygon(hexagon.vertices)
        total_flux_in_hex = 0
        npix_hex = 0
        flux_squared = []
        
        min_x, min_y, max_x, max_y = hex_polygon.bounds
        #print("minx", min_x, "miny", min_y, "maxx",max_x, "maxy",max_y)
    
        for y in range(math.ceil(min_y)-1, math.floor(max_y)+1):
            for x in range(math.ceil(min_x)-1, math.floor(max_x)+1):
                if hex_polygon.contains(Point(x,y)):
                    total_flux_in_hex += data[y-1,x-1]
                    npix_hex += 1
                    flux_squared.append(data[y-1,x-1]**2)


        #calculate the integrated flux
        print(f"   Number of pixels in hex {hexagon.name}: {npix_hex} Aperture size: {npix_hex*pixarea*3600**2:.2f} arcsec^2")
        int_flux = (total_flux_in_hex * pixarea / barea) #-TODO (bkg_flux * npix_hex * pixarea / barea) #bkg_flux*(nbeams inside hex)
        total_fluxes.append(int_flux)
        #uncertainties
        rms = np.sqrt(np.mean(np.array(flux_squared)))
        uncertainty = np.sqrt(rms**2 + (0.02 * int_flux)**2 + (std_bkg * npix_hex * pixarea / barea)**2) #TODO **IMPORTANT**check with supv if error is associated with each pix or final flux
        uncertainties.append(uncertainty)
        #add flux to hex
        hexagon.add_flux(int_flux,uncertainty)

    for hexagon, flux, uncertainty in zip(hexagons, total_fluxes, uncertainties):
        print(f"   Hexagon {hexagon.name}: Integrated Flux = {flux*10**3:.4f} +\- {uncertainty*10**3:.6f} mJy, ")

    return source_polygons, rms_bkg, bkg_pixels
 

def plotPolygons(region, hexagons, wcs, data, source_polygons, rms, bkg_pixels): 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1,projection=wcs)
    im=ax.imshow(data, cmap='gray')

    #contours
    contour_levels = rms * np.array([3,6,9]) #[-3, 3, 6, 9]
    contours = ax.contour(data, levels=contour_levels, colors='white', linewidths=0.5, linestyles='dashed')
    ax.add_patch(region)

    for hexagon in hexagons:
        patch = MtPltPolygon(hexagon.vertices, closed=True, edgecolor=hexagon.color, fill=False)
        ax.add_patch(patch)
        ax.annotate(hexagon.name, xy=hexagon.centre, ha='center', va='center', color=hexagon.color)   
    
    #plot background
    for poly in source_polygons:
        bkg_patch = MtPltPolygon(poly.exterior.coords, closed=True, edgecolor='blue', fill=False)
        ax.add_patch(bkg_patch)
    
    #plot bkgpixels
    #bkg_x, bkg_y = zip(*bkg_pixels)
   # ax.plot(bkg_x, bkg_y, 'ro', markersize=1)
    ax.grid(color='white', ls='dotted')

    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    #plot and save fig    
    plt.savefig("hex_grid_1.png")
#    plt.show()



def plot_sed(hexagons, frequencies):
    plt.figure(figsize=(10, 6))
    for hexagon in hexagons:
        flux_values = []
        flux_errors = []
        for (flux, error) in hexagon.fluxes:
            flux_values.append(flux)
            flux_errors.append(error)
        amp, alpha, damp ,dalpha = WLLS(flux_values, frequencies, flux_errors)
        plt.plot(frequencies, flux_values, 'o', label=f'Hex {hexagon.name}', color=hexagon.color)
        #plt.errorbar(frequencies, flux_values, yerr=flux_errors, fmt='o', label=f'Hex {hexagon.name}', capsize=5, markersize=5, color=hexagon.color)
        #fit_freqs = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), num=100)
        fit_fluxes = (10.0 **amp) * frequencies ** alpha
        chi2 = np.sum(((flux_values - fit_fluxes) / flux_errors) ** 2)
        chi2red = chi2 / (len(frequencies) -2)
        print(f"WLLS: Hex {hexagon.name}: alpha = {alpha:.2f} ± {dalpha:.2f}, amp = {10 ** amp:.2f} ± {10 ** damp:.2f}, chi2red = {chi2red:.3f}")
        plt.plot(frequencies, fit_fluxes, '-', color=hexagon.color)
   
	#plot lobe1/2 
    filename = "J01445_Lobe2.dat"
    S, f, Serr = read_data(filename)
    plt.plot(f,S, 'o', color='Black')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Integrated Flux (Jy)')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("sed_plot.png")
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
    return beta[0], beta[1] , stderr[0], stderr[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hexagonal grid flux measurement")
    parser.add_argument('--width', type=float, required=True, help="Width of the hexagons in arcseconds")
    parser.add_argument('--infits', type=str, required=True, help="Text file containing list of FITS files")
    args = parser.parse_args()
    width = args.width

    #read in FITS files
    with open(args.infits, 'r') as file:
        fits_files = [line.strip() for line in file.readlines()]
    
    reg_file = 'botlobe.reg'
    bkg_file = 'bkg.reg'
    

    frequencies = []
    hexagons=None

    for f in fits_files:

        data, wcs, header, pixd1, pixd2, barea, freq = extract_header_data(f)
        poly_pix, hexagons = polygons(reg_file, header, width, pixd1, hexagons)
        #measure flux in each hexagon
        bkg_polygon, rms_bkg, bkg_pixels = measure_flux(header, hexagons, data, abs(pixd2*pixd1), barea, bkg_file)
        
        frequencies.append(freq)

        #plotting
        region = MtPltPolygon(poly_pix, closed=True, edgecolor='r', linewidth=1, fill=False)
        plotPolygons(region, hexagons, wcs, data, bkg_polygon, rms_bkg, bkg_pixels) 
        print("\n")
    #plot sed for each hex
    print(frequencies)
    plot_sed(hexagons, frequencies)

