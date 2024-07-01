import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

from matplotlib.patches import RegularPolygon
from matplotlib.patches import Polygon as MtPltPolygon
import pyregion
from shapely.geometry import Point, Polygon

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from sf.synchrofit import spectral_fitter, spectral_model

import warnings
import sys
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
    def __init__(self, name):
        self.name = name
        self.vertices = None
        self.centre = None
        self.fluxes = []
    def set_attributes(self, vertices, centre):
        self.vertices = vertices
        self.centre = centre
    def add_flux(self, flux, uncertainty):
        self.fluxes.append((flux,uncertainty))

#function for calculating hexagon vertices
def calc_hex_vert(centre, width):
    vertices = []
    r = width*(1/np.sqrt(3))
    for n in range(0,6,1):
        x = centre[0] + r*np.cos(n*2*np.pi/6)
        y = centre[1] + r*np.sin(n*2*np.pi/6)
        vertices.append((x,y))
    return vertices

def extract_header_data(filename):
    print(f"{Colors.OKCYAN}>> Reading in FITS file: {filename}{Colors.ENDC} \n")
    with fits.open(filename) as hdu:
	
        data = hdu[0].data	
        header = hdu[0].header
        naxis = header["NAXIS"]
        bmaj = header["BMAJ"]
        bmin = header["BMIN"]
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
        print(f"   Beam area: {barea*3600**2:.2f} arcsec^2")        
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

    print(source_polygons)

    #calculate the mean background flux
    npix_bkg=0
    bkg_flux_values = []
    bkg_pixels = []
    min_x, min_y = data.shape[1], data.shape[0]
    max_x, max_y = 1, 1
    print(min_x,min_y,max_x,max_y)
    for y in range(min_y, max_y):
        print(y)
        for x in range(min_x, max_x):
            #point = Point(x,y)
            print(x,y)
            #corners
            #if all(not poly.contains(point) for poly in source_polygons) and not np.isnan(data[y - 1, x - 1]):
             #   print(x,y) 
             #   bkg_flux_values.append(data[y-1,x-1])
             #   npix_bkg+=1
             #   bkg_pixels.append((x, y))

    bkg_flux = np.mean(bkg_flux_values)
    std_bkg = np.std(bkg_flux_values)

    print(f"   {Colors.OKGREEN}Background mean flux: {bkg_flux} +\- {std_bkg} Jy/beam{Colors.ENDC}")
    print("   Number of pixels in background: ",npix_bkg) 
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
        int_flux = (total_flux_in_hex * pixarea / barea) - (bkg_flux * npix_hex * pixarea / barea) #bkg_flux*(nbeams inside hex)
        total_fluxes.append(int_flux)
        #uncertainties
        rms = np.sqrt(np.mean(np.array(flux_squared)))
        uncertainty = np.sqrt(rms**2 + (0.02 * int_flux)**2 + (std_bkg * npix_hex * pixarea / barea)**2)
        uncertainties.append(uncertainty)
        #add flux to hex
        hexagon.add_flux(int_flux,uncertainty)

    for hexagon, flux, uncertainty in zip(hexagons, total_fluxes, uncertainties):
        print(f"   Hexagon {hexagon.name}: Integrated Flux = {flux*10**3:.4f} +\- {uncertainty*10**3:.6f} mJy, ")

    return source_polygons
 

def plotPolygons(region, hexagons, wcs, data, source_polygons): 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1,projection=wcs)
    im=ax.imshow(data, cmap='hot')

    ax.add_patch(region)

    for hexagon in hexagons:
        patch = MtPltPolygon(hexagon.vertices, closed=True, edgecolor='white', fill=False)
        ax.add_patch(patch)
        ax.annotate(hexagon.name, xy=hexagon.centre, ha='center', va='center', color='white')   
    
    #plot background
    for poly in source_polygons:
        bkg_patch = MtPltPolygon(bkg_polygon.exterior.coords, closed=True, edgecolor='blue', fill=False)
        ax.add_patch(bkg_patch)
    
    ax.grid(color='white', ls='dotted')

    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    #plot and save fig    
    plt.savefig("hex_grid_1.png")
    plt.show()



def plot_sed(hexagons, frequencies):
    for hexagon in hexagons:
        flux_values = []
        flux_errors = []
        for (flux, error) in hexagon.fluxes:
            flux_values.append(flux)
            flux_errors.append(error)

        plt.errorbar(frequencies, flux_values, yerr=flux_errors, fmt='o', label=f'Hex {hexagon.name}')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Integrated Flux (Jy)')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig("sed_plot.png")
    plt.show()

def fit_spectra(hexagons, frequencies):
    results = []
    for hexagon in hexagons:
        flux_values = [flux for flux, _ in hexagon.fluxes]
        flux_errors = [error for _, error in hexagon.fluxes]

        params = spectral_fitter(
                frequency=frequencies,
                luminosity=flux_values,
                dluminosity=flux_errors,
                fit_type='TJP',
                n_breaks=31,
                break_range=[8, 11],
                n_injects=31,
                inject_range=[2.01, 2.99],
                n_iterations=3,
                b_field=0.35e-9,
                redshift=0.1
        )
        results.append(params)
        print(f"Hexagon {hexagon.name}: Injection Index = {params[3]} +/- {params[4]}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hexagonal grid flux measurement")
    parser.add_argument('--width', type=float, required=True, help="Width of the hexagons in arcseconds")
    parser.add_argument('--infits', type=str, required=True, help="Text file containing list of FITS files")
    args = parser.parse_args()
    width = args.width

    #read in FITS files
    with open(args.infits, 'r') as file:
        fits_files = [line.strip() for line in file.readlines()]
    
    reg_file = 'toplobe.reg'
    bkg_file = 'bkg.reg'
    
    #reg_file = '../data/reg1_deg_icrs.reg'
    #bkg_file = 'bkg_test.reg'

    frequencies = []
    hexagons=None

    for f in fits_files:

        data, wcs, header, pixd1, pixd2, barea, freq = extract_header_data(f)
        poly_pix, hexagons = polygons(reg_file, header, width, pixd1, hexagons)
        #measure flux in each hexagon
        bkg_polygon = measure_flux(header, hexagons, data, abs(pixd2*pixd1), barea, bkg_file)
        
        frequencies.append(freq)

        #plotting
        region = MtPltPolygon(poly_pix, closed=True, edgecolor='r', linewidth=1, fill=False)
        plotPolygons(region, hexagons, wcs, data, bkg_polygon) 
        print("\n")
    #plot sed for each hex
    print(frequencies)
    plot_sed(hexagons, frequencies)
    
    # Fit spectra for each hexagon and calculate injection index
    #fit_results = fit_spectra(hexagons, frequencies)
    #for result in fit_results:
#        print(result)




