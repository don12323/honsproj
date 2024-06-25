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

class Hexagon:
    def __init__(self, name, vertices,centre):
        self.name = name
        self.vertices = vertices
        self.centre = centre
        self.polygon = MtPltPolygon(vertices, closed=True, edgecolor='white', fill=False)

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
    print("**READING IN FITS FILE**")
    with fits.open(filename) as hdu:
	
        data = hdu[0].data	
        header = hdu[0].header
        naxis = header["NAXIS"]
        bmaj = header["BMAJ"]
        bmin = header["BMIN"]
        
        #find pixel size in degrees. Pixel area in degrees = (pixd1*pixd2)
        try:
            pixd1, pixd2 = header["CDELT1"], header["CDELT2"]
        except KeyError:
            pixd1, pixd2 = header["CD1_1"], header["CD2_2"]

        
        print("pixd1", pixd1, "pixd2", pixd2)
        barea = np.pi*bmaj*bmin/(4.0*np.log(2))
        npixpb = barea/(abs(pixd1*pixd2))
        
        print(f"NAXIS: {naxis}")
        print(f"Beam area: {barea}")
        print(f"Number of pixels per beam: {npixpb}")

        wcs = WCS(header, naxis= naxis)
    return data, wcs, header, pixd1, pixd2, barea

def polygons(reg_file, header, width):
    
    #read polygon region file and get pixel coordinates
    print("---------------------------------------------")
    print("**READING POLYGON REGION**")
    regions = pyregion.open(reg_file).as_imagecoord(header=header)
    poly_coords=[]
    for region in regions:
        for i in range(0, len(region.coord_list), 2):
            poly_coords.append((region.coord_list[i], region.coord_list[i+1])) 
    
    polygon = Polygon(poly_coords)
    #create bounding rectangle on polygon
    xmin, ymin, xmax, ymax = polygon.bounds
    #find hexagon centres that fit inside rectangle
    centres = []
    for y in np.arange(ymin-width,ymax, width):
        for x in np.arange(xmin-width*np.sqrt(3), xmax, width*np.sqrt(3)):
            centres.append((x,y))
            centres.append((x+width*np.sqrt(3)/2, y + width/2))

    #filter out centres that lie inside polygon
    new_centres = [point for point in centres if polygon.contains(Point(point))]
    print("Number of hexagons formed:", len(new_centres))

    #calculate hex vertices
    hexagons=[]
    for i, centre in enumerate(new_centres):
        name = f"{i+1}"
        hex_vertices = calc_hex_vert(centre, width)
        hexagon = Hexagon(name, hex_vertices, centre)
        if all(polygon.contains(Point(v)) for v in hex_vertices):
            hexagons.append(hexagon)
    
    return poly_coords, hexagons

    
def measure_flux(header, hexagons, data, pixarea, barea, bkg_file):
    #read in background polygon in image coordinates
    print("---------------------------------------------")
    print("**CALCULATING BACKGROUND FLUX**")
    bkg_regions = pyregion.open(bkg_file).as_imagecoord(header=header)
    bkg_coords = []
    for region in bkg_regions:
        for i in range(0, len(region.coord_list), 2):
            bkg_coords.append((region.coord_list[i], region.coord_list[i+1]))

    bkg_polygon = Polygon(bkg_coords)
    
    #calculate the mean background flux
    npix_bkg=0
    bkg_flux_values = []
    bkg_pixels = []
    min_x, min_y, max_x, max_y = bkg_polygon.bounds
    for y in range(math.ceil(min_y)-1, math.floor(max_y)+1):
        for x in range(math.ceil(min_x)-1, math.floor(max_x)+1):
            #corners
            if bkg_polygon.contains(Point(x,y)):
                bkg_flux_values.append(data[y-1,x-1])
                npix_bkg+=1
                bkg_pixels.append((x, y))

    
    print("\n")
    bkg_flux = np.mean(bkg_flux_values)
    std_bkg = np.std(bkg_flux_values)

    print("mean bkg flux:", bkg_flux, "Jy/beam. Number of pixels in background: ",npix_bkg) 
    print("std bkg flux:", std_bkg, "Jy/beam.")
    
    #calculate integrated flux for each hexagon
    print("---------------------------------------------")
    print("**CALCULATING FLUXES FOR HEX**")
    
    total_fluxes = []
    uncertainties = []
    for hexagon in hexagons:
        hex_polygon = Polygon(hexagon.vertices)
        total_flux_in_hex = 0
        npix_hex = 0
        flux_squared = []
        
        min_x, min_y, max_x, max_y = hex_polygon.bounds
        #print("minx", min_x, "miny", min_y, "maxx",max_x, "maxy",max_y)
    
        for y in range(math.ceil(min_y), math.floor(max_y)):
            for x in range(math.ceil(min_x), math.floor(max_x)):
                if hex_polygon.contains(Point(x,y)):
                    total_flux_in_hex += data[y-1,x-1]
                    npix_hex += 1
                    flux_squared.append(data[y-1,x-1]**2)


        #calculate the integrated flux
        print(f"Number of pixels in hex {hexagon.name}: {npix_hex} Aperture size: {npix_hex*pixarea}")
        int_flux = (total_flux_in_hex * pixarea / barea) - (bkg_flux * npix_hex * pixarea / barea) #bkg_flux*(nbeams inside hex)
        total_fluxes.append(int_flux)
        #uncertainties
        rms = np.sqrt(np.mean(np.array(flux_squared)))
        uncertainties.append(np.sqrt(rms**2 + (0.02 * int_flux)**2 + (std_bkg * npix_hex * pixarea / barea)**2))
    for hexagon, flux, uncertainty in zip(hexagons, total_fluxes, uncertainties):
        print(f"Hexagon {hexagon.name}: Integrated Flux = {flux} +- {uncertainty}Jy, ")

    return bkg_polygon
 

def plotPolygons(region, hexagons, wcs, data, bkg_polygon): 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1,projection=wcs)
    im=ax.imshow(data, cmap='gray')

    ax.add_patch(region)

    for hexagon in hexagons:
        ax.add_patch(hexagon.polygon)
        ax.annotate(hexagon.name, xy=hexagon.centre, ha='center', va='center', color='white')   
    
    #plot background
    bkg_patch = MtPltPolygon(bkg_polygon.exterior.coords, closed=True, edgecolor='blue', fill=False)
    ax.add_patch(bkg_patch)

    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    #plot and save fig    
    plt.savefig("hex_grid_1.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hexagonal grid flux measurement")
    parser.add_argument('width', type=float, help="Width of the hexagons")
    args = parser.parse_args()
    width = args.width
    #fits_files = ['../J01445/J01445_Combined_C_split_channel_block0.smalltest2.fits']
    #reg_file = 'toplobe.reg'
    #bkg_file = 'bkg.reg'
    fits_files = ['../data/3c391_ctm_spw0_multiscale_fixed.fits']
    reg_file = '../data/reg1_deg_icrs.reg'
    bkg_file = 'bkg_test.reg'



    for f in fits_files:

        data, wcs, header, pixd1, pixd2, barea = extract_header_data(f)
        poly_pix, hexagons = polygons(reg_file, header, width)
        #measure flux in each hexagon
        bkg_polygon = measure_flux(header, hexagons, data, abs(pixd2*pixd1), barea, bkg_file)
        
        #plotting
        region = MtPltPolygon(poly_pix, closed=True, edgecolor='r', linewidth=1, fill=False)
        plotPolygons(region, hexagons, wcs, data, bkg_polygon) 





