import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.patches import RegularPolygon
from matplotlib.patches import Polygon as MtPltPolygon
import pyregion
from shapely.geometry import Point, Polygon

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

#list to store x and y coordinates
class coords:
    def __init__(self):
        self.coords = []


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
        #find beam area
        barea = np.pi*bmaj*bmin/(4.0*np.log(2))
        #pixels per beam
        npixpb = barea/(abs(pixd1*pixd2))

        print("number of pixels per beam:", npixpb)

        wcs = WCS(header, naxis= naxis)
    return data, wcs, header, pixd1, pixd2, barea

def polygons(reg_file, header, width):
    
    poly_pix = coords()
    x = []
    y = []
    #read polygon region file and get pixel coordinates
    regions = pyregion.open(reg_file).as_imagecoord(header=header)
    for region in regions:
        print("scanning region")
        policoords = region.coord_list
        for i in range(0, len(policoords), 2):
            x.append(float(policoords[i]))
            y.append(float(policoords[i+1]))
        
    poly_pix.coords = [(x[i], y[i]) for i in range(len(x))]
    print("printing pixels for polygon region")
    print(poly_pix.coords)
    #create bounding rectangle on polygon
    xmin = xmax = poly_pix.coords[0][0]
    ymin = ymax = poly_pix.coords[0][1]
    
    for coord in poly_pix.coords:
        x, y = coord
        xmin = min(xmin,x)
        xmax = max(xmax,x)
        ymin = min(ymin,y)
        ymax = max(ymax,y)

    #find hexagon centres that fit inside rectangle
    centres = []
    for y in np.arange(ymin-width,ymax, width):
        for x in np.arange(xmin-width*np.sqrt(3), xmax, width*np.sqrt(3)):
            centres.append((x,y))
            centres.append((x+width*np.sqrt(3)/2, y + width/2))

    #filter out centres that lie inside polygon
    polygon = Polygon(poly_pix.coords)
    new_centres = [point for point in centres if polygon.contains(Point(point))]
    print("\n") 
    print("centres for hexagons", new_centres)

    #calculate hex vertices
    hexagons=[]
    for i, centre in enumerate(new_centres):
        name = f"{i+1}"
        hex_vertices = calc_hex_vert(centre, width)
        hexagon = Hexagon(name, hex_vertices, centre)
        hexagons.append(hexagon)
    
    return poly_pix, hexagons

    
def measure_flux(header, hexagons, data, pixarea, barea, bkg_file):
    #read in background polygon in image coordinates
    bkg_regions = pyregion.open(bkg_file).as_imagecoord(header=header)
    bkg_coords = []
    for region in bkg_regions:
        for i in range(0, len(region.coord_list), 2):
            bkg_coords.append((region.coord_list[i], region.coord_list[i+1]))

        #bkg_coords.extend([(region.coord_list[i], region.coord_list[i + 1]) for i in range(0, len(region.coord_list), 2)])
                        
    bkg_polygon = Polygon(bkg_coords)
    
    #calculate the mean background flux
    bkg_flux_values = []
    min_x, min_y, max_x, max_y = bkg_polygon.bounds
    print("minx", min_x, "miny", min_y, "maxx",max_x, "maxy",max_y)
    # loop to collect bkg flux values within polygon
    for y in range(math.ceil(min_y), math.floor(max_y)):
        for x in range(math.ceil(min_x), math.floor(max_x)):
            if bkg_polygon.contains(Point(x,y)):
                bkg_flux_values.append(data[y-1,x-1])
    
    print("\n")
    print("background flux values:", bkg_flux_values)
    bkg_flux = np.mean(bkg_flux_values)
    print("mean bkg flux Jy/beam", bkg_flux) 

   


 

def plotPolygons(region, hexagons, wcs, data): 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1,projection=wcs)
    im=ax.imshow(data, cmap='gray')

    ax.add_patch(region)

    for hexagon in hexagons:
        ax.add_patch(hexagon.polygon)
        ax.annotate(hexagon.name, xy=hexagon.centre, ha='center', va='center', color='white')   
    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    #plot and save fig    
    plt.savefig("hex_grid_2.png")

if __name__ == "__main__":
    
    fits_files = ['../data/3c391_ctm_spw0_multiscale_fixed.fits']
    reg_file = '../data/reg1_deg_icrs.reg'
    bkg_file = 'bkg_test.reg'
    width = 30

    for f in fits_files:

        data, wcs, header, pixd1, pixd2, barea = extract_header_data(f)
        poly_pix, hexagons = polygons(reg_file, header, width)
        #measure flux in each hexagon
        measure_flux(header, hexagons, data, pixd2*pixd1, barea, bkg_file)
        #plotting
        region = MtPltPolygon(poly_pix.coords, closed=True, edgecolor='r', linewidth=1, fill=False)
        plotPolygons(region, hexagons, wcs, data) 


        



        #for i in range(0,len(X)):

         #   print(X[i],Y[i])

    



