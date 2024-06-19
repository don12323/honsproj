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


class coords:
    def __init__(self):
        self.coords = []


class Hexagon:
    def __init__(self, name, vertices,centre):
        self.name = name
        self.vertices = vertices
        self.centre = centre
        self.polygon = MtPltPolygon(vertices, closed=True, edgecolor='white', fill=False)

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
        xmax = header["NAXIS1"]
        ymax = header["NAXIS2"]
        
        #find pixel size

        try:
            pixd1, pixd2 = header["CDELT1"], header["CDELT2"]
        except KeyError:
            pixd1, pixd2 = header["CD1_1"], header["CD2_2"]


        barea = np.pi*bmaj*bmin/(4.0*np.log(2))

        npixpb = barea/(abs(pixd1*pixd2))

        print(npixpb)

        wcs = WCS(header, naxis= naxis)
         
        #impix_coord = wcs.all_world2pix(ra,dec,0)
	 
        #plt.imshow(data,cmap='gray')
        #plt.colorbar()  # Add colorbar for intensity scale
        #plt.savefig('adsd.png')
            
    return xmax, ymax, data, wcs, header

def polygons(reg_file, wcs, header, width):

    poly_pix = coords()
    poly_idx = coords()
    
    
    x = []
    y = []
    #read region file and get pixel coordinates
    regions = pyregion.open(reg_file).as_imagecoord(header=header)
    for region in regions:
        policoords = region.coord_list
        for i in range(0, len(policoords), 2):
            x.append(float(policoords[i]))
            y.append(float(policoords[i+1]))
        
    poly_pix.coords = [(x[i], y[i]) for i in range(len(x))]
    poly_idx.coords = [(int(x[i]), int(y[i])) for i in range(len(x))]
    
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

    #calculate hex vertices
    hexagons=[]
    for i, centre in enumerate(new_centres):
        name = f"{i+1}"
        hex_vertices = calc_hex_vert(centre, width)
        hexagon = Hexagon(name, hex_vertices, centre)
        hexagons.append(hexagon)
    
    return poly_pix, poly_idx, hexagons

    
#def measure_flux(hexagons, data, pixare, barea):
    
   # for index in hexagon:
        #total_flux_in_hex += data[index]

    #total_flux = total_flux_in_hex*pixarea/barea

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
    
    #save fig    
    plt.savefig("hex_grid_2.png")

if __name__ == "__main__":
    
    fits_files = ['3c391_ctm_spw0_multiscale_fixed.fits']
    reg_file = 'reg1_deg_icrs.reg'
    width = 20

    for f in fits_files:

        xmax, ymax, data, wcs, header = extract_header_data(f)
        poly_pix, poly_idx, hexagons = polygons(reg_file, wcs, header, width)
        region = MtPltPolygon(poly_pix.coords, closed=True, edgecolor='r', linewidth=1, fill=False)
        plotPolygons(region, hexagons, wcs, data) 


        



        #for i in range(0,len(X)):

         #   print(X[i],Y[i])

    



