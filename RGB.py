import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt

data_R = fits.getdata('../fits/J01445_Combined_C_split_channel_block0.smalltest2.fits')
data_G = fits.getdata('../fits/J01445_Combined_C_split_channel_block1.smalltest2.fits')
data_B = fits.getdata('../fits/J01445_Combined_C_split_channel_block2.smalltest2.fits')
print('R data min:', np.min(data_R), 'max:', np.max(data_R))
print('G data min:', np.min(data_G), 'max:', np.max(data_G))
print('B data min:', np.min(data_B), 'max:', np.max(data_B))

# Normalize the data
data_R = (data_R - np.min(data_R)) / (np.max(data_R) - np.min(data_R))
data_G = (data_G - np.min(data_G)) / (np.max(data_G) - np.min(data_G))
data_B = (data_B - np.min(data_B)) / (np.max(data_B) - np.min(data_B))

stretch = 0.5  
Q = 10         

rgb_image = make_lupton_rgb(data_R, data_G, data_B, stretch=stretch, Q=Q)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(data_R, origin='lower', cmap='gray')
plt.title('R')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(data_G, origin='lower', cmap='gray')
plt.title('G')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(data_B, origin='lower', cmap='gray')
plt.title('B')
plt.colorbar()

plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(rgb_image, origin='lower')
plt.axis('off')
plt.show()
