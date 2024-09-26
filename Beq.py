import numpy as np

def calculate_Bme(k, eta, z, alpha, theta_x, theta_y_s, s, phi, F0, nu_0, nu_1, nu_2):
    constant = 5.69e-5
    
    part1 = (1 + k) / eta
    part2 = (1 + z)**(3 - alpha)
    part3 = 1 / (theta_x * theta_y_s * s * (np.sin(phi)**(3/2)))
    part4 = F0 / (nu_0**alpha)
    part5 = (nu_2**(alpha + 1/2) - nu_1**(alpha + 1/2)) / (alpha + 1/2)
    
    Bme = constant * (part1 * part2 * part3 * part4 * part5)**(2/7)
    
    return Bme

k = 1
eta = 1
z = 0.032303145
alpha = -1.31
theta_min = 20  # in arcsec
theta_max = 36# in arcsec
s = theta_min * 4.671
phi = np.pi / 2  # rad
F0 =0.065902169 #in Jy
nu_0 = 8.486751  # in GHz 
nu_1 = 8.48  # in GHz
nu_2 = 11.5  # in GHz

# Calculate Bme
Bme_value = calculate_Bme(k, eta, z, alpha, theta_min, theta_max,s, phi, F0, nu_0, nu_1, nu_2)
print(f"B_me = {Bme_value*1e-4 * 1e+9} nT")
print(f"B_me = {Bme_value*1e-4} T")
