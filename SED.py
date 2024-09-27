#!/usr/bin/env python3

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['serif']})



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
            f.append(float(row[1]) / 1e9)
            S.append(float(row[2]))
            Serr.append(float(row[3]))
   
    return np.array(S), np.array(f), np.array(Serr)

def splitWLLS(S, f, Serr, bfreq):
    lower = np.where(f <= bfreq) 
    upper = np.where(f >= bfreq)

    f_lower, S_lower, Serr_lower  = f[lower], S[lower], Serr[lower] 
    f_upper, S_upper, Serr_upper  = f[upper], S[upper], Serr[upper]

    beta_lower, Stderr_lower = WLLS(S_lower, f_lower, Serr_lower)
    beta_upper, Stderr_upper = WLLS(S_upper, f_upper, Serr_upper)
   
    return beta_lower, Stderr_lower, beta_upper, Stderr_upper, lower, upper

def WLLS(S1, f1, Serr1):
    f = np.log10(f1)
    S = np.log10(S1)
    Serr = Serr1 / S1*np.log(10)

    X = np.column_stack((np.ones_like(f), f))
    

    W = np.diag(1 / Serr**2)
    
    XtWX = np.linalg.inv(X.T @ W @ X)
    beta = XtWX @ X.T @ W @ S
    r = S - X @ beta
    df = len(f) - 2
    sumR = np.sum(r**2)
    sigma = sumR / df
    beta_var = sigma * XtWX
    stderr = np.sqrt(np.diag(beta_var))

    return beta, stderr


    
def plot_sed(S, f, Serr, input_file, output_dir, beta_lower, Stderr_lower, beta_upper, Stderr_upper, lower, upper):
    
    #fitted curve
    S_lower =(10.0 ** beta_lower[0]) * f[lower] ** beta_lower[1]
    S_upper=(10.0 ** beta_upper[0]) * f[upper] ** beta_upper[1]
    
    # Plot measured data
    alpha_low= f"{beta_lower[1]:.3f}"+ f"\pm {Stderr_lower[1]:.3f}"
    alpha_upper=f"{beta_upper[1]:.3f}"+ f"\pm {Stderr_upper[1]:.3f}"

    label_low = r'$\alpha_{'+str(int(np.min(f[lower])*1e3))+'MHz}^{'+str(int(np.max(f[lower])*1e3))+'MHz}='+ alpha_low+r'$'
    #label_low = r'$\alpha_{\scriptsize'+str(int(np.min(f[lower])*1e3))+'MHz}^{\tiny'+str(int(np.max(f[lower])*1e3))+'MHz}='+ alpha_low+r'$'

    label_up = r'$\alpha_{'+str(int(np.min(f[upper])))+'GHz}^{'+str(int(np.max(f[upper])))+'GHz}='+ alpha_upper + r'$'
    
    filename = (os.path.basename(input_file)).replace('.csv', '_sed').replace('.dat', '_sed').replace('.txt','_sed')

    plt.errorbar(f[:8], S[:8], yerr=Serr[:8], fmt='o', color='black', markersize=4, label='NVSS/TGSS/GLEAM/RACS')
    plt.errorbar(f[8:], S[8:], yerr=Serr[8:], fmt='o', color='blue', markersize=4, label='VLA')
    plt.plot(f[lower], S_lower, label=label_low, linestyle='-', linewidth=0.8, color='red')
    plt.plot(f[upper], S_upper, label=label_up,linestyle='-', linewidth=0.8,color ='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Frequency (GHz)')
    plt.ylabel(r'Flux Density (Jy)')
    plt.title(filename)
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    
    output_filename = input_file.replace('.csv', '_sed.png').replace('.dat', '_sed.png').replace('.txt','_sed.png')
    #plt.savefig(os.path.join(output_dir, output_filename)i,dpi=300)
    plt.savefig('fit'+output_filename)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect argument")
        sys.exit(1)
    
    input_file = sys.argv[1]  # Path to the input text file
    S, f, Serr = read_data(input_file)
    
    
    output_dir = '/data/donwijesinghe/results'   # Path to the directory where the plot will be saved

    beta_lower, Stderr_lower, beta_upper, Stderr_upper, lower, upper = splitWLLS(S, f, Serr, 0.2)
    plot_sed(S, f, Serr, input_file, output_dir,beta_lower, Stderr_lower, beta_upper, Stderr_upper, lower, upper)














