#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt

from sf.synchrofit import spectral_fitter, spectral_model_, spectral_ages
plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

# Define the function to read the .dat file
def read_dat_file(file_name):
    data = pd.read_csv(file_name, delimiter=',')
    obsname = data['Spectra'].values
    frequency = data['Frequency (Hz)'].values
    photometry = data['Photometry (Jy)'].values
    uncertainty = data['Uncertainty (Jy)'].values


    return obsname, frequency, photometry, uncertainty


def plot_spectrum(observed_data, obsname, model_data, plotting_data, fit_type, params, ages):
    frequency, photometry, uncertainty = map(np.asarray, observed_data)
    plotting_frequency, plotting_data, err_plotting_data, plotting_data_min, plotting_data_max = map(np.asarray, plotting_data)
    model_data = np.asarray(model_data)
    
    # Get params
    v_b = (10**params[1]) / 1e9 # Ghz
    dv_b = (10**params[1])*(np.log(10)*params[2]) / 1e9 
    inj_index = (params[3] - 1.0) / 2.0
    dinj_index = params[4] / 2.0
    t_on = ages[1]
    dt = params[6] * ages[0]
    t_off = ages[2]
    
    
    fig, ax =plt.subplots(figsize=(8, 8))

    unq_obsname = np.unique(obsname)
    for name in unq_obsname:
        idx = obsname == name
        handle = ax.errorbar(frequency[idx], photometry[idx], yerr=uncertainty[idx], capsize=0.5, label=name, fmt='o', alpha=0.9)

    # Plot model dat
    model_fit = ax.plot(plotting_frequency, plotting_data, 'k-', label='{}-off Model fit'.format(fit_type))
    error_fit = ax.fill_between(plotting_frequency, plotting_data_min, plotting_data_max, color='blue', alpha=0.2, label='Model error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.2 * np.min(photometry), 1.5 * np.max(photometry))
    ax.set_xlim(np.min(plotting_frequency), np.max(plotting_frequency))
    ax.set_xlabel(r'Frequency (Hz)')
    ax.set_ylabel(r'Flux Density (Jy)')
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=2, pad=5)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
 #   ax.grid(True)
    str2 = r'$\alpha_{inj} = $'
    str1 = r'$\nu_b$ = '
    str3 = r'$t_{on}$ = '
    str4 = r'$t_{off}$ = '

    legend_text = (str1+f'{v_b:.2f} ± {dv_b:.2f} GHz\n'+
                   str2+f'{inj_index:.3f} ± {dinj_index:.3f}\n'+
                   str3+f'{t_on:.2f} ± {dt:.2f} Myr\n'+
                   str4+f'{t_off:.2f} ± {dt:.2f} Myr')
    #blank = ax.plot([], [], ' ', label=legend_text)
    first_legend = ax.legend(handles=[plt.Line2D([], [], color='none', label=legend_text)], loc='lower left', handlelength=0, fontsize=11, frameon=False)
    ax.add_artist(first_legend)

    # Add the main legend for the model and dat
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=11, frameon=False)
    fig.tight_layout()
    plt.savefig('model_fit.png')

def main(file_name):
    # Read the data
    obsname, frequency, photometry, uncertainty = read_dat_file(file_name)
    #calculate Beq
    p = 0.5
    alpha_thin = -1.8
    nu_peak =7.510961461177
    S_peak=0.038872512411101165 
    theta_src_min = 46                 # J01445 21.9767
    theta_src_max = 55                 # J01445 30.476
    z = 0.097000                               # J01445 0.366972


    X_p = ((10**(p+alpha_thin))-(nu_peak**(p+alpha_thin)))/(p+alpha_thin)
    redshift_part = ((1+1)/1)*((1+z)**(3+alpha_thin))
    source_size_part = 1/(theta_src_min*theta_src_max*theta_src_max)
    freq_flux_part = S_peak/(nu_peak**(-alpha_thin))
    B_eq = 5.69e-5 * (redshift_part * source_size_part * freq_flux_part * X_p)**(2/7) 

    B_eq = B_eq / 1e4

    print("Beq is :",B_eq,"nT")
    fit_type = 'CI' 
    params, _, _, _, _, _= spectral_fitter(frequency, photometry, uncertainty, fit_type)
    
    # Generate the model spectrum
    model_data, err_model_data, model_data_min, model_data_max = spectral_model_(params, frequency)

    # Prepare observed data tuple
    observed_data = (frequency, photometry, uncertainty)

    # Prepare plotting data tuple
    plotting_frequency = np.geomspace(10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency)))), 100)
    plotting_data, err_plotting_data, plotting_data_min, plotting_data_max = spectral_model_(params, plotting_frequency)
    plotting_data = (plotting_frequency, plotting_data, err_plotting_data, plotting_data_min, plotting_data_max)

    # Plot the fitted spectrum using spectral_plotter
    
    #spectral_plotter(observed_data, model_data, plotting_data, fit_type=fit_type)

    #ages

    params_ages = (params[0], 10**params[1], params[5])
    ages = spectral_ages(params_ages, B_eq, z)

    plot_spectrum(observed_data, obsname, model_data, plotting_data, fit_type, params, ages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit synchrotron spectrum to radio data")
    parser.add_argument("file_name", type=str, help="Name of the data file (.dat format)")
    
    

    args = parser.parse_args()
    main(args.file_name)


















