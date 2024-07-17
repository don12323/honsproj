import pandas as pd
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from sf.synchrofit import spectral_fitter, spectral_model_, spectral_ages
plt.style.use('seaborn-v0_8-bright')

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
    dv_b = (10**params[2]) / 1e9 
    inj_index = (params[3] - 1.0) / 2.0
    dinj_index = params[4] / 2.0
    t_on = ages[1]
    t_off = ages[2]
    
    fig, ax =plt.subplots(figsize=(10, 10))

    unq_obsname = np.unique(obsname)
    for name in unq_obsname:
        idx = obsname == name
        handle = ax.errorbar(frequency[idx], photometry[idx], yerr=uncertainty[idx], capsize=0.8, linestyle='-', label=name, fmt='o', alpha=0.9)

    # Plot model dat
    model_fit = ax.plot(plotting_frequency, plotting_data, 'k-', label='{}-off Model fit'.format(fit_type))
    error_fit = ax.fill_between(plotting_frequency, plotting_data_min, plotting_data_max, color='blue', alpha=0.2, label='Model Uncertainty')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.2 * np.min(photometry), 5 * np.max(photometry))
    ax.set_xlim(np.min(plotting_frequency), np.max(plotting_frequency))
    ax.set_xlabel(r'Frequency (Hz)')
    ax.set_ylabel(r'Flux Density (Jy)')
 #   ax.grid(True)
    str2 = r'$\alpha_{inj} = $'
    str1 = r'$\nu_b$ = '
    str3 = r'$t_{on}$ = '
    str4 = r'$t_{off}$ = '

    legend_text = (str1+f'{v_b:.3f} ± {dv_b:.3f} GHz\n'+
                   str2+f'{inj_index:.3f} ± {dinj_index:.3f}\n'+
                   str3+f'{t_on:.2f} Myr\n'+
                   str4+f'{t_off:.2f} Myr')
    #blank = ax.plot([], [], ' ', label=legend_text)
    first_legend = ax.legend(handles=[plt.Line2D([], [], color='none', label=legend_text)], loc='upper left', handlelength=0, fontsize=11, frameon=False)
    ax.add_artist(first_legend)

    # Add the main legend for the model and dat
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=11, frameon=False)
    plt.savefig('model_fit.png')

def main(file_name):
    # Read the data
    obsname, frequency, photometry, uncertainty = read_dat_file(file_name)

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
    ages = spectral_ages(params_ages, 0.35e-9, 0.36692)

    plot_spectrum(observed_data, obsname, model_data, plotting_data, fit_type, params, ages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit synchrotron spectrum to radio data")
    parser.add_argument("file_name", type=str, help="Name of the data file (.dat format)")
    
    args = parser.parse_args()
    main(args.file_name)


















