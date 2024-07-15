import pandas as pd
import numpy as np
import math
import argparse
from sf.synchrofit import spectral_fitter, spectral_model_, spectral_plotter, spectral_ages

# Define the function to read the .dat file
def read_dat_file(file_name):
    data = pd.read_csv(file_name, delimiter=',')
    frequency = data['Frequency (Hz)'].values
    photometry = data['Photometry (Jy)'].values
    uncertainty = data['Uncertainty (Jy)'].values
    return frequency, photometry, uncertainty


def plot_spectrum(frequency, photometry, uncertainty, model_data, model_data_min, model_data_max):
    plt.errorbar(frequency, photometry, yerr=uncertainty, fmt='o', label='Observed Data')
    plt.plot(frequency, model_data, label='Model Data', color='red')
    plt.fill_between(frequency, model_data_min, model_data_max, color='red', alpha=0.3, label='Model Uncertainty')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux Density (Jy)')
    plt.legend()
    plt.title('Spectral Fitting with CI Model')
    plt.show()

def main(file_name):
    # Read the data
    frequency, photometry, uncertainty = read_dat_file(file_name)

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
    spectral_plotter(observed_data, model_data, plotting_data, fit_type=fit_type)

    #ages

    params = (params[0], 10**params[1], params[5])
    spectral_ages(params, 0.35e-9, 0.36692)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit synchrotron spectrum to radio data")
    parser.add_argument("file_name", type=str, help="Name of the data file (.dat format)")
    
    args = parser.parse_args()
    main(args.file_name)


















