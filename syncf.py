#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import os
import datetime
from scipy.optimize import curve_fit

from sf.synchrofit import spectral_fitter, spectral_model_, spectral_ages #https://github.com/synchrofit/synchrofit
plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

def WLLS(S1, f1, Serr1):

    f = np.log10(np.array(f1))
    S = np.log10(np.array(S1))
    Serr = np.array(Serr1) / np.array(S1)*np.log(10)

    X = np.column_stack((np.ones_like(f), f))
    W = np.diag(1/Serr**2)
    XtWX = np.linalg.inv(X.T @ W @ X)
    beta = XtWX @ X.T @ W @ S
    r = S - X @ beta
    df = len(f) - 2
    sumR = np.sum(r**2)
    sigma = sumR / df
    beta_var = sigma * XtWX
    stderr = np.sqrt(np.diag(beta_var))
    chi2red = (r.T @ W @ r) / df
    return beta[0], beta[1] , stderr[0], stderr[1], chi2red

def powerlaw(amp,frequency,alpha):
    return (10.0 **amp) * frequency ** alpha

def read_dat_file(file_name):
    data = pd.read_csv(file_name, delimiter=',')
    obsname = data['Spectra'].values
    frequency = data['Frequency (Hz)'].values
    photometry = data['Photometry (Jy)'].values
    uncertainty = data['Uncertainty (Jy)'].values


    return obsname, frequency, photometry, uncertainty


def plot_spectrum(observed_data, obsname, model_data, plotting_data, fit_type, params, ages, WLLS_params):
    frequency, photometry, uncertainty = map(np.asarray, observed_data)
    plotting_frequency, plotting_data, err_plotting_data, plotting_data_min, plotting_data_max = map(np.asarray, plotting_data)
    model_data = np.asarray(model_data)
    
    # Get params
    v_b = (10**params[1]) / 1e9 # Ghz
    dv_b = (10**params[1])*(np.log(10)*params[2]) / 1e9 
    inj_index = (params[3] - 1.0) / 2.0
    dinj_index = params[4] / 2.0
    Rrem = params[5]
    dRrem = params[6]
    if ages is not None:
        t_on = ages[1]
        dt = params[6] * ages[0]
        t_off = ages[2]
    
    
    fig, ax =plt.subplots(figsize=(6, 5)) #was 7,6


    unq_obsname = np.unique(obsname)
    for name in unq_obsname:
        idx = obsname == name
        handle = ax.errorbar(frequency[idx], photometry[idx], yerr=uncertainty[idx], capsize=0.5, label=name, fmt='o', alpha=0.9)
   # Plot model data
    model_suffix = ""
    if fit_type in ['CI', 'TCI']:
        if Rrem == 0:
            model_suffix = '-on'
        else:
            model_suffix = '-off'
    
    # Plot model dat
    model_fit = ax.plot(plotting_frequency, plotting_data, 'k-', label=f'{fit_type}{model_suffix} Model fit')
    error_fit = ax.fill_between(plotting_frequency, plotting_data_min, plotting_data_max, color='blue', alpha=0.2, label='Model error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.2 * np.min(photometry), 1.5 * np.max(photometry))
    ax.set_xlim(np.min(plotting_frequency), np.max(plotting_frequency))
    ax.set_xlabel(r'Frequency (Hz)')
    ax.set_ylabel(r'Flux Density (Jy)')
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=1.5, pad=5, right=True, top=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.5, pad=5, right=True, top=True)
	
	# plot fitted line
    if WLLS_params is not None:
		#unpack WLLS params
        amp, alpha, damp ,dalpha, chi2red = WLLS_params
        fitted_phot = [powerlaw(amp,f,alpha) for f in plotting_frequency]
        label_str = fr'$\alpha_{{4.5GHz}}^{{11.5GHz}} = {-alpha:.1f} \pm {dalpha:.1f}$'
        ax.plot(plotting_frequency,fitted_phot, '--', label=label_str)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
 #   ax.grid(True)
    str2 = r'$\alpha_{inj} = $'
    str1 = r'$\nu_b$ = '
    str3 = r'$R_{rem}$='
    if fit_type in ['CI', 'TCI'] and ages is not None and Rrem != 0: 
        legend_text = (str1 + f'{v_b:.3f} ± {dv_b:.3f} GHz\n' +
                       str2 + f'{inj_index:.3f} ± {dinj_index:.3f}\n' +
                       r'$t_{on} = $' + f'{t_on:.2f} ± {dt:.2f} Myr\n' +
                       r'$t_{off} = $' + f'{t_off:.2f} ± {dt:.2f} Myr')
    elif fit_type in ['CI', 'TCI'] and ages is not None:
        legend_text = (str1 + f'{v_b:.3f} ± {dv_b:.3f} GHz\n' +
                       str2 + f'{inj_index:.3f} ± {dinj_index:.3f}\n' +
                       r'$\tau = $' + f'{t_on:.2f} Myr')	
    else: 
        legend_text = (str1 + f'{v_b:.3f} ± {dv_b:.3f} GHz\n' +
                str2 + f'{inj_index:.3f} ± {dinj_index:.3f}\n' +
                str3 + f'{Rrem:.3f} ± {dRrem:.3f}')

    first_legend = ax.legend(handles=[plt.Line2D([], [], color='none', label=legend_text)], loc='lower left', handlelength=0, fontsize=11, frameon=False)
    ax.add_artist(first_legend)

    # Add the main legend for the model and dat
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=11, frameon=False)
    fig.tight_layout()
    plt.savefig(f'{fit_type}{model_suffix}_model_fit.pdf')
	



def log_results(file_name, B_eq, z, remnant_range, fit_type, params, ages):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/{fit_type}_{timestamp}.log'
    
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Input Parameters:\n")
        log_file.write(f"Data File: {file_name}\n")
        if B_eq is not None and z is not None:
            log_file.write(f"Magnetic Field (B_eq): {B_eq*1e9} nT\n")
            log_file.write(f"Redshift: {z}\n")
        log_file.write(f"Remnant Range: {remnant_range}\n")
        log_file.write(f"Fit Type: {fit_type}\n\n")
        
        log_file.write(f"fit_type         : {fit_type}\n")
        log_file.write(f"break_predict    : {10**(params[1]):.6f} (Hz)\n")
        log_file.write(f"dbreak_predict   : {10**(params[2]):.6f} (Hz)\n")
        log_file.write(f"inject_predict(s)   : {params[3]:.6f}\n")
        log_file.write(f"dinject_predict  : {params[4]:.6f}\n")
        log_file.write(f"remnant_predict  : {params[5]:.6f}\n")
        log_file.write(f"dremnant_predict : {params[6]:.6f}\n")
        
        if ages is not None:
            log_file.write("Spectral Ages:\n")
            log_file.write(f"tau   : {ages[0]:.6f} Myr\n")
            log_file.write(f"t_on  : {ages[1]:.6f} Myr\n")
            log_file.write(f"t_off : {ages[2]:.6f} Myr\n")

def main(file_name, fit_type, B_eq, z, remnant_range, extra_data):
    obsname, frequency, photometry, uncertainty = read_dat_file(file_name)
    obsname

    #X_p = ((10**(p+alpha_thin))-(nu_peak**(p+alpha_thin)))/(p+alpha_thin)
    #redshift_part = ((1+1)/1)*((1+z)**(3+alpha_thin))
    #source_size_part = 1/(theta_src_min*theta_src_max*theta_src_max)
    #freq_flux_part = S_peak/(nu_peak**(-alpha_thin))
    #B_eq = 5.69e-5 * (redshift_part * source_size_part * freq_flux_part * X_p)**(2/7) 

    
    if B_eq is not None:
        print("Beq is :",B_eq,"nT")
    
    params, _, _, _, _, _= spectral_fitter(frequency, photometry, uncertainty, fit_type, remnant_range=remnant_range,b_field=B_eq,redshift=z)
    
    # Generate the model spectrum
    model_data, err_model_data, model_data_min, model_data_max = spectral_model_(params, frequency)

    # Add extra data points
    WLLS_params = None
    if extra_data is not None:
        extra_obsname, extra_frequency, extra_photometry, extra_uncertainty = read_dat_file(extra_data)
        obsname = np.append(obsname, extra_obsname)
        frequency = np.append(frequency, extra_frequency)
        photometry = np.append(photometry, extra_photometry)
        uncertainty = np.append(uncertainty, extra_uncertainty)
		#WLLS
        amp, alpha, damp ,dalpha, chi2red = WLLS(extra_photometry, extra_frequency, extra_uncertainty)
        WLLS_params = (amp,alpha,damp,dalpha,chi2red)

	

    # Prep obsved data tuple
    observed_data = (frequency, photometry, uncertainty)

    # Prep plotting data tuple
    plotting_frequency = np.geomspace(10**(math.floor(np.min(np.log10(frequency)))),10**(math.ceil(np.max(np.log10(frequency)))), 100)
    plotting_data, err_plotting_data, plotting_data_min, plotting_data_max = spectral_model_(params, plotting_frequency)
    plotting_data = (plotting_frequency, plotting_data, err_plotting_data, plotting_data_min, plotting_data_max)

    # Plot the fitted spectrum using spectral_plotter
    
    #spectral_plotter(observed_data, model_data, plotting_data, fit_type=fit_type)

    #ages

    params_ages = (params[0], 10**params[1], params[5])
    ages = None
    if B_eq is not None and z is not None:
        ages = spectral_ages(params_ages, B_eq, z)

    plot_spectrum(observed_data, obsname, model_data, plotting_data, fit_type, params, ages, WLLS_params)

    log_results(file_name, B_eq, z, remnant_range, fit_type, params, ages)

if __name__ == "__main__":
    #TODO Make it work for other models as well
    parser = argparse.ArgumentParser(description="Fit synchrotron spectrum to radio data")
    parser.add_argument("--data", type=str, required=True,help="Name of the data file (.dat format)")
    parser.add_argument("--fit_type", type=str, required=True, help="Name of the fit type (CI, TCI, ...)")
    parser.add_argument("-B", type=float, required=False, help="Magnetic field strength in T")
    parser.add_argument("-z", type=float, required=False, help="Redshift of the source") 
    parser.add_argument("--remnant_range", required=False, dest='remnant_range', type=float, nargs='+', default=[0, 1],
                        help="(Default = [0, 1]) Accepted range for the remnant ratio")
    parser.add_argument("--data2", type=str, required=False, help="Any extra data to be plotted but not fitted")
    args = parser.parse_args()
    main(args.data,args.fit_type, args.B, args.z, args.remnant_range, args.data2)


















