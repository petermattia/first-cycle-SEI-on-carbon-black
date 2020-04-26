#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:37:48 2019

@author: peter

NOTE: I collected some linear scan voltammetry (LSV) data to get physical constants of the first cycle reaction, but I couldn't get the equations to match up.
This script has my attempts to do so

Reference: p236 of Bard and Faulkner
"""

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.constants import R
from scipy.constants import physical_constants
from scipy import interpolate
from scipy.signal import find_peaks

# Physical constants
F = physical_constants['Faraday constant'][0] # C mol^-1
T = 273.15 + 30 # deg C
f = F / (R * T) # V^-1

# Assumed peak
assumed_peak = 1.0 # V

# Script
file_list = sorted(glob.glob('*.txt'))

UCV = 1.2
LCV = 0.01
voltage_basis = np.linspace(LCV + 0.02, UCV - 0.025, 500)

colors = cm.Blues(np.linspace(0.3, 1, 5))[:,0:3]

peak_voltages = np.zeros((4,))
FWHM_voltages = np.zeros((4,))
peak_currents = np.zeros((4,))
baseline_currents = np.zeros((4,))

# Manually entered
baseline_voltage_endpoints = [[0.59, 1.15],
                              [0.45, 1.10],
                              [0.41, 0.89],
                              [0.20, 0.61],
                              [0.02, 0.47]] # def invalid


#plt.figure()
for k, file in enumerate(file_list[:-1]):
    
    print(f'Starting {file}')
    
    speed = re.findall(r'\d+', file.split('_')[3])[0]
    
    # Filter relevant data
    data = np.genfromtxt(file, delimiter='\t', skip_header=True)
    
    # Get first LSV sweep between 1.2 and end of negative sweep
    idx1 = np.where(data[:,2] < UCV)[0][0]
    data = data[idx1:]
    idx2 = np.where(data[:,0] == 1)[0][0]
    data = data[:idx2]
        
    # Get columns
    redox = data[:,0]
    time = data[:,1]
    voltage = data[:,2]
    current = data[:,3]
    
    # Interpolate
    idx = np.diff(voltage) < 0
    I_V = interpolate.interp1d(voltage[:-1][idx][::-1],
                               current[:-1][idx][::-1],
                               kind='linear')
    I_interp = I_V(voltage_basis)
    
    # Get peak
    idx3 = np.where(voltage_basis > 0.2)[0][0]
    peak_idx = idx3 + find_peaks(-I_interp[idx3:], distance=10000)[0]
    
    # Get baseline currents (baseline voltages are manual)
    idx_bl1 = np.where(voltage < baseline_voltage_endpoints[k][0])[0][0]
    idx_bl2 = np.where(voltage < baseline_voltage_endpoints[k][1])[0][0]
    baseline_current_endpoints = [current[idx_bl1], current[idx_bl2]]
    baseline_slope = (baseline_current_endpoints[1] \
                      - baseline_current_endpoints[0]) \
        / (baseline_voltage_endpoints[k][1] \
           - baseline_voltage_endpoints[k][0])
    baseline_current = baseline_current_endpoints[0] \
        + baseline_slope * (voltage_basis[peak_idx]\
                            - baseline_voltage_endpoints[k][0])
            
    # Get FWHM idx
    halfmax_current = I_interp[peak_idx] - baseline_current_endpoints[1]
    FWHM_idx = np.where(I_interp[::-1] < halfmax_current)[0][0]
        
    # Append values
    peak_voltages[k] = voltage_basis[peak_idx]
    FWHM_voltages[k] = voltage_basis[::-1][FWHM_idx]
    peak_currents[k] = I_interp[peak_idx]
    baseline_currents[k] = baseline_current
    
    # Plot
    plt.figure()
    plt.plot(voltage_basis, I_interp,
             color=colors[k],
             label=str(speed) + ' mV/min')
    plt.plot(voltage_basis[peak_idx], I_interp[peak_idx], 'ok')
    
    plt.plot(voltage_basis[peak_idx], baseline_current, 'sk')
    plt.plot(voltage_basis[::-1][FWHM_idx], halfmax_current, '^k')
    plt.plot(baseline_voltage_endpoints[k], baseline_current_endpoints, '--', 
             color='tab:gray')
    


plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.legend(frameon=False)
plt.xlim([0, 1.2])

## New figure
peak_currents_corrected = peak_currents - baseline_currents
peak_currents_neglog = np.log(-peak_currents_corrected)
peak_voltages_corrected = assumed_peak - peak_voltages

fit = np.polyfit(peak_voltages_corrected.T, peak_currents_neglog.T, 1)
alpha = fit[0] / f

x = np.linspace(0, 0.5, 100)
y = fit[0] * x + fit[1]

plt.figure()
plt.plot(peak_voltages_corrected, peak_currents_neglog, 'ok')
plt.plot(x, y)

## E - E_FWHM
FWHM_voltages = np.array([0.85, 0.775, 0.71, 0.50])
E_minus_E_FWHM = np.abs(peak_voltages - FWHM_voltages)
plt.figure()
plt.plot(0.0477 / E_minus_E_FWHM, 'ok')