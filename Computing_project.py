#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:37:43 2024

@author: chinmayib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#calculates and returns a suitable value for the mu in initial guess using the corresponding x-value of the maximum y-value
def mu_guess(x, y):
    index = 0
    i = 0
    while i != len(y):
        if y[i] == np.max(y):
            index = i
        i += 1
    guess = x[index]
    return guess

# The following four lines of code are taken from the Imperial College London, Department of Phyics, Year 1 Computing 2024-2025 Core Worksheet 2:
#equation of a straight line for the background and a Gaussian function for the feature
def fit_func(x, a, mu, sig, m, c):
    gaus = a * np.exp(-(x - mu)**2 / (2 * sig**2))
    line = m * x + c
    return gaus + line

#calculates and returns velocity in kps using the redshift formula
def calc_velocity(mu):
    lightSpeed_ms = 2.9979e8
    lambdaE_m = 486.1e-9
    lambdaO_m = lightSpeed_ms/mu
    lambdaRatio = lambdaO_m/lambdaE_m
    velocity = (lightSpeed_ms*(lambdaRatio**2 - 1))/(1000*(lambdaRatio**2 + 1))
    return velocity



distData = np.loadtxt("Data/Distance_Mpc.csv", dtype=str, delimiter=',') #opens and reads distance data into a matrix of dtype string
obsNumTemp = np.array(distData[0,1:]) #each column in the matrix is split into a separate array excluding the first element
distance_mpcTemp = np.array(distData[1,1:])
valInstRes = np.array(distData[2,1:])

obsNum = np.array([]) #initiates empty arrays
distance_mpcStr = np.array([])

#loops through distance data and does the following:
for i in range(len(obsNumTemp)):
    if valInstRes[i] == 'y': #checks if data point is valid
        obsNum = np.append(obsNum, obsNumTemp[i]) #appends valid data into arrays
        distance_mpcStr = np.append(distance_mpcStr, distance_mpcTemp[i])
distance_mpc = distance_mpcStr.astype(float) #changes dtype of array from string to float to draw graph


velocity_kps = np.array([])
velAbsUnc = np.array([])

#loops through H beta data files and does the following:
for i in range(len(obsNum)):
    fname = "Data/" + obsNum[i] + ".txt"
    index, frequency_hz, spectralIntensity_au = np.loadtxt(fname, skiprows=2, unpack=True) #opens and reads each column into a different array of index, frequency and intensity
    
    plt.figure()
    
    plt.plot(frequency_hz, spectralIntensity_au, '.') #plots graph of intensity against frequency
    plt.grid()
    plt.title(obsNum[i])
    plt.xlabel("Frequency/Hz")
    plt.ylabel("Spectral intensity/A.U.")
    
    if fname != "Data/18381.txt": #18381 is an exception to mu_guess() as its max y-value is not the peak intensity
        initialGuess = np.array([40, mu_guess(frequency_hz, spectralIntensity_au), 1e12, 1e-13, 100])
        popt, pcov = curve_fit(fit_func, frequency_hz, spectralIntensity_au, p0=initialGuess)
        plt.plot(frequency_hz, fit_func(frequency_hz, popt[0], popt[1], popt[2], popt[3], popt[4]), label='Fit results') #overlays the fitted curve with optimum parameters onto the existing graph
        plt.savefig("Output/" + obsNum[i] + ".png")
        velocity_kps = np.append(velocity_kps, calc_velocity(popt[1])) #appends each value of velocity corresponding to each mu of the graph into the array velocity_kps
        muPerUnc = np.sqrt(pcov[1,1])/popt[1] 
        velAbsUnc = np.append(velAbsUnc, calc_velocity(popt[1])*muPerUnc) #calculates absolute uncertainty in velocity using percentage uncertainty in mu
    
    else:
        initialGuess = np.array([40, 5.8e14, 1e12, 1e-13, 100])
        popt, pcov = curve_fit(fit_func, frequency_hz, spectralIntensity_au, p0=initialGuess)
        plt.plot(frequency_hz, fit_func(frequency_hz, popt[0], popt[1], popt[2], popt[3], popt[4]), label='Fit results')
        plt.savefig("Output/18381.png")
        velocity_kps = np.append(velocity_kps, calc_velocity(popt[1]))
        muPerUnc = np.sqrt(pcov[1,1])/popt[1]
        velAbsUnc = np.append(velAbsUnc, calc_velocity(popt[1])*muPerUnc)
    plt.show()
    plt.close()


#plots velocity against distance as a straight line with errorbars
plt.figure()
fit_vel, cov_vel = np.polyfit(distance_mpc, velocity_kps, 1, w=1/velAbsUnc, cov=True)
pVel = np.poly1d(fit_vel)
plt.plot(distance_mpc, velocity_kps, 'o')
plt.plot(distance_mpc, pVel(distance_mpc))
plt.errorbar(distance_mpc, velocity_kps, yerr=velAbsUnc, fmt='o', mew=2, ms=3, capsize=4)
plt.grid()
plt.title("Velocity against distance")
plt.xlabel("Distance/Mpc")
plt.ylabel("Velocity/kms-1")
plt.savefig("Output/Hubble_constant.png")
plt.show()
plt.close()
print(f"Hubble constant = gradient = {fit_vel[0]:.3f} +/- {np.sqrt(cov_vel[0,0]):.3f}")


