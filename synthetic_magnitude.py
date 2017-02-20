#absolute flux  function

import pandas as pd
import math
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import csv
from re import sub
from scipy import integrate
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unumpy


c = 2.99e8 * 1e6 # speed of light in microns

#define the binning function
#taking average of values with a bin
# bins to the highest

def get_vals(df,v):
	value_to_find = v
	Min = df['wavelength'] <= value_to_find
	Max = df['wavelength'] >= value_to_find
	#idx_Min = df.ix[Min, 'wavelength'].idxmax()
	idx_Max = df.ix[Max, 'wavelength'].idxmin()
	return df.ix[idx_Max, ['wavelength']]

#right now only for wise
#df, string
def synthetic_magnitude(spitzer_spec,wise_band):
#try:
	if wise_band == 'W4':
		w0 = 8.363
		response_curve = pd.read_csv("/home/erini/AGN_zoo/Data_Mine/wise_xmatch_casssis/resolution_curves/w4_curve.csv")
	elif wise_band == 'W3':
		w0 = 31.674
		response_curve = pd.read_csv("/home/erini/AGN_zoo/Data_Mine/wise_xmatch_casssis/resolution_curves/w3_curve.csv")
	elif wise_band == 'W2':
		w0 = 171.787
		response_curve = pd.read_csv("/home/erini/AGN_zoo/Data_Mine/wise_xmatch_casssis/resolution_curves/w2_curve.csv")
	response_curve_cut = pd.DataFrame()
	#only include portion of response curve that mathces with spitzer spec
	response_curve_cut = response_curve[(response_curve.wave.astype('float64') >= spitzer_spec.sort('wavelength').astype('float64').wavelength.iloc[0]) & (response_curve.wave.astype('float64') <= spitzer_spec.sort('wavelength').astype('float64').wavelength.iloc[-1])]

	value_match= lambda x: get_vals(spitzer_spec,x)[0]
	response_curve_cut['matched']=response_curve_cut['wave'].map(value_match)

	response_binned = pd.DataFrame()
	response_binned=response_curve_cut.groupby('matched').mean()







	temp = pd.DataFrame(np.zeros(len(spitzer_spec)))
	temp.index=spitzer_spec.sort('wavelength').wavelength
	temp[0]=response_binned['response']
	#temp.fillna(0,inplace=True)
	temp.columns=['response']
	response_binned=temp

	#calculating the magnitude

	a = spitzer_spec.flux_jy.values * response_binned.response.values/(response_binned.index.values**2)


	b = a[~np.isnan(a)]
	top = integrate.cumtrapz(b,response_binned.index[~np.isnan(a)].values,initial=0)


	bottom = integrate.cumtrapz(response_binned.response.values[~np.isnan(a)]/(response_binned.index[~np.isnan(a)].values**2),response_binned.index[~np.isnan(a)].values,initial=0)


	#test6ing

	#top = sum(spitzer_spec.flux_jy.values * response_binned.response.values )* 0.01
	#bottom = sum(response_binned.response.values)* 0.01
	#dis should be a number
	#synthetic_flux = top[-1]/bottom[-1]

	synthetic_flux = max(top)/max(bottom)

	# 8.363 is wise 4 zero magnitude attribute f_nu

	try:

		synthetic_mag = -2.5*math.log10(synthetic_flux/w0)


	except ValueError:

		print "some sort of math error"
		synthetic_mag = np.nan

	#print "Out of range wavelength scale"


	return synthetic_mag
