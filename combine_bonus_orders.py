#combine_bonus_orders.py


import pandas as pd
import numpy as np

def bonus_comb(spitzer_spec):

    wave = spitzer_spec.wavelength
    flux = spitzer_spec.flux_jy
    flux_err = spitzer_spec.flux_jy_err

    wave_dup = wave[wave.duplicated(keep=False)]
    flux_dup = flux[wave.duplicated(keep=False)]
    flux_dup_err = flux_err[wave.duplicated(keep=False)]
    if len(wave[wave.duplicated(keep=False)]) > 0.:
        i=0
        while len(wave[wave.duplicated(keep=False)]) > 0.:
            #print i
            new_flux = ((flux_dup.iloc[i]/flux_dup_err.iloc[i]**2) + (flux_dup.iloc[i+1]/flux_dup_err.iloc[i+1]**2))/((1/flux_dup_err.iloc[i]**2) + (1/flux_dup_err.iloc[i+1]**2))
            #error of weighted mean
            new_flux_err = np.sqrt(1/ ((1/flux_dup_err.iloc[i]**2) + (1/flux_dup_err.iloc[i+1]**2)))
            wave_loc_keep = wave_dup.index[i]
            wave_loc_drop = wave_dup.index[i+1]
            #drop first
            wave = wave.drop(wave_loc_drop)
            flux = flux.drop(wave_loc_drop)
            flux_err = flux_err.drop(wave_loc_drop)
            # now keep weighted mean flux
            flux[wave_loc_keep] = new_flux
            flux_err[wave_loc_keep] = new_flux_err
            i=i+2

    spitz_spec=pd.DataFrame(index=range(len(wave)))
    spitz_spec['wavelength'] = wave
    spitz_spec['flux_jy'] = flux
    spitz_spec['flux_jy_err'] = flux_err
    spitz_spec=spitz_spec.sort('wavelength')
    spitz_spec = spitz_spec.dropna()

    return spitz_spec
