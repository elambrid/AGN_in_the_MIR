#scale_sl1_ll2.py

import pandas as pd



def shift(spitzer_spec,scale):

    spitzer_spec = spitzer_spec.dropna()
    wave = spitzer_spec.wavelength
    flux = spitzer_spec.flux_jy
    flux_err = spitzer_spec.flux_jy_err

    if scale != 1.:
        shifted_1 = flux[wave >=  14.06935]
        shifted_2 = flux[wave < 14.06935] * scale
        flux = shifted_2.append(shifted_1)


    spitzer_spec_shifted = pd.DataFrame()
    spitzer_spec_shifted['wavelength'] = wave
    spitzer_spec_shifted['flux_jy'] = flux
    spitzer_spec_shifted['flux_jy_err'] = flux_err

    return(spitzer_spec_shifted)
