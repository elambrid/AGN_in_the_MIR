#rebin spectra
import pandas as pd
import numpy as np
from pysynphot import observation
from pysynphot import spectrum


def spectra_rebin(wave,flux,rebin_wave):

    spect=spectrum.ArraySourceSpectrum(wave=wave.values,flux=flux.values)
    f = np.ones(len(wave))
    filt=spectrum.ArraySpectralElement(wave.values,f,waveunits='microns')
    obs=observation.Observation(spect,filt,binset=rebin_wave,force='taper')

    return obs.binflux
