
'''
Erini Lambrides
A pythonic version of SM's linfit
- find the coefficants for a system of vectors whose coefficants are linear


function Reqs: statsmodels, math, numpy
i.e pip install statsmodels

Input:
vecs: a list of vectors of the linear system
known: the data your are trying to fit, or the right side vector

Returns:
ols.results: Contains coefficient values and errors, and suite of other functionality,
below is useful snippets from the data model

fit values: ols_result.fitted_values
#if you want to look deeper at what the coefficients actually are
const: ols_result.params.const
params: ols_result.params.xi where i goes from 1 to the amount of vectors in your input, in the order
of the list
errors: (standard errors of each coeff estimate) ols_result.bse.xi where i goes from 1 to the amount of vectors in your input, in the order
of the list,

EXAMPLE FOR USAGE:

import p_linfit as lf

#some line
line = np.arange(1,50,.5)
#some sin

sin = np.sin(line)

vecs = [line,sin]

#make some noisy fake data

knowns = sin + line + np.random.normal(0,1,98)

result = lf.linfit(vecs,knowns)

#plotting the fit, and showing usage of the specific parameters
import matplotlib.pyplot as plt
plt.plot(line, result.fittedvalues,label='FIT')
plt.plot(line, knowns,label='DATA')
plt.legend()
plt.show()

#a summary of the regression results
result.summary()

#R^2 statistic
result.rsqaured

'''


import statsmodels.api as sm
import math
import numpy as np

def linfit(vecs,knowns):

    #getting it in the right dimensions
    vectors =  np.column_stack(vecs)
    #including a fit for intercept
    vectors = sm.add_constant(vectors)

    ols = sm.OLS(knowns, vectors)
    ols_result = ols.fit()

    return ols_result
