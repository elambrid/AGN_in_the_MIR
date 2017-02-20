#linfit in python
#from sklearn.linear_model import LinearRegression # a linear regression function does a simple ols
import statsmodels.api as sm
import math
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt
import scipy as sp
import seaborn
import uncertainties
'''
spectrum: a pandas dataframe with at least 3 columns labeled wavelength, flux_jy, flux_jy_err
line: the feature line center in microns, list form even in one
z: redshift

INPUT SPECTRUM IS ASSUMED TO BE BONUS ORDER COMBINED ,AND SLLL1 SCALE APPLIED

'''

#line = [12.813,15.555,10.511,14.322]
#(S(7),5,4,3,2,1,0)
#line = [5.511,6.909,8.025,9.665,12.279,17.03,28.218]

#spectrum = spitzer_spec
def linfit(objid,aorkey,detlvl,spectrum,scale,line,z,filepath):

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    dl_mpc = cosmo.luminosity_distance(z) #dl in Mpc


    spectrum = spectrum.dropna()
    wave = spectrum.wavelength * (1+z)**-1
    flux = spectrum.flux_jy * (1+z)**-1
    flux_err = spectrum.flux_jy_err * (1+z)**-1

    line_lums = []
    line_lums_err = []

    for i in range(len(line)):

        #account for different line resolution at sl vs ll
        if line[i] * (1+z) < 14.2:
            sig = .1/2.35/(1+z)
        elif (line[i] * (1+z) >= 14.2) and (line[i] * (1+z) <= 20.6)  :
            sig =  .14/2.35/(1+z)
        elif  (line[i] * (1+z) > 20.6):
            sig =  .34/2.35/(1+z)

        # determine a fitting window on either side of function
        #cutout a 3sigma region on either side of feature
        sub_wave = wave[(wave >= line[i]-3.5*sig) & (wave <= line[i]+3.5*sig)]
        sub_flux = flux[(wave >= line[i]-3.5*sig) & (wave <= line[i]+3.5*sig)]
        sub_flux_err = flux_err[(wave >= line[i]-3.5*sig) & (wave <= line[i]+3.5*sig)]

        if (len(sub_wave) > 3) and (len(sub_flux[np.isfinite(sub_flux)]) == len(sub_flux)):

            #fine scale x
            x= pd.Series(data=np.arange(line[i] - 3*sig,line[i] + 3*sig,.01))

            #fit with underlying continuum
            one = sub_wave*0 + 1.0
            two = sub_wave - line[i]

            #assume some centroid noise - find the actual line center

            delta=pd.Series(data=np.arange(-.03,.03,.005))
            rms = pd.Series(data=np.zeros(len(delta)))

            for k in range(len(delta)):
                j = delta[k]
                line_center = line[i] + j
                gauss = np.exp(-((sub_wave-line_center)**2) /(2*sig**2))

                A = [(a,b,c) for a,b,c in zip(one,two,gauss)]

                ols = sm.OLS(sub_flux, A)
                ols_result = ols.fit()

                model =  ols_result.params.const + ols_result.params.x1*two + ols_result.params.x2 * gauss

                rms[k] = np.sqrt(np.sum(((sub_flux-model)**2)/sub_flux_err**2))

            min_delta = delta[rms == rms.min()].values[0]
            line_center = line[i] + min_delta
            gauss = np.exp(-((sub_wave-line_center)**2) /(2*sig**2))

            A = [(a,b,c) for a,b,c in zip(one,two,gauss)]

            ols = sm.OLS(sub_flux, A)

            ols_result = ols.fit()

            model =  ols_result.params.const + ols_result.params.x1*(x-line[i]) + ols_result.params.x2 *  np.exp(-((x-line_center)**2) /(2*sig**2))

            model_cont = ols_result.params.const + ols_result.params.x1*(x - line[i])


            #conver Fnu (Jy) to Flamb(ergs/s/cm**2/Hz)
            #Fnu * c/lambda**2 * 1e-23

            #line luminosity in 1e42 ergs/sec
            dl = dl_mpc.value * 3.086e24 #Mpc to cm

            if z == 0:
                dl = 1.


            line_flux_nu =  (ols_result.params.x2)*np.sqrt(2*np.pi)*sig
            line_flux_nu_cgs = 1e-23 * line_flux_nu
            line_flux_lam = line_flux_nu_cgs *2.99e14/(line_center)**2


            line_lum = dl**2* 4*np.pi* line_flux_lam
            line_lum_err = dl**2* 4*np.pi* ols_result.bse.x2* 2.99e14*sig/((line_center**2))* 1e-23 *np.sqrt(2*np.pi)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.title(" Aorkey: "+str(aorkey[:5]) + ' Line ' + str(line[i]))
            plt.suptitle("Detlvl: " + str(detlvl))
            ax.plot(sub_wave,sub_flux,'.')
            ax.fill_between(sub_wave,sub_flux+sub_flux_err,sub_flux-sub_flux_err,alpha=0.4)
            ax.plot(x,model)
            ax.plot(x,model_cont)
            ax.annotate("Line Lum 10^42 ergs/s: " + str(round(line_lum/1e42,3))+ " err: "+ str(round(line_lum_err/1e42,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.8),xy=(.1,.8))
            ax.annotate("Mean Squared Error: "+str(round(ols_result.mse_model,6)), xytext=(.1,.6),xy=(.1,.6))
            ax.annotate("R^2: " + str(round(ols_result.rsquared,3)),xytext=(.1,.4),xy=(.1,.4))
            plt.savefig(filepath+'/'+aorkey[:5]+'_'+str(line[i])+'.png')
            plt.close()

            line_lums.append(line_lum)
            line_lums_err.append(line_lum_err)
        else:
            print "Not enough points in region"
            line_lums.append(np.nan)
            line_lums_err.append(np.nan)


    return line_lums,line_lums_err

def pah_6_2(objid,aorkey,detlvl,spectrum,scale,z,filepath):
    #rest frame wavelenth

    spectrum = spectrum.dropna()
    wave = spectrum.wavelength * (1/(z+1.))
    shifted_1 = spectrum.flux_jy[spectrum.wavelength >=  14.06935]
    shifted_2 = spectrum.flux_jy[spectrum.wavelength < 14.06935] * scale
    flux = shifted_2.append(shifted_1) * (1/(z+1.))

    flux_err = spectrum.flux_jy_err * (1/(z+1.))
    wave_dup = wave[wave.duplicated(keep=False)]
    flux_dup = flux[wave.duplicated(keep=False)]
    flux_dup_err = flux_err[wave.duplicated(keep=False)]

    if len(wave[wave.duplicated(keep=False)]) > 0.:
        #cut the first bad ones
        #wave_cut[wave_cut.dupyerlicated(keep='first')]

        i=0
        while len(wave[wave.duplicated(keep=False)]) > 0.:




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

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    dl_mpc = cosmo.luminosity_distance(z) #dl in Mpc
    gammar_r = 0.030
    line = 6.22
    temp_wave = np.arange(5.5,7.0,0.01)

        # determine a fitting window on either side of function
        #cutout a 3sigma region on either side of feature
    sub_wave = wave[(wave >= 5.95) & (wave <= 6.55)]
    sub_flux = flux[(wave >= 5.95) & (wave <= 6.55)]
    sub_flux_err = flux_err[(wave >= 5.95) & (wave <= 6.55)]

    if len(sub_wave) > 3:

        #fit with underlying continuum
        one = sub_wave*0 + 1

        #some centroid noise

        delta=pd.Series(data=np.arange(-.1,.1,.005))
        rms = pd.Series(data=np.zeros(len(delta)))
        for k in range(len(delta)):
            center = line + delta[k]
            drude = gammar_r**2 /((((sub_wave/center) - (center/sub_wave))**2) + gammar_r**2)

            A = [(a,b,c) for a,b,c in zip(one,sub_wave,drude)]

            #can't use sklearn BECAUSE THEY DONT GIVE COVARIANCE MATRIX WTH
            #LR = LinearRegression(fit_intercept=True)
            #LR.fit(A,sub_flux)
            #soln_coeffs = LR.coef_
            #model =  soln_coeffs[0] + soln_coeffs[1]*two + soln_coeffs[2] * gauss

            ols = sm.OLS(sub_flux, A)
            ols_result = ols.fit()

            model =  ols_result.params.const + ols_result.params.x1*sub_wave + ols_result.params.x2 * drude

            rms[k] = np.sqrt(np.sum(((sub_flux-model)**2)/sub_flux_err**2))

        min_delta = delta[rms == rms.min()].values[0]
        center = line + min_delta
        drude = gammar_r**2 /((((sub_wave/center) - (center/sub_wave))**2) + gammar_r**2)


        #A = [(b,c) for b,c in zip(sub_wave,drude)]
        A = [(a,b,c) for a,b,c in zip(one,sub_wave,drude)]

        ols = sm.OLS(sub_flux, A)
        ols_result = ols.fit()

        model =  ols_result.params.const + ols_result.params.x1*sub_wave + ols_result.params.x2 * drude

        temp_model = ols_result.params.const + ols_result.params.x1*temp_wave + ols_result.params.x2 * gammar_r**2 /((((temp_wave/center) - (center/temp_wave))**2) + gammar_r**2)

        model_cont =  ols_result.params.const + ols_result.params.x1 * sub_wave

        #line luminosity in 1e42 ergs/sec

        dl = dl_mpc.value * 3.086e24 #Mpc to cm

        if z == 0:
            dl = 1.

        line_ew = np.pi*.5 *(ols_result.params.x2) * gammar_r * line /(ols_result.params.const + ols_result.params.x1 * line)
        #line_ew_err1 = ((ols_result.params.x2)/(ols_result.params.const + ols_result.params.x1 * line)) * np.sqrt(abs(((ols_result.bse.x2))/ols_result.params.x2) + abs((flux_err[pah]/wave[pah]**2)/use_c_no_ice))


        a = ols_result.params.x2
        b = ols_result.params.const
        c = ols_result.params.x1

        a_err = ols_result.bse.x2
        b_err = ols_result.bse.const
        c_err = ols_result.bse.x1



        a_err_sq = ols_result.bse.x2**2
        b_err_sq = ols_result.bse.const**2
        c_err_sq = ols_result.bse.x1**2

        c1 = (np.pi/2.)* (gammar_r*line)
        c2 = line
        partial_a_sq = (c1 * (b+(c2*c))**-1)**2
        partial_b_sq = (c1 * a * (b+(c2*c))**-2)**2
        partial_c_sq = (c1 * c2 *a * (b+(c2*c))**-2)**2

        line_ew_err = np.sqrt((a_err_sq*partial_a_sq) + (b_err_sq*partial_b_sq) + (c_err_sq*partial_c_sq))

        '''
        ew_err = np.pi*.5 *(a) * gammar_r * line /(b + c * line)

        line_ew_err = unumpy.std_devs(ew_err)
        '''

        line_lum = dl**2*(1/(1+z))* 4*np.pi* (a)* (2.99e14)*gammar_r/((line))* 1e-23* np.pi/2
        line_lum_err = dl**2*(1/(1+z))* 4*np.pi* (ols_result.bse.x2)* (2.99e14)*gammar_r/((line))* 1e-23* np.pi/2
        snr = ols_result.params.x2 / ols_result.bse.x2


        '''HEY LOOK HERE OVERWRITING LINE LUM TO LINE FLUX BECAUSE IM LAZY FOR TEST'''

        line_flux = (1/(1+z))* (ols_result.params.x2)* (2.99e14)*gammar_r/((line))*(1e-23)* np.pi/2 #W m-2
        line_flux_err = (1/(1+z))* (ols_result.bse.x2)* (2.99e14)*gammar_r/((line))*(1e-23)* np.pi/2# W m-2

        #line_lum = line_flux
        #line_lum_err = line_flux_err



        '''

        #DIRECT INTEGRATION

        #ice

        x2 = wave[(wave > 5.9) & (wave < 6.0)]
        y2 = flux[(wave > 5.9) & (wave < 6.0)]

        if len(x2) > 2:


            x2 = (x2[y2==min(y2)]).values[0]
            y2 = (y2[y2==min(y2)]).values[0]


            x3 = wave[(wave > 6.4) & (wave < 6.56)]
            y3 = flux[(wave > 6.4) & (wave < 6.56)]
            x3 = (x3[y3==min(y3)]).values[0]
            y3 = (y3[y3==min(y3)]).values[0]


            spx=[x2,x3]
            spy=[y2,y3]
            points = zip(spx, spy)

            # Sort list of tuples by x-value
            points = sorted(points, key=lambda point: point[0])

            # Split list of tuples into two list of x values any y values
            spx, spy = zip(*points)

            mx = wave[(wave >= x2) & (wave <= x3)]
            my = flux[(wave >= x2) & (wave <= x3)]

            #continuum flux
            flux_lin = sp.interpolate.interp1d(spx, spy, kind='linear')(mx)

            df_flxspl=pd.DataFrame()
            df_flxspl['mx'] = mx
            df_flxspl['flux_lin'] = flux_lin

            pah = (wave >= x2) & (wave <= x3)

            use_flux = (flux[pah] - df_flxspl.flux_lin)/wave[pah]**2

            wave_no_ice = mx[(mx < 6.56) & (mx > 5.9)]

            use_c_no_ice =df_flxspl.flux_lin[(mx < 6.56) & (mx > 5.9)]
            EW_ice=max(sp.integrate.cumtrapz(wave_no_ice,use_flux/use_c_no_ice))

            err_1 = (use_flux/use_c_no_ice) * np.sqrt(abs((flux_err[pah]/wave[pah]**2)/use_flux) + abs((flux_err[pah]/wave[pah]**2)/use_c_no_ice))
            pah_err_ice =(wave[1]-wave[0])*np.sqrt(err_1**2).sum()

            print "EW_ice: " + str(EW_ice)
            print "err: " + str(pah_err_ice)


        # no ice 5.5 to 6.8

        x2 = wave[(wave > 5.5) & (wave < 6.0)]
        y2 = flux[(wave > 5.5) & (wave < 6.0)]

        if len(x2) > 2:


            x2 = (x2[y2==min(y2)]).values[0]
            y2 = (y2[y2==min(y2)]).values[0]


            x3 = wave[(wave > 6.4) & (wave < 6.8)]
            y3 = flux[(wave > 6.4) & (wave < 6.8)]
            x3 = (x3[y3==min(y3)]).values[0]
            y3 = (y3[y3==min(y3)]).values[0]


            spx=[x2,x3]
            spy=[y2,y3]
            points = zip(spx, spy)

            # Sort list of tuples by x-value
            points = sorted(points, key=lambda point: point[0])

            # Split list of tuples into two list of x values any y values
            spx, spy = zip(*points)

            mx = wave[(wave >= x2) & (wave <= x3)]
            my = flux[(wave >= x2) & (wave <= x3)]

            #continuum flux
            flux_lin = sp.interpolate.interp1d(spx, spy, kind='linear')(mx)

            df_flxspl=pd.DataFrame()
            df_flxspl['mx'] = mx
            df_flxspl['flux_lin'] = flux_lin

            pah = (wave >= x2) & (wave <= x3)

            use_flux = (flux[pah] - df_flxspl.flux_lin)/wave[pah]**2

            wave_no_ice = mx[(mx <= x3) & (mx >= x2)]

            use_c_no_ice =df_flxspl.flux_lin[(mx <= x3) & (mx >= x2)]
            EW_ice_wide=max(sp.integrate.cumtrapz(wave_no_ice,use_flux/use_c_no_ice))

            err_1 = (use_flux/use_c_no_ice) * np.sqrt(abs((flux_err[pah]/wave[pah]**2)/use_flux) + abs((flux_err[pah]/wave[pah]**2)/use_c_no_ice))
            pah_err_ice_wide =(wave[1]-wave[0])*np.sqrt(err_1**2).sum()

            print "EW_ice_wide: " + str(EW_ice_wide)
            print "err: " + str(pah_err_ice_wide)

        else:

            err_1 = np.nan
            EW_ice = np.nan
            pah_err_ice = np.nan

                '''



        fig = plt.figure()
        #plt.ion()
        ax = fig.add_subplot(111)
        plt.title(" Aorkey: "+str(aorkey) + ' Line: 6.2)
        plt.suptitle("Detlvl: " + str(detlvl))
        ax.plot(sub_wave,sub_flux,'.')
        ax.errorbar(sub_wave,sub_flux,yerr=sub_flux_err,fmt=None)
        ax.plot(sub_wave,model)
        ax.plot(sub_wave,model_cont)
        ax.annotate("Line Lum 10^42 ergs/s: " + str(round(line_lum/1e42,3))+ " err: "+ str(round(line_lum_err/1e42,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.8),xy=(.1,.8))
        ax.annotate("Mean Squared Error: "+str(round(ols_result.mse_model,6)), xytext=(.1,.6),xy=(.1,.6))
        ax.annotate("R^2: " + str(round(ols_result.rsquared,3)),xytext=(.1,.4),xy=(.1,.4))
        plt.savefig(filepath+'/'+aorkey+'_'+str(line)+'.png')
        plt.close()

        '''
        fig = plt.figure()
        #plt.ion()
        ax = fig.add_subplot(111)
        plt.title(" Aorkey: "+str(aorkey) + ' Line' + str(line))
        plt.suptitle("Detlvl: " + str(detlvl))
        ax.plot(sub_wave,sub_flux,'.',label='Points fit via Drude')
        ax.plot(mx,my,'.',label ='Points used via Direct Int')
        ax.errorbar(sub_wave,sub_flux,yerr=sub_flux_err,fmt=None)
        ax.plot(sub_wave,model,label='Drude model')
        ax.plot(sub_wave,model_cont,label='No Ice Continuum')
        ax.plot(mx,df_flxspl.flux_lin,label='Ice Continuum')
        ax.plot(spx,spy,'o',label ='Ice Anchors')
        ax.annotate("Continuum Treatment Chosen:  " + cont_flag,xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.8),xy=(.1,.8))
        plt.legend()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(filepath+'/'+aorkey+'_'+str(line)+'.png')
        plt.close()
        '''

    else:
        line_ew = np.nan
        line_ew_err = np.nan
        line_lum=np.nan
        line_lum_err = np.nan
        snr = np.nan
        line_flux = np.nan
        line_flux_err = np.nan

    return line_lum,line_lum_err,line_ew,line_ew_err,line_flux,line_flux_err


def pah_11_3(objid,aorkey,detlvl,spectrum,scale,z,filepath):
    #rest frame wavelenth
    spectrum = spectrum.dropna()
    wave = spectrum.wavelength * (1/(z+1.))
    shifted_1 = spectrum.flux_jy[spectrum.wavelength >=  14.06935]
    shifted_2 = spectrum.flux_jy[spectrum.wavelength < 14.06935] * scale
    flux = shifted_2.append(shifted_1) * (1/(z+1.))

    flux_err = spectrum.flux_jy_err * (1/(z+1.))
    wave_dup = wave[wave.duplicated(keep=False)]
    flux_dup = flux[wave.duplicated(keep=False)]
    flux_dup_err = flux_err[wave.duplicated(keep=False)]

    if len(wave[wave.duplicated(keep=False)]) > 0.:
        #cut the first bad ones
        #wave_cut[wave_cut.dupyerlicated(keep='first')]

        i=0
        while len(wave[wave.duplicated(keep=False)]) > 0.:
            print i



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

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    dl_mpc = cosmo.luminosity_distance(z) #dl in Mpc
    gammar_r = 0.032
    gammar_n = 0.012
    fratio = 1.25
    line = 11.33
    line_n = 11.23
    temp_wave = np.arange(10.0,12.5,0.01)

        # determine a fitting window on either side of function
        #cutout a 3sigma region on either side of feature
    sub_wave = wave[((wave >= 10.0) & (wave <= 10.4)) | ((wave >= 10.6) & (wave <= 12.5))]
    sub_flux = flux[((wave >= 10.0) & (wave <= 10.4)) | ((wave >= 10.6) & (wave <= 12.5))]
    sub_flux_err = flux_err[((wave >= 10.0) & (wave <= 10.4)) | ((wave >= 10.6) & (wave <= 12.5))]

    sub_wave2= (sub_wave-line)**2
    sub_wave3 = (sub_wave - line)**3
    if len(sub_wave) > 3:

            #fine scale x
        #fit with underlying continuum
        one = sub_wave*0 + 1.0

        #some centroid noise

        delta=pd.Series(data=np.arange(-.03,.03,.005))
        rms = pd.Series(data=np.zeros(len(delta)))
        for k in range(len(delta)):
            center = line + delta[k]
            center_n = line_n + delta[k]
            drude = (gammar_r**2 /((((sub_wave/center) - (center/sub_wave))**2) + gammar_r**2)) + (fratio * gammar_n**2 /((((sub_wave/center_n) - (center_n/sub_wave))**2) + gammar_n**2))

            A = [(a,b,c,d,e) for a,b,c,d,e in zip(one,sub_wave,sub_wave2,sub_wave3,drude)]

            #can't use sklearn BECAUSE THEY DONT GIVE COVARIANCE MATRIX WTH
            #LR = LinearRegression(fit_intercept=True)
            #LR.fit(A,sub_flux)
            #soln_coeffs = LR.coef_
            #model =  soln_coeffs[0] + soln_coeffs[1]*two + soln_coeffs[2] * gauss

            ols = sm.OLS(sub_flux, A)
            ols_result = ols.fit()

            model =  ols_result.params.const + ols_result.params.x1*sub_wave + ols_result.params.x2 * sub_wave2 + ols_result.params.x3 * sub_wave3 + ols_result.params.x4 * drude

            rms[k] = np.sqrt(np.sum(((sub_flux-model)**2)/sub_flux_err**2))

        min_delta = delta[rms == rms.min()].values[0]
        center = line + min_delta
        center_n = line_n + min_delta
        drude = (gammar_r**2 /((((sub_wave/center) - (center/sub_wave))**2) + gammar_r**2)) + (fratio * gammar_n**2 /((((sub_wave/center_n) - (center_n/sub_wave))**2) + gammar_n**2))
        A = [(a,b,c,d,e) for a,b,c,d,e in zip(one,sub_wave,sub_wave2,sub_wave3,drude)]
        ols = sm.OLS(sub_flux, A)
        ols_result = ols.fit()

        model =  ols_result.params.const + ols_result.params.x1*sub_wave + ols_result.params.x2 * sub_wave2 + ols_result.params.x3 * sub_wave3 + ols_result.params.x4 * drude

        temp_model = ols_result.params.const + ols_result.params.x1 * temp_wave +ols_result.params.x2 * (temp_wave - line)**2 +ols_result.params.x3 * (temp_wave - line)**3 + ols_result.params.x4 *(gammar_r**2)/(((temp_wave/center) - (center/temp_wave))**2 + gammar_r**2)+ols_result.params.x4 *fratio*(gammar_n**2)/ (((temp_wave/center_n) - (center_n/temp_wave))**2 + gammar_n**2)

        temp_cont =  ols_result.params.const + ols_result.params.x1 * temp_wave +ols_result.params.x2 * (temp_wave - line)**2 +ols_result.params.x3 * (temp_wave - line)**3

        #line luminosity in 1e42 ergs/sec
        import uncertainties.unumpy as unumpy
        dl = dl_mpc.value * 3.086e24 #Mpc to cm
        #line_ew_err1 = ((ols_result.params.x2)/(ols_result.params.const + ols_result.params.x1 * line)) * np.sqrt(abs(((ols_result.bse.x2))/ols_result.params.x2) + abs((flux_err[pah]/wave[pah]**2)/use_c_no_ice))
        a = ols_result.params.x4 #drude
        b = ols_result.params.const #baseline
        c = ols_result.params.x1 #0th order coeff
        d = ols_result.params.x2 #2nd order coeff
        e = ols_result.params.x3 # 3rd order coeff

        a_err = ols_result.bse.x4
        b_err = ols_result.bse.const
        c_err = ols_result.bse.x1
        d_err = ols_result.bse.x2
        e_err = ols_result.bse.x3



        a = unumpy.uarray(( a,a_err ))
        b = unumpy.uarray(( b,b_err ))
        c = unumpy.uarray(( c,c_err ))
        d = unumpy.uarray(( d,d_err ))
        e = unumpy.uarray(( e,e_err ))


        line_ew = np.pi*.5 *(a) * (gammar_r*line) /(b + (c * line) + d*(line - line_n)**2 + e*(line - line_n)**3)
        line_ew_err = unumpy.std_devs(line_ew)

        line_ew = unumpy.nominal_values(a)



        '''
        a_err_sq = ols_result.bse.x4**2
        b_err_sq = ols_result.bse.const**2
        c_err_sq = ols_result.bse.x1**2
        d_err_sq = ols_result.bse.x2**2
        e_err_sq = ols_result.bse.x3**2



        c1 = (np.pi/2.) * (gammar_r*line + gammar_n*fratio*line_n)
        c2 = line
        c3 = line - line_n
        c4 = line - line_n

        c_parens = c1 * (b + c2*c + c3*d**2 + c4*e**3)**-1
        c_parens_sq = c1 * (b + c2*c + c3*d**2 + c4*e**3)**-1

        partial_a_sq = (c_parens)**2
        partial_b_sq = (a * c_parens_sq)**2
        partial_c_sq = (a*c2 * c_parens_sq)**2
        partial_d_sq = (2*a*c3*d * c_parens_sq)**2
        partial_e_sq = (a*c4*3*(e**2) * c_parens_sq)**2

        ew_err = np.sqrt((a_err_sq*partial_a_sq) + (b_err_sq*partial_b_sq) + (c_err_sq*partial_c_sq) + (d_err_sq*partial_d_sq) + (e_err_sq*partial_e_sq))
        '''
        #ew_err =  ((np.pi/2.)/(ols_result.params.const + line_1*ols_result.params.x1)**2)*((ols_result.bse.x2**2) + (ols_result.params.x2**2)*((ols_result.params.const)**2 + (line_1*ols_result.params.x1)**2))



        if z == 0:
            dl = 1.


        line_lum = dl**2*(1/(1+z))* 4*np.pi* (a)* (2.99e14)*(gammar_r + gammar_n*fratio)/((line))* 1e-23* np.pi/2
        line_lum_err = unumpy.std_devs(line_lum)

        line_lum = unumpy.nominal_values(line_lum)


        '''
        line_lum = dl**2*(1/(1+z))* 4*np.pi* (ols_result.params.x4)* (2.99e14)*(gammar_r + gammar_n*fratio)/((line))* 1e-23* np.pi/2
        line_lum_err = dl**2*(1/(1+z))* 4*np.pi* (ols_result.bse.x4)* (2.99e14)*(gammar_r + gammar_n*fratio)/((line))* 1e-23* np.pi/2

        line_flux = (1/(1+z))* (ols_result.params.x2)* (2.99e14)*gammar_r/((line))*(1e-23)* np.pi/2 #W m-2
        line_flux_err = (1/(1+z))* (ols_result.bse.x2)* (2.99e14)*gammar_r/((line))*(1e-23)* np.pi/2# W m-2


        '''
        snr = ols_result.params.x2 / ols_result.bse.x2


        line_flux = (1/(1+z))* (ols_result.params.x2)* (2.99e14)*gammar_r/((line))*(1e-23)* np.pi/2 #W m-2
        line_flux_err = (1/(1+z))* (ols_result.bse.x2)* (2.99e14)*gammar_r/((line))*(1e-23)* np.pi/2# W m-2

        #line_lum = line_flux
        #line_lum_err = line_flux_err

        fig = plt.figure()
        #plt.ion()
        ax = fig.add_subplot(111)
        plt.title(" Aorkey: "+str(aorkey) + ' Line: 11.3')
        plt.suptitle("Detlvl: " + str(detlvl))
        ax.plot(sub_wave,sub_flux,'.')
        ax.errorbar(sub_wave,sub_flux,yerr=sub_flux_err,fmt=None)
        ax.plot(temp_wave,temp_model)
        ax.plot(temp_wave,temp_cont)
        ax.annotate("Line Lum 10^42 ergs/s: " + str(round(line_lum/1e42,3))+ " err: "+ str(round(line_lum_err/1e42,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.8),xy=(.1,.8))
        ax.annotate("Mean Squared Error: "+str(round(ols_result.mse_model,6)),xycoords='axes fraction',textcoords='axes fraction', xytext=(.1,.6),xy=(.1,.6))
        ax.annotate("R^2: " + str(round(ols_result.rsquared,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.4),xy=(.1,.4))
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(filepath+'/'+aorkey+'_'+str(line)+'.png')
        plt.close()


    else:
        line_ew = np.nan
        line_ew_err = np.nan
        line_lum=np.nan
        line_lum_err = np.nan
        snr = np.nan

    return line_lum,line_lum_err,line_ew,line_ew_err

def pah_7_7(objid,aorkey,detlvl,spectrum,scale,z,filepath):
    #rest frame wavelenth

    spectrum = spectrum.dropna()
    wave = spectrum.wavelength * (1/(z+1.))
    shifted_1 = spectrum.flux_jy[spectrum.wavelength >=  14.06935]
    shifted_2 = spectrum.flux_jy[spectrum.wavelength < 14.06935] * scale
    flux = shifted_2.append(shifted_1) * (1/(z+1.))

    flux_err = spectrum.flux_jy_err * (1/(z+1.))
    wave_dup = wave[wave.duplicated(keep=False)]
    flux_dup = flux[wave.duplicated(keep=False)]
    flux_dup_err = flux_err[wave.duplicated(keep=False)]

    if len(wave[wave.duplicated(keep=False)]) > 0.:
        #cut the first bad ones
        #wave_cut[wave_cut.dupyerlicated(keep='first')]

        i=0
        while len(wave[wave.duplicated(keep=False)]) > 0.:
            print i



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


    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    dl_mpc = cosmo.luminosity_distance(z) #dl in Mpc
    gammar_1 = 0.126
    gammar_2 = 0.044
    gammar_3 = 0.053
    fratio1 = 2.50
    line_1 = 7.417
    line_2 = 7.598
    line_3 = 7.850
    fratio2 = 2.36

    temp_wave = np.arange(6.5,10,0.01)
    sub_wave = wave[(wave >= 7.0) & (wave <= 8.2)]
    sub_flux = flux[(wave >= 7.0) & (wave <= 8.2)]
    sub_flux_err = flux_err[(wave >= 7.0) & (wave <= 8.2)]

        # determine a fitting window on either side of function
        #cutout a 3sigma region on either side of feature


    sub_wave2 = (sub_wave - line_1)**2
    sub_wave3 = (sub_wave - line_1)**3




    if len(sub_wave) > 3:

            #fine scale x
        #fit with underlying continuum
        one = sub_wave*0 + 1.0

        #some centroid noise

        delta=pd.Series(data=np.arange(-.03,.03,.005))
        rms = pd.Series(data=np.zeros(len(delta)))
        for k in range(len(delta)):
            center_1 = line_1 + delta[k]
            center_2 = line_2 + delta[k]
            center_3 = line_3 + delta[k]
            drude = (gammar_1**2 /((((sub_wave/center_1) - (center_1/sub_wave))**2) + gammar_1**2))
            drude = drude +  (gammar_2**2 /((((sub_wave/center_2) - (center_2/sub_wave))**2) + gammar_2**2))
            drude = drude +  (gammar_3**2 /((((sub_wave/center_3) - (center_3/sub_wave))**2) + gammar_3**2))

            A = [(a,b,c) for a,b,c in zip(one,sub_wave,drude)]


            ols = sm.OLS(sub_flux, A)
            ols_result = ols.fit()


            model =  ols_result.params.const + ols_result.params.x1*sub_wave+  ols_result.params.x2*drude
            rms[k] = np.sqrt(np.sum(((sub_flux-model)**2)/sub_flux_err**2))

        min_delta = delta[rms == rms.min()].values[0]
        center_1 = line_1 + min_delta
        center_2 = line_2 + min_delta
        center_3 = line_3 + min_delta
        drude = (gammar_1**2 /((((sub_wave/center_1) - (center_1/sub_wave))**2) + gammar_1**2))
        drude = drude +  (gammar_2**2 /((((sub_wave/center_2) - (center_2/sub_wave))**2) + gammar_2**2))
        drude = drude +  (gammar_3**2 /((((sub_wave/center_3) - (center_3/sub_wave))**2) + gammar_3**2))

        A = [(a,b,c) for a,b,c in zip(one,sub_wave,drude)]


        ols = sm.OLS(sub_flux, A)
        ols_result = ols.fit()


        model =  ols_result.params.const + ols_result.params.x1*sub_wave+  ols_result.params.x2*drude
#        temp_model = ols_result.params.const +ols_result.params.x1*temp_wave + ols_result.params.x2 * ((gammar_1**2 /((((temp_wave/center_1) - (center_1/temp_wave))**2) + gammar_1**2)) + ((gammar_2**2)/((((temp_wave/center_2) - (center_2/temp_wave))**2) + gammar_2**2)) + ((gammar_3**2)/((((temp_wave/center_3) - (center_3/temp_wave))**2) + gammar_3**2)))

#        temp_cont =  ols_result.params.const + temp_wave*ols_result.params.x1
        #line luminosity in 1e42 ergs/sec

        dl = dl_mpc.value * 3.086e24 #Mpc to cm

#        line_lum = dl**2*(1/(1+z))* 4*np.pi* (ols_result.params.x2)* (2.99e14)*(fratio2*gammar_2+fratio1*gammar_1+ gammar_r)/((line_1))* 1e-23* np.pi/2
#        line_lum_err = dl**2*(1/(1+z))* 4*np.pi* (ols_result.bse.x2)* (2.99e14)*(fratio2*gammar_2+fratio1*gammar_1+ gammar_r)/((line_1))* 1e-23* np.pi/2


        if z == 0:
            dl = 1.

        line_lum = dl**2*(1/(1+z))* 4*np.pi* (ols_result.params.x2)* (2.99e14)*(gammar_1)/((line_1))* 1e-23* np.pi/2
        line_lum_err = dl**2*(1/(1+z))* 4*np.pi* (ols_result.bse.x2)* (2.99e14)*(gammar_1)/((line_1))* 1e-23* np.pi/2


        #use_c_no_ice =temp_cont/wave**2
        EW= (np.pi/2.) * ((ols_result.params.x2)*(gammar_1*line_1))/(ols_result.params.const + line_1*ols_result.params.x1)
        '''
        a = ols_result.params.x2
        b = ols_result.params.const
        c = ols_result.params.x1

        a_err_sq = ols_result.bse.x2**2
        b_err_sq = ols_result.bse.const**2
        c_err_sq = ols_result.bse.x1**2

        c1 = (np.pi/2.) *(fratio2*gammar_3*line_3+fratio1*gammar_2*line_2+ gammar_1*line_1)
        c2 = line_1
        partial_a_sq = (c1 * (b+(c2*c))**-1)**2
        partial_b_sq = (c1 * a * (b+(c2*c))**-2)**2
        partial_c_sq = (c1 * c2 *a * (b+(c2*c))**-2)**2

        ew_err = np.sqrt((a_err_sq*partial_a_sq) + (b_err_sq*partial_b_sq) + (c_err_sq*partial_c_sq))
        '''
        ew_err =  ((np.pi/2.)/(ols_result.params.const + line_1*ols_result.params.x1)**2)*((ols_result.bse.x2**2) + (ols_result.params.x2**2)*((ols_result.params.const)**2 + (line_1*ols_result.params.x1)**2))




        fig = plt.figure()
        #plt.ion()
        ax = fig.add_subplot(111)
        plt.title(" Aorkey: "+str(aorkey) + ' PAH 7_7' )
        plt.suptitle("Detlvl: " + str(detlvl))
        ax.errorbar(sub_wave,sub_flux,yerr=sub_flux_err,fmt='.')
        ax.errorbar(wave[(wave >= 6.5) & (wave <= 9.0)],flux[(wave >= 6.5) & (wave <= 9.0)],yerr=flux_err[(wave >= 6.5) & (wave <= 9.0)],fmt='.')
        ax.plot(sub_wave,model)
        #ax.plot(temp_wave,temp_cont)
        ax.annotate("Line Lum 10^42 ergs/s: " + str(round(line_lum/1e42,3))+ " err: "+ str(round(line_lum_err/1e42,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.8),xy=(.1,.8))
        ax.annotate("Mean Squared Error: "+str(round(ols_result.mse_model,6)),xycoords='axes fraction',textcoords='axes fraction', xytext=(.1,.6),xy=(.1,.6))
        ax.annotate("R^2: " + str(round(ols_result.rsquared,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.4),xy=(.1,.4))
        plt.savefig(filepath+'/'+str(aorkey)+'_7_7'+'.png')
        plt.close()



    else:
        line_lum=np.nan
        line_lum_err = np.nan
        EW = np.nan
        ew_err = np.nan

    return line_lum,line_lum_err,EW,ew_err

def pah_8_5(objid,aorkey,detlvl,spectrum,scale,z,fittype,filepath):
    #rest frame wavelenth

    spectrum = spectrum.dropna()
    wave = spectrum.wavelength * (1/(z+1.))
    shifted_1 = spectrum.flux_jy[spectrum.wavelength >=  14.06935]
    shifted_2 = spectrum.flux_jy[spectrum.wavelength < 14.06935] * scale
    flux = shifted_2.append(shifted_1)
    flux = spectrum.flux_jy
    flux_err = spectrum.flux_jy_err

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    dl_mpc = cosmo.luminosity_distance(z) #dl in Mpc
    gammar_r = 0.050
    gammar_n = 0.039
    fratio = 2.18
    line = 8.33
    line_n = 8.61
    temp_wave = np.arange(7.8,9.5,0.01)

        # determine a fitting window on either side of function
        #cutout a 3sigma region on either side of feature
    sub_wave = wave[(wave >= 7.8) & (wave <= 9.5)]
    sub_flux = flux[(wave >= 7.8) & (wave <= 9.5)]
    sub_flux_err = flux_err[(wave >= 7.8) & (wave <= 9.5)]

    sub_wave2= (sub_wave-line)**2
    sub_wave3 = (sub_wave - line)**3
    if len(sub_wave) > 3:

            #fine scale x
        #fit with underlying continuum
        one = sub_wave*0 + 1.0

        #some centroid noise

        delta=pd.Series(data=np.arange(-.03,.03,.005))
        rms = pd.Series(data=np.zeros(len(delta)))
        for k in range(len(delta)):
            center = line + delta[k]
            center_n = line_n + delta[k]
            drude = (gammar_r**2 /((((sub_wave/center) - (center/sub_wave))**2) + gammar_r**2)) + (fratio * gammar_n**2 /((((sub_wave/center_n) - (center_n/sub_wave))**2) + gammar_n**2))

            A = [(a,b,c,d,e) for a,b,c,d,e in zip(one,sub_wave,sub_wave2,sub_wave3,drude)]

            #can't use sklearn BECAUSE THEY DONT GIVE COVARIANCE MATRIX WTH
            #LR = LinearRegression(fit_intercept=True)
            #LR.fit(A,sub_flux)
            #soln_coeffs = LR.coef_
            #model =  soln_coeffs[0] + soln_coeffs[1]*two + soln_coeffs[2] * gauss

            ols = sm.OLS(sub_flux, A)
            ols_result = ols.fit()

            model =  ols_result.params.const + ols_result.params.x1*sub_wave + ols_result.params.x2 * sub_wave2 + ols_result.params.x3 * sub_wave3 + ols_result.params.x4 * drude

            rms[k] = np.sqrt(np.sum(((sub_flux-model)**2)/sub_flux_err**2))

        min_delta = delta[rms == rms.min()].values[0]
        center = line + min_delta
        center_n = line_n + min_delta
        drude = (gammar_r**2 /((((sub_wave/center) - (center/sub_wave))**2) + gammar_r**2)) + (fratio * gammar_n**2 /((((sub_wave/center_n) - (center_n/sub_wave))**2) + gammar_n**2))
        A = [(a,b,c,d,e) for a,b,c,d,e in zip(one,sub_wave,sub_wave2,sub_wave3,drude)]
        ols = sm.OLS(sub_flux, A)
        ols_result = ols.fit()

        model =  ols_result.params.const + ols_result.params.x1*sub_wave + ols_result.params.x2 * sub_wave2 + ols_result.params.x3 * sub_wave3 + ols_result.params.x4 * drude

        temp_model = ols_result.params.const + ols_result.params.x1 * temp_wave +ols_result.params.x2 * (temp_wave - line)**2 +ols_result.params.x3 * (temp_wave - line)**3 + ols_result.params.x4 *(gammar_r**2)/(((temp_wave/center) - (center/temp_wave))**2 + gammar_r**2)+ols_result.params.x4 *fratio*(gammar_n**2)/ (((temp_wave/center_n) - (center_n/temp_wave))**2 + gammar_n**2)

        temp_cont =  ols_result.params.const + ols_result.params.x1 * temp_wave +ols_result.params.x2 * (temp_wave - line)**2 +ols_result.params.x3 * (temp_wave - line)**3

        #line luminosity in 1e42 ergs/sec

        dl = dl_mpc.value * 3.086e24 #Mpc to cm

        line_lum = dl**2*(1/(1+z))* 4*np.pi* (ols_result.params.x4)* (2.99e14)*(fratio*gammar_n+ gammar_r)/((line))* 1e-23* np.pi/2
        line_lum_err = dl**2*(1/(1+z))* 4*np.pi* (ols_result.bse.x4)* (2.99e14)*(fratio*gammar_n+ gammar_r)/((line))* 1e-23* np.pi/2
        snr = ols_result.params.x4 / ols_result.bse.x4
        fig = plt.figure()
        #plt.ion()
        ax = fig.add_subplot(111)
        plt.title(" Aorkey: "+str(aorkey) + ' PAH' + str(line))
        plt.suptitle("Detlvl: " + str(detlvl))
        ax.plot(sub_wave,sub_flux,'.')
        ax.errorbar(sub_wave,sub_flux,yerr=sub_flux_err,fmt=None)
        ax.plot(temp_wave,temp_model)
        ax.plot(temp_wave,temp_cont)
        ax.annotate("Line Lum 10^42 ergs/s: " + str(round(line_lum/1e42,3))+ " err: "+ str(round(line_lum_err/1e42,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.8),xy=(.1,.8))
        ax.annotate("Mean Squared Error: "+str(round(ols_result.mse_model,6)),xycoords='axes fraction',textcoords='axes fraction', xytext=(.1,.6),xy=(.1,.6))
        ax.annotate("R^2: " + str(round(ols_result.rsquared,3)),xycoords='axes fraction',textcoords='axes fraction',xytext=(.1,.4),xy=(.1,.4))
        plt.savefig(filepath+'/'+aorkey+'_'+str(line)+'.png')
        plt.close()



    else:
        line_lum=np.nan
        line_lum_err = np.nan
        snr = np.nan

    return line_lum,line_lum_err,snr
