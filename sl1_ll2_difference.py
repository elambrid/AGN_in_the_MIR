import pandas as pd
import numpy as np
from astropy.io import fits
import heapq
import scipy.stats as scp
import matplotlib.pyplot as plt
from combine_bonus_orders import bonus_comb as bcomb


'''
Flag Key:
-222: probably no problem, noisy spec, jump detected away from index 300
-444 : no SL data
-555 : an abs feature on LL2 side
-666 : Object not detected in slit (detlvl <4)
-888 : a point is 6 sigma above std, right on the slit jump - probably a feature, flagging for now
-999 : Spectra has no jump at SL1 LL2 boundary


'''
keyMaps = [key for key in plt.rcParams.keys() if 'keymap.' in key]
for keyMap in keyMaps:
    plt.rcParams[keyMap] = ''

q_flag = ''

def press(event):
    global q_flag
    if event.key in ['1','2','3']:
        q_flag = event.key

def flux_diff(aorkey):
    try:
        spec_fits = fits.open("/home/erini/AGN_zoo/Data_Mine/cassis/cassis_lowres_fits/"+aorkey)
        spitzer_spec = pd.DataFrame()
        spitzer_spec['wavelength'] = spec_fits[0].data[:,0]
        spitzer_spec['flux_jy'] = spec_fits[0].data[:,1]
        spitzer_spec['flux_jy_err'] = spec_fits[0].data[:,2]
        spitzer_spec=spitzer_spec.sort('wavelength')
        extent = spec_fits[0].header['EXTENT']
        print aorkey
        spitzer_spec=spitzer_spec.sort('wavelength')
        spitzer_spec = bcomb(spitzer_spec)

        #First we check if this object was observed in SL1 abd LL2

        if spec_fits[0].header['DETLVL'] < 3:
            print "spectra is not well detected"
            return 1.

        if (spitzer_spec.wavelength.min() > 13.0) or (spitzer_spec.wavelength.max() < 14.0 ) or (len(spitzer_spec.wavelength[(spitzer_spec.wavelength > 14.0) & (spitzer_spec.wavelength < 15.0)]) == 0):
            print "No SL1 Data, or no LL2 Data"
            return 1.
        else:

            #make a cut and only work with points in between 13.5, and 14.6 microns - to ensure we get the slit and
            #some continuum

            a = spitzer_spec.flux_jy[(spitzer_spec.wavelength < 14.5) & (spitzer_spec.wavelength > 13.5)]
            d = spitzer_spec.wavelength.loc[a.index]

            dnorm = d.iloc[1]-d.iloc[0]

            #1st order finite difference
            da = a.diff(1) / dnorm

            #if there is a max of the 1st order finite difference anywhere other then the SL1 LL2 boundary
            # then there probably is not a significant jump and/or we need a more sophisticated stitching method



            d_after = spitzer_spec.wavelength[spitzer_spec.wavelength > 14.0]

            y_avg = (spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.069350].values[0] + spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.154030].values[0])/2.
            y_diff = spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.069350].values[0] - y_avg
            scale = (spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.069350].values[0] +y_diff)/spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.154030].values[0]
            if scale < 0:
                print " an abs feature, greater than a possible flux jump"
                scale = 1.
                return scale



            if ((d[da == da.max()] > 14.1).bool() or (d[da == da.max()] < 14.0).bool()):
                '''
                if (scale > 1.01) & spec_fits[0].header['DETLVL'] <= 6:
                    print "probably no problem, noisy spec, jump detected away from index 300"

                '''
                print "Spectra is Fine - just scaling to diference of points"
                #hardcoding where i know the jump is, this number should be close to 1
                '''
                d_after = spitzer_spec.wavelength[spitzer_spec.wavelength > 14.0]

                y_avg = (spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.069350].values[0] + spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.154030].values[0])/2.
                y_diff = spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.069350].values[0] - y_avg
                scale = (spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.069350].values[0] +y_diff)/spitzer_spec.flux_jy[spitzer_spec.wavelength == 14.154030].values[0]

                if (scale < 0) or (scale < 1.01):
                    print "no jump or nothing"
                    return 1, 1

                if scale > 1:
                    print "Abs feature on other side, y_avg doesnt doesnt work"
                    return -555,-555
                '''
                scale =1.
            else:

                #split the region into before jump (pre) and after jump (after)
                #since we are fitting continuum, we also want to remove points that may be on a feature for the continuum
                #fit, so we do another finite difference and choose the points then are less the spread of the finite difference
                #in this region

                #there also might a partial line ON the jump

                #forward

                a_pre_with = a.loc[:a[da == da.max()].index[0]-1]
                d_pre_with = spitzer_spec.wavelength.loc[a_pre_with.index]
                dd_pre_with = d_pre_with.iloc[1] -  d_pre_with.iloc[0]

                da_pre_with = a_pre_with.diff()/dd_pre_with
                a_pre_without =a_pre_with[da_pre_with.abs() < da_pre_with.std()]
                d_pre_without = spitzer_spec.wavelength.loc[a_pre_without.index]


                a_after_with = a.loc[a[da == da.max()].index[0]:]
                d_after_with = spitzer_spec.wavelength.loc[a_after_with.index]

                dd_after_with = d_after_with.iloc[1] - d_after_with.iloc[0]
                da_after_with = a_after_with.diff()/dd_after_with

                a_after_without=a_after_with[da_after_with.abs() < da_after_with.std()]
                d_after_without = spitzer_spec.wavelength.loc[a_after_without.index]





                #now lets see if our point exlusion did a good job, if there aren't any jumps, then a line fit of
                #all the points should be a better fit than less points, we determine better fit with an R^2

                a_pre_with_fit = np.polyfit(d_pre_with,a_pre_with,1)
                p_pre_with = np.poly1d(a_pre_with_fit)

                yhat_pre= p_pre_with(d_pre_with)
                ybar_pre = np.sum(a_pre_with)/len(a_pre_with)
                ssreg_pre = np.sum((yhat_pre-ybar_pre)**2)
                sstot_pre = np.sum((a_pre_with - ybar_pre)**2)
                r_squared_pre = ssreg_pre / sstot_pre

                if r_squared_pre < .9:
                    #print "pre w/o"
                    a_pre_fit = np.polyfit(d_pre_without,a_pre_without,1)
                    p_pre = np.poly1d(a_pre_fit)
                    a_pre = a_pre_without
                    d_pre = d_pre_without

                else:

                    a_pre_fit = a_pre_with_fit
                    p_pre = p_pre_with
                    a_pre = a_pre_with
                    d_pre = d_pre_with


                a_after_with_fit = np.polyfit(d_after_with,a_after_with,1)
                p_after_with = np.poly1d(a_after_with_fit)

                yhat_after= p_after_with(d_after_with)
                ybar_after = np.sum(a_after_with)/len(a_after_with)
                ssreg_after = np.sum((yhat_after-ybar_after)**2)
                sstot_after = np.sum((a_after_with - ybar_after)**2)
                r_squared_after = ssreg_after / sstot_after

                if r_squared_after < .9:
                    #print " after w/o"
                    a_after_fit = np.polyfit(d_after_without,a_after_without,1)
                    p_after = np.poly1d(a_after_fit)
                    a_after = a_after_without
                    d_after = d_after_without

                else:

                    a_after_fit = a_after_with_fit
                    p_after = p_after_with
                    a_after = a_after_with
                    d_after = d_after_with


                #if the slopes of the both lines aren't within the line fit error
                #equal to one another AND the slope of the region on the other
                #side of the diff slit is greater, then remove points from left of
                #rightmost line until it is, if it gets down to 3 points then flag it

                a_after_fit = np.polyfit(d_after,a_after,1)
                p_after = np.poly1d(a_after_fit)

                a_after_fit_r = np.polyfit(d_after,a_after,1)
                p_after = np.poly1d(a_after_fit)


                a_pre_fit = np.polyfit(d_pre,a_pre,1)
                p_pre = np.poly1d(a_pre_fit)

                a_after_r = a_after
                slope_ratio_r = 0
                slope_ratio_l = 0
                while (abs(a_after_fit[0]/a_pre_fit[0]) >= 2) & (len(a_after) >= 4.):
                        print len(a_after)
                        a_after = a_after[1:]
                        d_after = d[a_after.index]
                        a_after_fit = np.polyfit(d_after,a_after,1)
                        p_after = np.poly1d(a_after_fit)
                        slope_ratio_l = abs(a_after_fit[0]/a_pre_fit[0])

                while (abs(a_after_fit_r[0]/a_pre_fit[0]) >= 2) & (len(a_after_r) >= 4.):
                        print len(a_after)
                        a_after_r = a_after_r[:-1]
                        d_after_r = d[a_after_r.index]
                        a_after_fit_r = np.polyfit(d_after_r,a_after_r,1)
                        p_after_r = np.poly1d(a_after_fit_r)
                        slope_ratio_r = abs(a_after_fit_r[0]/a_pre_fit[0])

                if (slope_ratio_r > 0) & (slope_ratio_r < slope_ratio_l):
                        a_after = a_after_r
                        d_after = d_after_r
                        a_after_fit = a_after_fit_r
                        p_after = p_after_r



                '''
                For plotting

                plt.plot(d,a,'.',d_pre,p_pre(d_pre),d,p_after(d))
                plt.plot(d_pre.iloc[-1],a_pre.iloc[-1],'r.')
                plt.plot(d_after.iloc[0],a_after.iloc[0],'g.')
                plt.show()


                plt.plot(d_pre,p_pre(d_pre),'.')
                plt.plot(d_after,p_after(d_after),'.')
                plt.show()




                plt.plot(d_pre_without,a_pre_fit[0]*d_pre_without + a_pre_fit[1])
                plt.plot(d_without,a_without_fit[0]*d_without+ a_without_fit[1])

                plt.plot(d,a,'.')
                plt.show()
                '''

                #now shift
                #y_pred = a_without_fit[0]*d_pre_without.iloc[-1]+ a_without_fit[1]
                #y_diff = a_pre_without.iloc[-1] - y_pred

                y_pred = p_after(d_pre.iloc[-1])
                y_diff = a_pre.iloc[-1] - y_pred
                scale = (spitzer_spec.flux_jy[spitzer_spec.wavelength == d_after.iloc[0]].values[0] +y_diff)/spitzer_spec.flux_jy[spitzer_spec.wavelength == d_after.iloc[0]].values[0]


                #now we also want to always fall back on...if the predicted value is greater than the actual value..to use
                # the actual value

                if y_pred > a_after.iloc[0]:
                    y_diff = a_pre_with.iloc[-1] - a_after_with.iloc[0]
                    scale = (spitzer_spec.flux_jy[spitzer_spec.wavelength == d_after.iloc[0]].values[0] +y_diff)/spitzer_spec.flux_jy[spitzer_spec.wavelength == d_after.iloc[0]].values[0]
                    if scale < 0.:
                        y_diff = a_after.iloc[0] - a_pre.iloc[-1]
                        return 1.

                if y_diff > 0.: #scaling up, probably do to an abs feature
                    print scale
                    return 1.

                if extent < .4:
                    #our whole hypothesis fails if this is a compact point source object, other problems must be going on
                    scale = 1.
                '''
                #get scale factor
                scale = (spitzer_spec.flux_jy[spitzer_spec.wavelength == d_after.iloc[0]].values[0] +y_diff)/spitzer_spec.flux_jy[spitzer_spec.wavelength == d_after.iloc[0]].values[0]
                #print d_after.loc[
                '''
                if 1./scale < 1.:
                    print " an abs feature, greater than a possible flux jump"
                    scale = 1.
                '''
                if (abs(y_diff) < spitzer_spec.flux_jy_err[spitzer_spec.wavelength >= d_after.iloc[0]].iloc[0]) and (extent < 2):
                    print " Scale jump close to error and the object isn't extended, unlikely flux difference"
                    scale = 1.
            For plotting:

                a_shifted = a
                a_shifted[d_after.index] += y_diff

                #shifteverything

                shifted_1 = spitzer_spec.flux_jy[spitzer_spec.wavelength >= 14.06935]
                shifted_2 = spitzer_spec.flux_jy[spitzer_spec.wavelength < 14.06935] * 1./scale
                shifted = shifted_2.append(shifted_1)
                fig =plt.figure()
                ax = fig.add_subplot(111)

                plt.errorbar(spitzer_spec.wavelength,spitzer_spec.flux_jy,yerr=spitzer_spec.flux_jy_err ,label='Original')
                plt.plot(spitzer_spec.wavelength,shifted, label = 'Erini')
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)

                plt.show()

                shifted = spitzer_spec.flux_jy[spitzer_spec.wavelength >= 14.154030] * scale

                plt.plot(spitzer_spec.wavelength,spitzer_spec.flux_jy)
                plt.plot(spitzer_spec.wavelength[spitzer_spec.wavelength >= 14.154030],shifted)
                plt.show()


            if (spec_fits[0].header['DETLVL'] >= 4) and ((scale < 0) or (scale > 1)):
                if (spec_fits[0].header['DETLVL'] == 6):
                    print "still too noisy to say anything, but i tried"
                    return -222, -222
                else:
                    print "scale: ", scale
                    stop

            print "here"
            '''
            import matplotlib.ticker as mticker


            shifted_1 = spitzer_spec.flux_jy[spitzer_spec.wavelength >=  14.06935]
            shifted_2 = spitzer_spec.flux_jy[spitzer_spec.wavelength < 14.06935] * 1./scale
            shifted = shifted_2.append(shifted_1)
            '''
            shifted_1_n = spitzer_spec.flux_jy[spitzer_spec.wavelength >= 14.06935]
            shifted_2_n = spitzer_spec.flux_jy[spitzer_spec.wavelength < 14.06935] * nadia_scale
            shifted_n = shifted_2_n.append(shifted_1_n)
            '''
            plt.title('Scaled vs Original of AOR: '+aorkey)


            ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)


            ax1.plot(spitzer_spec.wavelength,shifted,label="Scaled",color='b')

            ax1.annotate("Detlvl: "+str(spec_fits[0].header['DETLVL']),xycoords='axes fraction',textcoords='axes fraction',xy=(.2,.9),xytext=(.2,.9))
            ax1.annotate("Scale: "+str(round(1./scale,3)),xycoords='axes fraction',textcoords='axes fraction',xy=(.2,.8),xytext=(.2,.8))
            ax1.annotate("Extent (arcseconds): "+str(round(extent,3)),xycoords='axes fraction',textcoords='axes fraction',xy=(.2,.6),xytext=(.2,.6))
            ax1.errorbar(spitzer_spec.wavelength,spitzer_spec.flux_jy,yerr=spitzer_spec.flux_jy_err ,label='Original',color='r')

            plt.xlabel('microns')
            plt.ylabel('Flux (Jy)')
            ax1.legend(loc=4)

            if (scale != 1):
                ax2 = plt.subplot2grid((3,3), (2, 0))



                ax2.set_xlim([13.8,14.2])
                ax2.set_ylim([spitzer_spec.flux_jy[spitzer_spec.wavelength > 13].values[0],shifted[spitzer_spec.wavelength < 16].values[-1]])
                ax2.errorbar(spitzer_spec.wavelength,spitzer_spec.flux_jy,yerr=spitzer_spec.flux_jy_err ,label='Original',color='r')
                ax2.plot(spitzer_spec.wavelength,shifted,label="Erini Scaled",color='b')

                myLocator = mticker.MultipleLocator(4)
                ax2.xaxis.set_major_locator(myLocator)

            plt.savefig('/home/erini/erini_stitch_examples/'+aorkey+'.jpg',dpi=1000)
            plt.close()

            if scale == 1:
                q_flag = 1.
            else:
                q_flag = -2.
            print "Scale: ", 1./scale
            return 1./scale
    except TypeError:
        print "I dont have it", aorkey
        return -1
