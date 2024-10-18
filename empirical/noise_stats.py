import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window
import pickle
from scipy.optimize import least_squares
import os

def calc_inner_prod(strain_dict, psd_dict, window_dur):

    """
    calculate the detector-wise noise-weighted inner product summed over frequency bins

    strain_dict : a dictionary of spectrogram objects
    """

    inner_prod = {}

    for detector in strain_dict.keys():

        whitened = np.abs(strain_dict[detector])**2 / psd_dict[detector]

        # sum over frequencies
        inner_prod[detector] = 4 * np.real(whitened.sum(axis=1)) / window_dur

    return inner_prod


def draw_gaussian_inner_products(time, frequency):

    n_R = np.random.normal(size=(time, frequency))
    n_I = np.random.normal(size=(time, frequency))


    gaussian_whitened = n_R**2 + n_I**2

    return gaussian_whitened.sum(axis=1)


def get_data_spcgram(channel,
                     tstart,
                     tstop,
                     fmin,
                     fmax,
                     window_dur,
                     resample_freq, 
                     outdir):

    """
    Fetches data from channel and creates spectrogram object.
    Returns spectrogram as a 2d numpy array.
    """


    # data resampled to the lower freq
    data = TimeSeries.get(channel,
                          tstart,
                          tstop + 0.1,
                          ).resample(resample_freq)


    window = get_window(('tukey', 0.4), Nx=window_dur*resample_freq)
    window_factor = np.mean(window**2)

    specgram_data = 2.55*np.sqrt(1/window_factor) * data.fftgram(window_dur, 
                                                            overlap=None, 
                                                            window=window)
    specgram_data =  specgram_data.crop_frequencies(low=fmin, high=fmax)

    

    spcgram = np.abs(specgram_data).plot(norm='log', vmin=1e-23, vmax=1e-19)
    ax = spcgram.gca()
    ax.colorbar(label='GW strain ASD [strain/$\sqrt{\mathrm{Hz}}$]')
    ax.set_yscale('log')
    ax.set_title(channel)
    plt.savefig(outdir + channel[0:3] + '_' + str(tstart) + '.png', dpi=300)
    plt.close()

    return specgram_data.value

def logistic(vec, xvals, cdf):

    k, x0 = vec[0], vec[1]


    logistic = 1 / (1 + np.exp(- k*(xvals - x0)))

    return logistic - cdf

    return

def calc_empirical_background(chirptime,
                                bknd_dur,
                                sampling_freq,
                                window_dur,
                                fmin,
                                fmax,
                                psd_file):

    """
    calculate the background.


    bknd_dur: duration to use for the empirical background
    window_dur : analysis window duration. We will assume the
                 same duration for the background estimation too
    fmin, fmax = frequencies

    The background will be calculated excising a window around chirptime
    """

    outdir = './window_' + str(window_dur) + '_bknd_' + str(bknd_dur) + '/'
    try:
        os.mkdir(outdir)
    except:
        pass

    tstart1 = chirptime - 0.5*bknd_dur - 0.5*window_dur
    tstop1 = chirptime - 0.5*window_dur

    tstart2 = chirptime + 0.5*window_dur
    tstop2 = chirptime + 0.5*bknd_dur + 0.5*window_dur


    H1_strain_specgram = get_data_spcgram("H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                                          tstart1,
                                          tstop1,
                                          fmin,
                                          fmax,
                                          window_dur,
                                          sampling_freq, 
                                          outdir)

    H1_strain_specgram = np.append(H1_strain_specgram,
                                   get_data_spcgram("H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                                          tstart2,
                                          tstop2,
                                          fmin,
                                          fmax,
                                          window_dur,
                                          sampling_freq, 
                                          outdir),
                                          axis=0)


    L1_strain_specgram = get_data_spcgram("L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                                          tstart1,
                                          tstop1,
                                          fmin,
                                          fmax,
                                          window_dur,
                                          sampling_freq, 
                                          outdir)

    L1_strain_specgram = np.append(L1_strain_specgram,
                                   get_data_spcgram("L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                                          tstart2,
                                          tstop2,
                                          fmin,
                                          fmax,
                                          window_dur,
                                          sampling_freq, 
                                          outdir),
                                          axis=0)


    with open(psd_file, 'rb') as f:
        psd_dict = pickle.load(f)

    strain_dict = {}
    strain_dict['H1'] = H1_strain_specgram
    strain_dict['L1'] = L1_strain_specgram

    psd_dict.pop('V1')
    psd_dict['H1'] = psd_dict['H1'][np.logical_and(psd_dict['frequencies']>=fmin,
                                                   psd_dict['frequencies']<fmax,)]
    psd_dict['L1'] = psd_dict['L1'][np.logical_and(psd_dict['frequencies']>=fmin,
                                                   psd_dict['frequencies']<fmax,)]

    inner_prod = calc_inner_prod(strain_dict, psd_dict, window_dur)

    gamma = np.array([])

    #for ii in range(inner_prod['H1'].size - 2):
    for ii in range(inner_prod['H1'].size - 5):
        time_shift_gamma = np.sqrt(inner_prod['H1'] + np.roll(inner_prod['L1'], ii + 1))
        gamma = np.append(gamma, time_shift_gamma)

    print('Calculating empirical background using ' + str(gamma.size) + ' data points')

    gaussian_inner_prod_H1 = draw_gaussian_inner_products(gamma.size, strain_dict['H1'].shape[1])
    gaussian_inner_prod_L1 = draw_gaussian_inner_products(gamma.size, strain_dict['L1'].shape[1])

    gaussian_gamma = np.sqrt(gaussian_inner_prod_H1 + gaussian_inner_prod_L1)



    quantile_arr = np.linspace(0, 1, int(5e3))

    gamma_quantiles = np.quantile(gamma, quantile_arr)
    gaussian_gamma_quantiles = np.quantile(gaussian_gamma, quantile_arr)

    plt.semilogx(gamma_quantiles, quantile_arr, label='gamma CDF')
    plt.semilogx(gaussian_gamma_quantiles, quantile_arr, label='gaussian gamma CDF')
    plt.legend()
    plt.xlabel('gamma')
    plt.ylabel('CDF')
    plt.savefig(outdir + 'gamma_cdf_' + str(window_dur) + 's.png', dpi=250)
    plt.close()



if __name__ == "__main__":

    gw200129_time = 1264316116.4
    bknd_dur = 320
    window_dur=8
    sampling_freq = 2048

    fmin, fmax = 20, 80

    psd_file = "/fred/oz006/sbanagiri/emp_like/GW200129/GW200129_psd_" + str(int(window_dur)) +    "s.pkl"
    #psd_file = "/fred/oz006/sbanagiri/emp_like/GW200129/GW200129_psd_4s.pkl"

    calc_empirical_background(gw200129_time,
                              bknd_dur,
                              sampling_freq,
                              window_dur,
                              fmin, fmax,
                              psd_file, )



