import numpy as np
from .base import GravitationalWaveTransient
import pickle
from ..utils import noise_weighted_inner_product, zenith_azimuth_to_ra_dec, ln_i0
from scipy.stats import chi2
import bilby
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


class EmpiricalGravitationalWaveTransient(GravitationalWaveTransient):


    def __init__(
            self, interferometers, waveform_generator, likelihood_file, time_marginalization=False,
            distance_marginalization=False, phase_marginalization=False, calibration_marginalization=False, priors=None,
            distance_marginalization_lookup_table=None, calibration_lookup_table=None,
            number_of_response_curves=1000, starting_index=0, jitter_time=True, reference_frame="sky",
            time_reference="geocenter", generate_gaussian_background=False
        ):


        super(EmpiricalGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference)


        if generate_gaussian_background:
            self.loglikelihood_object = self._gaussian_background_distribution(interferometers)

        else:
            with open(likelihood_file, 'rb') as f:
                gamma_dists = pickle.load(f)
                self.loglikelihood_object = gamma_dists['kde']


    def _gaussian_background_distribution(self, interferometers):
        ifo_list = []
        for ifo in interferometers:
            ifo_list.append(ifo.name)

        
        interferometers_bknd = bilby.gw.detector.InterferometerList(ifo_list)

        # Assume that background duration is 64 times the analysis duration
        # This will give O(64*64) = 4000 time slides for two detectors
        analysis_duration = int(ifo.duration)
        background_duration = 128 * analysis_duration
        sampling_frequency = int(ifo.sampling_frequency)
        minimum_frequency, maximum_frequency = ifo.minimum_frequency, ifo.maximum_frequency
        start_time = ifo.start_time - 0.5*background_duration

        interferometers_bknd.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=background_duration,
            start_time=start_time,
        )

        strain_series = []
        for ifo in interferometers_bknd:
            ifo.minimum_frequency = minimum_frequency
            ifo.maximum_frequency = maximum_frequency

            strain_series.append(TimeSeries(ifo.strain_data.time_domain_strain, 
                    t0=start_time, dt=1.0/sampling_frequency, name=ifo.name),), 
 
        n_segments = int(background_duration / analysis_duration)
 
        for i in range(n_segments):
            idxmin = i * sampling_frequency * analysis_duration
            idxmax = (i + 1) * sampling_frequency * analysis_duration

            interferometers_segment = bilby.gw.detector.InterferometerList(ifo_list)
            interferometers_segment[0].set_strain_data_from_gwpy_timeseries(strain_series[0][idxmin:idxmax])
            interferometers_segment[1].set_strain_data_from_gwpy_timeseries(strain_series[1][idxmin:idxmax])

            try:
                H1_spectrogram = np.vstack(
                    ((H1_spectrogram, interferometers_segment[0].frequency_domain_strain))
                )   
                L1_spectrogram = np.vstack(
                    ((L1_spectrogram, interferometers_segment[1].frequency_domain_strain))
                )   

            except:
                H1_spectrogram = interferometers_segment[0].frequency_domain_strain
                L1_spectrogram = interferometers_segment[1].frequency_domain_strain
                frequency_array = interferometers_segment[1].frequency_array


        H1_spectrogram = Spectrogram(H1_spectrogram, 
                        frequencies=frequency_array,
                        t0=start_time, dt=analysis_duration,)

        L1_spectrogram = Spectrogram(L1_spectrogram, 
                          frequencies=frequency_array,
                          t0=start_time, dt=analysis_duration,)

        H1_spectrogram =  H1_spectrogram.crop_frequencies(low=minimum_frequency,
                                                        high=maximum_frequency)
        L1_spectrogram =  L1_spectrogram.crop_frequencies(low=minimum_frequency,  
                                                            high=maximum_frequency)


        frequency_array = frequency_array[np.logical_and(
                    frequency_array>=minimum_frequency, 
                    frequency_array<maximum_frequency,
                    )]

        H1_psd = interferometers_bknd[0].power_spectral_density.power_spectral_density_interpolated(frequency_array)
        L1_psd = interferometers_bknd[1].power_spectral_density.power_spectral_density_interpolated(frequency_array)

        log_probability_object = self._calcuate_2det_background_from_spectrogram(H1_spectrogram.value, L1_spectrogram.value, H1_psd, L1_psd, analysis_duration)

        return log_probability_object


    def _calcuate_2det_background_from_spectrogram(self, 
                                                   H1, 
                                                   L1, 
                                                   H1_psd, 
                                                   L1_psd, 
                                                   analysis_duration):


        H1_inner = (4 * np.abs(H1)**2 / H1_psd ).sum(axis=1) / analysis_duration
        L1_inner = (4 * np.abs(L1)**2 / L1_psd ).sum(axis=1) / analysis_duration


        gamma = np.array([])


        #for ii in range(20):
        for ii in range(int(0.5*H1_inner.size)):
            time_shift_gamma = np.sqrt(H1_inner + np.roll(L1_inner, ii + 1))
            gamma = np.append(gamma, time_shift_gamma)

        #gamma = np.sqrt(np.random.chisquare(2*2*H1_psd.size, size=5000))
        kde = KernelDensity(kernel='gaussian', 
                        bandwidth='scott').fit(gamma.reshape(gamma.size, 1))

        return kde

        
            
    def _compute_gamma(self, waveform_polarizations):

        d_inner_h, h_inner_h = self._calculate_inner_products(waveform_polarizations)

        # calling method from GravitationalWaveTransient which is really calcualting the data inner product
        d_inner_d = -2 * self._calculate_noise_log_likelihood()

        return  np.sqrt(d_inner_d  + h_inner_h - 2 * np.real(d_inner_h))


    def noise_log_likelihood(self):
        
        if self._noise_log_likelihood_value is None:

            # calling method from GravitationalWaveTransient which is really calcualting the data inner product
            noise_gamma = np.sqrt(-2 * self._calculate_noise_log_likelihood())

            n_frequency_bins = self.interferometers[0].frequency_mask.sum()
            n_detectors = len(self.interferometers)

            n_dof = 2*n_frequency_bins*n_detectors

            noise_log_likelihood_gamma = self.loglikelihood_object.score_samples(np.array(noise_gamma).reshape(1, -1) )[0]

            self._noise_log_likelihood_value = noise_log_likelihood_gamma - (n_dof - 1) * np.log(noise_gamma)

        return self._noise_log_likelihood_value

    def _empirical_log_likelihood(self, parameters):
        
        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(parameters)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        self.parameters.update(self.get_sky_frame_parameters())

        gamma = self._compute_gamma(waveform_polarizations)
        
        loglike_gamma = self.loglikelihood_object.score_samples(np.array(gamma).reshape(1, -1) )[0]

        n_frequency_bins = self.interferometers[0].frequency_mask.sum()
        n_detectors = len(self.interferometers)

        n_dof = 2*n_frequency_bins*n_detectors

        loglike_data = loglike_gamma - (n_dof - 1) * np.log(gamma)

        return loglike_data

        
    def log_likelihood(self):

        return self._empirical_log_likelihood(self.parameters)


    def save_likelihood(self, outdir):

        dof = 2 * len(self.interferometers) * self.interferometers[0].duration * \
            (self.interferometers[0].maximum_frequency - self.interferometers[0].minimum_frequency )

        ## this is the mode of the standard Gaussian distributon
        ## we calculate it to find plotting bounds
        mode = np.sqrt(dof - 2)

        lower_limit = max( np.sqrt(dof - 8 * np.sqrt(2 *dof)) , 0)
        upper_limit = np.sqrt(dof + 8 * np.sqrt(2 *dof))

        gamma_array = np.arange(lower_limit, upper_limit, 0.01)
        gamma_array = gamma_array.reshape(gamma_array.size, 1)
        delta_gamma = gamma_array[1, 0] - gamma_array[0, 0]

        # the Gaussian likelihood in gamma is a chi^2
        gaussian_loglike = (dof - 2 ) * np.log(gamma_array) - gamma_array**2 / 2
        gaussian_like = np.exp(gaussian_loglike - gaussian_loglike.max())
 
        gaussian_like /= np.trapz(gaussian_like, dx=delta_gamma, axis=0)

        plt.plot(gamma_array, 
                np.exp(self.loglikelihood_object.score_samples(gamma_array)), 
                label='Empirical likelihood', 
                lw=1.0, 
                color='k')

        plt.plot(gamma_array, gaussian_like, 
                 label='gaussian likelihood', 
                 lw=1.0, color='#D55E00', ls='-.')

        plt.xlabel('$\gamma$')
        plt.ylabel('PDF')
        plt.axvline(mode, label='theoretical mode', color='b', ls='--')
        plt.ylim([0, 0.6])
        plt.xlim([lower_limit, upper_limit])
        plt.legend()
        plt.grid(visible=None)
        plt.savefig(outdir + '/gamma_histogram.png', dpi=250)
        plt.close()



        with open(outdir + '/likelihoods.pkl',  'wb') as f:
        
            data = {}
            data['kde'] = self.loglikelihood_object
            data['gamma_arr'] = gamma_array
            data['gaussian_like'] = gaussian_like