import numpy as np
from .base import GravitationalWaveTransient
import pickle
from ..utils import noise_weighted_inner_product, optimal_snr_squared
from scipy.stats import chi2
import bilby
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


class EmpiricalGravitationalWaveTransient(GravitationalWaveTransient):


    def __init__(
            self, interferometers, waveform_generator, likelihood_file, time_marginalization=False,
            distance_marginalization=False, phase_marginalization=False, calibration_marginalization=False, 
            priors=None, distance_marginalization_lookup_table=None, calibration_lookup_table=None,
            number_of_response_curves=1000, starting_index=0, jitter_time=True, reference_frame="sky",
            time_reference="geocenter", generate_gaussian_background=False, empirical_minimum_frequency=20.0, 
            empirical_maximum_frequency=100.0
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

        self.empirical_minimum_frequency = empirical_minimum_frequency
        self.empirical_maximum_frequency = empirical_maximum_frequency

        self.calc_frequency_masks_and_ndof()

        if generate_gaussian_background:
            self.loglikelihood_object = self._gaussian_background_distribution(interferometers)

        else:
            with open(likelihood_file, 'rb') as f:
                gamma_dists = pickle.load(f)
                self.loglikelihood_object = gamma_dists['kde']



    def calc_frequency_masks_and_ndof(self):

        for interferometer in self.interferometers:
            interferometer.gamma_frequency_mask = np.logical_and(
                interferometer.frequency_array>=self.empirical_minimum_frequency, 
                interferometer.frequency_array<=self.empirical_maximum_frequency,
                )
            
            interferometer.gaussian_frequency_mask = np.logical_and(
                ~interferometer.gamma_frequency_mask, interferometer.frequency_mask)

        n_frequency_bins = self.interferometers[0].gamma_frequency_mask.sum()
        n_detectors = len(self.interferometers)

        self.n_dof = 2*n_frequency_bins*n_detectors

        return

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
        minimum_frequency = self.empirical_minimum_frequency
        maximum_frequency = self.empirical_maximum_frequency
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
            
    def _masked_inner_products(self, frequency_domain_strain, 
                               signal, power_spectral_density, mask):

        inner_products = {'d_inner_d':0.0, 
                          'd_inner_h':0.0, 
                          'h_inner_h': 0.0}

        inner_products['d_inner_d'] = noise_weighted_inner_product(frequency_domain_strain[mask], 
                                                                     frequency_domain_strain[mask], 
                                                                     power_spectral_density[mask], 
                                                                     self.waveform_generator.duration).real

        inner_products['d_inner_h'] = noise_weighted_inner_product(frequency_domain_strain[mask], 
                                                                   signal[mask], power_spectral_density[mask], 
                                                                   self.waveform_generator.duration) 
        
        inner_products['h_inner_h'] = optimal_snr_squared(signal=signal[mask], 
            power_spectral_density=power_spectral_density[mask], duration=self.waveform_generator.duration).real

        return inner_products


    def noise_log_likelihood(self):
        
        if self._noise_log_likelihood_value is None:

            noise_gamma_squared = 0
            gaussian_noise_logl = 0

            for interferometer in self.interferometers:

                noise_gamma_squared += noise_weighted_inner_product(
                    interferometer.frequency_domain_strain[interferometer.gamma_frequency_mask], 
                    interferometer.frequency_domain_strain[interferometer.gamma_frequency_mask], 
                    interferometer.power_spectral_density_array[interferometer.gamma_frequency_mask], 
                    self.waveform_generator.duration).real


                gaussian_noise_logl -= 0.5*noise_weighted_inner_product(
                    interferometer.frequency_domain_strain[interferometer.gaussian_frequency_mask], 
                    interferometer.frequency_domain_strain[interferometer.gaussian_frequency_mask], 
                    interferometer.power_spectral_density_array[interferometer.gaussian_frequency_mask], 
                    self.waveform_generator.duration).real

            noise_gamma = np.sqrt(noise_gamma_squared)

            gamma_log_likelihood = self.loglikelihood_object.score_samples(
                                    np.array(noise_gamma).reshape(1, -1))[0] - (self.n_dof- 1) * np.log(noise_gamma)


            self._noise_log_likelihood_value = gamma_log_likelihood + gaussian_noise_logl

        return self._noise_log_likelihood_value

    def _empirical_log_likelihood(self, waveform_polarizations):
        
        d_inner_d = 0.0 
        d_inner_h = 0.0 
        h_inner_h = 0.0

        for interferometer in self.interferometers:
            signal = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                )

            inner_products = self._masked_inner_products(interferometer.frequency_domain_strain, 
                               signal, interferometer.power_spectral_density_array, 
                               interferometer.gamma_frequency_mask)

            d_inner_d += inner_products['d_inner_d']
            d_inner_h += inner_products['d_inner_h']
            h_inner_h += inner_products['h_inner_h']


        gamma = np.sqrt(d_inner_d  + h_inner_h - 2 * np.real(d_inner_h))
        #gamma = self._compute_gamma(waveform_polarizations)
        
        loglike_gamma = self.loglikelihood_object.score_samples(np.array(gamma).reshape(1, -1) )[0]

        loglike_data = loglike_gamma - (self.n_dof - 1) * np.log(gamma)

        return loglike_data

    def _gaussian_log_likelihood(self, waveform_polarizations):

        d_inner_d = 0.0 
        d_inner_h = 0.0 
        h_inner_h = 0.0

        for interferometer in self.interferometers:
            signal = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                )


            inner_products = self._masked_inner_products(interferometer.frequency_domain_strain, 
                               signal, interferometer.power_spectral_density_array,
                               interferometer.gaussian_frequency_mask)

            d_inner_d += inner_products['d_inner_d']
            d_inner_h += inner_products['d_inner_h']
            h_inner_h += inner_products['h_inner_h']


        log_likelihood = -0.5 * (d_inner_d + h_inner_h - 2 * np.real(d_inner_h))

        return log_likelihood

        
    def log_likelihood(self):

        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(self.parameters)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        self.parameters.update(self.get_sky_frame_parameters())

        empirical_log_likelihood = self._empirical_log_likelihood(waveform_polarizations)
        gaussian_log_likelihood = self._gaussian_log_likelihood(waveform_polarizations)

        return empirical_log_likelihood + gaussian_log_likelihood

    def save_likelihood(self, outdir):

        ## this is the mode of the standard Gaussian distributon
        ## we calculate it to find plotting bounds
        mode = np.sqrt(self.n_dof - 2)

        lower_limit = max( np.sqrt(self.n_dof - 8 * np.sqrt(2*self.n_dof)) , 0)
        upper_limit = np.sqrt(self.n_dof + 8 * np.sqrt(2*self.n_dof))

        gamma_array = np.arange(lower_limit, upper_limit, 0.01)
        gamma_array = gamma_array.reshape(gamma_array.size, 1)
        delta_gamma = gamma_array[1, 0] - gamma_array[0, 0]

        # the Gaussian likelihood in gamma is a chi^2
        gaussian_loglike = (self.n_dof - 2 ) * np.log(gamma_array) - gamma_array**2 / 2
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