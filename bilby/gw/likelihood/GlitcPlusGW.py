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


class GlitchIncludedGravitationalWaveTransient(GravitationalWaveTransient):

    def __init__(
        self,
        interferometers,
        waveform_generator,
        time_marginalization=False,
        distance_marginalization=False,
        phase_marginalization=False,
        calibration_marginalization=False,
        priors=None,
        distance_marginalization_lookup_table=None,
        calibration_lookup_table=None,
        number_of_response_curves=1000,
        starting_index=0,
        jitter_time=True,
        reference_frame="sky",
        time_reference="geocenter",
        ifos_with_glitch=None,
    ):

        super(GlitchIncludedGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )

        self.ifos_with_glitch = ifos_with_glitch

    @classmethod
    def error_fxn_glitch_frequency_domain(cls, amplitude, t0, sigma, frequencies):
        '''
        frequency domain error fucntion glitch model.
        '''

        wavelet_term1 = amplitude * 1j  * np.exp(- 2 * np.pi * 1j * frequencies * t0 ) / (2 * np.pi * frequencies)
        wavelet_term2 = np.exp( - (np.pi * frequencies * sigma)**2)

        wavelet =  wavelet_term1 * wavelet_term2
        wavelet = np.where(np.isnan(wavelet), 0, wavelet)

        return wavelet



    def _inner_products(
        self, frequency_domain_strain, signal, power_spectral_density, 
    ):

        inner_products = {"d_inner_d": 0.0, "d_inner_h": 0.0, "h_inner_h": 0.0}

        inner_products["d_inner_d"] = noise_weighted_inner_product(
            frequency_domain_strain,
            frequency_domain_strain,
            power_spectral_density,
            self.waveform_generator.duration,
        ).real

        inner_products["d_inner_h"] = noise_weighted_inner_product(
            frequency_domain_strain,
            signal,
            power_spectral_density,
            self.waveform_generator.duration,
        )

        inner_products["h_inner_h"] = optimal_snr_squared(
            signal=signal,
            power_spectral_density=power_spectral_density,
            duration=self.waveform_generator.duration,
        ).real

        return inner_products

    def _glitch_inclusive_likelihood(self, waveform_polarizations):

        d_inner_d = 0.0
        d_inner_h = 0.0
        h_inner_h = 0.0

        for ifo in self.interferometers:

            signal = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=ifo,
            )

            ifo_data = ifo.frequency_domain_strain

            if ifo.name in self.ifos_with_glitch:
                glitch_waveform = self.__class__.error_fxn_glitch_frequency_domain(
                                    self.parameters[f"glitch_A_{ifo.name}"], 
                                    self.parameters[f"glitch_t0_{ifo.name}"], 
                                    1e-4,
                                    ifo.frequency_array,
                )

                ifo_data += glitch_waveform

            inner_products = self._inner_products(
                ifo_data, signal,
                ifo.power_spectral_density_array,
            )

            d_inner_d += inner_products["d_inner_d"]
            d_inner_h += inner_products["d_inner_h"]
            h_inner_h += inner_products["h_inner_h"]



        if self.distance_marginalization:
            log_likelihood_ratio =  self.distance_marginalized_likelihood(d_inner_h, h_inner_h)  
        else:
            log_likelihood_ratio = -0.5 * (h_inner_h - 2 * np.real(d_inner_h))

        return log_likelihood_ratio

    def log_likelihood(self):

        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            self.parameters
        )

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        self.parameters.update(self.get_sky_frame_parameters())

        log_likelihood_ratio = self._glitch_inclusive_likelihood(
            waveform_polarizations
        )
        noise_log_likelihood = self.noise_log_likelihood()

        return log_likelihood_ratio + noise_log_likelihood


    ############# ------- Distance marginalization code ----------------

    # def distance_marginalized_likelihood(self, d_inner_h, h_inner_h):
    #     d_inner_h_ref, h_inner_h_ref = self._setup_rho(
    #         d_inner_h, h_inner_h)
    #     if self.phase_marginalization:
    #         d_inner_h_ref = np.abs(d_inner_h_ref)
    #     else:
    #         d_inner_h_ref = np.real(d_inner_h_ref)

    #     return self._interp_dist_margd_loglikelihood(
    #         d_inner_h_ref, h_inner_h_ref, grid=False)



    # def _setup_distance_marginalization(self, lookup_table=None):
    #     if isinstance(lookup_table, str) or lookup_table is None:
    #         self.cached_lookup_table_filename = lookup_table
    #         lookup_table = self.load_lookup_table(
    #             self.cached_lookup_table_filename)
    #     if isinstance(lookup_table, dict):
    #         if self._test_cached_lookup_table(lookup_table):
    #             self._dist_margd_loglikelihood_array = lookup_table[
    #                 'lookup_table']
    #         else:
    #             self._create_lookup_table()
    #     else:
    #         self._create_lookup_table()
    #     self._interp_dist_margd_loglikelihood = BoundedRectBivariateSpline(
    #         self._d_inner_h_ref_array, self._optimal_snr_squared_ref_array,
    #         self._dist_margd_loglikelihood_array.T, fill_value=-np.inf)




    # @property
    # def _d_inner_h_ref_array(self):
    #     """ Matched filter snr at fiducial distance of ref_dist Mpc """
    #     if self.phase_marginalization:
    #         return np.logspace(-5, 10, self._dist_margd_loglikelihood_array.shape[1])
    #     else:
    #         n_negative = self._dist_margd_loglikelihood_array.shape[1] // 2
    #         n_positive = self._dist_margd_loglikelihood_array.shape[1] - n_negative
    #         return np.hstack((
    #             -np.logspace(3, -3, n_negative), np.logspace(-3, 10, n_positive)
    #         ))



    # def _create_lookup_table(self):
    #     """ Make the lookup table """
    #     from tqdm.auto import tqdm
    #     logger.info('Building lookup table for distance marginalisation.')

    #     self._dist_margd_loglikelihood_array = np.zeros((400, 800))
    #     scaling = self._ref_dist / self._distance_array
    #     d_inner_h_array_full = np.outer(self._d_inner_h_ref_array, scaling)
    #     h_inner_h_array_full = np.outer(self._optimal_snr_squared_ref_array, scaling ** 2)
    #     if self.phase_marginalization:
    #         d_inner_h_array_full = ln_i0(abs(d_inner_h_array_full))
    #     prior_term = self.distance_prior_array * self._delta_distance
    #     for ii, optimal_snr_squared_array in tqdm(
    #             enumerate(h_inner_h_array_full), total=len(self._optimal_snr_squared_ref_array)
    #     ):
    #         for jj, d_inner_h_array in enumerate(d_inner_h_array_full):
    #             self._dist_margd_loglikelihood_array[ii][jj] = logsumexp(
    #                 d_inner_h_array - optimal_snr_squared_array / 2,
    #                 b=prior_term
    #             )
    #     log_norm = logsumexp(
    #         0 / self._distance_array, b=self.distance_prior_array * self._delta_distance
    #     )
    #     self._dist_margd_loglikelihood_array -= log_norm
    #     self.cache_lookup_table()