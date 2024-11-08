import numpy as np
from .base import GravitationalWaveTransient
import pickle
from ..utils import noise_weighted_inner_product, zenith_azimuth_to_ra_dec, ln_i0
from scipy.stats import chi2

class EmpiricalGravitationalWaveTransient(GravitationalWaveTransient):


    def __init__(
            self, interferometers, waveform_generator, likelihood_file, time_marginalization=False,
            distance_marginalization=False, phase_marginalization=False, calibration_marginalization=False, priors=None,
            distance_marginalization_lookup_table=None, calibration_lookup_table=None,
            number_of_response_curves=1000, starting_index=0, jitter_time=True, reference_frame="sky",
            time_reference="geocenter",
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

        with open(likelihood_file, 'rb') as f:
            gamma_dists = pickle.load(f)
            self.loglikelihood_object = gamma_dists['kde']


    
    def _compute_gamma(self, waveform_polarizations):

        d_inner_h, h_inner_h = self._calculate_inner_products(waveform_polarizations)

        # calling method from GravitationalWaveTransient which is really calcualting the data inner product
        d_inner_d = -self._calculate_noise_log_likelihood()

        return  np.sqrt(d_inner_d  + h_inner_h - 2 * np.real(d_inner_h))


    def noise_log_likelihood(self):
        
        if self._noise_log_likelihood_value is None:

            # calling method from GravitationalWaveTransient which is really calcualting the data inner product
            noise_gamma = np.sqrt(-self._calculate_noise_log_likelihood())

            n_frequency_bins = self.interferometers[0].frequency_mask.sum()
            n_detectors = len(self.interferometers)

            n_dof = 2*n_frequency_bins*n_detectors

            self._noise_log_likelihood_value = np.log(2*noise_gamma) + chi2.logpdf(noise_gamma, n_dof)

        return self._noise_log_likelihood_value

    def empirical_log_likelihood(self):
        
        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(self.parameters)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        self.parameters.update(self.get_sky_frame_parameters())

        gamma = self._compute_gamma(waveform_polarizations)

        return self.loglikelihood_object.score_samples(np.array(gamma).reshape(1, -1) )[0]

        
    def log_likelihood(self):

        return self.empirical_log_likelihood()
