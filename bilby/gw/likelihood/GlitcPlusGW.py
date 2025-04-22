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
        ifo_with_glitch=None,
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

        self.ifo_with_glitch = ifo_with_glitch

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

    def _frequency_domain_glitch_waveform(self, interferometer):

        wavelet_Q = (
            2 * np.pi * self.parameters["wavelet_f0"] * self.parameters["wavelet_tau"]
        )

        psd_at_f0 = (
            interferometer.power_spectral_density.power_spectral_density_interpolated(
                self.parameters["wavelet_f0"]
            )
        )

        wavelet_A = self.parameters["wavelet_snr"] * np.sqrt(
            np.sqrt(8 * np.pi) * self.parameters["wavelet_f0"] * psd_at_f0 / wavelet_Q
        )

        amplitude = (
            np.sqrt(np.pi / 4)
            * wavelet_A
            * self.parameters["wavelet_tau"]
            * np.exp(
                   -(
                        np.pi
                        * self.parameters["wavelet_tau"]
                        * (interferometer.frequency_array
                            - self.parameters["wavelet_f0"]
                        )
                    )** 2

            )
        )

        wave = np.exp(
            2
            * np.pi
            * 1j
            * self.parameters["wavelet_t0"]
            * (interferometer.frequency_array - self.parameters["wavelet_f0"])
        )

        return amplitude * wave

    def _glitch_inclusive_likelihood(self, waveform_polarizations):

        d_inner_d = 0.0
        d_inner_h = 0.0
        h_inner_h = 0.0

        for interferometer in self.interferometers:
            signal = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
            )

            if interferometer.name == self.ifo_with_glitch:
                glitch_waveform = self._frequency_domain_glitch_waveform(interferometer)
                signal = signal + glitch_waveform

            inner_products = self._inner_products(
                interferometer.frequency_domain_strain,
                signal,
                interferometer.power_spectral_density_array,
            )

            d_inner_d += inner_products["d_inner_d"]
            d_inner_h += inner_products["d_inner_h"]
            h_inner_h += inner_products["h_inner_h"]

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
