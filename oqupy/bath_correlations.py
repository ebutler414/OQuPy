# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module for environment correlations.
"""

from typing import Callable, Optional, Text,Dict
from typing import Any as ArrayLike
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy import special

from oqupy.base_api import BaseAPIClass
from oqupy.config import INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.util import check_true

#np.seterr(all='warn')
# --- spectral density classes ------------------------------------------------

class BaseCorrelations(BaseAPIClass):
    """Base class for environment auto-correlations. """

    def correlation(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Auto-correlation function.

        .. math::

            C(\tau) = C(t, t-\tau) \
                    = \langle F(t) F(t-\tau) \rangle_\mathrm{env}

        where :math:`\tau` is the time difference `tau` and :math:`F(t)` is the
        the environment part of the coupling operator in Heisenberg picture with
        respect to the environment Hamiltonian.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        raise NotImplementedError(
            "{} has no correlation implementation.".format(type(self).__name__))

    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        raise NotImplementedError(
            "{} has no correlation_2d_integral implementation.".format(
                type(self).__name__))


class CustomCorrelations(BaseCorrelations):
    r"""
    Encodes a custom auto-correlation function

    .. math::

        C(\tau) = C(t, t-\tau) = \langle F(t) F(t-\tau) \rangle_\mathrm{env}

    with time difference `tau` :math:`\tau` and :math:`F(t)` is the
    the environment part of the coupling operator in Heisenberg picture with
    respect to the environment Hamiltonian. We assume that :math:`C(\tau) = 0`
    for all :math:`\tau > \tau_\mathrm{max}`.

    Parameters
    ----------
    correlation_function : callable
        The correlation function :math:`C`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    """

    def __init__(
            self,
            correlation_function: Callable[[float], float],
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Creates a CustomCorrelations object. """

        # check input: j_function
        try:
            tmp_correlation_function = np.vectorize(correlation_function)
            complex(tmp_correlation_function(1.0))
        except Exception as e:
            raise AssertionError("Correlation function must be vectorizable " \
                                 + "and must return float.") from e
        self.correlation_function = tmp_correlation_function

        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        return "".join(ret)

    def correlation(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = None,
            subdiv_limit: Optional[int] = None) -> ArrayLike:
        r"""
        Auto-correlation function.

        .. math::

            C(\tau) = C(t, t-\tau) \
                    = \langle F(t) F(t-\tau) \rangle_\mathrm{env}

        with time difference `tau` :math:`\tau` and :math:`F(t)` is the
        the environment part of the coupling operator in Heisenberg picture with
        respect to the environment Hamiltonian.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tolerance (has no effect here).
        subdiv_limit : int
            Maximal number of interval subdivisions for numerical integration
            (has no effect here).

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        return self.correlation_function(tau)

    @lru_cache(maxsize=2 ** 10, typed=False)
    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        c_real = lambda y, x: np.real(self.correlation(x - y))
        c_imag = lambda y, x: np.imag(self.correlation(x - y))

        if time_2 is None:
            time_2 = time_1 + delta
        else:
            assert shape == 'rectangle', \
                "parameter 'time_2' can only be used in conjunction with " \
                "'shape' = ``rectangle`` !"

        lower_boundary = {'square': lambda x: 0.0,
                          'upper-triangle': lambda x: 0.0,
                          'rectangle': lambda x: 0.0}
        upper_boundary = {'square': lambda x: delta,
                          'upper-triangle': lambda x: x - time_1,
                          'rectangle': lambda x: delta, }

        int_real = integrate.dblquad(func=c_real,
                                     a=time_1,
                                     b=time_2,
                                     gfun=lower_boundary[shape],
                                     hfun=upper_boundary[shape],
                                     epsrel=epsrel)[0]
        int_imag = integrate.dblquad(func=c_imag,
                                     a=time_1,
                                     b=time_2,
                                     gfun=lower_boundary[shape],
                                     hfun=upper_boundary[shape],
                                     epsrel=epsrel)[0]
        return int_real + 1.0j * int_imag


# === CORRELATIONS FROM SPECTRAL DENSITIES ====================================


# --- the cutoffs -------------------------------------------------------------

def _hard_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Hard cutoff function."""
    return np.heaviside(omega_c - omega, 0)


def _exponential_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Exponential cutoff function."""
    return np.exp(-omega / omega_c)


def _gaussian_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Gaussian cutoff function."""
    return np.exp(-(omega / omega_c) ** 2)


# dictionary for the various cutoffs in the form:
#   'cutoff_name': cutoff_function
CUTOFF_DICT = {
    'hard': _hard_cutoff,
    'exponential': _exponential_cutoff,
    'gaussian': _gaussian_cutoff,
}


# --- the spectral density classes --------------------------------------------

def _complex_integral(
        integrand: Callable[[float], complex],
        a: Optional[float] = 0.0,
        b: Optional[float] = 1.0,
        epsrel: Optional[float] = INTEGRATE_EPSREL,
        limit: Optional[int] = SUBDIV_LIMIT) -> complex:
    re_int = integrate.quad(lambda x: np.real(integrand(x)),
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit)[0]
    im_int = integrate.quad(lambda x: np.imag(integrand(x)),
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit)[0]

    return re_int + 1j * im_int

class CustomSD(BaseCorrelations):
    r"""
    Correlations corresponding to a custom spectral density for a thermal
    system with known temperature. The resulting spectral density is

    .. math::

        J(\omega) = j(\omega) X(\omega,\omega_c) ,

    with `j_function` :math:`j`, `cutoff` :math:`\omega_c` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.


    Parameters
    ----------
    j_function : callable
        The spectral density :math:`j` without the cutoff.
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    cutoff_type : str (default = ``'exponential'``)
        The cutoff type. Types are: {``'hard'``, ``'exponential'``,
        ``'gaussian'``}
    temperature: float
        The environment's temperature.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    """

    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            cutoff_type: Optional[Text] = 'exponential',
            temperature: Optional[float] = 0.0,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a CustomFunctionSD (spectral density) object. """

        # check input: j_function
        try:
            tmp_j_function = np.vectorize(j_function)
            float(tmp_j_function(1.0))
        except Exception as e:
            raise AssertionError("Spectral density must be vectorizable " \
                                 + "and must return float.") from e
        self.j_function = tmp_j_function

        # check input: cutoff
        try:
            tmp_cutoff = float(cutoff)
        except Exception as e:
            raise AssertionError("Cutoff must be a float.") from e
        self.cutoff = tmp_cutoff

        # check input: cutoff_type
        assert cutoff_type in CUTOFF_DICT, \
            "Cutoff type must be one of: {}".format(CUTOFF_DICT.keys())
        self.cutoff_type = cutoff_type

        # input check for temperature.
        try:
            tmp_temperature = float(temperature)
        except Exception as e:
            raise AssertionError("Temperature must be a float.") from e
        if tmp_temperature < 0.0:
            raise ValueError("Temperature must be >= 0.0 (but is {})".format(
                tmp_temperature))
        self.temperature = tmp_temperature

        self._cutoff_function = \
            lambda omega: CUTOFF_DICT[self.cutoff_type](omega, self.cutoff)
        self._spectral_density = \
            lambda omega: self.j_function(omega) * self._cutoff_function(omega)

        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  cutoff        = {} \n".format(self.cutoff))
        ret.append("  cutoff_type   = {} \n".format(self.cutoff_type))
        ret.append("  temperature   = {} \n".format(self.temperature))

        return "".join(ret)

    def spectral_density(self, omega: ArrayLike) -> ArrayLike:
        r"""
        The resulting spectral density (including the cutoff).

        Parameters
        ----------
        omega : ndarray
            The frequency :math:`\omega` for which we want to know the
            spectral density.

        Returns
        -------
        spectral_density : ndarray
            The resulting spectral density :math:`J(\omega)` at the frequency
            :math:`\omega`.
        """
        return self._spectral_density(omega)

    def correlation(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False) -> ArrayLike:
        r"""
        Auto-correlation function associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            C(\tau) = \int_0^{\infty} J(\omega) \
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if matsubara:
            tau = -1j * tau
        # convention is tau.imag < 0
        if self.temperature == 0.0:
            check_true(
                matsubara is False,
                'Matsubara correlations only defined for temperature > 0')
            def integrand(w):
                return self._spectral_density(w) * np.exp(-1j * w * tau)
        else:
            def integrand(w):
                # this is to stop overflow
                if np.exp(-w / self.temperature) > np.finfo(float).eps:
                    inte = self._spectral_density(w) \
                        * (np.exp(-1j * tau * w)
                           + np.exp(-(1 / self.temperature * w \
                                      - 1j * tau * w))) \
                        / (1 - np.exp(-w / self.temperature))
                else:
                    inte = self._spectral_density(w) * np.exp(-1j * w * tau)
                return inte

        integral = _complex_integral(integrand,
                                     a=0.0,
                                     b=self.cutoff,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)

        if self.cutoff_type != "hard":
            integral += _complex_integral(integrand,
                                          a=self.cutoff,
                                          b=np.inf,
                                          epsrel=epsrel,
                                          limit=subdiv_limit)
        if matsubara:
            integral = integral.real
        return integral

    @lru_cache(maxsize=2 ** 10, typed=False)
    def eta_function(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False) -> ArrayLike:
        r"""
        Auto-correlation function associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            C(\tau) = \int_0^{\infty} J(\omega) \
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        if self.cutoff_type == 'exponential' and self.zeta == 1: 

            i1 = 2.0*self.alpha*sum([k2*np.real(special.loggamma(self.temperature/self.cutoff+(k1+1.0)/2+1j*self.temperature*tau*(k2+1.0)/2.0)) for k1 in [1.0,-1.0] for k2 in [1.0,-1.0]])
            i2 = -1j*2.0*self.alpha*(np.arctan(self.cutoff*tau)-self.cutoff*tau)

            return -(i1+i2)
        
        
        # real and imaginary part of the integrand
        if matsubara:
            tau = -1j * tau
        # convention is tau.imag < 0
        if self.temperature == 0.0:
            check_true(
                matsubara is False,
                'Matsubara correlations only defined for temperature > 0')
            def integrand(w):
                return self._spectral_density(w) / w ** 2 * (
                    (np.exp(-1j * w * tau) - 1) + 1j * w * tau)
        else:
            def integrand(w):
                # this is to stop overflow
                if np.exp(-w / self.temperature) > np.finfo(float).eps:
                    inte = self._spectral_density(w) / w ** 2 \
                        * (((np.exp(-1j*tau * w) \
                             + np.exp(-(w / self.temperature - 1j*tau * w))) \
                            - np.exp(- w / self.temperature) - 1) \
                        / (1 - np.exp(-w / self.temperature)) + 1j*tau * w)
                else:
                    inte = self._spectral_density(w) / w ** 2 \
                        * (np.exp(-1j * w * tau) - 1 + 1j * w * tau)
                return inte

        integral = _complex_integral(integrand,
                                     a=0.0,
                                     b=self.cutoff,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)

        if self.cutoff_type != "hard":
            integral += _complex_integral(integrand,
                                          a=self.cutoff,
                                          b=np.inf,
                                          epsrel=epsrel,
                                          limit=subdiv_limit)
        if matsubara:
            integral = integral.real
        return -integral

    def eta_function_analytic(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False) -> ArrayLike:
        r""" obtained for ohmic baths with exponential cutoff"""
        i1 = 2.0*self.alpha*sum([k2*np.real(special.loggamma(self.temperature/self.cutoff+(k1+1.0)/2+1j*self.temperature*tau*(k2+1.0)/2.0)) for k1 in [1.0,-1.0] for k2 in [1.0,-1.0]])
        i2 = -1j*2.0*self.alpha*(np.arctan(self.cutoff*tau)-self.cutoff*tau)
        
        return -(i1+i2)

    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        kwargs = {
            'epsrel': epsrel,
            'subdiv_limit': subdiv_limit,
            'matsubara': matsubara}

        if shape == 'upper-triangle':
            integral = self.eta_function(time_1 + delta, **kwargs) \
                       - self.eta_function(time_1, **kwargs)
        elif shape == 'square':
            integral = self.eta_function(time_1 + delta, **kwargs) \
                       - 2.0 * self.eta_function(time_1, **kwargs) \
                       + self.eta_function(time_1 - delta, **kwargs)
        elif shape == 'rectangle':
            integral = self.eta_function(time_2, **kwargs) \
                       - self.eta_function(time_1, **kwargs) \
                       - self.eta_function(time_2 - delta, **kwargs) \
                       + self.eta_function(time_1 - delta, **kwargs)
        else:
            raise NotImplementedError("Shape '{shape}' not implemented.")

        if matsubara:
            integral = integral.real
        return integral
    

class CustomCountingSD(CustomSD):
    r"""
    Correlations corresponding to a custom spectral density, with the counting fields. The resulting
    spectral density is

    .. math::

        J(\omega) = j(\omega) X(\omega,\omega_c) ,

    with `j_function` :math:`j`, `cutoff` :math:`\omega_c` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.


    Parameters
    ----------
    j_function : callable
        The spectral density :math:`j` without the cutoff.
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    u : float
        The value of the counting field :math: u at which to compute the correlation functions.
    cutoff_type : str (default = ``'exponential'``)
        The cutoff type. Types are: {``'hard'``, ``'exponential'``,
        ``'gaussian'``}
    temperature: float
        The environment's temperature.
    max_correlation_time : float
        The maximal occuring correlation time :math:`\tau_\mathrm{max}`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    description_dict: dict
        An optional dictionary with descriptive data.
    """

    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            u: float,
            cutoff_type: Optional[Text] = 'exponential',
            temperature: Optional[float] = 0.0,
            max_correlation_time: Optional[float] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a CustomFunctionSD (spectral density) object. """
        self._u=u
        super().__init__(j_function,cutoff,cutoff_type,temperature,name,description)
        self.alpha = j_function(0.5)

    def correlationA1(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Counting-field dressed auto-correlation function A1 associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            A_1(\tau) = \int_0^{\infty} J(\omega) \cos^2(u\omega/2)\
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: self._spectral_density(w) * (np.cos(self._u*w/2)**2.0) * np.cos(w*tau)
        else:
            re_integrand = lambda w: self._spectral_density(w) * (np.cos(self._u*w/2)**2.0) * np.cos(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.cos(self._u*w/2)**2.0) \
                                      * np.sin(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    def correlationA2(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Counting-field dressed auto-correlation function A2 associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            A_2(\tau) = \int_0^{\infty} J(\omega) \sin^2(u\omega/2)\
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: self._spectral_density(w) * (np.sin(self._u*w/2)**2.0) * np.cos(w*tau)
        else:
            re_integrand = lambda w: self._spectral_density(w) * (np.sin(self._u*w/2)**2.0) * np.cos(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w/2)**2.0) \
                                      * np.sin(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    def correlationC(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Counting-field dressed auto-correlation function C associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            A_1(\tau) = \int_0^{\infty} J(\omega) \cos^2(u\omega/2)\
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w)/2.0) * np.sin(w*tau)
        else:
            re_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w)/2.0) * np.sin(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w)/2.0) \
                                      * np.cos(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    @lru_cache(maxsize=2**10, typed=False)
    def correlation_2d_integral_marked_old(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            which_corr: Optional[Text] = 'A1',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{lower-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{t'-t_1}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        ``'lower-triangle'``, or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intevals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'lower-triangle'``, ``'lower-triangle'``}
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        
        selected_corr={
            'A1': self.correlationA1,
            'A2': self.correlationA2,
            'C': self.correlationC,
        }
        
        c_real = lambda y, x: np.real(selected_corr[which_corr](x-y))
        c_imag = lambda y, x: np.imag(selected_corr[which_corr](x-y))

        if time_2 is None:
            time_2 = time_1 + delta
        else:
            assert shape == 'rectangle', \
                "paramter 'time_2' can only be used in conjunction with " \
                "'shape' = ``rectangle`` !"

        lower_boundary = {'square': lambda x: 0.0,
                          'upper-triangle': lambda x: 0.0,
                          'lower-triangle': lambda x: x-time_1,
                          'rectangle': lambda x: 0.0}
        upper_boundary = {'square': lambda x: delta,
                          'upper-triangle': lambda x: x-time_1,
                          'lower-triangle': lambda x: delta,
                          'rectangle': lambda x: delta,}

        int_real = integrate.dblquad(func=c_real,
                                     a=time_1,
                                     b=time_2,
                                     gfun=lower_boundary[shape],
                                     hfun=upper_boundary[shape],
                                     epsrel=epsrel)[0]
        int_imag = integrate.dblquad(func=c_imag,
                                     a=time_1,
                                     b=time_2,
                                     gfun=lower_boundary[shape],
                                     hfun=upper_boundary[shape],
                                     epsrel=epsrel)[0]
        return int_real+1.0j*int_imag

    @lru_cache(maxsize=2**10, typed=False)
    def correlation_2d_integral_marked(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            which_corr: Optional[Text] = 'A1',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{lower-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{t'-t_1}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        ``'lower-triangle'``, or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intevals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'lower-triangle'``, ``'lower-triangle'``}
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        
        if (which_corr=='A1'):   
                # real and imaginary part of the integrand
                if self.temperature == 0.0:
                    re_integrand = lambda w: \
                        (np.cos(self._u*w/2.0)**2)*self._spectral_density(w) \
                        * INTEGRAND_DICT[shape][0](w,
                                                   delta,
                                                   time_1,
                                                   time_2,
                                                   self.temperature)
                else:
                    re_integrand = lambda w: \
                        (np.cos(self._u*w/2.0)**2)*self._spectral_density(w) \
                        * INTEGRAND_DICT[shape][1](w,
                                                   delta,
                                                   time_1,
                                                   time_2,
                                                   self.temperature)
                im_integrand = lambda w: \
                    (np.cos(self._u*w/2.0)**2)*self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][2](w,
                                               delta,
                                               time_1,
                                               time_2,
                                               self.temperature)
                
        elif (which_corr=='A2'):                
                if self.temperature == 0.0:
                    re_integrand = lambda w: \
                        (np.sin(self._u*w/2)**2)*self._spectral_density(w) \
                        * INTEGRAND_DICT[shape][0](w,
                                                   delta,
                                                   time_1,
                                                   time_2,
                                                   self.temperature)
                else:
                    re_integrand = lambda w: \
                        (np.sin(self._u*w/2.0)**2)*self._spectral_density(w) \
                        * INTEGRAND_DICT[shape][1](w,
                                                   delta,
                                                   time_1,
                                                   time_2,
                                                   self.temperature)
                im_integrand = lambda w: \
                    (np.sin(self._u*w/2.0)**2)*self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][2](w,
                                               delta,
                                               time_1,
                                               time_2,
                                               self.temperature)
        elif (which_corr=='C'):        
                if self.temperature == 0.0:
                    re_integrand = lambda w: \
                        (np.sin(self._u*w)/2.0)*self._spectral_density(w) \
                        * INTEGRAND_DICT_C[shape][0](w,
                                                   delta,
                                                   time_1,
                                                   time_2,
                                                   self.temperature)
                else:
                    re_integrand = lambda w: \
                         (np.sin(self._u*w)/2.0)*self._spectral_density(w) \
                         * INTEGRAND_DICT_C[shape][1](w,
                                                   delta,
                                                   time_1,
                                                   time_2,
                                                   self.temperature)
                im_integrand = lambda w: \
                     -1.0*(np.sin(self._u*w)/2.0)*self._spectral_density(w) \
                     * INTEGRAND_DICT_C[shape][2](w,
                                               delta,
                                               time_1,
                                               time_2,
                                               self.temperature)

        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]

        return re_int+1j*im_int     
        


class CustomCountingSD_analytical(CustomSD):
    r"""
    Correlations corresponding to a custom spectral density, with the counting fields. The resulting
    spectral density is

    .. math::

        J(\omega) = j(\omega) X(\omega,\omega_c) ,

    with `j_function` :math:`j`, `cutoff` :math:`\omega_c` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.


    Parameters
    ----------
    j_function : callable
        The spectral density :math:`j` without the cutoff.
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    u : float
        The value of the counting field :math: u at which to compute the correlation functions.
    cutoff_type : str (default = ``'exponential'``)
        The cutoff type. Types are: {``'hard'``, ``'exponential'``,
        ``'gaussian'``}
    temperature: float
        The environment's temperature.
    max_correlation_time : float
        The maximal occuring correlation time :math:`\tau_\mathrm{max}`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    description_dict: dict
        An optional dictionary with descriptive data.
    """

    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            u: float,
            cutoff_type: Optional[Text] = 'exponential',
            temperature: Optional[float] = 0.0,
            max_correlation_time: Optional[float] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a CustomFunctionSD (spectral density) object. """
        self._u=u
        super().__init__(j_function,cutoff,cutoff_type,temperature,name,description)
        self.alpha = j_function(0.5)

    def correlationA1(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Counting-field dressed auto-correlation function A1 associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            A_1(\tau) = \int_0^{\infty} J(\omega) \cos^2(u\omega/2)\
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: self._spectral_density(w) * (np.cos(self._u*w/2)**2.0) * np.cos(w*tau)
        else:
            re_integrand = lambda w: self._spectral_density(w) * (np.cos(self._u*w/2)**2.0) * np.cos(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.cos(self._u*w/2)**2.0) \
                                      * np.sin(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    def correlationA2(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Counting-field dressed auto-correlation function A2 associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            A_2(\tau) = \int_0^{\infty} J(\omega) \sin^2(u\omega/2)\
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: self._spectral_density(w) * (np.sin(self._u*w/2)**2.0) * np.cos(w*tau)
        else:
            re_integrand = lambda w: self._spectral_density(w) * (np.sin(self._u*w/2)**2.0) * np.cos(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w/2)**2.0) \
                                      * np.sin(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    def correlationC(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Counting-field dressed auto-correlation function C associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            A_1(\tau) = \int_0^{\infty} J(\omega) \cos^2(u\omega/2)\
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w)/2.0) * np.sin(w*tau)
        else:
            re_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w)/2.0) * np.sin(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * (np.sin(self._u*w)/2.0) \
                                      * np.cos(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    @lru_cache(maxsize=2**10, typed=False)
    def eta_function_marked(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False,
            which_corr: Optional[Text] = 'A1' ) -> ArrayLike:
        r""" obtained for ohmic baths with exponential cutoff"""



        if (which_corr=='A1'):

            i1 = 0.5*self.alpha*sum([k4*np.real(special.loggamma(self.temperature/self.cutoff+(k1+1.0)/2+1j*(k2+k3)/2*self._u*self.temperature+1j*self.temperature*tau*(k4+1.0)/2.0)) for k1 in [1.0,-1.0] for k2 in [1.0,-1.0] for k3 in [1.0,-1.0] for k4 in [1.0,-1.0]])
            i2 = -1j*self.alpha*(np.arctan(self.cutoff*tau)+0.5*np.arctan(self.cutoff*(self._u+tau))-0.5*np.arctan(self.cutoff*(self._u-tau)))\
                +1.0j*self.alpha*self.cutoff*tau*(1.0+1.0/(1+(self._u*self.cutoff)**2))
            
        
        elif (which_corr=='A2'):

            i1 = 2.0*self.alpha*sum([k4*np.real(special.loggamma(self.temperature/self.cutoff+(k1+1.0)/2+1j*self.temperature*tau*(k4+1.0)/2.0)) for k1 in [1.0,-1.0] for k4 in [1.0,-1.0]]) \
                    -0.5*self.alpha*sum([k4*np.real(special.loggamma(self.temperature/self.cutoff+(k1+1.0)/2+1j*(k2+k3)/2*self._u*self.temperature+1j*self.temperature*tau*(k4+1.0)/2.0)) for k1 in [1.0,-1.0] for k2 in [1.0,-1.0] for k3 in [1.0,-1.0] for k4 in [1.0,-1.0]]) 
            i2 = -2j*self.alpha*np.arctan(self.cutoff*tau)+2.0j*self.alpha*self.cutoff*tau \
                    +1j*self.alpha*(np.arctan(self.cutoff*tau)+0.5*np.arctan(self.cutoff*(self._u+tau))-0.5*np.arctan(self.cutoff*(self._u-tau)))\
                    -1.0j*self.alpha*self.cutoff*tau*(1.0+1.0/(1+(self._u*self.cutoff)**2))
            
        elif (which_corr=='C'):

            i1 = -self.alpha/2*sum([k2*k3*np.real(special.loggamma(self.temperature/self.cutoff+(k1+1.0)/2+1j*self.temperature*(self._u-k3*tau)*(k2+1.0)/2.0)) for k1 in [1.0,-1.0] for k2 in [1.0,-1.0] for k3 in [1.0,-1.0] ])
            i2 = -self.alpha*self.temperature*tau*sum([np.imag( special.digamma(self.temperature/self.cutoff+(k1+1.0)/2-1.0j*self._u*self.temperature) ) for k1 in [1.0,-1.0]])
            i2 += -1j*self.alpha/2*(np.arctan(self.cutoff*(tau+self._u))-np.arctan(self.cutoff*(tau-self._u))-2.0*np.arctan(self._u*self.cutoff))


        return -i1-i2
    
    def correlation_2d_integral_marked(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            which_corr: Optional[Text] = 'A1',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        kwargs = {
            'epsrel': epsrel,
            'subdiv_limit': subdiv_limit,
            'matsubara': matsubara,
            'which_corr': which_corr}

        if shape == 'upper-triangle':
            integral = self.eta_function_marked(time_1 + delta, **kwargs) \
                       - self.eta_function_marked(time_1, **kwargs)
        elif shape == 'square':
            integral = self.eta_function_marked(time_1 + delta, **kwargs) \
                       - 2.0 * self.eta_function_marked(time_1, **kwargs) \
                       + self.eta_function_marked(time_1 - delta, **kwargs)
        elif shape == 'rectangle':
            integral = self.eta_function_marked(time_2, **kwargs) \
                       - self.eta_function_marked(time_1, **kwargs) \
                       - self.eta_function_marked(time_2 - delta, **kwargs) \
                       + self.eta_function_marked(time_1 - delta, **kwargs)
        else:
            raise NotImplementedError("Shape '{shape}' not implemented.")

        if matsubara:
            integral = integral.real
        return integral

class PowerLawSD(CustomSD):
    r"""
    Correlations corresponding to the spectral density of the standard form

    .. math::

        J(\omega) = 2 \alpha \frac{\omega^\zeta}{\omega_c^{\zeta-1}} \
                    X(\omega,\omega_c)

    with `alpha` :math:`\alpha`, `zeta` :math:`\zeta` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.

    Parameters
    ----------
    alpha : float
        The coupling strength :math:`\alpha`.
    zeta : float
        The exponent :math:`\zeta` (corresponds to the dimensionality of the
        environment). The environment is called *ohmic* if :math:`\zeta=1`,
        *superohmic* if :math:`\zeta>1` and *subohmic* if :math:`\zeta<1`
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    cutoff_type : str (default = ``'exponential'``)
        The cutoff type. Types are: {``'hard'``, ``'exponential'``,
        ``'gaussian'``}
    temperature: float
        The environment's temperature.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    """

    def __init__(
            self,
            alpha: float,
            zeta: float,
            cutoff: float,
            cutoff_type: Text = 'exponential',
            temperature: Optional[float] = 0.0,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a StandardSD (spectral density) object. """

        # check input: alpha
        try:
            tmp_alpha = float(alpha)
        except Exception as e:
            raise AssertionError("Alpha must be a float.") from e
        self.alpha = tmp_alpha

        # check input: zeta
        try:
            tmp_zeta = float(zeta)
        except Exception as e:
            raise AssertionError("Nu must be a float.") from e
        self.zeta = tmp_zeta

        # check input: cutoff
        try:
            tmp_cutoff = float(cutoff)
        except Exception as e:
            raise AssertionError("Cutoff must be a float.") from e
        self.cutoff = tmp_cutoff

        # use parent class for all the rest.
        j_function = lambda w: 2.0 * self.alpha * w ** self.zeta \
                               * self.cutoff ** (1 - zeta)

        super().__init__(j_function,
                         cutoff=cutoff,
                         cutoff_type=cutoff_type,
                         temperature=temperature,
                         name=name,
                         description=description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  alpha         = {} \n".format(self.alpha))
        ret.append("  zeta          = {} \n".format(self.zeta))

        return "".join(ret)


# === CORRELATIONS FROM SPECTRAL DENSITIES ====================================


# --- the cutoffs -------------------------------------------------------------

def _hard_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Hard cutoff function."""
    return np.heaviside(omega_c - omega, 0)

def _exponential_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Exponential cutoff function."""
    return np.exp(-omega/omega_c)

def _gaussian_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Gaussian cutoff function."""
    return np.exp(-(omega/omega_c)**2)

# dictionary for the various cutoffs in the form:
#   'cutoff_name': cutoff_function
CUTOFF_DICT = {
    'hard':_hard_cutoff,
    'exponential':_exponential_cutoff,
    'gaussian':_gaussian_cutoff,
    }

# --- 2d integrals ------------------------------------------------------------

def _2d_square_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * 2 * np.cos(time_1*omega) * (1 - np.cos(delta*omega))

def _2d_square_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    integrand = 1.0/omega**2 * 2 * np.cos(time_1*omega) \
                * (1 - np.cos(delta*omega))
    return integrand / np.tanh(omega/(2*temperature))

def _2d_square_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    return -1.0/omega**2 * 2 * np.sin(time_1*omega) * (1 - np.cos(delta*omega))

def _2d_upper_triangle_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of upper triangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1+delta)) \
                           - omega * delta * np.sin(omega*time_1))

def _2d_upper_triangle_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of upper triangle 2D time integral at finite
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1+delta)) \
                           - omega * delta * np.sin(omega*time_1)) \
                        / np.tanh(omega/(2*temperature))

def _2d_upper_triangle_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of upper triangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (np.sin(omega*time_1) \
                            - np.sin(omega*(time_1+delta)) \
                            + omega * delta * np.cos(omega*time_1))

def _2d_lower_triangle_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of lower triangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1-delta)) \
                           + omega * delta * np.sin(omega*time_1))

def _2d_lower_triangle_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of lower triangle 2D time integral at finite
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1-delta)) \
                           + omega * delta * np.sin(omega*time_1)) \
                        / np.tanh(omega/(2*temperature))

def _2d_lower_triangle_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of lower triangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (np.sin(omega*time_1)
                            - np.sin(omega*(time_1-delta))
                            - omega * delta * np.cos(omega*time_1))

def _2d_rectangle_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of rectangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos((time_2 - delta) * omega)
                           - np.cos((time_1 - delta) * omega)
                           - np.cos(time_2 * omega)
                           + np.cos(time_1 * omega))

def _2d_rectangle_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of rectangle 2D time integral at finite
    temperature without J(omega). """
    integrand = 1.0/omega**2 * (np.cos((time_2 - delta) * omega)
                                - np.cos((time_1 - delta) * omega)
                                - np.cos(time_2 * omega)
                                + np.cos(time_1 * omega))
    return integrand / np.tanh(omega/(2*temperature))

def _2d_rectangle_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of rectangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (np.sin((time_2 - delta) * omega)
                            - np.sin((time_1 - delta) * omega)
                            - np.sin(time_2 * omega)
                            + np.sin(time_1 * omega))


# dictionary for the various integrands for the 2d time integral
INTEGRAND_DICT = {
    'square': (_2d_square_integrand_real,
               _2d_square_integrand_real_t,
               _2d_square_integrand_imag),
    'upper-triangle': (_2d_upper_triangle_integrand_real,
                       _2d_upper_triangle_integrand_real_t,
                       _2d_upper_triangle_integrand_imag),
    'lower-triangle': (_2d_lower_triangle_integrand_real,
                       _2d_lower_triangle_integrand_real_t,
                       _2d_lower_triangle_integrand_imag),
    'rectangle': (_2d_rectangle_integrand_real,
                  _2d_rectangle_integrand_real_t,
                  _2d_rectangle_integrand_imag),
    }


## new class to calculate the correlation functions for the case with heat markers/counting fields

# --- 2d integrals for the C correlation function (the A1 and A2 use the same as before) --- 
# note that the 2d integral for C and those for A1 are related as
# Re part of integral for C is Im part of that for A1 times the temperature factor
# Im part of integral for C is Re part of that for A1 without the temperature factor
# This may be a slightly clunky way to do it but hope it is clear. This code was
# constructed by editing the square_integrand case, then cut-and-paste and replace square with the relevant 
# text for the other cases.

def _2d_square_integrand_C_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    #return 1.0/omega**2 * 2 * np.cos(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_square_integrand_imag(omega,delta,time_1,time_2,temperature)

def _2d_square_integrand_C_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    #integrand = 1.0/omega**2 * 2 * np.cos(time_1*omega) \
    #            * (1 - np.cos(delta*omega))
    integrand = _2d_square_integrand_imag(omega,delta,time_1,time_2,temperature)
    return integrand / np.tanh(omega/(2*temperature))

def _2d_square_integrand_C_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    #return -1.0/omega**2 * 2 * np.sin(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_square_integrand_real(omega,delta,time_1,time_2,temperature)

# above code copied then search and replace square->upper_triangle
def _2d_upper_triangle_integrand_C_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    #return 1.0/omega**2 * 2 * np.cos(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_upper_triangle_integrand_imag(omega,delta,time_1,time_2,temperature)

def _2d_upper_triangle_integrand_C_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    #integrand = 1.0/omega**2 * 2 * np.cos(time_1*omega) \
    #            * (1 - np.cos(delta*omega))
    integrand = _2d_upper_triangle_integrand_imag(omega,delta,time_1,time_2,temperature)
    return integrand / np.tanh(omega/(2*temperature))

def _2d_upper_triangle_integrand_C_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    #return -1.0/omega**2 * 2 * np.sin(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_upper_triangle_integrand_real(omega,delta,time_1,time_2,temperature)

# above code copied then search and replace square->lower_triangle
def _2d_lower_triangle_integrand_C_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    #return 1.0/omega**2 * 2 * np.cos(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_lower_triangle_integrand_imag(omega,delta,time_1,time_2,temperature)

def _2d_lower_triangle_integrand_C_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    #integrand = 1.0/omega**2 * 2 * np.cos(time_1*omega) \
    #            * (1 - np.cos(delta*omega))
    integrand = _2d_lower_triangle_integrand_imag(omega,delta,time_1,time_2,temperature)
    return integrand / np.tanh(omega/(2*temperature))

def _2d_lower_triangle_integrand_C_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    #return -1.0/omega**2 * 2 * np.sin(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_lower_triangle_integrand_real(omega,delta,time_1,time_2,temperature)

# above code copied then search and replace square->rectangle

def _2d_rectangle_integrand_C_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    #return 1.0/omega**2 * 2 * np.cos(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_rectangle_integrand_imag(omega,delta,time_1,time_2,temperature)

def _2d_rectangle_integrand_C_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    #integrand = 1.0/omega**2 * 2 * np.cos(time_1*omega) \
    #            * (1 - np.cos(delta*omega))
    integrand = _2d_rectangle_integrand_imag(omega,delta,time_1,time_2,temperature)
    return integrand / np.tanh(omega/(2*temperature))

def _2d_rectangle_integrand_C_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    #return -1.0/omega**2 * 2 * np.sin(time_1*omega) * (1 - np.cos(delta*omega))
    return _2d_rectangle_integrand_real(omega,delta,time_1,time_2,temperature)


INTEGRAND_DICT_C = {
    'square': (_2d_square_integrand_C_real,
               _2d_square_integrand_C_real_t,
               _2d_square_integrand_C_imag),
    'upper-triangle': (_2d_upper_triangle_integrand_C_real,
                       _2d_upper_triangle_integrand_C_real_t,
                       _2d_upper_triangle_integrand_C_imag),
    'lower-triangle': (_2d_lower_triangle_integrand_C_real,
                       _2d_lower_triangle_integrand_C_real_t,
                       _2d_lower_triangle_integrand_C_imag),
    'rectangle': (_2d_rectangle_integrand_C_real,
                  _2d_rectangle_integrand_C_real_t,
                  _2d_rectangle_integrand_C_imag),
    }