import numpy as np
import pyccl as ccl
from scipy.integrate import cumulative_trapezoid, simpson


# Epsilon-mass relation
def epsilon_bfc(cosmo, M, a):
    eps = 4 - 0.5*1.686/ccl.sigmaM(cosmo, M, a)
    eps = eps * (eps > 1) + (eps < 1) * 1.0
    return eps


class ConcentrationDutton14(ccl.halos.Concentration):
    """Concentrations form Dutton+Maccio. 2014)
    c200. 200 times RHOC)
    Assumes PLANCK cosmology
    """
    name = 'Dutton14'

    def __init__(self, *, mass_def="200c"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ["200c"]

    def _concentration(self, cosmo, M, a):
        z = 1/a-1
        M_pivot_inv = cosmo["h"] * 1E-12
        A = 0.520 + (0.905 - 0.520) * np.exp(-0.617*z**1.21)
        B = -0.101 + 0.026*z
        return 10.0**A*(M*M_pivot_inv)**(B)


class Mvir2MtotNFWt(object):
    def __init__(self, concentration_f,
                 c_range=[1.0, 30.], nc=40,
                 eps_range=[1.0, 5.0], neps=20,
                 xv_min=1E-3, xv_inf=1E3, dl10xv=0.001,
                 interp_kind='linear', analytic=True):
        from scipy.interpolate import RegularGridInterpolator

        self.analytic = analytic
        self.cf = concentration_f
        self.ef = epsilon_bfc

        self.l10xv_min = np.log10(xv_min)
        self.dl10xv = dl10xv
        self.xv_inf = 1E3
        self.cs = np.linspace(c_range[0], c_range[1], nc)
        self.epss = np.linspace(eps_range[0], eps_range[1], neps)

        self.int_1 = np.array([[self.int_NFW(1, c, eps) for eps in self.epss]
                               for c in self.cs])
        self.int_inf = np.array([[self.int_NFW(self.xv_inf, c, eps)
                                  for eps in self.epss]
                                 for c in self.cs])
        self.conv_arr = self.int_inf/self.int_1
        self.conv_int = RegularGridInterpolator([self.cs, self.epss],
                                                self.conv_arr,
                                                method=interp_kind)
        self.int1_int = RegularGridInterpolator([self.cs, self.epss],
                                                self.int_1,
                                                method=interp_kind)

    def int_NFW(self, xv_max, c, eps):
        if self.analytic:
            x = c*xv_max
            tau = c*eps
            t2 = tau**2
            x2 = x**2
            pre = t2/(2*(t2+1)**3*(1+x)*(t2+x2))
            f1 = (t2+1)*x*(x*(x+1)-t2*(x-1)*(2+3*x)-2*t2**2)
            f2 = tau*(x+1)*(t2+x2)*(2*(3*t2-1)*np.arctan(x/tau) +
                                    tau*(t2-3)*np.log(t2*(1+x)**2/(t2+x2)))
            return pre * (f1+f2)
        else:
            l10xv_max = np.log10(xv_max)
            l10xvs = np.arange(self.l10xv_min, l10xv_max, self.dl10xv)
            xvs = 10**l10xvs
            integrand = xvs**2/((1+xvs*c)**2*(1+(xvs/eps)**2)**2)
            integral = np.trapezoid(integrand, x=l10xvs*np.log(10))
            return integral

    def get_Mtot(self, Mv, a, cosmo):
        M_use = np.atleast_1d(Mv)
        cs = self.cf(cosmo, M_use, a)
        epss = self.ef(cosmo, M_use, a)
        coords = np.array([cs, epss]).T
        ratio = self.conv_int(coords)
        Mtot = M_use * ratio
        if np.ndim(Mv) == 0:
            Mtot = np.squeeze(Mtot, axis=0)
        return Mtot

    def get_I1(self, Mv, a, cosmo):
        M_use = np.atleast_1d(Mv)
        cs = self.cf(cosmo, M_use, a)
        epss = self.ef(cosmo, M_use, a)
        coords = np.array([cs, epss]).T
        I1 = self.int1_int(coords)
        if np.ndim(Mv) == 0:
            I1 = np.squeeze(I1, axis=0)
        return I1

    def get_Mcumul(self, Mv, a, cosmo, xv):
        M_use = np.atleast_1d(Mv)
        xv_use = np.atleast_2d(xv)
        assert ((xv_use.shape[0] == len(M_use)) or
                (xv_use.shape[0] == 1))
        cs = self.cf(cosmo, M_use, a)[:, None]
        epss = self.ef(cosmo, M_use, a)[:, None]
        Ix = self.int_NFW(xv_use, cs, epss)
        I1 = self.get_I1(M_use, a, cosmo)[:, None]
        Mcumul = M_use[:, None] * Ix/I1
        if np.ndim(Mv) == 0:
            Mcumul = np.squeeze(Mcumul, axis=0)
        return Mcumul


class HaloProfileCustom(ccl.halos.HaloProfile):
    def volint(self, cosmo, xmax, M, a, xmin=1E-3, nx=100):
        """Calculates the volume integral of the profile,
        Int[4 pi r^2 dr rho(r)]. This is used to calculate the total
        mass of the profile.

        Parameters
        ----------
        cosmo : ccl.Cosmology
            Cosmology object.
        xmax : float
            Maximum radius of the integral in units of Rvir.
        M : float or array_like
            Halo mass in units of Msun/h.
        a : float
            Scale factor.
        xmin : float, optional
            Minimum radius of the integral in units of Rvir. Default is 1E-3.
        nx : int, optional
            Number of points to use in the integral. Default is 100.
        Returns
        -------
        Vint : float or array_like
            Volume integral of the profile in units of Msun/h.
        """
        # Volume integral of the profile.
        M_use = np.atleast_1d(M)
        rvir = self.mass_def.get_radius(cosmo, M_use, a) / a

        xs = np.geomspace(xmin, xmax, nx)
        lxs = np.log(xs)
        Vint = []
        for Mm, rv in zip(M_use, rvir):
            r = xs * rv
            prof = self._real(cosmo, r, Mm, a)
            integrand = prof * xs**3
            Vint.append(simpson(integrand, x=lxs)*4*np.pi*rv**3)
        Vint = np.array(Vint)
        if np.ndim(M) == 0:
            Vint = np.squeeze(Vint, axis=0)
        return Vint


class HaloProfilePressureNFWBFCDavid(HaloProfileCustom):
    def __init__(self, *, mass_def, log10Mc, mu, delta, a0_nth=0.1, XH=0.76,
                 Mvir2Mtot, xv_min=1E-3, xv_max=1E3, nxv=260):
        super().__init__(mass_def=mass_def)
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=500,
                                     plaw_fourier=-2.)
        # All equations below are from 2507.07892, unless otherwise stated.
        self.log10Mc = log10Mc  # Mass scaling of intermediate-radius slope of hot gas profile. Eq. 2.12. # noqa
        self.mu = mu  # Mass slope of intermediate-radius slope of hot gas profile. Eq. 2.12. # noqa
        self.delta = delta  # Outer slope of the hot gas profile. Eq. 2.12. # noqa
        self.theta_c0 = 0.3  # Scaling of the core radius of the hot gas profile. Eq. 2.12 and page 18. # noqa
        self.alpha = 1.0  # Inner slope of the hot gas profile. Eq. 2.12. # noqa
        self.gamma = 1.5  # Outer slope of the hot gas profile. Eq. 2.12. # noqa
        self.ciga0 = 0.1  # Scaling of the central galaxy cold gas fraction at z=0. Eq. 2.23 and page 18. # noqa
        self.Mstar = 2.5E11  # Pivot mass forthe stellar mass fraction. Eq. 2.21. # noqa
        self.Nstar = 0.03  # Normalisation of the stellar mass fractions. Eq. 2.21. # noqa
        # Changed from 0.028 to 0.03 in Table 1 of 2507.07991
        self.eta = 0.1  # High-mass slope of the stellar mass fraction. Eq. 2.21. # noqa
        # Changed from 0.07 to 0.1 in Table 1 of 2507.07991
        self.deta = 0.22  # High-mass slope of the central galaxy stellar mass fraction. Eq. 2.21. # noqa
        self.zeta = 1.376  # Low-mass slope of the stellar mass fraction. Eq. 2.2. # noqa
        self.n_nth = 0.8  # Non-thermal pressure profile (P_nth = a_nth(z)*(r/rvir)^n_nth): slope # noqa
        self.a0_nth = a0_nth  # Non-thermal pressure profile (P_nth = a_nth(z)*(r/rvir)^n_nth): normalisation at z=0 # noqa
        self.xv_min = xv_min
        self.xv_max = xv_max
        self.nxv = nxv
        self.Mv2Mt = Mvir2Mtot
        self.XH = XH  # Mass fraction of hydrogen, used to convert total gas pressure into electron P  # noqa
        self.prefac_Pe = 2 * (self.XH + 1) / (5 * self.XH + 3)  # This is said prefactor  # noqa

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"

    def update_parameters(self, *, log10Mc=None, mu=None,
                          delta=None, a0_nth=None):
        if log10Mc is not None:
            self.log10Mc = log10Mc
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delta = delta
        if a0_nth is not None:
            self.a0_nth = a0_nth

    def _fhga(self, Mv, a, cosmo):
        xstar = Mv*cosmo['h']/self.Mstar
        fstar = self.Nstar/(xstar**self.eta+1/xstar**self.zeta)
        fcga = self.Nstar/(xstar**(self.eta+self.deta)+1/xstar**self.zeta)
        # fsga = fstar - fcga
        # fsga = fsga * (fsga > 0)
        ciga = self.ciga0  # /a**1.5  <- cancelling redshift dependence for now
        figa = ciga * fcga
        fb = cosmo['Omega_b'] / (cosmo['Omega_c'] + cosmo['Omega_b'])
        fhga = fb-fstar-figa
        fhga = fhga * (fhga > 0)
        return fhga

    def _int_hga(self, Mv, a, cosmo):
        xv = np.geomspace(self.xv_min, self.xv_max, 60)
        lxv = np.log(xv)
        prof = self._u_hga(xv[None, :], Mv, a, cosmo)
        integral = np.trapezoid(xv[None, :]**3*prof, x=lxv, axis=-1)
        return integral

    def _u_hga(self, xvir, Mv, a, cosmo):
        Mc = 10**self.log10Mc/cosmo['h']
        xM = (Mv/Mc)**self.mu

        beta = 3*xM/(1+xM)

        theta_c = self.theta_c0  # /a**0.5 <- TODO: cancelling redshift dependence for now  # noqa

        eps = epsilon_bfc(cosmo, Mv, a)

        return 1/((1+(xvir/theta_c)**self.alpha)**(beta[:, None]/self.alpha) *
                  (1+(xvir/eps[:, None])**self.gamma)**(self.delta/self.gamma))

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Get gas profile normalisation
        I_hga = self._int_hga(M_use, a, cosmo)
        fhga = self._fhga(M_use, a, cosmo)
        Mtot = self.Mv2Mt.get_Mtot(M_use, a, cosmo)
        rvir = self.mass_def.get_radius(cosmo, M_use, a) / a
        # Density normalisation in physical (non-comoving) units.
        rho_hga0 = fhga*Mtot/(4*np.pi*rvir**3*I_hga*a**3)

        # Get integration axis
        xv = np.geomspace(self.xv_min, self.xv_max, self.nxv)  # [Nx]
        lxv = np.log(xv)
        # Gas profile
        ugas = self._u_hga(xv[None, :], M_use, a, cosmo)  # [Nm, Nx]
        rhogas = rho_hga0[:, None] * ugas  # [Nm, Nx]
        # Total NFW mass profile.
        # TODO: account for baryonified mass profile (?)
        Mcumul = self.Mv2Mt.get_Mcumul(M_use, a, cosmo, xv)  # [Nm, Nx]
        # Hydrostatic equilibrium integral
        integrand = rhogas * Mcumul / xv[None, :]  # [Nm, Nx]
        Pint = cumulative_trapezoid(integrand, x=lxv,
                                    initial=0, axis=-1)  # [Nm, Nx]
        # The above is the integral from 0 to xv. We want the
        # integral from xv to infinity, so we need to subtract from
        # the total integral at infinity.
        Pint = Pint[:, -1][:, None] - Pint  # [Nm, Nx]

        # Now evaluate this at the desired radii via linear interpolation.
        # Note that we correct by an additional factor of a to put the
        # output in physical (non-comoving) units.
        lxvir = np.log(r_use[None, :] / rvir[:, None])  # [Nm, Nr]
        prof = np.array([np.interp(lxvir[i, :], lxv,
                                   Pint[i, :], right=0)/(a*rvir[i])
                         for i in range(len(M_use))])  # [Nm, Nr]

        # Account for non-thermal fraction using the parametrisation
        # in Section 2.1 of 2507.07991
        if self.a0_nth > 0:
            f1 = 1/a**0.5
            fmax = 1/(4**self.n_nth*self.a0_nth)
            f2 = (fmax-1)*np.tanh(0.5*(1/a-1))+1
            fnth = min(f1, f2)
            # Note/TODO, we are modelling the scale dependence as
            # (r/rvir)^n_nth, although 2507.07991 uses (r/r500)^n_nth.
            # Michael's code used rvir, so I'm sticking with this.
            frac_nth = (self.a0_nth * fnth *
                        np.exp(self.n_nth*lxvir))  # [Nm, Nr]
            # Truncate so frac_nth < 1
            sh = frac_nth.shape
            frac_nth = frac_nth.flatten()
            frac_nth[frac_nth > 1] = 1
            frac_nth = frac_nth.reshape(sh)
            # Keep only the thermal part of the pressure profile
            prof = prof * (1-frac_nth)

        # Account for units conversion to eV/cm^3
        # So far we've calculated Int[dr M(<r) rho(r) / r^2],
        # which has units of Msun^2/Mpc^4.
        # The quantity below is the gravitational constant in
        # in units such that the final profile is in eV/cm^3.
        G_to_eV_cm3 = 1.86031781E-27
        # We also then transform total pressure to electron pressure
        prof *= G_to_eV_cm3*self.prefac_Pe

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileGasBFCDavid(HaloProfileCustom):
    def __init__(self, *, mass_def, log10Mc, mu, delta,
                 Mvir2Mtot, xv_min=1E-3, xv_max=1E3):
        super().__init__(mass_def=mass_def)
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=500,
                                     plaw_fourier=-2.)
        # All equations below are from 2507.07892, unless otherwise stated.
        self.log10Mc = log10Mc  # Mass scaling of intermediate-radius slope of hot gas profile. Eq. 2.12.  # noqa
        self.mu = mu  # Mass slope of intermediate-radius slope of hot gas profile. Eq. 2.12.  # noqa
        self.delta = delta  # Outer slope of the hot gas profile. Eq. 2.12. # noqa
        self.theta_c0 = 0.3  # Scaling of the core radius of the hot gas profile. Eq. 2.12 and page 18. # noqa
        self.alpha = 1.0  # Inner slope of the hot gas profile. Eq. 2.12. # noqa
        self.gamma = 1.5  # Outer slope of the hot gas profile. Eq. 2.12. # noqa
        self.ciga0 = 0.1  # Scaling of the central galaxy cold gas fraction at z=0. Eq. 2.23 and page 18. # noqa
        self.Mstar = 2.5E11  # Pivot mass forthe stellar mass fraction. Eq. 2.21. # noqa
        self.Nstar = 0.028  # Normalisation of the stellar mass fractions. Eq. 2.21. # noqa
        self.eta = 0.07  # High-mass slope of the stellar mass fraction. Eq. 2.21. # noqa
        self.deta = 0.22  # High-mass slope of the central galaxy stellar mass fraction. Eq. 2.21. # noqa
        self.zeta = 1.376  # Low-mass slope of the stellar mass fraction. Eq. 2.2.  # noqa
        self.xv_min = xv_min
        self.xv_max = xv_max
        self.Mv2Mt = Mvir2Mtot

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"

    def update_parameters(self, *, log10Mc=None, mu=None, delta=None):
        if log10Mc is not None:
            self.log10Mc = log10Mc
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delta = delta

    def _fhga(self, Mv, a, cosmo):
        xstar = Mv*cosmo['h']/self.Mstar
        fstar = self.Nstar/(xstar**self.eta+1/xstar**self.zeta)
        fcga = self.Nstar/(xstar**(self.eta+self.deta)+1/xstar**self.zeta)
        # fsga = fstar - fcga
        # fsga = fsga * (fsga > 0)
        ciga = self.ciga0  # /a**1.5  <- cancelling redshift dependence for now
        figa = ciga * fcga
        fb = cosmo['Omega_b'] / (cosmo['Omega_c'] + cosmo['Omega_b'])
        fhga = fb-fstar-figa
        fhga = fhga * (fhga > 0)
        return fhga

    def _int_hga(self, Mv, a, cosmo):
        xv = np.geomspace(self.xv_min, self.xv_max, 60)
        prof = self._u_hga(xv[None, :], Mv, a, cosmo)
        integral = np.trapezoid(xv[None, :]**3*prof, x=np.log(xv), axis=-1)
        return integral

    def _u_hga(self, xvir, Mv, a, cosmo):
        Mc = 10**self.log10Mc/cosmo['h']
        xM = (Mv/Mc)**self.mu

        beta = 3*xM/(1+xM)

        theta_c = self.theta_c0  # /a**0.5 <- TODO: cancelling redshift dependence for now # noqa

        eps = epsilon_bfc(cosmo, Mv, a)

        return 1/((1+(xvir/theta_c)**self.alpha)**(beta[:, None]/self.alpha) *
                  (1+(xvir/eps[:, None])**self.gamma)**(self.delta/self.gamma))

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Get gas profile normalisation
        I_hga = self._int_hga(M_use, a, cosmo)
        fhga = self._fhga(M_use, a, cosmo)
        Mtot = self.Mv2Mt.get_Mtot(M_use, a, cosmo)
        rvir = self.mass_def.get_radius(cosmo, M_use, a) / a
        rho_hga0 = fhga*Mtot/(4*np.pi*rvir**3*I_hga)

        # Get shape
        xvir = r_use[None, :] / rvir[:, None]
        u = self._u_hga(xvir, M_use, a, cosmo)

        prof = rho_hga0[:, None] * u

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def get_normalization(self, cosmo, a, *, hmc):
        def Mhga_integrand(M):
            Mtot = self.Mv2Mt.get_Mtot(M, a, cosmo)
            fhga = self._fhga(M, a, cosmo)
            return Mtot * fhga

        return hmc.integrate_over_massfunc(Mhga_integrand, cosmo, a)


def bU(hmc, prof, z, cosmo, xmax=1000, xmin=0.001, nx=100):
    """
    Compute <bU>

    Parameters
    ----------
    hmc : HMCalculator
        The halo mass-concentration relation.
    profP : HaloProfile
        The pressure profile
    z : float
        The redshift at which to compute the bias.
    cosmo : Cosmology
        The cosmology object.
    xmax : float, optional
        The maximum radius to integrate to in units of the virial radius. Default is 1000.  # noqa
    xmin : float, optional
        The minimum radius to calculate the profile in units of the virial radius. Default is 0.001.  # noqa
    nx : int, optional
        The number of radial points to use for integration. Default is 100.

    Returns
    -------
    bU : ndarray
        <bU>
    """
    a = 1/(1 + z)
    hmc._get_ingredients(cosmo, a, get_bf=True)
    lMs = hmc._lmass  # [Nm]
    Ms = 10**lMs  # [Nm]
    Uvol = prof.volint(cosmo, xmax, Ms, a, xmin=xmin, nx=nx)  # [Nm]
    norm = prof.get_normalization(cosmo, a, hmc=hmc)
    bU = hmc._integrate_over_mbf(Uvol)/norm

    return bU
