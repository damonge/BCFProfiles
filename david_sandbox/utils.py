import numpy as np
import pyccl as ccl


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
            integral = np.trapz(integrand, x=l10xvs*np.log(10))
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


class HaloProfileGasBFCDavid(ccl.halos.HaloProfile):
    def __init__(self, *, mass_def, log10Mc, mu, delta,
                 Mvir2Mtot, xv_min=1E-3, xv_max=1E3):
        super().__init__(mass_def=mass_def)
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=500,
                                     plaw_fourier=-2.)
        # All equations below are from 2507.07892, unless otherwise stated.
        self.log10Mc = log10Mc  # Mass scaling of intermediate-radius slope of hot gas profile. Eq. 2.12.
        self.mu = mu  # Mass slope of intermediate-radius slope of hot gas profile. Eq. 2.12.
        self.delta = delta  # Outer slope of the hot gas profile. Eq. 2.12.
        self.theta_c0 = 0.3  # Scaling of the core radius of the hot gas profile. Eq. 2.12 and page 18.
        self.alpha = 1.0  # Inner slope of the hot gas profile. Eq. 2.12.
        self.gamma = 1.5  # Outer slope of the hot gas profile. Eq. 2.12.
        self.ciga0 = 0.1  # Scaling of the central galaxy cold gas fraction at z=0. Eq. 2.23 and page 18.
        self.Mstar = 2.5E11  # Pivot mass forthe stellar mass fraction. Eq. 2.21.
        self.Nstar = 0.028  # Normalisation of the stellar mass fractions. Eq. 2.21.
        self.eta = 0.07  # High-mass slope of the stellar mass fraction. Eq. 2.21.
        self.deta = 0.22  # High-mass slope of the central galaxy stellar mass fraction. Eq. 2.21.
        self.zeta = 1.376  # Low-mass slope of the stellar mass fraction. Eq. 2.2.
        self.xv_min = xv_min
        self.xv_max = xv_max
        self.Mv2Mt = Mvir2Mtot

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"
        super().__init__(mass_def=mass_def)

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
        prof = self._u(xv[None, :], Mv, a, cosmo)
        integral = np.trapz(xv[None, :]**3*prof, x=np.log(xv), axis=-1)
        return integral

    def _u(self, xvir, Mv, a, cosmo):
        Mc = 10**self.log10Mc/cosmo['h']
        xM = (Mv/Mc)**self.mu

        beta = 3*xM/(1+xM)

        theta_c = self.theta_c0  # /a**0.5 <- cancelling redshift dependence for now

        eps = epsilon_bfc(cosmo, Mv, a)

        return 1/((1+(xvir/theta_c)**self.alpha)**(beta[:, None]/self.alpha) *
                  (1+(xvir/eps[:, None])**self.gamma)**(self.delta/self.gamma))

    def _real(self, cosmo, r, M, a):
        # Real-space profile.
        # Output in units of eV/cm^3
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Get normalisation
        I_hga = self._int_hga(M_use, a, cosmo)
        fhga = self._fhga(M_use, a, cosmo)
        Mtot = self.Mv2Mt.get_Mtot(M_use, a, cosmo)
        rvir = self.mass_def.get_radius(cosmo, M_use, a) / a
        rho_hga0 = fhga*Mtot/(4*np.pi*rvir**3*I_hga)

        # Get shape
        xvir = r_use[None, :] / rvir[:, None]
        u = self._u(xvir, M_use, a, cosmo)

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
