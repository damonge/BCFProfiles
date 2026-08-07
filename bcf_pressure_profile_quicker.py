from params_bfc import par as default_par
from bcf_functions import (
    fSTAR_fct,
    uHGA_fct, uIGA_fct, uCGA_fct,
    MNFWtr_fct, mNFWtr_fct, mTOTtr_fct,
    rhoc_of_z,
    DELTAVIR,
)
from scipy.interpolate import splrep, splev, RectBivariateSpline
from scipy.integrate import simpson, cumulative_trapezoid
import numpy as np
import pyccl as ccl
import copy
import hashlib
 
# Newton's constant [Mpc / Msun * (km/s)^2]
G = 4.299e-9
 
# Unit-conversion factors
_MCONV  = 1.989e33      # Msun -> g
_DCONV  = 3.085678e24   # Mpc  -> cm
_KM2CM  = 1e5           # km   -> cm
_TO_CGS = _MCONV / _DCONV**3 * _KM2CM**2
_TO_EV  = 6.242e11      # erg/cm^3 -> eV/cm^3
 
 
def get_prefac_P(kind, XH=0.76):
    """Prefactor transforming gas pressure into a chosen number density."""
    if kind == "n_total":
        return 1.0
    elif kind == "n_H":
        return 4 * XH / (5 * XH + 3)
    elif kind == "n_baryon":
        return (3 * XH + 1) / (5 * XH + 3)
    elif kind == "n_electron":
        return (2 * (XH + 1)) / (5 * XH + 3)
    else:
        raise NotImplementedError(f"Pressure type {kind} not implemented")
 
 
def _cosmo_key(cosmo):
    return (cosmo['Omega_b'], cosmo['Omega_c'], cosmo['h'],
            cosmo['sigma8'], cosmo['n_s'])
 
 
def _baryon_hash(param):
    """Hash of baryon+code parameters — drives spline cache invalidation."""
    p = param.baryon
    c = param.code
    key = (
        p.Mc, p.mu, p.thco, p.alpha, p.beta, p.gamma, p.delta,
        p.eta, p.deta, p.rcga, p.ciga,
        c.eps0, c.eps1,
        p.a_nth, p.n_nth, p.b_nth,
    )
    return hashlib.md5(str(key).encode()).hexdigest()
 
 
def _cvir_dutton14(mvir, z):
    """Concentration from Dutton & Macciò (2014), c200c."""
    A = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    B = -0.101 + 0.026 * z
    return 10.0**A * (mvir / 1.0e12)**B
 
 
class HaloProfilePressureBFC(ccl.halos.HaloProfilePressure):
    """
    Pressure profile of the BFC model.
 
    Parameters
    ----------
    mass_def      : CCL mass definition
    bias_model    : callable — bias(cosmo, m [Msun], a)
    param         : BFC parameter object (default: params_bfc.par())
    little_h      : I/O in Mpc/h and Msun/h if True
    units         : 'cgs' -> erg/cm^3, 'eV' -> eV/cm^3, else internal
    comoving      : return comoving pressure if True
    full_output   : also return MDMB, MHGA, MACM, MCGA, MIGA (not for CCL)
    concentration : None -> Dutton & Maccio (2014); float -> fixed; callable
    kind          : 'n_total' | 'n_H' | 'n_baryon' | 'n_electron'
    use_nfw_mass  : use MNFW instead of MDMB in the pressure integral
    skip_ac       : skip adiabatic contraction entirely (xx = 1 everywhere)
    """
 
    _rmin_corr = 1e-4
    _rmax_corr = 1e3
    _nr_corr   = 256
    _na_corr   = 64
 
    def __init__(self, *, mass_def, bias_model, param=None,
                 little_h=False, units='cgs',
                 comoving=False, full_output=False,
                 concentration=None, kind='n_total',
                 use_nfw_mass=False, skip_ac=False):
 
        self.hmd          = mass_def
        self.bM           = bias_model
        self.little_h     = little_h
        self.units        = units
        self.comoving     = comoving
        self.output       = full_output
        self.cvir_func    = concentration
        self.kind         = kind
        self.use_nfw_mass = use_nfw_mass
        self.skip_ac      = skip_ac
        self.prefac       = get_prefac_P(kind)
        self.param        = copy.deepcopy(param if param is not None else default_par())
 
        self.cvir_first   = True
 
        # Runtime state
        self.h0:   float      = 1.0
        self.cvir: np.ndarray = None
        self.Mtot: np.ndarray = None
        self.fhga: np.ndarray = None
 
        # Baryon-profile spline cache
        self._baryon_hash_val  = None
        self._spline_cache     = {}
        self._spline_rbin_hash = None
        self._spline_mv_hash   = None
 
        # Correlation-function cache (built only when AC models 0-4 are active)
        self._corr_key    = None
        self._xi_spline   = None
        self._r_grid_corr = None
        self._a_grid_corr = None
 
        super().__init__(mass_def=mass_def)
 
    # Correlation-function table — only called for AC models 0-4
    def _ensure_corr_table(self, cosmo, r_needed_max_com):
        key = (_cosmo_key(cosmo), self._rmin_corr, self._rmax_corr,
               self._nr_corr, self._na_corr)
        need_build = (getattr(self, '_corr_key', None) != key)
        if not need_build and r_needed_max_com > self._rmax_corr * 0.95:
            self._rmax_corr *= 2
            need_build = True
        if not need_build:
            return
 
        r_grid = np.geomspace(self._rmin_corr, self._rmax_corr, self._nr_corr)
        ccl.spline_params.A_SPLINE_NA_PK   = 50
        ccl.spline_params.A_SPLINE_NLOG_PK = 11
        a_arr = ccl.get_pk_spline_a()
 
        nk    = int((6 - (-6)) * 60)
        k_arr = np.logspace(-6, 6, nk)
        Pk    = ccl.linear_matter_power(cosmo, k_arr, a_arr)
        pk2d  = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=Pk,
                         is_logp=False, extrap_order_lok=1, extrap_order_hik=2)
 
        xi = np.empty((len(a_arr), self._nr_corr))
        for i, a in enumerate(a_arr):
            xi[i] = ccl.correlation_3d(cosmo, r=r_grid, a=a, p_of_k_a=pk2d)
 
        self._xi_spline   = RectBivariateSpline(a_arr, np.log(r_grid), xi,
                                                kx=3, ky=3, s=0.0)
        self._r_grid_corr = r_grid
        self._a_grid_corr = a_arr
        self._corr_key    = key
 
    def _xi_ar(self, cosmo, a, r_com):
        """xi(a, r_comoving) via cached bivariate spline."""
        self._ensure_corr_table(cosmo, float(np.max(r_com)))
        a_c = float(np.clip(a, self._a_grid_corr[0], self._a_grid_corr[-1]))
        return self._xi_spline(a_c, np.log(r_com), grid=False)
 
    # Baryon-profile spline cache
    def _ensure_splines(self, rbin, mv, rvir, eps):
        """
        Return (MHGA, MIGA, MCGA, MHGA_tck, MIGA_tck, MCGA_tck, multi).
 
        Only recomputed when the baryon-parameter hash, r-grid, or mass array
        have changed. A pure cosmo MCMC step skips all cumtrapz/splrep work.
        """
        bh = _baryon_hash(self.param)
        rh = hashlib.md5(rbin.ravel().tobytes()).hexdigest()
        mh = hashlib.md5(mv.ravel().tobytes()).hexdigest()
 
        if (bh == self._baryon_hash_val and
                rh == self._spline_rbin_hash and
                mh == self._spline_mv_hash):
            c = self._spline_cache
            return (c['MHGA'], c['MIGA'], c['MCGA'],
                    c['MHGA_tck'], c['MIGA_tck'], c['MCGA_tck'], c['multi'])
 
        # --- recompute ---
        n_m  = mv.size
        r1d  = rbin.ravel()
        mv1d = mv.ravel()
        mv_col   = mv1d[:, None]
        rvir_col = rvir.ravel()[:, None]
        eps_col  = eps.ravel()[:, None] if np.ndim(eps) > 0 else eps
 
        # uHGA_fct can return (n_m, 1, n_r) due to broadcasting of cvir (n_m,1)
        # against rbin (1, n_r) inside the function; squeeze to (n_m, n_r).
        uHGA = np.squeeze(uHGA_fct(r1d[None, :], self.cvir, mv_col, eps_col, self.param))
        uIGA = np.squeeze(uIGA_fct(r1d[None, :], rvir_col))
        uCGA = np.squeeze(uCGA_fct(r1d[None, :], mv_col, self.param))
        # Re-ensure 2-D in the single-mass case (squeeze collapses to 1-D)
        if uHGA.ndim == 1: uHGA = uHGA[None, :]
        if uIGA.ndim == 1: uIGA = uIGA[None, :]
        if uCGA.ndim == 1: uCGA = uCGA[None, :]
 
        if n_m > 1:
            norm_H = (self.Mtot.ravel() /
                      (4.0 * np.pi * simpson(r1d**2 * uHGA, x=r1d, axis=1)))[:, None]
            norm_I = (self.Mtot.ravel() /
                      (4.0 * np.pi * simpson(r1d**2 * uIGA, x=r1d, axis=1)))[:, None]
            norm_C = (self.Mtot.ravel() /
                      (4.0 * np.pi * simpson(r1d**2 * uCGA, x=r1d, axis=1)))[:, None]
        else:
            norm_H = self.Mtot / (4.0 * np.pi * simpson(r1d**2 * uHGA[0], x=r1d))
            norm_I = self.Mtot / (4.0 * np.pi * simpson(r1d**2 * uIGA[0], x=r1d))
            norm_C = self.Mtot / (4.0 * np.pi * simpson(r1d**2 * uCGA[0], x=r1d))
 
        rhoHGA = norm_H * uHGA
        rhoIGA = norm_I * uIGA
        rhoCGA = norm_C * uCGA
 
        MHGA = cumulative_trapezoid(4.0 * np.pi * r1d**2 * rhoHGA,
                                    r1d, initial=0.0, axis=1)
        MIGA = cumulative_trapezoid(4.0 * np.pi * r1d**2 * rhoIGA,
                                    r1d, initial=0.0, axis=1)
        MCGA = cumulative_trapezoid(4.0 * np.pi * r1d**2 * rhoCGA,
                                    r1d, initial=0.0, axis=1)
 
        multi = n_m > 1
        if multi:
            kx       = min(3, n_m - 1)
            MHGA_tck = RectBivariateSpline(mv1d, r1d, MHGA, kx=kx)
            MIGA_tck = RectBivariateSpline(mv1d, r1d, MIGA, kx=kx)
            MCGA_tck = RectBivariateSpline(mv1d, r1d, MCGA, kx=kx)
        else:
            MHGA_tck = splrep(r1d, MHGA[0], s=0, k=1)
            MIGA_tck = splrep(r1d, MIGA[0], s=0, k=1)
            MCGA_tck = splrep(r1d, MCGA[0], s=0, k=1)
 
        self._spline_cache = dict(
            MHGA=MHGA, MIGA=MIGA, MCGA=MCGA,
            MHGA_tck=MHGA_tck, MIGA_tck=MIGA_tck, MCGA_tck=MCGA_tck,
            multi=multi,
        )
        self._baryon_hash_val  = bh
        self._spline_rbin_hash = rh
        self._spline_mv_hash   = mh
 
        return MHGA, MIGA, MCGA, MHGA_tck, MIGA_tck, MCGA_tck, multi
 
    def _real(self, cosmo, rr, mvir, ah):
        """
        Compute the thermal pressure profile.
 
        Parameters / return signature unchanged from original.
        """
        rdim = np.ndim(rr)
        mdim = np.ndim(mvir)

        Om = cosmo['Omega_c'] + cosmo['Omega_b']
        Ob = cosmo['Omega_b']
        self.h0 = cosmo['h']
        fb = Ob / Om
 
        self.cosmo = cosmo
 
        if not self.little_h:
            rr   = rr   * self.h0
            mvir = mvir * self.h0
 
        if ah is not None:
            self.param.cosmo.z = 1 / ah - 1
            zz = self.param.cosmo.z
        else:
            ah = 1 / (1 + self.param.cosmo.z)
            zz = self.param.cosmo.z
 
        r    = rr
        mv   = np.atleast_1d(mvir)[:, None]    # (n_m, 1)
        rbin = r[None, :]                       # (1,   n_r)
        rvir = (3.0 * mv /
                (4.0 * np.pi * DELTAVIR * rhoc_of_z(self.param)))**(1.0 / 3.0)
 
        # Baryon fractions
        eta  = self.param.baryon.eta
        deta = self.param.baryon.deta
 
        fcdm  = (Om - Ob) / Om
        fstar = fSTAR_fct(mv, self.param, eta)
        fcga  = fSTAR_fct(mv, self.param, eta + deta)
        fsga  = np.clip(fstar - fcga, 0.0, None)
        figa  = self.param.baryon.ciga * fcga
        self.fhga = fb - fcga - fsga - figa
 
        # Concentration, eps, tau
        eps0   = self.param.code.eps0
        eps1   = self.param.code.eps1
        mccl   = (mv / self.h0)[:, 0]          # Msun (no h), shape (n_m,)
 
        sigma_M    = ccl.sigmaM(cosmo, mccl, ah)
        peak_height = 1.686 / sigma_M
        eps = np.clip((eps0 - eps1 * peak_height)[:, None], 1.0, None)
 
        if self.cvir_first:
            if self.cvir_func is None:
                self.cvir = _cvir_dutton14(mv, zz)
            elif callable(self.cvir_func):
                self.cvir = self.cvir_func(cosmo, mccl, ah)[:, None]
            else:
                self.cvir = float(self.cvir_func) * np.ones_like(mv)
 
        tau = eps * self.cvir
 
        # Total (truncated-NFW) mass
        self.Mtot = mv * mTOTtr_fct(tau) / mNFWtr_fct(self.cvir, tau)
        self.Mtot = np.where(self.Mtot < mv, mv, self.Mtot)
 
        # NFW mass profile
        MNFW = MNFWtr_fct(rbin, self.cvir, tau, mv, self.param)
 
        #  rhoHGA - always needed for the pressure integrand
        uHGA = uHGA_fct(rbin, self.cvir, mv, eps, self.param)
        rv = (3.0*mv/(4.0*np.pi*DELTAVIR*rhoc_of_z(self.param)))**(1.0/3.0)  # [nm, 1]
        x = np.geomspace(1E-2, 1E2, 100)
        rx = x[None, :]*rv
        u = uHGA_fct(rx, self.cvir, mv, eps, self.param)
        norm = 4.0 * np.pi * simpson(rx**2 * u, x=rx, axis=-1)
        rho0HGA = (self.Mtot.ravel() / norm)[:, None]
        rhoHGA = rho0HGA * uHGA * self.fhga

        # Mass to use in the pressure integral (M_he) and MDMB
        # use_nfw_mass=True  -> M_he = MNFW
        # use_nfw_mass=False -> M_he = MDMB (need baryon profiles)
        # skip_ac=True  -> MACM = MNFW (xx = 1), no correlation function
        # skip_ac=False -> run the AC model in param.code.AC_model

        if self.use_nfw_mass and not self.output:
            M_he = MNFW
        else:
            # Need baryon profiles to build MDMB (or for full_output)
            MHGA, MIGA, MCGA, MHGA_tck, MIGA_tck, MCGA_tck, multi = \
                self._ensure_splines(rbin, mv, rvir, eps)
 
            if self.skip_ac or self.use_nfw_mass:
                # skip_ac: xx = 1 so MACM = MNFW, no correlation function needed
                # use_nfw_mass + full_output: still skip AC for MACM
                MACM = MNFW
            else:
                # Full AC path
                AC = self.param.code.AC_model
                ACM_q0     = self.param.code.q0
                ACM_q1     = self.param.code.q1
                ACM_q2     = self.param.code.q2
                ACM_q0_exp = self.param.code.q0_exp
                ACM_q1_exp = self.param.code.q1_exp
                ACM_q2_exp = self.param.code.q2_exp
 
                # AC models 0-4 need the two-halo mass M2h
                if AC in (0, 1, 2, 3, 4):
                    r_for_xi = rr / self.h0 if self.little_h else rr
                    corr  = self._xi_ar(cosmo, ah, r_for_xi)[None, :]
                    rhom  = ccl.rho_x(cosmo, ah, 'matter',
                                      is_comoving=True) / self.h0**2
                    bias  = self.bM(cosmo, mccl, ah)[:, None]
                    excl  = self.param.code.halo_excl
                    rho2h = ((1 - np.exp(-excl * rbin / rvir))
                             * (corr * bias + 1.0) * rhom)
                    M2h   = cumulative_trapezoid(4.0 * np.pi * rbin**2 * rho2h,
                                                 rbin, initial=0.0)
                    if multi:
                        M2h_tck = RectBivariateSpline(mv[:, 0], rbin[0, :],
                                                       M2h, kx=1)
                    else:
                        M2h_tck = splrep(rbin[0, :], M2h[0, :], s=0, k=1)
 
                from scipy.optimize import fsolve
 
                if AC == 0:
                    nn = ACM_q0 * (1 + zz)**ACM_q0_exp
                    aa = ACM_q1 * (1 + zz)**ACM_q1_exp
                    if multi:
                        func = lambda x: (x - 1.0) - aa * (
                            ((MNFW + M2h) /
                             ((fcdm + fsga) * MNFW
                              + fcga * MCGA_tck(mv, x * rbin)
                              + self.fhga * MHGA_tck(mv, x * rbin)
                              + figa * MIGA_tck(mv, x * rbin)
                              + M2h_tck(mv, x * rbin)))**nn - 1.0)
                    else:
                        func = lambda x: (x - 1.0) - aa * (
                            ((MNFW + M2h) /
                             ((fcdm + fsga) * MNFW
                              + fcga * splev(x * rbin, MCGA_tck, der=0, ext=3)
                              + self.fhga * splev(x * rbin, MHGA_tck, der=0, ext=3)
                              + figa * splev(x * rbin, MIGA_tck, der=0, ext=3)
                              + splev(x * rbin, M2h_tck, der=0, ext=3)))**nn - 1.0)
                    xx = fsolve(func, np.ones_like(rbin))
 
                elif AC == 1:
                    Q0 = ACM_q0 * (1 + zz)**ACM_q0_exp
                    Q1 = ACM_q1 * (1 + zz)**ACM_q1_exp
                    if multi:
                        func = lambda x: (x - 1.0) - Q1 * (
                            (MNFW + M2h) /
                            ((fcdm + fsga) * MNFW
                             + fcga * MCGA_tck(mv, x * rbin)
                             + self.fhga * MHGA_tck(mv, x * rbin)
                             + figa * MIGA_tck(mv, x * rbin)
                             + M2h_tck(mv, x * rbin)) - 1.0) - Q0
                    else:
                        func = lambda x: (x - 1.0) - Q1 * (
                            (MNFW + M2h) /
                            ((fcdm + fsga) * MNFW
                             + fcga * splev(x * rbin, MCGA_tck, der=0)
                             + self.fhga * splev(x * rbin, MHGA_tck, der=0)
                             + figa * splev(x * rbin, MIGA_tck, der=0, ext=3)
                             + splev(x * rbin, M2h_tck, der=0, ext=3)) - 1.0) - Q0
                    xx = fsolve(func, np.ones_like(rbin))
 
                elif AC == 2:
                    Q0 = ACM_q0 * (1 + zz)**ACM_q0_exp
                    Q1 = ACM_q1 * (1 + zz)**ACM_q1_exp
                    nn = 1.5
                    rc = eps * rvir / eps0
                    fstep_fn = lambda rr: 1 - Q0 / (1 + (rr / rc)**nn)
                    r1    = rbin * fstep_fn(rbin)
                    MNFW1 = MNFWtr_fct(r1, self.cvir, tau, mv, self.param)
                    if multi:
                        func = lambda x: x - (
                            (MNFW1 + M2h_tck(mv, r1)) /
                            ((fcdm + fsga + Q1 * fcga + Q1 * figa) * MNFW1
                             + (1 - Q1) * (fcga * MCGA_tck(mv, x * r1)
                                           + figa * MIGA_tck(mv, x * r1))
                             + self.fhga * MHGA_tck(mv, x * r1)
                             + M2h_tck(mv, x * r1)))
                    else:
                        func = lambda x: x - (
                            (MNFW1 + splev(r1, M2h_tck, der=0, ext=0)) /
                            ((fcdm + fsga + Q1 * fcga + Q1 * figa) * MNFW1
                             + (1 - Q1) * (fcga * splev(x * r1, MCGA_tck, der=0, ext=0)
                                           + figa * splev(x * r1, MIGA_tck, der=0, ext=0))
                             + self.fhga * splev(x * r1, MHGA_tck, der=0, ext=0)
                             + splev(x * r1, M2h_tck, der=0, ext=0)))
                    x2 = fsolve(func, np.ones_like(rbin))
                    xx = x2 * fstep_fn(rbin)
 
                elif AC == 3:
                    Q0 = ACM_q0 * (1 + zz)**ACM_q0_exp
                    Q1 = ACM_q1 * (1 + zz)**ACM_q1_exp
                    Q2 = ACM_q2 * (1 + zz)**ACM_q2_exp
                    nn = 1.5
                    rc = eps * rvir / eps0
                    fstep_fn = lambda rr: 1 - Q0 / (1 + (rr / rc)**nn)
                    r1    = rbin * fstep_fn(rbin)
                    MNFW1 = MNFWtr_fct(r1, self.cvir, tau, mv, self.param)
                    if multi:
                        func = lambda x: (x - 1.0) - Q2 * (
                            (MNFW1 + M2h_tck(mv, r1)) /
                            ((fcdm + fsga + Q1 * fcga + Q1 * figa) * MNFW1
                             + (1 - Q1) * (fcga * MCGA_tck(mv, x * r1)
                                           + figa * MIGA_tck(mv, x * r1))
                             + self.fhga * MHGA_tck(mv, x * r1)
                             + M2h_tck(mv, x * r1)) - 1.0)
                    else:
                        func = lambda x: (x - 1.0) - Q2 * (
                            (MNFW1 + splev(r1, M2h_tck, der=0, ext=0)) /
                            ((fcdm + fsga + Q1 * fcga + Q1 * figa) * MNFW1
                             + (1 - Q1) * (fcga * splev(x * r1, MCGA_tck, der=0, ext=0)
                                           + figa * splev(x * r1, MIGA_tck, der=0, ext=0))
                             + self.fhga * splev(x * r1, MHGA_tck, der=0, ext=0)
                             + splev(x * r1, M2h_tck, der=0, ext=0)) - 1.0)
                    x2 = fsolve(func, np.ones_like(rbin))
                    xx = x2 * fstep_fn(rbin)
 
                elif AC == 4:
                    Q0 = ACM_q0 * (1 + zz)**ACM_q0_exp
                    Q1 = ACM_q1 * (1 + zz)**ACM_q1_exp
                    Q2 = ACM_q2 * (1 + zz)**ACM_q2_exp
                    nn = 1.5
                    rc = eps * rvir / eps0
                    fstep_fn = lambda rr: 1 - Q0 / (1 + (rr / rc)**nn)
                    r1    = rbin * fstep_fn(rbin)
                    MNFW1 = MNFWtr_fct(r1, self.cvir, tau, mv, self.param)
                    if multi:
                        func = lambda x: x - (
                            (MNFW1 + M2h_tck(mv, r1)) /
                            ((fcdm + fsga + (1 - Q1) * (fcga + figa)
                              + (1 - Q2) * self.fhga) * MNFW1
                             + Q1 * fcga * MCGA_tck(mv, x * r1)
                             + Q1 * figa * MIGA_tck(mv, x * r1)
                             + Q2 * self.fhga * MHGA_tck(mv, x * r1)
                             + M2h_tck(mv, x * r1)))
                    else:
                        func = lambda x: x - (
                            (MNFW1 + splev(r1, M2h_tck, der=0, ext=0)) /
                            ((fcdm + fsga + (1 - Q1) * (fcga + figa)
                              + (1 - Q2) * self.fhga) * MNFW1
                             + Q1 * fcga * splev(x * r1, MCGA_tck, der=0, ext=0)
                             + Q1 * figa * splev(x * r1, MIGA_tck, der=0, ext=0)
                             + Q2 * self.fhga * splev(x * r1, MHGA_tck, der=0, ext=0)
                             + splev(x * r1, M2h_tck, der=0, ext=0)))
                    x2 = fsolve(func, np.ones_like(rbin))
                    xx = x2 * fstep_fn(rbin)
 
                elif AC == 5:
                    Q0 = ACM_q0 * (1 + zz)**ACM_q0_exp
                    Q1 = ACM_q1 * (1 + zz)**ACM_q1_exp
                    Q2 = ACM_q2 * (1 + zz)**ACM_q2_exp
                    nn = 1.5
                    rc    = eps * rvir / eps0
                    fstep = 1 + Q0 / (1 + (rbin / rc)**nn)
                    MNFW_safe = np.where(np.abs(MNFW) > 0, MNFW, 1.0)
                    if multi:
                        ri_ov_rf = (fstep
                                    + Q1 * fcga * (MCGA_tck(mv, rbin) / MNFW_safe - 1)
                                    + Q1 * figa * (MIGA_tck(mv, rbin) / MNFW_safe - 1)
                                    + Q2 * self.fhga * (MHGA_tck(mv, rbin) / MNFW_safe - 1))
                    else:
                        ri_ov_rf = (fstep
                                    + Q1 * fcga * (splev(rbin, MCGA_tck, der=0, ext=0) / MNFW_safe - 1)
                                    + Q1 * figa * (splev(rbin, MIGA_tck, der=0, ext=0) / MNFW_safe - 1)
                                    + Q2 * self.fhga * (splev(rbin, MHGA_tck, der=0, ext=0) / MNFW_safe - 1))
                    ri_ov_rf = np.where(np.abs(ri_ov_rf) > 1e-10, ri_ov_rf, 1e-10)
                    xx = np.nan_to_num(1.0 / ri_ov_rf, nan=1.0)
 
                else:
                    raise ValueError(f"AC_model {AC} not recognised.")
 
                # No adiabatic expansion
                if not self.param.code.adiab_exp:
                    xx = np.where(xx > 1.0, 1.0, xx)
 
                MACM = MNFWtr_fct(rbin / xx, self.cvir, tau, mv, self.param)
 
            MDMB = (fcdm + fsga) * MACM + self.fhga * MHGA + fcga * MCGA + figa * MIGA
            M_he = MNFW if self.use_nfw_mass else MDMB
 
        # Thermal pressure via hydrostatic equilibrium
        pinf = cumulative_trapezoid(G * rhoHGA * M_he / rbin**2,
                                    rbin, initial=0.0, axis=-1)
        ptot = pinf[:, -1:] - pinf
 
        # Non-thermal pressure correction
        a_nth = self.param.baryon.a_nth
        b_nth = self.param.baryon.b_nth
        n_nth = self.param.baryon.n_nth
 
        if a_nth > 0:
            fmax   = 4**(-0.8) / a_nth
            f1     = (1 + zz)**b_nth
            f2     = (fmax - 1) * np.tanh(b_nth * zz) + 1
            a_nthz = a_nth * min(f1, f2)
        else:
            a_nthz = 0.0
 
        pnth = a_nthz * (rbin / rvir)**n_nth * ptot
        pth  = ptot - pnth
 
        # Unit conversion
        if self.units == 'cgs' or self.units == 'eV':
            pth = pth * _TO_CGS
            if self.units == 'eV':
                pth = pth * _TO_EV
 
        if not self.little_h:
            pth = pth * self.h0**2
 
        if not self.comoving:
            # dP_phys = -G * (rho_gas,com / a^3) * M_tot / (r_com * a)^2 * dr_com * a
            #        = -G * rho_gas,com * M_tot / r_com^2 * dr_com / a^4
            pth = pth / ah**4
 
        pth = pth * self.prefac
 
        if rdim == 0:
            pth = np.squeeze(pth, axis=-1)
        if mdim == 0:
            pth = np.squeeze(pth, axis=0)

        # Output
        if self.output:
            if not self.little_h:
                MDMB = MDMB / self.h0
                MHGA_out = self.fhga * MHGA / self.h0
                MACM_out = (fcdm + fsga) * MACM / self.h0
                MCGA_out = fcga * MCGA / self.h0
                MIGA_out = figa * MIGA / self.h0
            else:
                MHGA_out = self.fhga * MHGA
                MACM_out = (fcdm + fsga) * MACM
                MCGA_out = fcga * MCGA
                MIGA_out = figa * MIGA
            return pth, MDMB, MHGA_out, MACM_out, MCGA_out, MIGA_out
 
        return pth
 
    # Helper functions
    def get_normalization(self, cosmo, ah, hmc):
        return 1
 
    def get_Mtot(self):
        if not self.little_h:
            return np.squeeze(self.Mtot / self.h0)
        return np.squeeze(self.Mtot)
 
    def approx_2halo(self, cosmo, rbin, mvir, ah):
        """Two-halo pressure term (mean-temperature approximation)."""
        from astropy import constants as const
        from astropy import units as u
 
        if self.little_h:
            raise SystemExit('approx_2halo is not implemented for little_h=True')
 
        T_mean = 2_611_442.968
        mu     = 0.6125
        mp     = const.m_p.to(u.Msun).value
        kb     = (const.k_B.to(u.erg / u.K)
                  .to(u.Msun * u.km**2 / u.s**2 / u.K)).value
        Om     = cosmo['Omega_c'] + cosmo['Omega_b']
 
        corr   = self._xi_ar(cosmo, ah, rbin)
        bias   = self.bM(cosmo, np.atleast_1d(mvir), ah)
        rhoc   = ccl.rho_x(cosmo, ah, 'critical', is_comoving=self.comoving)
        p_crit = rhoc / mu / mp * kb * T_mean
        bg     = (1 + bias * corr) * self.fhga[0, 0] * Om * p_crit
 
        rvir_val = (3.0 * mvir / (4.0 * np.pi * 200.0 * rhoc))**(1.0 / 3.0)
        excl     = 1.0 - np.exp(-self.param.code.halo_excl * rbin / rvir_val)
        bg      *= excl
 
        if self.units == 'cgs' or self.units == 'eV':
            bg = bg * _TO_CGS
            if self.units == 'eV':
                bg = bg * _TO_EV
        return bg
 
    def _update_params(self, cosmos, rbin, mvir, ah, params):
        """Update BFC parameters. Baryon changes auto-invalidate spline cache."""
        if 'Mc'    in params: self.param.baryon.Mc     = 10**params['Mc']
        if 'mu'    in params: self.param.baryon.mu     = params['mu']
        if 'thej'  in params: self.param.baryon.thej   = params['thej']
        if 'thco'  in params: self.param.baryon.thco   = params['thco']
        if 'alpha' in params: self.param.baryon.alpha  = params['alpha']
        if 'beta'  in params: self.param.baryon.beta   = params['beta']
        if 'gamma' in params: self.param.baryon.gamma  = params['gamma']
        if 'delta' in params: self.param.baryon.delta  = params['delta']
        if 'eta'   in params: self.param.baryon.eta    = params['eta']
        if 'deta'  in params: self.param.baryon.deta   = params['deta']
        if 'eps0'  in params: self.param.code.eps0     = params['eps0']
        if 'eps1'  in params: self.param.code.eps1     = params['eps1']
        if 'ant'   in params: self.param.baryon.a_nth  = params['ant']
        if 'nnt'   in params: self.param.baryon.n_nth  = params['nnt']
        if 'Nstar' in params: self.param.baryon.Nstar  = params['Nstar']
        if 'ciga'  in params: self.param.baryon.ciga   = params['ciga']
        if 'cvir'  in params:
            self.cvir       = params['cvir']
            self.cvir_first = False
        return None
    