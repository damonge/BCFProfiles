"""
Changes vs the original
------------------------
*  AC model 5 only — all other AC branches removed.
*  full_output removed — _real() always returns pressure only.
*  Spline cache: MHGA / MIGA / MCGA splines are rebuilt only when the baryon-
   parameter hash changes.  A cosmo-only MCMC step (Om, s8) skips all
   cumulative_trapezoid and splrep work.
*  MNFWtr_fct called once per _real() and reused everywhere.
*  xi(a, r) / sigmaM / rho_x / bias served from CosmoGrid when attached;
   falls back to live CCL calls otherwise (useful for testing / validation).
*  Unit conversion factors computed once at import, not inside _real().
*  Two-halo term, pressure integration, and unit conversion are each isolated
   in a private helper so _real() reads as a clear linear sequence.
*  self.tau stored on the instance so downstream diagnostics can access it.
*  approx_2halo() kept but cleaned up.
*  _update_params() unchanged in interface; now also resets baryon hash so
   the spline cache is invalidated automatically on parameter changes.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Optional

import numpy as np
import pyccl as ccl
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import RectBivariateSpline, splrep, splev

from params_bfc import par as default_par
from bfc_functions import (
    fSTAR_fct,
    uHGA_fct, uIGA_fct, uCGA_fct,
    MNFWtr_fct, mNFWtr_fct, mTOTtr_fct,
    rhoc_of_z,
    DELTAVIR,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# Newton's constant  [Mpc / Msun * (km/s)^2]
G = 4.299e-9

# Unit-conversion factors (computed once at import)
_MCONV  = 1.989e33          # Msun -> g
_DCONV  = 3.085678e24       # Mpc  -> cm
_KM2CM  = 1e5               # km   -> cm
# factor: Msun/Mpc^3 * (km/s)^2  ->  erg/cm^3  (= g cm^-3 cm^2 s^-2)
_TO_CGS = _MCONV / _DCONV**3 * _KM2CM**2
# factor: erg/cm^3  ->  eV/cm^3
_TO_EV  = 6.242e11


# ---------------------------------------------------------------------------
# Prefactor: gas pressure -> number density type
# ---------------------------------------------------------------------------
def get_prefac_P(kind: str, XH: float = 0.76) -> float:
    """Prefactor transforming gas pressure into a chosen number density."""
    if kind == "n_total":
        return 1.0
    elif kind == "n_H":
        return 4 * XH / (5 * XH + 3)
    elif kind == "n_baryon":
        return (3 * XH + 1) / (5 * XH + 3)
    elif kind == "n_electron":
        return 2 * (XH + 1) / (5 * XH + 3)
    else:
        raise NotImplementedError(f"Pressure type {kind!r} not implemented")


# ---------------------------------------------------------------------------
# Cosmology key (for external caching if needed)
# ---------------------------------------------------------------------------
def _cosmo_key(cosmo) -> tuple:
    return (cosmo["Omega_b"], cosmo["Omega_c"], cosmo["h"],
            cosmo["sigma8"], cosmo["n_s"])


# ---------------------------------------------------------------------------
# Baryon-parameter hash (drives spline cache invalidation)
# ---------------------------------------------------------------------------
def _baryon_hash(param) -> str:
    p = param.baryon
    c = param.code
    key = (
        p.Mc, p.mu, p.thco, p.alpha, p.beta, p.gamma, p.delta,
        p.eta, p.deta, p.rcga, p.ciga,
        c.eps0, c.eps1,
        p.a_nth, p.n_nth, p.b_nth,
    )
    return hashlib.md5(str(key).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Concentration — Dutton & Macciò 2014
# ---------------------------------------------------------------------------
def _cvir_dutton14(mvir: np.ndarray, z: float) -> np.ndarray:
    """
    Halo concentration c_200c from Dutton & Macciò (2014).

    Parameters
    ----------
    mvir : virial mass [Msun/h], shape (n_m, 1) or (n_m,)
    z    : redshift

    Returns
    -------
    c    : concentration, same shape as mvir
    """
    A = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    B = -0.101 + 0.026 * z
    return 10.0**A * (mvir / 1.0e12)**B


# ===========================================================================
# Main class
# ===========================================================================
class HaloProfilePressureBFC(ccl.halos.HaloProfilePressure):
    """
    Pressure profile of the BFC (Baryonic Feedback Correction) model.

    Inherits from ccl.halos.HaloProfilePressure so it plugs directly into
    the CCL halo-model machinery.

    Parameters
    ----------
    mass_def      : CCL mass definition object
    bias_model    : callable  bias(cosmo, m [Msun], a) -> ndarray
    param         : BFC parameter object (default: params_bfc.par())
    little_h      : if True, I/O in Mpc/h and Msun/h; else Mpc and Msun
    units         : 'cgs'  -> erg/cm^3
                    'eV'   -> eV/cm^3
                    else   -> Msun/Mpc^3 * (km/s)^2
    comoving      : if True, return comoving pressure; else physical
    concentration : None   -> Dutton & Macciò (2014)
                    float  -> fixed value for all masses
                    callable(cosmo, m [Msun], a) -> ndarray
    kind          : which number density to return (see get_prefac_P)
    cosmo_grid    : CosmoGrid instance for fast CCL-replacement lookups;
                    if None, live CCL calls are used (slower but exact)
    """

    # Default correlation-function grid (comoving Mpc)
    _rmin_corr: float = 1e-4
    _rmax_corr: float = 1e3
    _nr_corr:   int   = 256
    _na_corr:   int   = 64

    def __init__(
        self,
        *,
        mass_def,
        bias_model,
        param=None,
        little_h:     bool  = False,
        units:        str   = "cgs",
        comoving:     bool  = False,
        concentration        = None,
        kind:         str   = "n_total",
        cosmo_grid           = None,
    ):
        self.hmd         = mass_def
        self.bM          = bias_model
        self.little_h    = little_h
        self.units       = units
        self.comoving    = comoving
        self.cvir_func   = concentration
        self.kind        = kind
        self.prefac      = get_prefac_P(kind)
        self.param       = copy.deepcopy(param if param is not None else default_par())
        self.cosmo_grid  = cosmo_grid

        # Runtime state set during _real() — exposed for diagnostics
        self.h0:    float        = 1.0
        self.tau:   np.ndarray   = None   # eps * cvir, set in _real()
        self.cvir:  np.ndarray   = None   # concentration, set in _real()
        self.Mtot:  np.ndarray   = None   # total (truncated NFW) mass
        self.fhga:  np.ndarray   = None   # hot-gas fraction

        # Spline cache — invalidated by baryon hash
        self._baryon_hash_val: Optional[str] = None
        self._spline_cache: dict = {}           # keys: 'MHGA','MIGA','MCGA' + '_spl'
        self._spline_rbin_hash: Optional[str] = None
        self._spline_mv_hash:   Optional[str] = None

        # Correlation-function cache (CCL fallback, not used when cosmo_grid set)
        self._corr_key   = None
        self._xi_spline  = None
        self._r_grid_corr = None
        self._a_grid_corr = None

        super().__init__(mass_def=mass_def)

    # -----------------------------------------------------------------------
    # CCL-fallback: correlation function table (only used without cosmo_grid)
    # -----------------------------------------------------------------------
    def _ensure_corr_table(self, cosmo, r_needed_max_com: float):
        key = (_cosmo_key(cosmo),
               self._rmin_corr, self._rmax_corr,
               self._nr_corr,   self._na_corr)
        need_build = (self._corr_key != key)
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
        pk2d  = ccl.Pk2D(
            a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=Pk,
            is_logp=False, extrap_order_lok=1, extrap_order_hik=2,
        )
        xi = np.empty((len(a_arr), self._nr_corr))
        for i, a in enumerate(a_arr):
            xi[i] = ccl.correlation_3d(cosmo, r=r_grid, a=a, p_of_k_a=pk2d)

        from scipy.interpolate import RectBivariateSpline as RBS
        self._xi_spline   = RBS(a_arr, np.log(r_grid), xi, kx=3, ky=3, s=0.0)
        self._r_grid_corr = r_grid
        self._a_grid_corr = a_arr
        self._corr_key    = key

    def _xi_ar(self, cosmo, a: float, r_com: np.ndarray) -> np.ndarray:
        """xi(a, r_comoving).  Uses CosmoGrid if available, else CCL."""
        if self.cosmo_grid is not None:
            Om = cosmo["Omega_c"] + cosmo["Omega_b"]
            s8 = cosmo["sigma8"]
            return self.cosmo_grid.xi(Om, s8, a, r_com)
        self._ensure_corr_table(cosmo, float(np.max(r_com)))
        a_c = float(np.clip(a, self._a_grid_corr[0], self._a_grid_corr[-1]))
        return self._xi_spline(a_c, np.log(r_com), grid=False)

    # -----------------------------------------------------------------------
    # Concentration
    # -----------------------------------------------------------------------
    def _get_cvir(self, cosmo, mv: np.ndarray, ah: float, zz: float) -> np.ndarray:
        """
        Return concentration c_vir, shape (n_m, 1).

        mv is expected in Msun/h with shape (n_m, 1).
        """
        if self.cvir_func is None:
            return _cvir_dutton14(mv, zz)
        elif callable(self.cvir_func):
            # CCL concentration functions take Msun (no little_h)
            return self.cvir_func(cosmo, mv / self.h0, ah)[:, None]
        else:
            # Fixed scalar
            return float(self.cvir_func) * np.ones_like(mv)

    # -----------------------------------------------------------------------
    # Spline cache helpers
    # -----------------------------------------------------------------------
    def _invalidate_spline_cache(self):
        self._baryon_hash_val   = None
        self._spline_cache      = {}
        self._spline_rbin_hash  = None
        self._spline_mv_hash    = None

    def _ensure_splines(
        self,
        rbin: np.ndarray,   # shape (n_m, n_r) or (n_r,)
        mv:   np.ndarray,   # shape (n_m, 1)
        rvir: np.ndarray,   # shape (n_m, 1)
        eps:  np.ndarray,   # shape (n_m, 1) or scalar
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (MHGA, MIGA, MCGA) cumulative-mass arrays, shape (n_m, n_r).

        Rebuilds only when the baryon-parameter hash, r-grid, or mass array
        have changed since the last call.  A pure cosmo step hits the cache.
        """
        bh = _baryon_hash(self.param)
        rh = hashlib.md5(rbin.ravel().tobytes()).hexdigest()
        mh = hashlib.md5(mv.ravel().tobytes()).hexdigest()

        if (bh == self._baryon_hash_val and
                rh == self._spline_rbin_hash and
                mh == self._spline_mv_hash):
            return (self._spline_cache["MHGA"],
                    self._spline_cache["MIGA"],
                    self._spline_cache["MCGA"])

        # --- recompute ---
        # rbin always has shape (1, n_r) coming from _real(), so we work on
        # the squeezed 1-D r array throughout and reshape to (n_m, n_r) at the end.
        n_m = mv.size
        r1d = rbin.ravel()                                       # (n_r,)
        mv2d = mv.ravel()                                        # (n_m,)  [Msun/h]

        # Profile shapes: functions broadcast (n_m,1) x (n_r,) -> (n_m, n_r)
        mv_col  = mv2d[:, None]                                  # (n_m, 1)
        rvir_1d = rvir.ravel()[:, None]                          # (n_m, 1)
        eps_1d  = eps.ravel()[:, None] if np.ndim(eps) > 0 else eps

        uHGA = uHGA_fct(r1d[None, :], self.cvir, mv_col, eps_1d, self.param)  # (n_m, n_r)
        uIGA = uIGA_fct(r1d[None, :], rvir_1d)                                # (n_m, n_r)
        uCGA = uCGA_fct(r1d[None, :], mv_col, self.param)                     # (n_m, n_r)

        # Normalisation: integrate over r for each mass row -> (n_m, 1)
        norm_HGA = (self.Mtot.ravel() /
                    (4.0 * np.pi * simpson(r1d**2 * uHGA, x=r1d, axis=1)))[:, None]
        norm_IGA = (self.Mtot.ravel() /
                    (4.0 * np.pi * simpson(r1d**2 * uIGA, x=r1d, axis=1)))[:, None]
        norm_CGA = (self.Mtot.ravel() /
                    (4.0 * np.pi * simpson(r1d**2 * uCGA, x=r1d, axis=1)))[:, None]

        # Cumulative mass profiles: always (n_m, n_r)
        MHGA = cumulative_trapezoid(
            4.0 * np.pi * r1d**2 * norm_HGA * uHGA, r1d, initial=0.0, axis=1)
        MIGA = cumulative_trapezoid(
            4.0 * np.pi * r1d**2 * norm_IGA * uIGA, r1d, initial=0.0, axis=1)
        MCGA = cumulative_trapezoid(
            4.0 * np.pi * r1d**2 * norm_CGA * uCGA, r1d, initial=0.0, axis=1)
        # MHGA, MIGA, MCGA are now cleanly (n_m, n_r) in all cases

        # Build splines — single-mass uses splrep (1-D), multi uses RectBivariateSpline
        multi = n_m > 1

        if multi:
            kx = min(3, mv.size - 1)
            MHGA_spl = RectBivariateSpline(mv2d, r1d, MHGA, kx=kx)
            MIGA_spl = RectBivariateSpline(mv2d, r1d, MIGA, kx=kx)
            MCGA_spl = RectBivariateSpline(mv2d, r1d, MCGA, kx=kx)
        else:
            MHGA_spl = splrep(r1d, MHGA[0], s=0, k=3)
            MIGA_spl = splrep(r1d, MIGA[0], s=0, k=3)
            MCGA_spl = splrep(r1d, MCGA[0], s=0, k=3)

        # Store
        self._spline_cache = dict(
            MHGA=MHGA, MIGA=MIGA, MCGA=MCGA,
            MHGA_spl=MHGA_spl, MIGA_spl=MIGA_spl, MCGA_spl=MCGA_spl,
            multi=multi,
        )
        self._baryon_hash_val  = bh
        self._spline_rbin_hash = rh
        self._spline_mv_hash   = mh

        return MHGA, MIGA, MCGA

    # -----------------------------------------------------------------------
    # AC model 5 — explicit formula, fully vectorised
    # -----------------------------------------------------------------------
    def _ac5_compute(
        self,
        rbin:  np.ndarray,
        mv:    np.ndarray,
        rvir:  np.ndarray,
        eps:   np.ndarray,
        zz:    float,
        ACM_q0: float, ACM_q0_exp: float,
        ACM_q1: float, ACM_q1_exp: float,
        ACM_q2: float, ACM_q2_exp: float,
        fcga:  float,
        figa:  float,
        MCGA:  np.ndarray,
        MIGA:  np.ndarray,
        MHGA:  np.ndarray,
        MNFW:  np.ndarray,
    ) -> np.ndarray:
        """
        Compute AC-5 displacement factor xx = r_i / r_f.

        Everything is vectorised over (n_m, n_r) simultaneously.
        MNFWtr_fct is passed in (already computed in _real) — not recomputed here.

        Returns
        -------
        xx : ndarray, shape (n_m, n_r)
        """
        Q0 = ACM_q0 * (1 + zz) ** ACM_q0_exp
        Q1 = ACM_q1 * (1 + zz) ** ACM_q1_exp
        Q2 = ACM_q2 * (1 + zz) ** ACM_q2_exp

        nn = 1.5
        rc = eps * rvir / self.param.code.eps0          # (n_m, 1)

        fstep = 1.0 + Q0 / (1.0 + (rbin / rc) ** nn)   # (n_m, n_r)

        # Guard MNFW against zeros once
        MNFW_safe = np.where(np.abs(MNFW) > 0.0, MNFW, 1.0)

        ri_ov_rf = (
            fstep
            + Q1 * fcga       * (MCGA / MNFW_safe - 1.0)
            + Q1 * figa       * (MIGA / MNFW_safe - 1.0)
            + Q2 * self.fhga  * (MHGA / MNFW_safe - 1.0)
        )

        # Guard against zero / negative before inversion
        ri_ov_rf = np.where(np.abs(ri_ov_rf) > 1e-10, ri_ov_rf, 1e-10)

        xx = np.nan_to_num(1.0 / ri_ov_rf, nan=1.0)
        return xx

    # -----------------------------------------------------------------------
    # Non-thermal pressure fraction
    # -----------------------------------------------------------------------
    @staticmethod
    def _nth_factor(a_nth: float, b_nth: float, n_nth: float,
                    zz: float, rbin: np.ndarray, rvir: np.ndarray
                    ) -> np.ndarray:
        """
        Thermal fraction f_th = 1 - f_nth.

        Returns pth / ptot, shape matching rbin.
        """
        if a_nth <= 0:
            return np.ones_like(rbin)
        fmax = 4.0**(-0.8) / a_nth
        f1   = (1.0 + zz)**b_nth
        f2   = (fmax - 1.0) * np.tanh(b_nth * zz) + 1.0
        f_z  = min(f1, f2)
        a_nthz = a_nth * f_z
        fnth = a_nthz * (rbin / rvir)**n_nth
        return 1.0 - fnth       # thermal fraction

    # -----------------------------------------------------------------------
    # Unit conversion
    # -----------------------------------------------------------------------
    def _apply_units(self, pth: np.ndarray) -> np.ndarray:
        """Convert pressure from internal [Msun/Mpc^3 * (km/s)^2] to output units."""
        if self.units in ("cgs", "eV"):
            pth = pth * _TO_CGS
            if self.units == "eV":
                pth = pth * _TO_EV
        return pth

    # -----------------------------------------------------------------------
    # Core profile computation
    # -----------------------------------------------------------------------
    def _real(self, cosmo, rr, mvir, ah):
        """
        Compute the thermal pressure profile.

        Parameters
        ----------
        cosmo : ccl.Cosmology
        rr    : radii [Mpc or Mpc/h depending on little_h], shape (n_r,)
        mvir  : virial mass [Msun or Msun/h], scalar or shape (n_m,)
        ah    : scale factor

        Returns
        -------
        pth : pressure, shape (n_m, n_r) or (n_r,) for single mass
        """
        # ------------------------------------------------------------------
        # 0.  Setup
        # ------------------------------------------------------------------
        Om = cosmo["Omega_c"] + cosmo["Omega_b"]
        Ob = cosmo["Omega_b"]
        self.h0 = cosmo["h"]
        fb = Ob / Om

        zz = 1.0 / ah - 1.0
        self.param.cosmo.z = zz

        # Convert to little-h units for internal calculations
        if not self.little_h:
            rr   = rr   * self.h0
            mvir = mvir * self.h0

        # Ensure shapes (n_m, 1) and (1, n_r) for broadcasting
        mv   = np.atleast_1d(mvir)[:, None]           # (n_m, 1)
        rbin = rr[None, :]                             # (1,   n_r)  -> broadcasts
        multi = mv.size > 1

        # ------------------------------------------------------------------
        # 1.  Halo structural quantities
        # ------------------------------------------------------------------
        rvir = (3.0 * mv /
                (4.0 * np.pi * DELTAVIR * rhoc_of_z(self.param)))**(1.0 / 3.0)

        self.cvir = self._get_cvir(cosmo, mv, ah, zz)  # (n_m, 1)

        eps0 = self.param.code.eps0
        eps1 = self.param.code.eps1

        # Peak height -> eps
        mccl = (mv / self.h0)[:, 0]                   # Msun, no h, shape (n_m,)
        if self.cosmo_grid is not None:
            s8   = cosmo["sigma8"]
            sigM = self.cosmo_grid.sigmaM(Om, s8, ah, mccl)
        else:
            sigM = ccl.sigmaM(cosmo, mccl, ah)

        nu  = 1.686 / sigM                             # peak height, (n_m,)
        eps = np.clip((eps0 - eps1 * nu)[:, None],
                      1.0, None)                       # (n_m, 1), floor at 1

        self.tau = eps * self.cvir                     # (n_m, 1)

        # ------------------------------------------------------------------
        # 2.  Baryon fractions
        # ------------------------------------------------------------------
        eta  = self.param.baryon.eta
        deta = self.param.baryon.deta

        fcdm  = (Om - Ob) / Om
        fstar = fSTAR_fct(mv, self.param, eta)
        fcga  = fSTAR_fct(mv, self.param, eta + deta)
        fsga  = np.clip(fstar - fcga, 0.0, None)
        figa  = self.param.baryon.ciga * fcga
        self.fhga = fb - fcga - fsga - figa            # (n_m, 1)

        # ------------------------------------------------------------------
        # 3.  Total (truncated-NFW) mass
        # ------------------------------------------------------------------
        self.Mtot = mv * mTOTtr_fct(self.tau) / mNFWtr_fct(self.cvir, self.tau)
        self.Mtot = np.where(self.Mtot < mv, mv, self.Mtot)

        # ------------------------------------------------------------------
        # 4.  NFW mass profile — computed once, reused in AC-5 and pressure
        # ------------------------------------------------------------------
        MNFW = MNFWtr_fct(rbin, self.cvir, self.tau, mv, self.param)  # (n_m, n_r)

        # ------------------------------------------------------------------
        # 5.  Two-halo term
        # ------------------------------------------------------------------
        r_for_xi = rr / self.h0 if self.little_h else rr   # Mpc (no h) for CCL
        corr = self._xi_ar(cosmo, ah, r_for_xi)[None, :]   # (1, n_r)

        if self.cosmo_grid is not None:
            bias = self.cosmo_grid.bias(Om, s8, ah, mccl)[:, None]
            rhom = self.cosmo_grid.rho_x(Om, s8, ah, "matter") / self.h0**2
        else:
            bias = self.bM(cosmo, mccl, ah)[:, None]
            rhom = ccl.rho_x(cosmo, ah, "matter", is_comoving=True) / self.h0**2

        excl_scale = self.param.code.halo_excl
        excl  = 1.0 - np.exp(-excl_scale * rbin / rvir)    # (n_m, n_r)
        rho2h = excl * (corr * bias + 1.0) * rhom           # (n_m, n_r)

        M2h = cumulative_trapezoid(
            4.0 * np.pi * rbin**2 * rho2h, rbin, initial=0.0, axis=-1)  # (n_m, n_r)

        # ------------------------------------------------------------------
        # 6.  Baryon mass profiles (cached by baryon hash)
        # ------------------------------------------------------------------
        MHGA, MIGA, MCGA = self._ensure_splines(rbin, mv, rvir, eps)
        # shapes are (n_m, n_r); scale by fractions
        MHGA_f = self.fhga * MHGA
        MIGA_f = figa      * MIGA
        MCGA_f = fcga      * MCGA

        # ------------------------------------------------------------------
        # 7.  AC model 5 — explicit, vectorised
        # ------------------------------------------------------------------
        xx = self._ac5_compute(
            rbin, mv, rvir, eps, zz,
            self.param.code.q0, self.param.code.q0_exp,
            self.param.code.q1, self.param.code.q1_exp,
            self.param.code.q2, self.param.code.q2_exp,
            fcga,
            figa,
            MCGA, MIGA, MHGA,
            MNFW,
        )
        

        # No adiabatic expansion
        if not self.param.code.adiab_exp:
            xx = np.where(xx > 1.0, 1.0, xx)

        # ------------------------------------------------------------------
        # 8.  Contracted dark-matter + total baryonic mass
        # ------------------------------------------------------------------
        MACM = MNFWtr_fct(rbin / xx, self.cvir, self.tau, mv, self.param)
        MDMB = (fcdm + fsga) * MACM + MHGA_f + MCGA_f + MIGA_f

        # ------------------------------------------------------------------
        # 9.  Thermal pressure via hydrostatic equilibrium
        # ------------------------------------------------------------------
        # dP/dr = -G * rho_gas * M_tot / r^2
        # Integrate inward from r_max: P(r) = integral_r^inf G rho_gas M / r'^2 dr'
        if multi:
            rho0HGA = self.Mtot/(4.0*np.pi*simpson(rbin**2.0*uHGA_fct(rbin, self.cvir, mv, eps, self.param),x=rbin, axis=1))[:,None]
            rhoHGA   = rho0HGA*uHGA_fct(rbin, self.cvir, mv, eps, self.param)
        
        else:
            rho0HGA = self.Mtot/(4.0*np.pi*simpson(rbin**2.0*uHGA_fct(rbin, self.cvir, mv, eps, self.param),x=rbin))
            rhoHGA   = rho0HGA*uHGA_fct(rbin, self.cvir, mv, eps, self.param)
       
        rhoHGA *= self.fhga

        pinf = cumulative_trapezoid(
            G * rhoHGA * MDMB / rbin**2,
            rbin, initial=0.0, axis=-1,
        )                                               # (n_m, n_r)

        ptot = pinf[:, -1:] - pinf                     # (n_m, n_r), zero at r_max

        # ------------------------------------------------------------------
        # 10.  Non-thermal pressure correction
        # ------------------------------------------------------------------
        fth = self._nth_factor(
            self.param.baryon.a_nth,
            self.param.baryon.b_nth,
            self.param.baryon.n_nth,
            zz, rbin, rvir,
        )
        pth = ptot * fth                                # (n_m, n_r)

        # ------------------------------------------------------------------
        # 11.  Unit conversion
        # ------------------------------------------------------------------
        pth = self._apply_units(pth)

        # little-h and comoving corrections
        if not self.little_h:
            pth = pth * self.h0**2
        if not self.comoving:
            pth = pth / ah**3

        # ------------------------------------------------------------------
        # 12.  Output — squeeze to (n_r,) for single mass
        # ------------------------------------------------------------------
        pth = pth * self.prefac
        return pth if multi else pth.squeeze()

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------
    def get_normalization(self, cosmo, ah, hmc):
        """CCL interface — BFC profiles are not normalised externally."""
        return 1

    def get_Mtot(self) -> np.ndarray:
        """Total (truncated-NFW) mass after the last _real() call [Msun or Msun/h]."""
        if self.Mtot is None:
            raise RuntimeError("Call _real() before get_Mtot()")
        return np.squeeze(self.Mtot / self.h0) if not self.little_h else np.squeeze(self.Mtot)

    # -----------------------------------------------------------------------
    # Parameter update (MCMC interface)
    # -----------------------------------------------------------------------
    def _update_params(self, cosmo, rbin, mvir, ah, params: dict):
        """
        Update BFC parameters and recompute the profile.

        Baryon-param changes automatically invalidate the spline cache via
        the hash checked in _ensure_splines() — no manual reset needed.

        Parameters
        ----------
        params : dict mapping parameter names to new values.
                 Supported keys:
                   Mc, mu, thej, thco, alpha, beta, gamma, delta,
                   eta, deta, eps0, eps1, ant, nnt, Nstar, ciga, cvir
        """
        _BARYON = {
            "Mc":    lambda v: setattr(self.param.baryon, "Mc",    10**v),
            "mu":    lambda v: setattr(self.param.baryon, "mu",    v),
            "thej":  lambda v: setattr(self.param.baryon, "thej",  v),
            "thco":  lambda v: setattr(self.param.baryon, "thco",  v),
            "alpha": lambda v: setattr(self.param.baryon, "alpha", v),
            "beta":  lambda v: setattr(self.param.baryon, "beta",  v),
            "gamma": lambda v: setattr(self.param.baryon, "gamma", v),
            "delta": lambda v: setattr(self.param.baryon, "delta", v),
            "eta":   lambda v: setattr(self.param.baryon, "eta",   v),
            "deta":  lambda v: setattr(self.param.baryon, "deta",  v),
            "eps0":  lambda v: setattr(self.param.code,   "eps0",  v),
            "eps1":  lambda v: setattr(self.param.code,   "eps1",  v),
            "ant":   lambda v: setattr(self.param.baryon, "a_nth", v),
            "nnt":   lambda v: setattr(self.param.baryon, "n_nth", v),
            "Nstar": lambda v: setattr(self.param.baryon, "Nstar", v),
            "ciga":  lambda v: setattr(self.param.baryon, "ciga",  v),
        }
        for k, v in params.items():
            if k in _BARYON:
                _BARYON[k](v)
            elif k == "cvir":
                self.cvir = np.atleast_1d(v)[:, None]

        #return self._real(cosmo, rbin, mvir, ah)

    # -----------------------------------------------------------------------
    # Approximate two-halo term (standalone utility)
    # -----------------------------------------------------------------------
    def approx_2halo(self, cosmo, rbin: np.ndarray,
                     mvir: float, ah: float) -> np.ndarray:
        """
        Two-halo pressure term using a mean-temperature approximation.

        Parameters
        ----------
        rbin : radii [Mpc]
        mvir : virial mass [Msun]
        ah   : scale factor

        Returns
        -------
        bg : two-halo pressure [same units as _real()]
        """
        from astropy import constants as const
        from astropy import units as u

        if self.little_h:
            raise SystemExit("approx_2halo is not implemented for little_h=True")

        T_mean = 2_611_442.968  # mean IGM temperature [K]
        mu_mol = 0.6125
        mp   = const.m_p.to(u.Msun).value
        kb   = (const.k_B.to(u.erg / u.K)
                .to(u.Msun * u.km**2 / u.s**2 / u.K)).value

        Om   = cosmo["Omega_c"] + cosmo["Omega_b"]
        corr = self._xi_ar(cosmo, ah, rbin)

        if self.cosmo_grid is not None:
            s8   = cosmo["sigma8"]
            bias = self.cosmo_grid.bias(Om, s8, ah, np.atleast_1d(mvir))
            rhoc = self.cosmo_grid.rho_x(Om, s8, ah, "critical")
        else:
            bias = self.bM(cosmo, np.atleast_1d(mvir), ah)
            rhoc = ccl.rho_x(cosmo, ah, "critical", is_comoving=self.comoving)

        p_crit = rhoc / mu_mol / mp * kb * T_mean
        bg     = (1.0 + bias * corr) * self.fhga[0, 0] * Om * p_crit

        rvir_val = (3.0 * mvir /
                    (4.0 * np.pi * 200.0 * rhoc))**(1.0 / 3.0)
        excl = 1.0 - np.exp(-self.param.code.halo_excl * rbin / rvir_val)
        bg  *= excl

        bg = self._apply_units(bg)
        return bg