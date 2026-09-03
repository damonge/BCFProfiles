import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator

import sys
import os
from pathlib import Path
repo_path = Path("/mnt/users/rogers/CCL").resolve()

import pyccl as ccl
sys.path.append(os.path.abspath("repo_shortcut"))
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))




class HMCalculatorHarriet(ccl.halos.HMCalculator):
    def integrate_2d(self, prof1, prof2, Fint, k, a, cosmo, interp_ranges):
        lM_min_interp, lM_max_interp, lk_min_interp, lk_max_interp = interp_ranges
        # Integral of [n(M1) * n(M2) * b(M1) * b(M2) * F(M1, M2)]
        self._get_ingredients(cosmo, a, get_bf=True)
        # Logarithmic mass grid
        lM = self._lmass  # [Nm]
        M = 10**lM  # [Nm]
        # Halo mass function
        nM = self._mf  # [Nm]
        # Halo bias
        bM = self._bf  # [Nm]
        # Profile values
        p1 = prof1.fourier(cosmo, k, M, a)  # [Nm, Nk]
        p2 = prof2.fourier(cosmo, k, M, a)  # [Nm, Nk]

        norm1 = prof1.get_normalization(cosmo, a, hmc=self)
        norm2 = prof2.get_normalization(cosmo, a, hmc=self)

        ones = np.ones([len(lM), len(lM), len(k)])

        lM_trunc = lM.copy()
        lM_trunc[lM_trunc > lM_max_interp] = lM_max_interp
        lM_trunc[lM_trunc < lM_min_interp] = lM_min_interp
        lk_trunc = np.log10(k)
        #lk_trunc[lk_trunc > lk_max_interp] = lk_max_interp  ###
        lk_trunc[lk_trunc < lk_min_interp] = lk_min_interp
        lM1 = (lM_trunc[:, None, None]*ones).flatten()
        lM2 = (lM_trunc[None, :, None]*ones).flatten()
        lk = (lk_trunc[None, None, :]*ones).flatten()
        pts = np.array([lM1, lM2, lk]).T


        bnl_interp = Fint(pts)
        F = bnl_interp
        F[pts[:, 2] <= lk_min_interp] = 1
        F = F.reshape([len(lM), len(lM), len(k)])


        integrand = ((nM*bM)[:, None]*p1)[:, None, :] * ((nM*bM)[:, None]*p2)[None, :, :] * F  # [Nm, Nm, nk]
        integral = simpson(simpson(integrand, lM, axis=1), lM, axis=0)  # [Nk]

        integrand = F[:, 0, :] * p2[0, :] * self._mbf0 * ((nM*bM)[:, None]*p1)  # [Nm, Nk]
        integral += simpson(integrand, lM, axis=0)

        integrand = F[0, :, :] * p1[0, :] * self._mbf0 * ((nM*bM)[:, None]*p2)  # [Nm, Nk]
        integral += simpson(integrand, lM, axis=0)

        integral += F[0, 0, :]*self._mbf0**2*p1[0, :]*p2[0, :]


        integral = integral / (norm1*norm2)
        return integral
        

