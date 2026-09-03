"""
Extract bPe from FLAMINGO power spectra using the system
estimator of Sara Maleubre's paper (S. Maleubre et al. 2025), Appendix A.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import pickle
import pyccl as ccl # Core Cosmological Library.  - does the same as classy from SHP.




class Flamingo:
    
    """
    Loader + <bPe> extractor for one FLAMINGO suite directory.
    """

    default_variants = [
        "fgas+2sigma",  # low feedback
        "L1_m9", # standard feedback (fiducial)
        "fgas-2sigma", # stronger feedback
        "fgas-4sigma",
        "fgas-8sigma", # strongest feedback
        "LS8", # low-S8 cosmology, standard feedback
        "LS8_fgas-8sigma", # low-S8 + strongest feedback
    ]

    colours = {
        "fgas+2sigma":     "tab:orange",
        "L1_m9":           "k",
        "fgas-2sigma":     "tab:blue",
        "fgas-4sigma":     "tab:purple",
        "fgas-8sigma":     "tab:red",
        "LS8":             "tab:green",
        "LS8_fgas-8sigma": "olive",
    }

    # same cosmology as A.Laposta et al 2026.
    cosmo_params = {
        "D3A": dict(h=0.681, Omega_b=0.0486, Omega_c=0.306-0.0486,
                    A_s=2.099e-9, n_s=0.967, m_nu=0.06),
        "LS8": dict(h=0.682, Omega_b=0.0473, Omega_c=0.305-0.0473,
                    A_s=1.836e-9, n_s=0.965, m_nu=0.06),
    }

    def __init__(self, base_path, variants=None, kmax=0.3, boxsize = 1000.0):
        
        """
        constructor that initialises each instance of the class.

        """
        
        self.base = Path(base_path)
        self.variants = variants if variants is not None else self.default_variants
        self.kmax = kmax
        self.boxsize = boxsize
        

    # 1) file access and code admin - read, unpack, list, sort redshift etc.
    def _file(self, variant, field, snap):
        
        """
        function that defines a path to one spectrum file - function is meant for internal use only, (private file access), my specific path.
        """

        return self.base/variant/"power_spectra"/f"power_{field}_{snap:04d}.txt"
        
        

    def load_pk(self, variant, field, snap):
        
        """
        function that returns z, k, p(k) for one file. field: 'matter' or 'matter-pressure'.
        """
        

        # unpack = False returns 2D array, (n_rows, n_cols) - True transposes the array, so (3,n_rows) and we can unpack cols as z,k,P
        z, k, P = np.loadtxt(self._file(variant, field, snap), unpack=True)
        return z[0], k, P # choose z[0], as for a given .txt file, they are all at the same redshift. The z column is full of the same elements.

    def list_snapshots(self, variant, field="matter-pressure"):
        
        """
        function that specifies, for a given variant, what snapshot numbers exits.
        
        This function turns files in a directory into a sorted list of integers you can loop over.

        • Default field is matter-pressure to list snapshots, but could pass matter instead - good to check if matter and matter-pressure do not have matching snapshots (for us - there is 122 for both).
        """
        
        files = (self.base / variant / "power_spectra").glob(f"power_{field}_*.txt") # .glob() on a Path object returns all the matching files in that directory. # * means any characters here.
        
        # f.stem - filename without directory/extension - e.g: power_matter-pressre_0077.
        # .split("_") - break on underscores, e.g: ['power', 'matter-pressure', '0077']
        # [-1] - last piece - 0077
        # int - 0077 to 77.


        # sort, as glob returns file in an arbitrary order - doesnt promise alphabetical or numerical order. 
        # matters because when we plot later - self.listsnapshots(variant)[-1] - to mean, lowest redshift
        # its in order of 0001 - 0122 going from hightes to lowest redshift.
        
        return sorted(int(f.stem.split("_")[-1]) for f in files)

    
    # 2) plots and build model and assumed cosmology.

    def plot_spectra(self, snap=None):

        """
        • Function to plot each power spectra of constant feedback + graph of P,mPe of varying feedback.
        """

        if snap is None:
            snap = self.list_snapshots("L1_m9")[-1]
 
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
 
        z, k, P_mm = self.load_pk("L1_m9", "matter", snap)
        _, _, P_mPe = self.load_pk("L1_m9", "matter-pressure", snap)
        _, _, P_pp = self.load_pk("L1_m9", "pressure", snap)
        axes[0].loglog(k, P_mm, "k-", label=r"$P_{mm}$")
        axes[0].loglog(k, np.abs(P_mPe), "r-", label=r"$|P_{m,P_e}|$")
        axes[0].loglog(k, np.abs(P_pp), "b-", label=r"$|P_{P_e P_e}|$")
        axes[0].axvline(self.kmax, color="gray", ls=":")
        axes[0].set_title(f"L1_m9, z = {z:.2f}")
        axes[0].set_xlabel(r"$k$ [Mpc$^{-1}$]")
        axes[0].legend(frameon=False)
 
        for v in self.variants:
            z, k, P = self.load_pk(v, "matter-pressure", snap)
            axes[1].loglog(k, np.abs(P), color=self.colours.get(v), label=v)
        axes[1].axvline(self.kmax, color="gray", ls=":")
        axes[1].set_title(f"$P_{{m,P_e}}$ by variant, z = {z:.2f}")
        axes[1].set_xlabel(r"$k$ [Mpc$^{-1}$]")
        axes[1].legend(frameon=False, fontsize=8)
 
        plt.tight_layout()
        return fig
 

    def get_cosmo(self, variant):

        """
        ** unpacks a dict into key word arguments: it does this: ccl.Cosmology(h=0.681, Omega_b=0.0486, Omega_c=0.257, A_s=2.099e-9, n_s=0.967, m_nu=0.06)
        • Caching = compute something expensive once, store it, reuse it. Here instead of redoing 122 times.
        """

        key = "LS8" if variant.startswith("LS8") else "D3A" # pick what cosmology this variant requires.
        if not hasattr(self, "cosmo_cache"): # If this instance doesn't yet have a _cosmo_cache attribute, create it as an empty dict. hasattr checks existence.
            self.cosmo_cache = {}
        if key not in self.cosmo_cache:
            self.cosmo_cache[key] = ccl.Cosmology(**self.cosmo_params[key]) # unpack the cosmology for that key.
        return self.cosmo_cache[key]



    def linear_template(self, variant, k, z):

        """
        • Linear matter power spectrum at redshift z, on the given k grid.
        • Adrien's 512-point grid and 128 scale factors were for building an interpolatable object inside the likelihood - we are evaluating at specific points.
        """

        
        cosmo = self.get_cosmo(variant)
        return ccl.power.linear_matter_power(cosmo, k, 1.0 / (1.0 + z))

    def nonlin_template(self, variant, k, z):
        """
        Non-linear (halofit) matter power spectrum, same interface as linear_template.
        """

        cosmo = self.get_cosmo(variant)
        return ccl.power.nonlin_matter_power(cosmo, k, 1.0/(1.0+z))



    def build_system(self, variant, snap, nmax=1, kmin=0.0, kmax=None, template = "linear"):

        """
        Full two-block system (Eq. A5 - 2 row Template matrix). Returns z, k, d, T, C.

        •This function implements saras appendix A. - d (Pmm, PmPe)
        """



        kmax = self.kmax if kmax is None else kmax
        z, k, P_mm  = self.load_pk(variant, "matter", snap)
        _, _, P_mPe = self.load_pk(variant, "matter-pressure", snap)
        _, _, P_pp  = self.load_pk(variant, "pressure", snap)

        msk = (k > kmin) & (k < kmax)
        k, P_mm, P_mPe, P_pp = k[msk], P_mm[msk], P_mPe[msk], P_pp[msk]
        n = len(k)

        # turn spectra into one long data vector to match theory vector format
        d = np.concatenate([P_mm, P_mPe])

        if template == "sim":
            Pt = P_mm
        elif template == "linear":
            Pt = self.linear_template(variant, k, z)
        elif template == "nonlin":
            Pt = self.nonlin_template(variant, k, z)
        

        one, zero = np.ones(n), np.zeros(n)
        top, bot = [], []  # rows for the mm block / the mU block

        # T must have shape (2n, 6): one row per data-vector element, one column per parameter.
        # top = [[1,1,1,1...], [0,0,0,0,0...]] an array containing two arrays - same for bottom but opposite.
        # after loop: top = [one, zero, Pt, zero, k²Pt, zero]
                    # bot = [zero, one, zero, Pt, zero, k²Pt] -defined as vectors as in appendix A of saras paper.
        top.append(one);  bot.append(zero)  # N_mm 
        top.append(zero); bot.append(one) # N_mU
        for nn in range(0, nmax + 1):  # A^0 (linear bias), then A^1...
            shape = k**(2*nn) * Pt
            top.append(shape); bot.append(zero) # mm
            top.append(zero);  bot.append(shape) # mU
        # block allows you to nest smaller arrays - gives a matrix of vectors for example.
        T = np.block([[np.column_stack(top)], [np.column_stack(bot)]])

        dk = k[1] - k[0]
        Nk = self.boxsize**3 * k**2 * dk / (2*np.pi**2)  
        var_mm = 2 * P_mm**2 / Nk   # Wick: w=x=y=z=m
        var_mU = (P_mm * P_pp + P_mPe**2) / Nk
        cov_x  = 2 * P_mm * P_mPe / Nk               

        C = np.zeros((2*n, 2*n))
        C[:n, :n]   = np.diag(var_mm)
        C[n:, n:]   = np.diag(var_mU)
        C[:n, n:]   = np.diag(cov_x)                     
        C[n:, :n]   = np.diag(cov_x)
        
        return z, k, d, T, C

    def solve_full(self, variant, snap, nmax=1, template="linear", **kw):

        """
        Analytic solve of the full system, then Eq. A7 for bg and <bPe>.

        • Ctheta = inv(T^T Cinv T) is the parameter covariance, so the errors on
          bg and bPe come from propagating it through Eq. A7 (both are non-linear
          functions of theta, hence the derivatives below rather than a plain
          sqrt of a diagonal element).
        """

        z, k, d, T, C = self.build_system(variant, snap, nmax=nmax, template=template, **kw)
        Cinv = np.linalg.inv(C)
        Ctheta = np.linalg.inv(T.T @ Cinv @ T)
        theta = Ctheta @ T.T @ Cinv @ d
        A0_mm, A0_mU = theta[2], theta[3] # to extra params relevant to bPe and bias.
        bg  = np.sqrt(A0_mm)                                
        bPe = A0_mU / np.sqrt(A0_mm) * 1e3 * (1 + z)**3       


        # f = y/sqrt(x) - how we break down into x and y variables.
        # var_bPe = (df/dx)^2 * var_x^2 + (df/dy)^2 * var_y^2 + 2 * df/dx * df/dy * cov(xy) - the error propagation
        var_x  = Ctheta[2, 2]
        var_y  = Ctheta[3, 3]
        cov_xy = Ctheta[2, 3]

        dbg_dx = 1.0 / (2.0 * np.sqrt(A0_mm))
        err_bg = np.sqrt(dbg_dx**2 * var_x)

        df_dx = -A0_mU / (2.0 * A0_mm**1.5)
        df_dy = 1.0 / np.sqrt(A0_mm)
        var_f = (df_dx**2 * var_x + df_dy**2 * var_y
                 + 2.0 * df_dx * df_dy * cov_xy)
        err_bPe = np.sqrt(np.abs(var_f)) * 1e3 * (1 + z)**3

        return z, bg, bPe, err_bg, err_bPe, theta, Ctheta

    def bPe_of_z_full(self, variant, every=1, nmax=1, **kw):

        """
        (z, bPe, bg, err_bPe) over all snapshots via the full A5/A7 system.
        """

        zs, bs, gs, es = [], [], [], []
        for snap in self.list_snapshots(variant)[::every]:
            z, bg, bPe, err_bg, err_bPe, theta, Ctheta = self.solve_full(
                variant, snap, nmax=nmax, **kw)
            zs.append(z); bs.append(bPe); gs.append(bg); es.append(err_bPe)
        order = np.argsort(zs)
        return (np.array(zs)[order], np.array(bs)[order],
                np.array(gs)[order], np.array(es)[order])

    def plot_bPe_z_templates(self, every=5, zmax=2.5, nmax=1):

        """
        bPe(z) for all variants: sim template vs CCL linear vs CCL nonlin template.

        • three templates -> three columns (was 2, which threw on the third).
        """

        templates = ["sim", "linear", "nonlin"]
        fig, axes = plt.subplots(2, len(templates), figsize=(18, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        for col, template in enumerate(templates):
            for v in self.variants:
                print(f"{template}: {v} ...")
                z, b, bg, e = self.bPe_of_z_full(v, every=every, nmax=nmax,
                                                 template=template)
                sel = z <= zmax
                ls = "--" if v.startswith("LS8") else "-"
                axes[0, col].errorbar(z[sel], b[sel], yerr=e[sel], ls=ls,
                                      color=self.COLORS.get(v), label=v,
                                      elinewidth=0.6, capsize=0)
                axes[1, col].plot(z[sel], bg[sel], ls,
                                  color=self.COLORS.get(v))

            axes[0, col].set_title(f"template = {template}")
            axes[1, col].axhline(1.0, color="grey", lw=0.8, ls=":")
            axes[1, col].set_xlabel(r"$z$")
            axes[1, col].set_ylabel(r"$b_g$")

        axes[0, 0].set_ylabel(r"$\langle bP_e \rangle$ [meV cm$^{-3}$]")
        axes[0, 0].legend(frameon=False, fontsize=8)
        plt.tight_layout()
        return fig






















# MCMC - module level, not methods. These take a Flamingo instance as an argument, so they must sit outside the class or `self` would swallow the first parameter.


# NOTE: commented out for now. If reinstating, the imports it needs are: import pocomc as pc, import corner, from scipy.stats import uniform
# and runMCMC/derived_from_samples need updating for solve_full's new return signature (z, bg, bPe, err_bg, err_bPe, theta, Ctheta).
"""

def get_loglike(theta, d, T, Cinv):
    
    • In a standard fit, you minimise chi^2 (the error).
    • In Bayesian statistics (MCMC), we maximise Likelihood (L).
    • For Gaussian errors, the relationship is L propto e^{-0.5 chi^2}.
    • Computers hate dealing with tiny exponential numbers, so we take the Logarithm.
    • Therefore, ln(L) = -0.5 x chi^2. This function tells the MCMC: "Higher values are better fits".


    # logp(d|theta) = (d-t)^T * Cinv * (d-t)
    # nothing to interpolate (like with BAO CLASS model): every model shape is known exactly at your k values.
    r = d - T @ theta # t = T*theta, r = d-t - difference between data and theoretical vector. - model is linear.
    return -0.5 * (r @ Cinv @ r) # rearrnaging


def runMCMC(flam, variant="L1_m9", snap=None, nmax=1,
            filename="bPe_samples.pickle", overwrite=False):
    
    Sample the full Appendix A parameter vector for one snapshot.

    theta = (N_mm, N_mU, A0_mm, A0_mU, A1_mm, A1_mU, ...)
    with b_g = sqrt(A0_mm) and <bU> = A0_mU / sqrt(A0_mm)   (Eq. A7)
   
    if snap is None:
        snap = flam.list_snapshots(variant)[-1]

    labels = ["N_mm", "N_mU"]

    # for each n - add amplitude terms.
    for n in range(0, nmax + 1):
        labels += [f"A{n}_mm", f"A{n}_mU"]

    if overwrite and os.path.exists(filename):
        os.remove(filename)
    if os.path.isfile(filename):
        with open(filename, "rb") as handle:
            samples = pickle.load(handle)
        return samples, labels

    # build the linear system for this snapshot
    z, k, d, T, C = flam.build_system(variant, snap, nmax=nmax)
    Cinv = np.linalg.inv(C)

    # analytic solution first: it sets sensible prior ranges and is
    # the answer the chain should reproduce
    Ctheta = np.linalg.inv(T.T @ Cinv @ T)
    theta_BF = Ctheta @ T.T @ Cinv @ d
    sig = np.sqrt(np.diag(Ctheta))
    print("analytic best fit:")
    for lab, val, s in zip(labels, theta_BF, sig):
        print(f"  {lab:8s} = {val: .5e} +/- {s:.2e}")

    #  priors: wide uniforms centred on the analytic solution.
    # A0_mm must stay positive since we take its square root in A7.
    prior_list = []
    for i, lab in enumerate(labels):
        lo, hi = theta_BF[i] - 20 * sig[i], theta_BF[i] + 20 * sig[i]
        if lab == "A0_mm": # to avoid getting NaN values as div0 error.
            lo = max(lo, 1e-6)
        prior_list.append(uniform(loc=lo, scale=hi - lo)) # wide uniform distribution
    prior = pc.Prior(prior_list) # set prior for pocoMC.

    if hasattr(prior, "n_dim"):
        prior.n_dim = int(prior.n_dim)

    # sampler: sends out virtual explorers into the parameter space,
    # using get_loglike to sniff out the peak of the posterior.
    sampler = pc.Sampler(
        prior=prior,
        likelihood=get_loglike,
        likelihood_args=(d, T, Cinv),
        random_state=0,
        n_effective=1024, 
        n_active=512, # 512 virtual explorers, to "look" through the parameter space.
    )
    sampler.run() # actual search phase

    # once the explorers find the peak they spend most of their time walking
    # around it - that cloud of points is the posterior.
    # width of cloud = error (sigma), centre of cloud = mean value
    samples, logl, logp = sampler.posterior(resample=True)

    with open(filename, "wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return samples, labels


def derived_from_samples(samples, labels, z, nmax=1):

 
    Apply Eq. A7 sample by sample, so the derived posteriors carry the full
    (non-Gaussian) uncertainty rather than propagating errors on the mean.
    Returns b_g and <bPe> in meV cm^-3, physical units.
    

    i_A0mm = labels.index("A0_mm") #could skip this and just do, A0_mm = samples[:, 2] or 3 for mU, but this way ensures its correct.
    i_A0mU = labels.index("A0_mU")

    A0_mm = samples[:, i_A0mm] # samples is shape (n_samples, 6), so samples[:, 2] pulls the Amm^0 value from every sample.
    A0_mU = samples[:, i_A0mU]


    # estimates from best fit params.
    bg = np.sqrt(A0_mm)
    bPe = A0_mU / np.sqrt(A0_mm) * 1e3 * (1 + z) ** 3
    return bg, bPe 


def plot_mcmc(samples, labels, z, nmax=1):


    Corner plot of the sampled parameters, plus the derived quantities.
    

    fig = corner.corner(samples, labels=labels, show_titles=True,
                        title_fmt=".2e", quantiles=[0.16, 0.5, 0.84])

    bg, bPe = derived_from_samples(samples, labels, z, nmax=nmax)
    derived = np.column_stack([bg, bPe])
    fig2 = corner.corner(derived,
                         labels=[r"$b_g$", r"$\langle bP_e\rangle$ [meV cm$^{-3}$]"],
                         show_titles=True, title_fmt=".3f",
                         quantiles=[0.16, 0.5, 0.84],
                         truths=[1.0, None]) # draws a reference line at 1 for bg and none for bPe.

    print(f"b_g   = {np.median(bg):.4f} "
          f"+{np.percentile(bg,84)-np.median(bg):.4f} "
          f"-{np.median(bg)-np.percentile(bg,16):.4f}")
    print(f"<bPe> = {np.median(bPe):.4f} "
          f"+{np.percentile(bPe,84)-np.median(bPe):.4f} "
          f"-{np.median(bPe)-np.percentile(bPe,16):.4f}  meV cm^-3")
    return fig, fig2
"""


if __name__ == "__main__":

    import os
    flam = Flamingo(os.environ.get('FLAM_DATA', '/path/to/FLAMINGO') + "/L1_m9")
    snap = flam.list_snapshots("L1_m9")[-1] # redshift 0.0, snapshot 122, lowest redshift.

    flam.plot_spectra()
    
    plt.savefig("flamingo_spectra.png", dpi=150)
    

    # sanity check on one snapshot, all three templates.
    # b_g should come out ~1 for the matter field by construction - if the
    # linear template pushes it away from 1 while sim keeps it there, the
    # template mismatch is being absorbed into the bias rather than the k^2 terms.
    print("\nz=0, L1_m9:")
    for t in ["sim", "linear", "nonlin"]:
        z, bg, bPe, err_bg, err_bPe, _ , _ = flam.solve_full("L1_m9", snap, template=t)
        print(f"  {t:7s}: b_g = {bg:.4f} +/- {err_bg:.4f}, "
              f"<bPe> = {bPe:.4f} +/- {err_bPe:.4f} meV cm^-3")




    z, k, P_mm = flam.load_pk("L1_m9", "matter", 122)
    Pt = flam.linear_template("L1_m9", k[k < 0.3], z)
    print(Pt.shape)

   
    fig = flam.plot_bPe_z_templates(every=5)
    fig.savefig("flamingo_bPe_templates.png", dpi=150)




    
    # flamingo Pmm, CCL linear, CCL non-linear - with a ratio panel underneath,
    # since the ratio is where a units or cosmology mismatch would show up.
    snap = flam.list_snapshots("L1_m9")[-1] # z = 0
    z, k, P_mm = flam.load_pk("L1_m9", "matter", snap)
    cosmo = flam.get_cosmo("L1_m9")
    a = 1.0/(1.0+z)

    msk = k < 0.3
    P_lin = ccl.power.linear_matter_power(cosmo, k[msk], a)
    P_nl  = ccl.power.nonlin_matter_power(cosmo, k[msk], a)

    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                        gridspec_kw={"height_ratios": [3, 1]})
    ax[0].loglog(k[msk], P_mm[msk], 'k-',  lw=2, label='FLAMINGO $P_{mm}$')
    ax[0].loglog(k[msk], P_lin,     'b--', label='CCL linear')
    ax[0].loglog(k[msk], P_nl,      'r-.', label='CCL non-linear (halofit)')
    ax[0].axvline(flam.kmax, color='grey', ls=':')
    ax[0].set_ylabel(r'$P(k)$ [Mpc$^3$]'); ax[0].legend()
    ax[0].set_title(f'L1_m9, z = {z:.2f}')

    ax[1].semilogx(k[msk], P_mm[msk]/P_lin, 'b--', label='sim / linear')
    ax[1].semilogx(k[msk], P_mm[msk]/P_nl,  'r-.', label='sim / non-linear')
    ax[1].axhline(1, color='k', lw=0.8)
    ax[1].axvline(flam.kmax, color='grey', ls=':')
    ax[1].set_ylim(0.5, 1.5); ax[1].set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax[1].set_ylabel('ratio'); ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("flamingo_template_comparison.png", dpi=150)



    # bPe values for all three templates
    # skip the z>2 snapshots before solving, so we don't do the work then bin it.
    print(f"\n{'z':>6} {'sim':>9} {'linear':>9} {'nonlin':>9} "
          f"{'lin/sim':>9} {'nl/sim':>9} {'bg(sim)':>9}")
    for snap in flam.list_snapshots("L1_m9")[::4]:
        z_here, _, _ = flam.load_pk("L1_m9", "matter", snap) # cheap z lookup
        if z_here > 2.0:
            continue
        row, errs, bgs = {}, {}, {}
        for t in ["sim", "linear", "nonlin"]:
            zz, bg, bPe, err_bg, err_bPe, _, _ = flam.solve_full("L1_m9", snap, template=t)
            row[t] = bPe; errs[t] = err_bPe; bgs[t] = bg
        print(f"{zz:6.2f} {row['sim']:9.4f} {row['linear']:9.4f} {row['nonlin']:9.4f} "
              f"{row['linear']/row['sim']:9.3f} {row['nonlin']/row['sim']:9.3f} "
              f"{bgs['sim']:9.4f}")



    # same thing as a plot, with error bars from the propagated covariance,
    # against Sara's published polynomial fit.
    plt.figure(figsize=(7.5,5))
    styles = {"sim": ('k', '-', 2), "linear": ('b', '--', 1.5), "nonlin": ('r', '-.', 1.5)}
    curves = {}
    for t in ["sim", "linear", "nonlin"]:
        z_t, b_t, bg_t, e_t = flam.bPe_of_z_full("L1_m9", every=4, template=t)
        curves[t] = (z_t, b_t, e_t)
        sel = z_t <= 2.0
        c, ls, lw = styles[t]
        plt.errorbar(z_t[sel], b_t[sel], yerr=e_t[sel], color=c, ls=ls, lw=lw,
                     label=f'template = {t}', elinewidth=0.7, capsize=0)

    zz = np.linspace(0, 2, 100)
    plt.plot(zz, np.polyval([-0.06415216, 0.0334674, 0.34947284, 0.13396767], zz),
             color='grey', lw=1, label='Sara poly fit')
    plt.xlabel('z'); plt.ylabel(r'$\langle bP_e\rangle$ [meV cm$^{-3}$]')
    plt.legend(); plt.title(f'L1_m9, template comparison, kmax={flam.kmax}')
    plt.tight_layout()
    plt.savefig("flamingo_bPe_by_template.png", dpi=150)

    print("\ndone.")
    






    """
    samples, labels = runMCMC(flam, "L1_m9", snap=snap, overwrite=True)
    plot_mcmc(samples, labels, z)

    plt.savefig("flamingo_mcmc_corner.png", dpi=150)
    """