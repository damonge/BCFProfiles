import copy
from bfc_functions import rhoc_of_z, fSTAR_fct, cvir_fct_au, beta_fct, uHGA_fct, uIGA_fct, mTOTtr_fct, mNFWtr_fct
from params_bfc import par as default_par  # your params file
from scipy.interpolate import splrep, splev, interp1d, RectBivariateSpline, UnivariateSpline
from scipy.integrate import simpson, quad
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

""" Deltavir overdensity criterion """
DELTAVIR = 200.0 

""" Critical density at z=0 in [h^2 Msun/Mpc^3] """
RHOC = 2.776e11


class HaloProfileGasBFC(ccl.halos.HaloProfilePressure):

    def __init__(self, *, mass_def, param=None, 
                 little_h=False, cgs=False, IGA=False,
                 comoving=False, exclusion=True, model='new',
                 want_profile=True, bM = None, two_halo=False):
        '''
        Class to compute the Density profile of the BCM model.
        param: mass_def: Mass definition
        param: little_h: If true input in Msun and Mpc, else in Msun/h and Mpc/h
        param: cgs: If true output in g/cm^3, else in Msun/Mpc^3 (or Msun/h/Mpc/h^3)
        param: comoving: If true output in comoving units, else in physical units
        param: exclusion: If true include exclusion term for overlapping halos
        param: model: 'old' or 'new' model for the BCM profile
        param: want_profile: If true, the real density is given, if False the CCL HM
                            convention for normalisation is used
        param: bm: bias model 
        '''
        
        self.hmd = mass_def
        self.little_h = little_h
        self.cgs = cgs
        self.comoving = comoving
        self.exclusion = exclusion
        self.model = model
        self.IGA = IGA
        self.want_profile = want_profile
        self.param = copy.deepcopy(param if param is not None else default_par())
        self.bM = bM
        self.fhga = None
        self.two_halo = two_halo
        
        self._has_run_fft = False
        self._fft_cosmo_hash = None  # track when cosmo changes

        # Interpolator for dimensionless Fourier-space profile
        self._fourier_interp = None
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"
    
    def _fft(self, rbin):
        from scipy.interpolate import interp1d
        """
        Define pk2d and bias model for 2-halo term.
        """
        log10k_min=-6
        log10k_max=6
        nk_per_decade=60
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        k_arr = np.logspace(log10k_min, log10k_max, nk_total)
        ccl.spline_params.A_SPLINE_NA_PK = 50 #deafult is 50
        ccl.spline_params.A_SPLINE_NLOG_PK = 11 #default is 11, reduce time to compute
        a_arr = ccl.get_pk_spline_a()
        Pk_lin = ccl.linear_matter_power(self.cosmo, k_arr, a_arr)
        extrap_order_lok=1
        extrap_order_hik=2
        pk2d = ccl.Pk2D(a_arr=a_arr,
                            lk_arr=np.log(k_arr),
                            pk_arr=Pk_lin,
                            is_logp=False,
                            extrap_order_lok=extrap_order_lok,
                            extrap_order_hik=extrap_order_hik)
        corr_grid = np.zeros((len(a_arr), len(rbin)), dtype=np.float64)
        for i, a in enumerate(a_arr):
            corr_grid[i] = ccl.correlation_3d(self.cosmo, r=rbin, a=a, p_of_k_a=pk2d)

        self.corr_interp_funcs = [
            interp1d(a_arr, corr_grid[:, j], kind='linear', bounds_error=False, fill_value='extrapolate')
            for j in range(len(rbin))]
    
    def fast_corr3d(self, a_query):
        return np.array([f(a_query) for f in self.corr_interp_funcs])
    
    def _real(self, cosmos, rbin, mvir, ah=None, just_fhga=False):

        """
        params: cosmo: Cosmology
        params: rbin, Radius in Mpc
        params: mvir: Virial mass in Msun
        params: ah: Scale factor

        return: rhoHGA: Gas density profile in Msun / Mpc^3

        Comment 1: CCL is a library with no little_h dependence, here we modify the BCM model
        to exlude the little_h dependence in the output as well.

        Comment 2: Input mass and rbin is in Msun and Mpc but calculations are done in Msun/h and Mpc/h
        """
        
        h0 = cosmos['h']
        self.param.cosmo.h0 = h0
        self.param.cosmo.Om = cosmos['Omega_c'] + cosmos['Omega_b']
        self.param.cosmo.Ob = cosmos['Omega_b']
        self.param.cosmo.s8 = cosmos['sigma8']
        self.param.cosmo.ns = cosmos['n_s']
        
        self.fb = self.param.cosmo.Ob/self.param.cosmo.Om
        
        if ah is not None:
            #print('Using scale factor and not redshift from params')
            self.param.cosmo.z = 1/ah - 1
        else:
            ah = 1/(1+self.param.cosmo.z)

        if not self.little_h:
            #print('Values given are not little_h dependent')
            rbin = rbin*h0 #convert to Mpc/h
            mvir = mvir*h0 #convert to Msun/h

        if not self._has_run_fft and self.two_halo: 
            self.cosmo_params = cosmos
            self._fft(rbin) 
            self._has_run_fft = True

        #if not self.comoving:
        #    rbin = rbin/ah #convert to comoving units

        if np.max(rbin)<400 or np.min(rbin)>1e-2:
            ruse = rbin
            rbin = np.geomspace(1e-5, 400, 1000)

        else:
            ruse = rbin
            
        mvir = np.atleast_1d(mvir)[:,None]
        rvir = (3.0*mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(self.param)))**(1.0/3.0)
        rbin = np.atleast_1d(rbin)[None,:]                       
        
        eta   = self.param.baryon.eta
        deta  = self.param.baryon.deta

        #total fractions
        fstar = fSTAR_fct(mvir,self.param,eta)
        fcga  = fSTAR_fct(mvir,self.param,eta+deta) #Moster13
        fsga  = fstar-fcga #satellites and intracluster light

        if np.size(fsga)>1:
            fsga[np.where(fsga<0)] = 0
        elif fsga<0:
            fsga = 0

        figa  = self.param.baryon.ciga*fcga
        #self.fhga  = self.fb-fcga-fsga-figa
        self.fhga = self.fb-fstar-figa
        self.fhga[self.fhga<0.] = 0

        eps0 = self.param.code.eps0
        eps1 = self.param.code.eps1
        mccl = (mvir/h0)[:,0]
        sigma_M = ccl.sigmaM(cosmos, mccl, ah) #ccl needs mass in Msun not Msun/h
        peak_height = 1.686/sigma_M
        eps  = (eps0 - eps1*peak_height)

        # if eps < 1, set to 1
        if np.size(eps) == 1:
            if eps < 1:
                eps = 1.0
        elif np.size(eps) > 1:
            eps[np.where(eps < 1)] = 1.0

        if np.size(eps)>1:
            eps = eps[:,None]
            
        cvir = cvir_fct_au(mvir, self.param)
        tau  = eps*cvir
        
        #total dark-matter-only mass
        if not self.want_profile:
            self.Mtot = mvir#*(mTOTtr_fct(tau)/mNFWtr_fct(cvir,tau))
        else:
            self.Mtot = mvir*(mTOTtr_fct(tau)/mNFWtr_fct(cvir,tau))
        if just_fhga:
            return self.fhga

        uHGA =  uHGA_fct(rbin,cvir,mvir,eps,self.param)
        

        #FINAL IGA density and mass profile
        uIGA    = uIGA_fct(rbin,rvir)

        if np.size(mvir)>1:
            rho0HGA  = self.Mtot/(4.0*np.pi*simpson(rbin**2.0*uHGA,x=rbin, axis=1))[:,None]
            rho0IGA = self.Mtot/(4.0*np.pi*simpson(rbin**2.0*uIGA,x=rbin))[:,None]

            rhoHGA   = rho0HGA*uHGA
            rhoIGA   = rho0IGA*uIGA

            rhoHGA_fct = RectBivariateSpline(mvir[:,0], rbin[0,:], rhoHGA)
            rhoIGA_fct = RectBivariateSpline(mvir[:,0], rbin[0,:], rhoIGA)

            rhoHGA = rhoHGA_fct(mvir,ruse)
            rhoIGA = rhoIGA_fct(mvir,ruse)
       
        else:
            rho0HGA = self.Mtot/(4.0*np.pi*simpson(rbin**2.0*uHGA,x=rbin))
            rho0IGA = self.Mtot/(4.0*np.pi*simpson(rbin**2.0*uIGA,x=rbin))

            rhoHGA   = rho0HGA*uHGA
            rhoIGA   = rho0IGA*uIGA
            
            rhoHGA_fct = UnivariateSpline(np.log(rbin[0]),np.log(rhoHGA),k=1,s=0) 
            rhoHGA = np.exp(rhoHGA_fct(np.log(ruse)))
            rhoIGA = np.interp(ruse, rbin[0], rhoIGA[0])
            
        # 2-halo term, large scale approximation
        if self.two_halo:
            rhom = ccl.rho_x(cosmos, ah, 'matter', is_comoving=True)/h0**2
            exclusion = self.param.code.halo_excl
            corr = self.fast_corr3d(ah)[None,:]
            bias = self.bM(cosmos, np.atleast_1d(mvir)/h0, ah)[:,None]
            
            rho2h = (1-np.exp(-exclusion*rbin/rvir)) * (corr * bias + 1.0)*rhom #2-halo term in Msun/Mpc^3 * h0^2
                  
        if self.cgs:
            from astropy import units as u
            mconv  = u.Msun.to(u.g) #msun to g
            dconv  = u.Mpc.to(u.cm) #mpc to cm
            rhoHGA = rhoHGA * mconv / dconv**3 #in g/cm^3
            rhoIGA = rhoIGA * mconv / dconv**3 #in g/cm^3

        #in this step we transform back to Msun/Mpc^3
        if not self.little_h:
            rhoHGA = rhoHGA * h0**2
            rhoIGA = rhoIGA * h0**2    
            
        if not self.want_profile:
            norm = ccl.rho_x(cosmos, ah, 'matter', is_comoving=self.comoving)*self.fb
            rhoHGA /= norm
            
        if not self.comoving:
            am3 = 1/ah**3
            rhoHGA *= am3
            rhoIGA *= am3
            
        if np.size(mvir)>1:
            if self.IGA:
                return self.fhga * rhoHGA, figa * rhoIGA
            else:
                return self.fhga * rhoHGA #return gas density in single array
        
        else:
            if self.IGA:
                return self.fhga * rhoHGA, figa * rhoIGA
            else:
                return (self.fhga * rhoHGA.squeeze()) #return gas density in single array
      
            
    def approx_2halo(self, cosmos, rbin, mvir, ah):
        """
        Calculation of the two halo term using the approximation

        params: cosmo: Cosmology, ccl object
        params: rbin: Radius in Mpc
        params: mvir: Virial mass in Msun
        params: ah: Scale factor
        params: exclusion: Exclusion term for overlapping halos

        return: bg: Two halo term in Msun / Mpc^3
        """

        if self.little_h==True:
            raise SystemExit('ERROR: The two halo term is not implemented for little_h=True')
        
        if ah is not None:
            #print('Using scale factor and not redshift from params')
            self.param.cosmo.z = 1/ah - 1
        else:
            ah = 1/(1+self.param.cosmo.z)
            

        Om = cosmos['Omega_c']+cosmos['Omega_b'] #ccl.omega_x(cosmos, ah, 'matter') #

        rho_matter = ccl.rho_x(cosmos, ah, 'matter', is_comoving=self.comoving) #critical density no little h dependence
        rho_critical = ccl.rho_x(cosmos, ah, 'critical', is_comoving=self.comoving) #critical density no little h dependence

        #if self.corrb is None:
        bM = ccl.halos.HaloBiasSheth99(mass_def=ccl.halos.MassDefFof)
        b = bM(cosmos, mvir, ah)

        log10k_min=-6
        log10k_max=6
        nk_per_decade=60
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        k_arr = np.logspace(log10k_min, log10k_max, nk_total)
        ccl.spline_params.A_SPLINE_NA_PK = 20 #deafult is 50
        ccl.spline_params.A_SPLINE_NLOG_PK = 6 #default is 11, reduce time to compute
        a_arr = ccl.get_pk_spline_a()
        Pk_lin = ccl.linear_matter_power(cosmos, k_arr, a_arr)
        extrap_order_lok=1
        extrap_order_hik=2
        pk2d = ccl.Pk2D(a_arr=a_arr,
                            lk_arr=np.log(k_arr),
                            pk_arr=Pk_lin,
                            is_logp=False,
                            extrap_order_lok=extrap_order_lok,
                            extrap_order_hik=extrap_order_hik)
        corr = ccl.correlation_3d(cosmos, r=rbin, a=ah, p_of_k_a=pk2d)

        corr_b = corr*b

        #else:
        #    corr_b = self.corrb

        rvir = (3.0*mvir/(4.0*np.pi*200*rho_critical))**(1.0/3.0)
        exclscale = 0.5            

        if np.size(mvir)>1:
            excl = (1-np.exp(-exclscale*rbin/rvir[:,None]))
            b = b[:,None]
            bg = (1+corr_b)*rho_matter*self.fhga[0]
        
        else:
            excl = (1-np.exp(-exclscale*rbin/rvir))
            bg = (1+corr_b)*rho_matter*self.fhga

        if self.exclusion:
            bg *= excl

        if self.cgs:
            from astropy import units as u
            mconv  = u.Msun.to(u.g) #msun to g
            dconv  = u.Mpc.to(u.cm) #mpc to cm
            bg = bg * mconv / dconv**3 #in g/cm^3

        return bg

    def get_normalization(self, cosmos, ah, *, hmc):
        def rho_gas_integrand(mass):
            fhga = self._real(cosmos, 1.0 , mass, ah, just_fhga=True)
            Mhga = self.get_Mtot().squeeze()*fhga.squeeze()
            if not self.comoving:
                Mhga /= ah**3
            return Mhga

        return hmc.integrate_over_massfunc(rho_gas_integrand, cosmos, ah)
            
    # def get_normalization(self, cosmos, ah, *, hmc=None):
    #     """
    #     Normalization of the profile.
    #     """
    #     def fhga(mass):
    #         self._real(cosmos, ah, mass,1)
    #         return mass*self.fhga[:,0]*self.bM(cosmos,mass, ah)/self.fb

    #     def ones(mass):
    #         return mass*self.bM(cosmos,mass, 1.)
        
    #     result = hmc.integrate_over_massfunc(fhga,cosmos,ah)/hmc.integrate_over_massfunc(ones,cosmos,1.)
    #     return result
    
    #def get_normalization(self, cosmo=None, a=None, *, hmc=None):
    #    return 1.0
    
    def frac(self):
        return self.fhga
    
    def get_Mtot(self):
        """
        Total mass of the profile(s).
        """
        if not self.little_h:
            return np.squeeze(self.Mtot/self.param.cosmo.h0)
        else:
            return np.squeeze(self.Mtot)

    def _update_params(self, cosmos, rbin, mvir, ah, params):
        '''
        call this function to generate profiles with updated profile parameters
        '''
        
        cosmo_keys = {'Om', 'Ob', 's8'}  # params that invalidate the FFT
        if cosmo_keys & set(params.keys()):
            self._has_run_fft = False   # force FFT recomputation
            self._fft_cosmo_hash = None # reset FFT cosmology hash to ensure recomputation
        
        if 'Mc' in params:
            self.param.baryon.Mc = 10**params['Mc']
        if 'mu' in params:
            self.param.baryon.mu = params['mu']
        if 'thej' in params:
            self.param.baryon.thej = params['thej']
        if 'thco' in params:
            self.param.baryon.thco = params['thco']
        if 'alpha' in params:
            self.param.baryon.alpha = params['alpha']
        if 'beta' in params:
            self.param.baryon.beta = params['beta']
        if 'gamma' in params:
            self.param.baryon.gamma = params['gamma']
        if 'delta' in params:
            self.param.baryon.delta = params['delta']
        if 'eta' in params:
            self.param.baryon.eta = params['eta']
        if 'deta' in params:
            self.param.baryon.deta = params['deta']
        if 'eps0' in params:
            self.param.code.eps0 = params['eps0']
        if 'eps1' in params:
            self.param.code.eps1 = params['eps1']
        if 'ciga' in params:
            self.param.baryon.ciga = params['ciga']
        if 'Nstar' in params:
            self.param.baryon.Nstar = params['Nstar']
        
        # Rebuild CCL cosmology if needed
        if cosmo_keys & set(params.keys()):
            cosmos = ccl.Cosmology(
                Omega_c=self.param.cosmo.Om - self.param.cosmo.Ob,
                Omega_b=self.param.cosmo.Ob,
                h=self.param.cosmo.h0,
                n_s=self.param.cosmo.ns,
                sigma8=self.param.cosmo.s8,
                transfer_function='eisenstein_hu'
            )

        return self._real(cosmos, rbin, mvir, ah)