import numpy as np
from scipy.special import erf
from scipy.integrate import simpson as simps
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import fsolve
from scipy.interpolate import splrep,splev

""" Deltavir overdensity criterion """
DELTAVIR = 200.0 

""" Critical density at z=0 in [h^2 Msun/Mpc^3] """
RHOC = 2.776e11

""" Critical density in [Msun/h/(Mpc/h)^3] """
def rhoc_of_z(param):
    """
    Redshift dependence of critical density
    (in comoving units where rho_b=const; same as in AHF)
    """
    Om = param.cosmo.Om
    z  = param.cosmo.z
    
    return RHOC*(Om*(1.0+z)**3.0 + (1.0-Om))/(1.0+z)**3.0

def cvir_fct_au(mvir,param):
    """
    Concentrations form Dutton+Maccio (2014)
    c200 (200 times RHOC)
    Assumes PLANCK cosmology
    """
    z = param.cosmo.z
    A = 0.520 + (0.905 - 0.520)*np.exp(-0.617*z**1.21) #0.905 #1.025
    B = -0.101 + 0.026*z #-0.101 #0.097
    return 10.0**A*(mvir/1.0e12)**(B)

"""
GENERAL FUNCTIONS REALTED TO THE NFW PROFILE
"""

def r500_fct(r200,c):
    """
    From r200 to r500 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200


def rvir_fct(r200,c):
    """
    From r500 to r200 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 96.0/200.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200


def M500_fct(M200,c):
    """
    From M200 to M500 assuming a NFW profiles
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 5.0/2.0*M200*y0**3.0


"""
STELLAR FRACTIONS
"""

def fSTAR_fct(Mvir, param, eta=0.3):
    """
    Total stellar fraction (central and satellite galaxies).
    Free model parameter eta.
    (Function inspired by Moster+2013, Eq.2)
    """
    NN = param.baryon.Nstar
    M1 = param.baryon.Mstar
    zeta = 1.376
    return NN*((Mvir/M1)**(-zeta)+(Mvir/M1)**(eta))**(-1.0)


"""
Generalised (truncated) NFW profiles
"""

def uNFWtr_fct(rbin,cvir,t,Mvir,param):
    """
    Truncated NFW density profile. Normalised.
    """

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    x = cvir*rbin/rvir
    return 1.0/(x * (1.0+x)**2.0 * (1.0+x**2.0/t**2.0)**2.0)


def rhoNFW_fct(rbin,cvir,Mvir,param):
    """
    NFW density profile.
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rho0 = DELTAVIR*rhoc_of_z(param)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
    x = cvir*rbin/rvir
    return rho0/(x * (1.0+x)**2.0)


def mNFWtr_fct(x,t):
    """
    Truncated NFW mass profile. Normalised.
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = x/((1.0+x)*(t**2.0+x**2.0))*(x-2.0*t**6.0+t**4.0*x*(1.0-3.0*x)+x**2.0+2.0*t**2.0*(1.0+x-x**2.0))
    second = t*((6.0*t**2.0-2.0)*np.arctan(x/t)+t*(t**2.0-3.0)*np.log(t**2.0*(1.0+x)**2.0/(t**2.0+x**2.0)))
    return pref*(first+second)


def mNFW_fct(x):
    """
    NFW mass profile. Normalised.
    """
    return (np.log(1.0+x)-x/(1.0+x))


def mTOTtr_fct(t):
    """
    Normalised total mass (from truncated NFW)
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = (3.0*t**2.0-1.0)*(np.pi*t-t**2.0-1.0)
    second = 2.0*t**2.0*(t**2.0-3.0)*np.log(t)
    return pref*(first+second)


def MNFWtr_fct(rbin,cvir,t,Mvir,param):
    """
    Truncateed NFW mass profile.
    """
    
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    return Mvir*mNFWtr_fct(cvir*rbin/rvir,t)/mNFWtr_fct(cvir,t)


def MNFW_fct(rbin,cvir,Mvir,param):
    """
    NFW mass profile
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    x = cvir*rbin/rvir
    return (np.log(1.0+x) - x/(1.0+x))/(np.log(1.0+cvir)-cvir/(1.0+cvir))*Mvir


"""
GAS PROFILE
"""

def beta_fct(Mvir,param):
    """
    Parametrises slope of gas profile
    Two models (0), (1)
    """
    z  = param.cosmo.z
    Mc = param.baryon.Mc
    mu = param.baryon.mu
    nu = param.baryon.nu
    Mc_of_z = Mc*(1+z)**nu
    
    if (param.code.beta_model==0):

        dslope = 3.0
        beta = dslope - (Mc_of_z/Mvir)**mu
        if (beta<-10.0):
            beta = -10.0

    elif (param.code.beta_model==1):

        dslope = 3.0
        beta = dslope*(Mvir/Mc)**mu/(1+(Mvir/Mc)**mu)

    elif (param.code.beta_model==2):
        beta = param.baryon.beta
        

    else:
        print('ERROR: beta model not defined!')
        exit()
        
    return beta


def uHGA_fct(rbin,cvir,Mvir,eps,param):
    """
    Normalised gas density profile
    """
    thco = param.baryon.thco
    al   = param.baryon.alpha
    be   = beta_fct(Mvir,param)
    ga   = param.baryon.gamma
    de   = param.baryon.delta

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rsc  = rvir/cvir
    rco  = thco*rvir
    rej  = eps*rvir

    w = rbin/rco
    x = rbin/rsc
    y = rbin/rej

    return 1.0/(1.0+w**al)**(be/(al))/(1.0+y**ga)**(de/ga)
    
def uIGA_fct(rbin,rvir):
    """
    Inner (cold) density profile. Truncated at the virial radius
    """
    rco  = rvir
    w    = rbin/rco
    uIGA = np.exp(-w) / rbin ** 3.0
    rmin = 0.005 #Mpc/h
    uIGA = np.where(rbin < rmin, 1.0, uIGA)
    return uIGA


def uBAR_noF_fct(rbin,cvir,Mvir,eps,param):
    """
    Normalised gas+stellar density profile assuming 
    no feeedback (just cooling)
    """
    thco = 0.1 
    al   = 1.0
    be   = 3.0
    ga   = param.baryon.gamma
    de   = param.baryon.delta

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rsc  = rvir/cvir
    rco  = thco*rvir
    rej  = eps*rvir

    w = rbin/rco
    x = rbin/rsc
    y = rbin/rej

    return 1.0/(1.0+w**al)**(be/al)/(1.0+y**ga)**(de/ga)


"""
STELLAR PROFILE
"""

def uCGA_fct(rbin,Mvir,param):
    """
    Normalised density profile of central galaxy
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    R12 = param.baryon.rcga*rvir
    return np.exp(-rbin/R12)/rbin**2.0

def MCGA_fct(rbin,Mvir,param):
    """
    Normalised mass profile of central galaxy
    (needs to be multiplied with fcga*Mtot)
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    R12 = param.baryon.rcga*rvir
    return 1 - np.exp(-rbin/R12)