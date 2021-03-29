from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Black formulas
# =============================================================================

def iv_black(f, k, ttm, p):
    """The Black iv formula using bisection

    :param f: forward rate
    :param k: strike
    :param ttm: time to maturity
    :param p: price
    """    
    max_iter = 50
    sigma_lo = 0.000001
    sigma_hi = 1.0
    sigma = 0.5*(sigma_lo + sigma_hi)
    model_price = call_black(f,k, ttm, sigma)
    i = 0
    while (np.fabs(model_price - p)/p > 0.0001) and i < max_iter:
        if (model_price > p):
            sigma_hi = sigma
        else:
            sigma_lo = sigma
        sigma = 0.5*(sigma_lo + sigma_hi)
        model_price = call_black(f, k, ttm, sigma) 
        i = i + 1
    
    return sigma

def call_black(f, k, ttm, sigma):
    """The Black call formula.

    :param f: forward rate
    :param k: strike
    :param ttm: time to maturity
    :param sigma: volatility
    """
    V = sigma**2 * ttm
    if isinstance(V, np.ndarray):
        if V.any() <= 0.0:
            raise RuntimeError("vol must be non-negative")
    else:
        if V <= 0.0:
            raise RuntimeError("vol must be non-negative")

    d_p = (np.log(f/k) + 0.5*V)/np.sqrt(V)
    d_m = d_p - np.sqrt(V)

    return f*norm.cdf(d_p) - k*norm.cdf(d_m)

# =============================================================================
# Hagen Model
# =============================================================================
def vol(K,beta = 0.5, alpha = 0.085, vega = 0.44 , rho = -0.46, F = 0.04, t = 0, T = 10):
    """ The Hagen model
    
    :param K: strike price
    :param F: foward price
    :param b: beta
    :param sigma0: volatility
    :param z0: zt from zabr model
    :param epsilon: epsilon from zabr model
    :param rho: rho from zabr model
    :param t: initial time
    :param T: maturity time
    """
        
        sigma = alpha
        sigma /= (F*K)**((1-beta)/2)
        
        if K==F:
            sigma = (alpha/ F**(1-beta)) * l_term(beta, alpha , vega, rho, F, K, t, T)
            return sigma
        
        sigma /= 1 + (1-beta)**2 /24 * np.log(F/K)**2 + (1-beta)**4 /1920 * np.log(F/K)**4
        
        z = z_sabr(K, beta, alpha, vega, F)
        x = x_sabr(rho, z)
        
        sigma *= z/x
        
        sigma *= l_term(beta, alpha , vega, rho, F, K, t, T)
        
        return sigma
        
def z_sabr(K, beta = 0.5, alpha = 0.085, vega = 0.44, F= 0.04):
    return (vega/alpha) * (F*K)**((1-beta)/2) * np.log(F/K)

def x_sabr(rho, z):
    return np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho))
    
def l_term(beta, alpha , vega, rho, F, K, t, T):
    ret = 1 + (((((1-beta)**2) / 24) * (alpha **2) / ((F*K)**(1-beta)) \
            + 1/4 * rho*beta*vega*alpha / ((F*K)**((1-beta)/2)) \
            + (2-3*rho**2) *vega**2 / 24 )* (T-t))
    return ret


# =============================================================================
# SABR
# =============================================================================

def jy(epsilon, y, rho):
    return (1 + epsilon**2 * y**2 - 2*rho*epsilon*y)**0.5

def x_t(K, y, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46):
    if y != 0:
        return 1/epsilon * np.log((jy(epsilon, y, rho) - rho + epsilon * y)/(1-rho))
    else:
        return dy_t(beta, sigma0, F, K, z0)

def y_t(beta, sigma0, F, K, z0):
    if beta != 1:
        return 1/(sigma0*z0) * (F**(1-beta)-K**(1-beta))/(1-beta)
    else:
        return 1/(sigma0*z0) * np.log(F/K)

def dy_t(beta, sigma0, F, K, z0):
    if beta !=1:
        return 1/(sigma0*z0) * (-K**(-beta))
    else:
        return 1/(sigma0*z0) * (-F/(K**2))

def log_normal(K, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46, t=0, T=10):
    """The SABR price
    """
    y = y_t(beta, sigma0, F, K, z0)
    if K==F:
        return (-F/(K**2))/x_t(K, y, F=F, beta=beta, sigma0=sigma0, z0 = z0, epsilon = epsilon, rho=rho)
    return np.log(F/K)/x_t(K, y, F=F, beta=beta, sigma0=sigma0, z0 = z0, epsilon = epsilon, rho=rho)

# =============================================================================
# Copy
# =============================================================================

def x_t_e(K, y, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46):
    return 1/epsilon * np.log((jy(epsilon, y, rho) - rho + epsilon * y)/(1-rho))
    

def y_t_e(beta, sigma0, F, K, z0):
    if beta != 1:
        return 1/(sigma0*z0) * (F**(1-beta)-K**(1-beta))/(1-beta)
    else:
        return 1/(sigma0*z0) * np.log(F/K)

def log_normal_e(K, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46, t=0, T=10):
    y = y_t_e(beta, sigma0, F, K, z0)
#    if K==F:
#        return (-F/(K**2))/x_t(K, y, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46)
    return np.log(F/K)/x_t_e(K, y, F=F, beta=beta, sigma0=sigma0, z0 = z0, epsilon = epsilon, rho=rho)


# =============================================================================
# ZABR
# =============================================================================

def Zabr(K, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho = -0.46, t = 0, T = 10):
    """The Zabr implied volatility via back formula.
    
    :param K: strike price
    :param F: foward price
    :param b: beta
    :param sigma0: volatility
    :param z0: zt from zabr model
    :param epsilon: epsilon from zabr model
    :param rho: rho from zabr model
    :param t: initial time
    :param T: maturity time
    """
    return iv_black(F, K, T,
                    Zabr_price(K, F, beta, sigma0, z0, epsilon, rho, t, T))
#   

# =============================================================================
# Price
# =============================================================================

def Zabr_price(K, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho = -0.46, t = 0, T = 10):
    y = y_t(beta, sigma0, F, K, z0)
    """The Zabr price
    
    :param K: strike price
    :param F: foward price
    :param b: beta
    :param sigma0: volatility
    :param z0: zt from zabr model
    :param epsilon: epsilon from zabr model
    :param rho: rho from zabr model
    :param t: initial time
    :param T: maturity time
    """
    y = y_t(beta, sigma0, F, K, z0)
    phi_sqrt = (T-t)**0.5
    vega = (F-K)/x_t(K, y, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46)
    if F!=K:
       par_norm = (F-K)/(vega*phi_sqrt)
       return (F-K) * norm.cdf(par_norm) + vega * phi_sqrt * norm.pdf(par_norm)
       
#    else:
    vega = (-1)/x_t(K, y, F=0.04, beta=0.5, sigma0=0.085, z0 = 1, epsilon = 0.44, rho=-0.46)
    par_norm = 0/(vega*phi_sqrt)
    return (F-K) * norm.cdf(par_norm) + vega * phi_sqrt * norm.pdf(par_norm) 

    

# =============================================================================
# Density
# =============================================================================

def dens(func,K, *args,h=0.0001):
    term1 = func(K+h)
    term2 = func(K)
    term3 = func(K-h)
    return (term1 - 2 * term2 + term3)/(h**2)

def dens2(func,K, *args, h =0.0001):
    term1 = call_black(0.04, K+h, 10,func(K+h))
    term2 = call_black(0.04, K, 10,func(K))
    term3 = call_black(0.04, K-h, 10,func(K-h))
    return (term1 - 2 * term2 + term3)/(h**2)



def dif(func,K, beta, sigma0, *args,h=0.0001):
    term1 = func(K=K, beta = beta, sigma0 = sigma0+h)
    term2 = func(K, beta = beta, sigma0 = sigma0)
    return (term1 - term2)/(h)

def dif1(func,K, beta, epsilon, *args,h=0.0001):
    term1 = func(K=K, beta = beta, epsilon = epsilon+h)
    term2 = func(K, beta = beta, epsilon = epsilon)
    return (term1 - term2)/(h)

def dif2(func,K, beta, rho, *args,h=0.0001):
    term1 = func(K=K, beta = beta, rho = rho+h)
    term2 = func(K, beta = beta, rho=rho)
    return (term1 - term2)/(h)


def opt(param, vol, K, beta=0.5, F=0.04, z0 = 1, t=0, T=10):
    value = log_normal_e(K=K, F=F, beta=beta, sigma0=param[0], z0 = z0, epsilon = param[1], rho=param[2], t=t, T=T)
    return vol-value

def phi(T, k = 0.05):
    return 1/k * (1-np.exp(-k*T))

def v(T, t=5, S=5, k = 0.05,sigma = 0.01):
    return ((phi(T, k) - phi(S,k))**2) * (sigma**2 / (2*k)) * (np.exp(2*k*t) - 1)


def swap_option(strike, z_curve, T=10, t=5, S=5, k = 0.05,sigma = 0.01, simulation = 40000, alpha = 0.95):
    """Calculate the option price and the alpha confidence interval of the price.
    
    :param strike: Strike price
    :param z_curve: the discount curve
    :param T: Maturity of the above discount curve
    :param t: Start point of the discount curve
    :param S: Maturity of the below discount curve
    :param sigma: the volatility
    :param simulation: number of simulations
    :param alpha: the confidence interval ratio
    """
    time = np.arange(S+1,T+1)

    sig = v(time, t, S, k, sigma)
    shap = (simulation,len(sig))
    sim = np.ones(shap)
    for i,j in enumerate(sig):
        sim[:,i] = np.random.normal(size = shap[0],loc = 0, scale = j**0.5)
    
    z_6_10 = z_curve(time)
    z_ratio = z_6_10/z_curve(S)
    
    sim_z = np.multiply(z_ratio , np.exp(-0.5 * sig + sim))
    
    pv01 = np.cumsum(sim_z, axis = 1)[:,-1]
    
    swap_value = (1-sim_z[:,-1])/pv01
    
    call = (swap_value - strike)
    put = -call
    
    call[call<0] = 0
    put[put<0] = 0
    
    call *= z_curve(t) * pv01
    put *= z_curve(t) * pv01
    
    option = pd.DataFrame(columns = [" price", "Left Tail", "Right Tail"])
    mu = np.mean(call)
    std = np.std(call)/np.sqrt(simulation) * norm.ppf(alpha + (1-alpha)/2)
    option.loc["Call"] = [mu, mu - std, mu + std]
    
    mu = np.mean(put)
    std = np.std(put)/np.sqrt(simulation) * norm.ppf(alpha + (1-alpha)/2)
    option.loc["Put"] = [mu, mu - std, mu + std]
    
    return option
    

    