from bound import *
#from curves_assignment_2_students import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.stats as st
import pandas as pd


class InterpolationBase(object):
    """
    Base class for interpolation objects
    """
    def __init__(self, abscissae, ordinates):
        if not sorted(abscissae) or \
           len(abscissae) != len(ordinates):
               raise RuntimeError('abscissae/ordinates length mismatch')
        self.N = len(abscissae)
        self.abscissae, self.ordinates = abscissae, ordinates

    def _locate_single(self, x):
        i, j = bound(x, self.abscissae)
        x_lo, x_hi = self.abscissae[i], self.abscissae[j]
        y_lo, y_hi = self.ordinates[i], self.ordinates[j]

        return x_lo, x_hi, y_lo, y_hi

    def locate(self, x):
        if not isinstance(x, np.ndarray):
            return self._locate_single(x)
        else:
            n = x.shape[0]
            x_lo = np.zeros(n)
            x_hi = np.zeros(n)
            y_lo = np.zeros(n)
            y_hi = np.zeros(n)
            for i in range(0,n):
                x_lo[i], x_hi[i], y_lo[i], y_hi[i] = self._locate_single(x[i])

        return x_lo, x_hi, y_lo, y_hi

class InterpolationLinear(InterpolationBase):
    """
    Linear interpolation object
    """
    def __init__(self, abscissae, ordinates):
        InterpolationBase.__init__(self, abscissae, ordinates)

    def __call__(self, x):
        x_lo, x_hi, y_lo, y_hi = \
            InterpolationBase.locate(self, x)
        r = 1.0 - (x_hi - x)/(x_hi - x_lo)

        return r*(y_hi - y_lo) + y_lo


class InterpolationLoglinear(InterpolationBase):
    """
    Log Linear interpolation object
    """
    def __init__(self, abscissae, ordinates):
        InterpolationBase.__init__(self, abscissae, ordinates)

    def __call__(self, x):
        x_lo, x_hi, y_lo, y_hi = \
            InterpolationBase.locate(self, x)
        ln_ylo, ln_yhi = np.log(y_lo), np.log(y_hi)
        R = 1.0 - (x_hi - x)/(x_hi - x_lo)

        return np.exp(ln_ylo+(ln_yhi - ln_ylo)*R)


class Curve(object):
    """
    General curve object
    """
    def __init__(self, times, factors, interp):
        self.__impl = interp(times, factors)

    def __call__(self, t):
        return self.__impl(t)


def ois_swaps(f, maturities, rates):
    """
    compute valuation of fixed rate ois swap
    """
    # compute discrete set of discount factors
    df = np.ones(len(maturities))
    df[1:] = np.exp(-f*maturities[1:])
    # put in interpolator curve
    z = Curve(maturities, df, InterpolationLoglinear)
    # longest maturity
    max_maturity = maturities[-1]
    # annual period up to longest maturity
    times = np.arange(0.0, max_maturity+0.1)
    # pv01 up to longest maturity
    pv01 = z(times)
    pv01[0] = 0.0
    all_pv01 = np.cumsum(pv01)
    # filter maturities for the maturities needed
    pv01 = all_pv01[maturities.astype(np.int64)]
    # compute ois swap values
    swap_values = rates * pv01[1:] - (1.0 - df[1:])

    return swap_values

def forward(times, z):
    """

    :param times: times at which to compute the forward
    :param z: discount curve
    :return: forward rates from 0:1, ..., n-1:n
    """
    df = z(times)
    delta = times[1:] - times[:-1]
    fwd = (df[:-1] / df[1:] - 1) / delta

    return fwd

def fixed_floating_swaps(f_6m, maturities_fixed, rates, z):
    """

    :param f_6m: forwards for the Z_6m curve
    :param maturities_fixed: maturities of the swaps
    :param rates: swap rates on the fixed floating swaps
    :param z: discount curve (Z)
    :return: value of the fixed-floating swaps
    """
    # compute discrete set of discount factors
    df_6m = np.ones(len(maturities_fixed))
    df_6m[1:] = np.exp(-f_6m*maturities_fixed[1:])
    # put in interpolator curve
    z_6m = Curve(maturities_fixed, df_6m, InterpolationLoglinear)
    years = np.arange(0, maturities_fixed[-1]+0.1)
    half_years = np.arange(0, 2*maturities_fixed[-1]+0.1)*0.5
    df_fixed = z(years)
    df_fixed[0] = 0.0
    pv01_fixed = np.cumsum(df_fixed)
    pv01 = pv01_fixed[maturities_fixed.astype(np.int64)]

    fixed_leg = rates * pv01[1:]
    # floating leg cash flows
    forwards = forward(half_years, z_6m)
    float_cf = z(half_years[1:]) * forwards*0.5
    # select only full year sums
    float_leg = np.cumsum(float_cf)[1::2]

    return fixed_leg - float_leg


def tenor_basis_swaps(f_short, maturities, spreads, z, z_long):
    """

    :param f_short: forward rates for z_short (eg. 3m)
    :param maturities: maturities of the tenor basis swaps 
    :param spreads: spreads on the tenor basis swaps
    :param z: discount curve
    :param z_long: long (e.g. 6m) discount curve
    :return: value of the tenor basis swaps
    """
    # compute discrete set of discount factors
    df = np.ones(len(maturities))
    df[1:] = np.exp(-f_short*maturities[1:])
    # put in interpolator curve
    z_short = Curve(maturities, df, InterpolationLoglinear)
    half_years = np.arange(0, 2*maturities[-1]+0.1)*0.5
    quarter_years = np.arange(0, 4*maturities[-1]+0.1)*0.25
    # floating leg cash flows long leg
    forwards_long = forward(half_years, z_long)
    float_cf_long = forwards_long* z(half_years[1:])*0.5
    # select only full year sums
    float_leg_long = np.cumsum(float_cf_long)[1::2]
    float_leg_long = float_leg_long[(maturities[1:]-1).astype(np.int64)]



    # floating leg cash flows short leg
    forwards_short = forward(quarter_years, z_short)
    # discounted value of the floating coupons
    float_cf_short = forwards_short* z(quarter_years[1:])*0.25
    # select only full year sums
    float_leg_short =np.cumsum(float_cf_short)[1::4]
    float_leg_short = float_leg_short[(maturities[1:]-1).astype(np.int64)]
    # fixed (spread) discount curve
    df_fixed = z(quarter_years)*0.25
    df_fixed[0] = 0.0
    pv01_fixed = np.cumsum(df_fixed)[0::4]
    # select only pv01s for the necessary maturities
    pv01 = pv01_fixed[(maturities).astype(np.int64)]


    return float_leg_short +(spreads*pv01[1:]) - float_leg_long


def swap_rate(fow, maturities, z, s=0,f_time = 3):
    """
    Fow: Foward rates;
    maturities: Maturities;
    z: z curve;
    s: spread;
    f_time: 3 : 3months 
            6 : 6months;
    """
    df = np.ones(len(maturities))
    df[1:] = np.exp(-fow*maturities[1:])
    z_short = Curve(maturities, df, InterpolationLoglinear)
    years = np.arange(0, maturities[-1]+0.1)
    
    if f_time==3:
        di_years = np.arange(0, 4*maturities[-1]+0.1)*0.25
        aux = 4
    else:
        di_years = np.arange(0, 2*maturities[-1]+0.1)*0.5
        aux = 2
    
    df_fixed = z(years)
    df_fixed[0] = 0.0
    pv01_fixed = np.cumsum(df_fixed)
    pv01 = pv01_fixed[maturities.astype(np.int64)]
    
    forwards = forward(di_years, z_short)
    float_cf = z(di_years[1:]) * (forwards+s)*1/aux
    
    # select only full year sums
    float_leg = np.cumsum(float_cf)[aux-1::aux]   
    
    return float_leg/pv01[1:]

    