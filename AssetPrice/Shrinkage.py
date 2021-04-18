# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:11:28 2019

@author: pedro
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:53:50 2019

@author: pedro
"""

import numpy as np
import pandas as pd
import numpy.linalg as la
import openpyxl as oxl
from scipy.optimize import minimize
from numpy.testing import assert_allclose
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks",color_codes=True)
from arch import arch_model
from sklearn.covariance import LedoitWolf


def fig_latex(file_name, figure, graph_title = None,label = None, file_type = ".png", force = True, center = True, latex = True, type_fig = "pd"):
    """ This function save a pandas figure and create a latex file for the 
        that 
        file_name: name of the file;
        figure: variable if the figure;
        Graph title: title for graph in latex;
        label: create a label for latex;
        file_type: Type of file (x. '.png' or '.pdf')
        force: to force the position in latex;
        Center: to center the image of plot in latex;
        latex: Create a latex file
    """
    #import matplotlib.pyplot as plt
    
    if type_fig == "pd":
        fig = figure.get_figure()
        fig.savefig(file_name + file_type)    
    else:
        figure.savefig(file_name+file_type)
        
    
    
    graph = "\\begin{figure}"
    if force == True:
        graph = graph + "[H]" 
        
    graph = graph + "\n"
        
    if center == True:
        graph = graph + "\\centering \n"
        
    if latex == True:
        
        text_file = open(file_name + '.tex', "w")
        
        graph = graph + "\\includegraphics{" + file_name + file_type + "} \n"
        
        if graph_title != None:
            graph = graph + "\\caption{"+ str(graph_title) + "} \n" 
        
        if label != None:
            graph = graph + "\\label{" + str(label) + "}\n"
        
        graph = graph + "\\end{figure}"
        
        text_file.write(graph)
        text_file.close()


def d_file(file_name, df, table_Title = None, center = True, longtable = None):
    """
    takes a pandas DataFrame and creates a file_name.tex with LaTeX table data
    file_name: name of the file;
    df: Data Frame;
    table_Title: title of the table to latex;
    center: centering the table in latex;
    longtable: Put tables with large rows in latex
    Remark: if longtable == True so center and table_Title will not work.
    """
    # create and open file
    text_file = open(file_name, "w")
    # data frame to LaTeX
    if longtable != True:
        ##
        if table_Title != None and center == True:
            df_latex = '\\begin{table}[H]\n\centering\n' + '\\caption{'+ table_Title + '}\\\n' + df.to_latex() + '\\end{table}\n' 
        else:
            if  table_Title != None:
                df_latex = '\\begin{table}[H]\n' + '\\caption{'+ table_Title + '}\n' + df.to_latex() + '\\end{table}\n' 
            else:
                df_latex = '\\begin{table}[H]\n' +  df.to_latex() + '\\end{table}\n'  
    
    else:
        ##
        df_aux = df.to_latex(longtable = longtable)
        if table_Title != None:
            df_latex = df_aux[:22] + "\\caption{" + table_Title + "}\\\ \n" + df_aux[22:]
        
        if center == True:
            df_latex = "\\begin{center} \n" + df_latex + "\\end{center}"

    # Consider extensions (see later in class)
    # write latex string to file
    text_file.write(df_latex)
    # close file
    text_file.close()
    

# =============================================================================
# Geting the data
# =============================================================================

returns = pd.read_excel('Exam_app.xlsx', index_col = 0, usecols = 'A:U', skiprows = 3)
returns.columns = ['all equity', 'small', 'value', 'europe', 'us', 'japan', 'emerging market',
'frontier market', 'all bonds', 'all tb', '1-3y', '3-5y', '7-10y',
'inflation linked', 'inv Grade', 'high yield', '1-3yc', '3-5yc',
'7-10yc', 'market']


# =============================================================================
# Fixing the data
# =============================================================================

mkt_ret_full = returns["market"]

returns_full = returns['2014-09-29':].drop(labels = 'market', axis = 1)

returns_full = returns_full.astype(float)

returns = returns_full.iloc[-104:]



mkt_returns = mkt_ret_full.iloc[-104:]

# =============================================================================
# Calculating the volatility
# =============================================================================

hh = returns.describe().T

hh['mean'] = hh['mean']*52

annualized_vols = returns.std()*np.sqrt(52)

hh['std'] = annualized_vols


col = list(hh.columns)
col[1] = "A. mean"
col[2] = "A. std"

hh.columns = col 

# =============================================================================
# New Cov
# =============================================================================

# =============================================================================
# Garch
# =============================================================================

#vol = pd.DataFrame()
#for i in returns_full:
#     am = arch_model(returns_full[i]*100)
#     result = am.fit()
#     vol[i] = result.conditional_volatility
#
#variance_nw = 52 * (vol/100)**2
#std_nw = (variance_nw**0.5)
#
#(std_nw.T@std_nw)/std_nw.shape[0]


# =============================================================================
# Shrikage
# =============================================================================

rho_hat = returns_full['2014-09-29':].corr()

cov = LedoitWolf().fit(returns_full)

cov_hat = pd.DataFrame(cov.covariance_*52, columns = rho_hat.columns, index = rho_hat.index)

#d_file("shr_cov", round(cov_hat,4), "Shrinkage Covariance Matrix")

# =============================================================================
# 
# =============================================================================


#d_file("vol", round(hh,4), "Data description")

#rho_hat = returns_full['2014-09-29':].corr()
#
##sns.pairplot(returns_full['2014-09-29':])
##sns.plt.show()
##
##sns.pairplot(returns_full['2014-09-29':], kind="scatter")
##plot.show()
#
#a = sns.pairplot(returns_full['2014-09-29':], kind="reg")
#fig
#plt.show()
#
##g = sns.PairGrid(returns_full)
##g = g.map_diag(plt.hist, edgecolor="w")
##g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
#
#
## Generate a mask for the upper triangle
#mask = np.zeros_like(rho_hat, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#
## Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(8, 6))
#
## Generate a custom diverging colormap
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
## Draw the heatmap with the mask and correct aspect ratio
##sns.heatmap(rho_hat, mask=mask)
#sns.heatmap(rho_hat, mask=mask, cmap=cmap, vmax=1, vmin=-1,center=0,
#            square=True, linewidths=0.5, cbar_kws={"shrink": .7})


#d_file("corr", round(rho_hat,2), "Correlation Matrix")

# create covariance matrix (L21:R26)
#cov_hat = pd.DataFrame(np.multiply(np.matrix(annualized_vols).T@\
#                                   np.matrix(annualized_vols),rho_hat),
#    columns = rho_hat.columns, index = rho_hat.index)


#d_file("cov_hat", round(cov_hat,4), "Covariance Matrix in percentage")

###############################################################################
# market caps page
###############################################################################

market_weights = pd.Series(np.zeros(returns.shape[1]), index = returns.columns)

market_weights["all equity", "all bonds"] = 0.6,0.4

aux = market_weights.plot(kind='bar', title='Representative Investor Weights', 
                                grid='True')

#fig_latex("mktw", aux, "Representative Investor Weights")

weights = market_weights



###############################################################################
# Main inputs page
###############################################################################
rf = 0.01
global_risk_aversion = (0.045/(cov_hat["all equity"].T@weights))

client_risk_aversion = global_risk_aversion

implied_ER = pd.Series(global_risk_aversion *\
                       cov_hat.values @ weights.values,
                       index=weights.index)


inv_cov = la.inv(cov_hat)

# average annual excess returns
avg_returns = returns.mean()*52 - rf

exp = pd.DataFrame([implied_ER.values], 
             index = ["Implied Excess Returns(%)"], 
             columns = implied_ER.index).T
                   
#d_file("skret",round(exp*100,4) ,"Expected Implied Excess Sk-Returns in Percentage", longtable=True)

                   

implied_ER_opt = pd.Series(1 / global_risk_aversion *\
                           inv_cov @ implied_ER.values)


avg_ER_opt = pd.Series(1 / global_risk_aversion *\
                       inv_cov @ avg_returns.values)

#avg_ER_opt.index= avg_returns.index
#
#aux = avg_ER_opt.plot(kind='bar', title='Optimum Weights', 
#                                grid='True')                

#fig_latex("avg_w", aux, "Avarege optimum weights")

tau = 0.05
###############################################################################
# views page
###############################################################################

views = pd.DataFrame(np.zeros((7,len(weights.index))),columns=weights.index,
                     index=np.array([1,2,3,4,5,6,7]))

# Equities (MSCI World) are going to outperform bonds (Treasury+ CB portfolio) with 4% (90%
# confidence)
views.loc[1,["all equity","all bonds"]] =  np.array([1,-1])

# Small caps are going to outperform MSCI World with 0.5% (30% confidence)
views.iloc[1,0:2] = np.array([-1, 1])

# Value stocks will have same return as MSCI world (30% confidence)
views.loc[3,"value"] = np.array([1])
views.loc[3,"all equity"] = np.array([-1])

# A portfolio of 70% EM and 30% FM will outperform a portfolio of 45% US equity, 35% European
# equities, and 20% Japanese Equities with 1% (75% confidence)
views.loc[4,["emerging market","frontier market"]] = np.array([0.7,0.3])
views.loc[4,"europe":"japan"] = np.array([-0.35,-0.45,-0.2])

# Nominal bonds will underperform Inflation-Linked bonds with 25bps (40% confidence)
views.loc[5,"1-3y":] = -1*np.ones(len(views.loc[5,"1-3y":]))/(len(views.loc[5,"1-3y":]) -1)
views.loc[5,"inflation linked"] = 1

# Long-term (7-10Y) Corporate bonds will outperform long-term treasury bonds with 75bps (60%
# confidence)
views.loc[6,["7-10y", "7-10yc"]] = [-1,1]

# High Yield bonds will outperform Investment Grade Bonds with 50bps (50% confidence)
views.loc[7,["inv Grade", "high yield"]] = [-1,1]


#d_file("view",views.T,"Views")

P = views

ER = pd.Series(np.array([0.04, 0.005, 0, 0.01,0.0025,0.0075, 0.005]),
               index=np.array(range(1,8)))

#bb = (ER - views@implied_ER).to_frame("Values")
#aux = []
#for i in bb["Values"]:
#    if i>0:
#        aux.append("Bullish")
#    else:
#        aux.append("Bearish")
#
#bb['Results'] = aux
#
#d_file("bullbear", bb, "Bullish and Bearish")

        


Q = ER

conf_views = pd.Series(np.array([0.9, 0.3, 0.3, 0.75, 0.4, 0.6, 0.5]),
                 index=np.array(range(1,8)))

#aux = pd.DataFrame([ER.values, conf_views.values], columns = views.index,
#             index = ["Outperform", "Confidence"])

#d_file("exp_out", aux, "Expected outperformance and confidence based on views")

alpha = (1 - conf_views)/conf_views

omega_diag = np.diag(np.diag(alpha)@P@(tau*cov_hat)@P.T)
omega = pd.DataFrame(np.eye(7)*omega_diag, index = P.index, columns = P.index) 

#d_file("shkomega", pd.DataFrame(omega_diag, columns= ["Values"], index = views.index), "Diagonal of Sk-Omega", longtable=True)



inv_omega = la.inv(omega)


P_aux = P
P_aux.index = (P.T @ inv_omega).columns

sigma_bar = inv_cov / tau + P.T @ inv_omega @ P_aux
inv_sigma_bar = la.inv(sigma_bar)
v1 = inv_sigma_bar @ inv_cov / tau
v2 = inv_sigma_bar @ P.T @ inv_omega

# Black-Litterman expected returns (D29:D35)
bl = v1 @ implied_ER.values + v2 @ Q.values


#d_file("skbl_exr",round(pd.DataFrame([bl,implied_ER], index=["BL", "Market neutral"], columns = weights.index).T *100,4), "Excess returns (SK)", longtable = True)
#
#
#aux = pd.DataFrame(views@bl, columns = ["BL"])
#
#aux.index = ER.index
#
#aux["Implied"] = ER
#
#aux["Confidence"] = conf_views
#
#d_file("view_p", round(aux*100,4), "Views Performance in percentage")


#aux[]

# Black-Litterman optimal weights (D29:D35) 
bl_opt = inv_cov @ bl / client_risk_aversion

test_a = sigma_bar @ bl / client_risk_aversion


#aux = pd.Series(bl_opt, index = weights.index).plot(kind='bar', title='Optimum Weights for BL',grid='True')
#df_aux = pd.Series(bl_opt, index = weights.index)
#df_aux["Sum"] = sum(df_aux)
#d_file("sk_wbl", round(df_aux.to_frame("Weights")*100,4), "Optimum Weights with no constraint and with SK covariance in percentage", longtable =True)

#fig_latex("SK_ER_weigts", aux, "BL optimum weights with no constraint")



#d_file("sk_shp_rn",
#       pd.DataFrame([round((bl@bl_opt)/((bl_opt@cov_hat@bl_opt)**0.5),4),
#                     round(bl@bl_opt.T/((bl_opt@la.inv(sigma_bar)@bl_opt)**0.5),4)],
#              columns = ["Shape Ratio (SK)"], index = ["Returns std", "BL std"]),
#                     "Sharpe Ratio")

#bl@test_a/((test_a@la.inv(sigma_bar)@test_a)**0.5)


###############################################################################
# Constrained optimization
###############################################################################

mu = bl[:, None]

def mv(w, gamma, mu, vcv):
    """ Objective function """
    return -(np.dot(w.T, mu) - 0.5 * gamma * w.T @ vcv @ w)[0]

def dmv(w, gamma, mu, vcv):
    """ Objective function """
    return -(mu.T[0] - gamma * w @ vcv)


cons = ({'type': 'ineq',
         'fun' : lambda x: np.sum(x)-1,
         'jac' : lambda x: np.ones(len(x))},
         {'type': 'ineq',
         'fun' : lambda x: np.array([x[i] for i in range(len(x))]),
         'jac' : lambda x: np.eye(len(x))})

res = minimize(mv, np.ones(len(bl))/len(bl),
               args=(client_risk_aversion ,mu, cov_hat),
               jac=dmv, constraints=cons, method='SLSQP',
               options={'disp': True})


#res1 = minimize(mv, ww,
#               args=(client_risk_aversion, mu, cov_hat),
#               jac=dmv, constraints=cons, method='SLSQP',
#               options={'disp': True})

y = res.x

#a = pd.Series(y, index = weights.index).plot(kind='bar', title='Optimum Weights for BL',grid='True')
#fig_latex("skc_weig",a, "BL Optimum Weights with Constraint")
#
#d_file("skw_blc", round(pd.Series(y, index = weights.index)*100,4).to_frame("Weights SK"), "BL Weights with Constraint", longtable = True)




#d_file("skshp_rc",
#       pd.DataFrame([round((bl@y)/((y@cov_hat@y)**0.5),4),
#                     round(bl@y/((y@la.inv(sigma_bar)@y)**0.5),4)],
#              columns = ["SK Shape Ratio"], index = ["Returns std", "BL std"]),
#                     "Sharpe Ratio")