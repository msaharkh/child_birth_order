
# import Python scientific stack;
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# import local classes of households with 1 and 2 children;
import hhold1 as hh1
import hhold2 as hh2

# style commands for graphics;
plt.close('all')
plt.style.use('ggplot')
mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
pd.options.display.float_format = '{:,.4f}'.format

##############################################################################;
# Monte Carlo: 1-child Households;
##############################################################################;

# true parameters; 
#   row 1: cognitive elasticities
#   row 2: non-cognitive elasticities
#   row 3: preference parameters
param1 = np.array([-2.00,  0.08, -3.10,  0.05, -1.30, -0.05,
                   -2.10,  0.05, -3.00,  0.10, -2.40,  0.10,
                    0.00,  0.00,  0.00])

# initial point for the optimization; 
start1 = np.array([-1.80,  0.03, -3.00,  0.01, -1.20, -0.05,
                   -2.00,  0.02, -3.10,  0.05, -2.30,  0.02,
                    0.20, -0.20,  0.10])

# Simulate the data;
dgp_t     = hh1.hhold(*hh1.pmap(param1))           # True DGP;
data1     = pd.read_stata('data1.dta')             # The intial conditions;
sim1      = hh1.simulate(dgp_t,data1)              # Simulated data;

# Estimate the model;
param1hat = opt.fmin_bfgs(hh1.gmm,start1,args=(sim1,))

####################################################
# Graph the estimates, true, and initial parameters;
####################################################

# create the DGPs;
dgp_e = hh1.hhold(*hh1.pmap(param1hat))            # Estimated DGP;
dgp_i = hh1.hhold(*hh1.pmap(start1))               # Initial DGP;

# plot the graphs;
hh1.fig_u(dgp_e,dgp_t,dgp_i)                       # Preferences;
hh1.fig_e(dgp_e,dgp_t,dgp_i,skill='k')             # Cognitive elasticities;
hh1.fig_e(dgp_e,dgp_t,dgp_i,skill='n')             # Non-cognitive elasticities;


##############################################################################;
# Monte Carlo: 2-child Households;
##############################################################################;

# true parameters; 
#   row 1: cognitive elasticities
#   row 2: non-cognitive elasticities
#   row 3: preference parameters
param2 = np.array([-2.10,  0.05, -3.00,  0.06, -1.20, -0.06, -1.60, -0.02,
                   -2.00,  0.04, -3.20,  0.10, -2.10,  0.08, -1.90, -0.02,
                    0.00,  0.00,  0.00,  0.00])

# initial point for the optimization; 
start2 = np.array([-1.80,  0.01, -3.00,  0.01, -1.20, -0.05, -1.40, -0.05,
                   -1.90,  0.02, -3.10,  0.05, -2.30,  0.02, -1.70, -0.04, 
                    0.20, -0.20,  0.20,  0.20])


# Simulate the data;
dgp_t = hh2.encapsulate(param2,param1)             # True DGP;
data2 = pd.read_stata('data2.dta')                 # The intial conditions;
sim2  = hh2.simulate(dgp_t,data2)                  # Simulated data;

# Estimate the model;
param2hat = opt.fmin_bfgs(hh2.gmm,start2,args=(sim2,param1))

####################################################
# Graph the estimates, true, and initial parameters;
####################################################

# create the DGPs for hhold with birth spacing of 2 years;
dgp_t = hh2.hhold(2,*hh2.pmap(param2,param1))      # True DGP;
dgp_e = hh2.hhold(2,*hh2.pmap(param2hat,param1))   # Estimated DGP;
dgp_i = hh2.hhold(2,*hh2.pmap(start2,param1))      # Initial DGP;

# plot the graphs;
hh2.fig_u(dgp_e,dgp_t,dgp_i)                       # Preferences
hh2.fig_e(dgp_e,dgp_t,dgp_i,skill='k',child='1')   # Cognitive elasticities; child 1;
hh2.fig_e(dgp_e,dgp_t,dgp_i,skill='k',child='2')   # Cognitive elasticities; child 2;
hh2.fig_e(dgp_e,dgp_t,dgp_i,skill='n',child='1')   # Non-cognitive elasticities; child 1;
hh2.fig_e(dgp_e,dgp_t,dgp_i,skill='n',child='2')   # Non-cognitive elasticities; child 2;


