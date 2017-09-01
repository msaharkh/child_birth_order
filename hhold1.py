
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

##############################################################################;
# define a class for households with 1 child;
##############################################################################;

class hhold:
    
    def __init__(self,u,f,g): 
        """
        instantiates a household with one child; then, computes cognitive
        and non-cognitive elasticities at each age; finally, computes
        household's optimal choices at each period.
        
        u: preferences; a pd.Series; 
        --------------------------------
        alpha_1 = weight on consumption;
        alpha_2 = weight on leisure;
        alpha_3 = weight on cognitive skills;
        alpha_4 = weight on non-cognitive skills;
            
        f: cognitive technology; a pd.DataFrame; 
        --------------------------------------------
            it has 2 columns: 1a, 1b
            and 3 rows: k, e, a
            1a  = the intercept;
            1b  = the slope in age;
            k   = cognitive skills;
            e   = monetary investments;
            a   = time investments;
        
        g: non-cognitive technology; a pd.DataFrame;
        ------------------------------------------------
            it has 2 columns: 1a, 1b
            and 3 rows: n, e, a
            1a  = the intercept;
            1b  = the slope in age;
            n   = non-cognitive skills;
            e   = monetary investments;
            a   = time investments;
        """
        
        ########################################################
        # Set M and beta;
        ########################################################
        
        # set the lenght of development;
        self.M = 17

        # set the discount factor;
        self.beta = 0.95

        # alias M and beta;        
        M = self.M
        beta = self.beta
        
        ########################################################
        # Set the preferences;
        ########################################################
        
        self.u = u
    
        ########################################################
        # Set the cognitive elasticities at each age;
        ########################################################
        
        # t denotes the time in the model; also child's age;
        self.delta = pd.DataFrame(np.arange(M),columns=['t'])
        for key in f.index :
            self.delta[key+'1'] = np.exp(f.loc[key,'1a'] + f.loc[key,'1b']*self.delta['t'])
        
        ########################################################
        # Set the non-cognitive elasticities at each age;
        ########################################################
        
        # t denotes the time in the model; also child's age;
        self.eta = pd.DataFrame(np.arange(M),columns=['t'])
        for key in g.index :
            self.eta[key+'1'] = np.exp(g.loc[key,'1a'] + g.loc[key,'1b']*self.eta['t'])

        ########################################################
        # Compute the optimal share of resources for each use;
        ########################################################
        
        # compute the CV of future flows of investments at each period;
        seq=np.cumsum(np.tri(M,k=-1),axis=0)
        I1 = (seq > 0)
        x1 = np.cumprod( self.delta['k1'].values.reshape(M,1)*I1+(1-I1) , axis=0)*I1
        x1 = np.sum(x1*(beta**seq), axis=0)
        y1 = np.cumprod( self.eta['n1'].values.reshape(M,1)*I1+(1-I1) , axis=0)*I1
        y1 = np.sum(y1*(beta**seq), axis=0)
        
        # compute the shares of hhold resources allocated to each choice;
        self.phi = pd.DataFrame(np.arange(M),columns=['t'])
        self.shares = pd.DataFrame(np.arange(M),columns=['t'])

        # compute the lifetime marginal benefit of each choice;
        self.phi['c']   = u['alpha_1']*np.ones(M)
        self.phi['e1']  = u['alpha_3']*self.delta['e1']*(1+x1)
        self.phi['e1'] += u['alpha_4']*self.eta['e1']*(1+y1)
        self.phi['l']   = u['alpha_2']*np.ones(M)
        self.phi['a1']  = u['alpha_3']*self.delta['a1']*(1+x1)
        self.phi['a1'] += u['alpha_4']*self.eta['a1']*(1+y1)
        
        # share of household's lifetime income (PV) allocated to c and e;
        for key in ['c','e1']:
            self.shares[key] = self.phi[key] / self.phi[['c','e1']].sum(axis=1).sum()

        # share of household's periodic time endowment allocated to l and a;
        for key in ['l','a1']:
            self.shares[key] = self.phi[key] / self.phi[['l','a1']].sum(axis=1)
            
###############################################################################
# "GMM" computes the GMM objective function value;
###############################################################################

def gmm(params,data):
    """ 
    the Simulated Method of Moments objective function.
    
    inputs
    -------
    params: a vector of parameters; np.array;
    
    data: the data; a pd.DataFrame; 
    its columns are: 
        t    = period and 1st child's age;
        inc  = household's lifetime income;
        k1   = cognitive score at time t;
        n1   = non-cognitive score at time t;
        k1p  = cognitive score at time t+5;
        n1p  = non-cognitive score at time t+5;
        a1   = parental time;
        
    output
    ------
    GMM objective function value
    """
    
    model = hhold(*pmap(params))
    obj = moments(model,data)
    print(obj)
    return obj

###############################################################################
# "pmap" maps a vector of parameters into u, f, and g;
###############################################################################

def pmap(params):
    """
    maps the parameters into the preference weights and technology elasticities;
    
    inputs
    -------
    params: 15-element np.array;
    
    outputs
    -------
    u: preferences; pd.Series
    f: cognitive technology; pd.DataFrame
    g: non-cognitive technology; pd.DataFrame
    """
    
    f = pd.DataFrame(params[0:6].reshape(3,2), index=['k','e','a'], columns=['1a','1b'])
    g = pd.DataFrame(params[6:12].reshape(3,2), index=['n','e','a'], columns=['1a','1b'])
    u = pd.Series(np.exp(params[12:14]),index=['alpha_2','alpha_3'])
    zeta = np.exp(params[14])/(1+np.exp(params[14]))
    
    # impose the preference restrictions;
    u /= (1+u.sum())
    u['alpha_3'], u['alpha_4'] = u['alpha_3']*zeta, u['alpha_3']*(1-zeta)
    u['alpha_1'] = 1-u.sum()
    u = u.reindex(sorted(u.index))

    return (u, f, g)

##############################################################################;
# "moments" computes moments from data and model;      
##############################################################################;

def moments(model,data):
    """
    computes moments from the model and the data;
    
    inputs
    ------
    model: an instant of hhold;
    data: check gmm's docstring;
    
    outputs
    -------
    obj: gmm's objective function value;
    mom: a list of moments from the model and the data;
    """
    
    dat = data.copy()
    sim = simulate(model,dat[['id','t','inc','k1','n1']]) 
    
    dat['gt']=dat['t'].map(lambda x: x if x%2!=0 else x-1)
    sim['gt']=sim['t'].map(lambda x: x if x%2!=0 else x-1)
    
    # compute moments;
    mom = list()
    mom.append(dat.groupby(by=['t']).mean()[['a1','k1p','n1p']])
    mom.append(sim.groupby(by=['t']).mean()[['a1','k1p','n1p']])
    mom.append(dat.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['k1p']].unstack())
    mom.append(sim.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['k1p']].unstack())
    mom.append(dat.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['n1p']].unstack())
    mom.append(sim.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['n1p']].unstack())

    obj  = (((mom[0] - mom[1])**2)/mom[0].max()).sum().sum()
    obj += (((mom[2] - mom[3])**2)/mom[2].max()).sum().sum()
    obj += (((mom[4] - mom[5])**2)/mom[4].max()).sum().sum()
    
    return obj

##############################################################################;
# define function that simulates the model at t+5
##############################################################################;

def simulate(model,data):   
    """
    simulate takes state variables at time t and returns choices at time t 
    and state values at time t+5;

    inputs
    ------
    check moments's docstring
    
    outputs
    -------
    simulated dataset comparable with observables in the CDS-PSID
    """
    
    # compute k and n in t+5;
    sim = data.copy()
    for period in [0,1,1,1,1]:
        sim['t'] = sim['t']+period
        
        # get the optimal shares and multiply by endowemnts to get the choices;
        sim = sim.merge(model.shares[['t','a1','e1']], how='left', on=['t'])
        sim['e1'] *= sim['inc']*10*model.M / 52
        sim['a1'] *= 7*24

        # get the elasticities;        
        pf = sim[['t']].merge(model.delta, how='left', on=['t'])
        pn = sim[['t']].merge(model.eta, how='left', on=['t'])

        # compute the next period's skills;
        sim['k1'] = (sim['k1']**pf['k1'])*(sim['e1']**pf['e1'])*(sim['a1']**pf['a1']) 
        sim['n1'] = (sim['n1']**pn['n1'])*(sim['e1']**pn['e1'])*(sim['a1']**pn['a1'])
        sim.drop(['a1','e1'],axis=1,inplace=True) 
                    
    # clean the simulated data;
    sim.rename_axis(axis=1,mapper={'k1':'k1p', 'n1':'n1p'},inplace=True)
    sim['t'] -= 4
    
    # get the choices at t;
    sim = sim.merge(model.shares[['t','a1','e1']], how='left',on=['t'])
    sim['a1'] *= 7*24
    
    # add the skills at t;
    sim = sim.merge(data[['id','k1','n1']],on=['id'])
    
    return sim 

##############################################################################;
# Graphs
##############################################################################;

def fig_u(dgp,dgp_t=None,dgp_i=None):
    """
    Bar plot of a DGP preference parameters;

    inputs
    ------
    dgp:   estimated DGP; an instant of hhold;
    dgp_t: true DGP; an instant of hhold;
    dgp_i: initial DGP; an instant of hhold;

    
    outputs
    -------
    Bar plot of preferences;
    """
    
    if dgp_t is None:
        axes = dgp.u.plot(kind='bar', color=['black']*4) 
    else:
        frame = pd.DataFrame({'estimate':dgp.u,'true':dgp_t.u,'initial':dgp_i.u},columns=['estimate','true','initial'])
        axes = frame.plot(kind='bar', color=['black','dimgray','darkgray'])
    mapper = dict(alpha_1 = r'$\alpha_1$', 
                  alpha_2 = r'$\alpha_2$', 
                  alpha_3 = r'$\alpha_3\zeta$', 
                  alpha_4 = r'$\alpha_3(1-\zeta)$')
    axes.set_xticklabels([mapper[x] for x in dgp.u.index],rotation='horizontal')


def fig_e(dgp,dgp_t=None,dgp_i=None,skill='k'):
    """
    Line plot of a DGP elasticities;

    inputs
    ------
    dgp:   estimated DGP; an instant of hhold;
    dgp_t: true DGP; an instant of hhold;
    dgp_i: initial DGP; an instant of hhold;
    skill: 'k' for cognitive or 'n' for non-cognitive;
    
    outputs
    -------
    Line plot of elasticities;
    """
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.7)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    axes = [ax1, ax2, ax3]
    elas = 'delta' if skill == 'k' else 'eta'
    x = 't'
    y = [skill+'1','e1','a1'] 
    xlabels = ['','age','age']
    titles = ['Initial Skills', 'Monetary Investments', 'Alone Time Investments']
    graph = dict(x=x, y=y, ax=axes, legend=False, subplots=True, rot=0)
    if dgp_t is not None:
        dgp_t.__dict__[elas].plot(color='dimgray',ls='solid',**graph)
        dgp_i.__dict__[elas].plot(color='darkgray',ls='dotted',**graph)
    dgp.__dict__[elas].plot(color='black',ls='dashed',**graph)
    if dgp_t is not None:
        fig.legend((ax1.lines[2],ax1.lines[0],ax1.lines[1]),('estimate','true','initial'),loc=(0.65,0.2),framealpha=0.5)
    for ax, xlabel, title in zip(axes,xlabels,titles):
        ax.set_xlabel(xlabel)
        ax.set_title(title)
