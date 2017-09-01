
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################;
# define a class for households with 2 children;
##############################################################################;

class hhold:
    
    def __init__(self,S,u,f2,g2,f1,g1):
        """
        instantiate a household with two children with spacing of S; a scalar;
                
        u: preferences; a pd.Series;
        --------------------------------------------
            alpha_1 = weight on consumption;
            alpha_2 = weight on leisure;
            alpha_3 = weight on the first child's cognitive;
            alpha_4 = weight on the first child's non-cognitive;
            alpha_5 = weight on the second child's cognitive;
            alpha_6 = weight on the second child's non-cognitive;
            
        f2: cognitive technology when both children live in the hhold; a pd.DataFrame;
        --------------------------------------------
            4 rows: k, e, a, s
                k   = cognitive skills;
                e   = monetary investments;
                a   = time investments - alone;
                s   = time investments - shared;
            2 columns: 2a, 2b
                2a = the intercept;
                2b = the slope;
                
        g2: non-cognitive technology when both children live in the hhold; a pd.DataFrame;
        ------------------------------------------------
            4 rows: n, e, a, s
                n   = non-cognitive skills;
                e   = monetary investments;
                a   = time investments - alone;
                s   = time investments - shared;
            2 columns: 2a, 2b
                2a = the intercept;
                2b = the slope;
                
        f1: cognitive technology when one child lives in the hhold; a pd.DataFrame;
        --------------------------------------------
            it has 2 columns: 1a, 1b
            and 3 rows: k, e, a
            1a  = the intercept;
            1b  = the slope in age;
            k   = cognitive skills;
            e   = monetary investments;
            a   = time investments;
        
        g1: non-cognitive technology when one child lives in the hhold; a pd.DataFrame;
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
        # Set S, M, and beta;
        ########################################################
        
        # set the spacing;
        assert S <= 4
        self.S = S
        
        # set the lenght of development;
        self.M = 17
        
        # set the discount factor;
        self.beta = 0.95
        
        # alias M and beta;        
        M = self.M
        beta = self.beta
        T = M+S
        
        ########################################################
        # Set the preferences;
        ########################################################
        
        self.u = u
        
        ########################################################
        # Set the cognitive elasticities at each age;
        ########################################################
        
        # _i1 indicates periods that the first child lives in the hhold alone;
        # _i2 indicates periods that both children live in the hhold;
        # t denotes the time in the model; also first child's age;
        # v denotes second child's age;
       
        _i1 = np.arange(T) < S
        _i2 = ( S<=np.arange(T) ) & ( np.arange(T)<M )   
        
        self.delta = pd.DataFrame(np.arange(T),columns=['t'])   
        self.delta['v'] = self.delta['t']-np.ones(T)*S   

        # elasticities for the periods that child lives in the hhold alone;
        for key in f1.index :  
            self.delta[key+'1']  = np.exp(f1.loc[key,'1a'] + f1.loc[key,'1b']*self.delta['t']) * _i1 
            self.delta[key+'2']  = np.exp(f1.loc[key,'1a'] + f1.loc[key,'1b']*self.delta['v']) * np.flipud(_i1) 
        
        self.delta['s1'] = 0
        self.delta['s2'] = 0
        # elasticities for the periods that both children live in the hhold;
        for key in f2.index :  
            self.delta[key+'1'] += np.exp(f2.loc[key,'2a'] + f2.loc[key,'2b']*self.delta['t']) * _i2
            self.delta[key+'2'] += np.exp(f2.loc[key,'2a'] + f2.loc[key,'2b']*self.delta['v']) * _i2 
        
        # elasticities for the periods that child doesn't live in the hhold;
        self.delta['k1'] += np.flipud(_i1)  
        self.delta['k2'] += _i1 
        
        del self.delta['v'] 
        self.delta.insert(1,'spacing',S)
        
        ########################################################
        # Set the non-cognitive elasticities at each age;
        ########################################################
        
        # identitical to cognitive elasticities; see above for comments;
        self.eta = pd.DataFrame(np.arange(T),columns=['t'])
        self.eta['v'] = self.eta['t']-np.ones(T)*S
        for key in g1.index :
            self.eta[key+'1']  = np.exp(g1.loc[key,'1a'] + g1.loc[key,'1b']*self.eta['t']) * _i1 
            self.eta[key+'2']  = np.exp(g1.loc[key,'1a'] + g1.loc[key,'1b']*self.eta['v']) * np.flipud(_i1) 
        
        self.eta['s1'] = 0
        self.eta['s2'] = 0
        for key in g2.index :
            self.eta[key+'1'] += np.exp(g2.loc[key,'2a'] + g2.loc[key,'2b']*self.eta['t']) * _i2
            self.eta[key+'2'] += np.exp(g2.loc[key,'2a'] + g2.loc[key,'2b']*self.eta['v']) * _i2
        
        self.eta['n1'] += np.flipud(_i1)
        self.eta['n2'] += _i1
        
        del self.eta['v']
        self.eta.insert(1,'spacing',S)
        
        ########################################################
        # Compute the optimal share of resources for each use;
        ########################################################
        
        # compute the CV of future flows of investments at each period;
        seq= np.cumsum(np.tri(T,k=-1),axis=0)
        I1 = (seq != 0)*((_i1+_i2).reshape(1,T))
        I2 = (seq != 0)*((np.flipud(_i1)+_i2).reshape(1,T))
        
        # future flows for child 1;
        x1 = np.cumprod( self.delta['k1'].values.reshape(T,1)*I1+(1-I1) , axis=0)*I1
        x1 = np.sum(x1*(beta**seq), axis=0)
        y1 = np.cumprod( self.eta['n1'].values.reshape(T,1)*I1+(1-I1) , axis=0)*I1
        y1 = np.sum(y1*(beta**seq), axis=0)
        
        # future flows for child 2;
        x2 = np.cumprod( self.delta['k2'].values.reshape(T,1)*I2+(1-I2) , axis=0)*I2
        x2 = np.sum(x2*(beta**seq), axis=0)
        y2 = np.cumprod( self.eta['n2'].values.reshape(T,1)*I2+(1-I2) , axis=0)*I2
        y2 = np.sum(y2*(beta**seq), axis=0)
                       
        # compute the shares of hhold resources allocated to each choice;
        self.phi = pd.DataFrame(np.arange(T),columns=['t'])       
        self.shares = pd.DataFrame(np.arange(T),columns=['t'])   
        self.shares.insert(1,'spacing',S)

        # compute the lifetime marginal benefit of each choice;
        self.phi['c']   = u['alpha_1']* np.ones(T)
        self.phi['e1']  = u['alpha_3']*self.delta['e1']*(1+x1)
        self.phi['e1'] += u['alpha_4']*self.eta['e1']*(1+y1)
        self.phi['e2']  = u['alpha_5']*self.delta['e2']*(1+x2)
        self.phi['e2'] += u['alpha_6']*self.eta['e2']*(1+y2)
        self.phi['l']   = u['alpha_2']* np.ones(T)
        self.phi['a1']  = u['alpha_3']*self.delta['a1']*(1+x1)
        self.phi['a1'] += u['alpha_4']*self.eta['a1']*(1+y1)
        self.phi['a2']  = u['alpha_5']*self.delta['a2']*(1+x2)
        self.phi['a2'] += u['alpha_6']*self.eta['a2']*(1+y2)
        self.phi['s']   = u['alpha_3']*self.delta['s1']*(1+x1) 
        self.phi['s']  += u['alpha_4']*self.eta['s1']*(1+y1) 
        self.phi['s']  += u['alpha_5']*self.delta['s2']*(1+x2)
        self.phi['s']  += u['alpha_6']*self.eta['s2']*(1+y2)
        
        # share of household's lifetime income (PV) allocated to c, e1, and e2;
        for key in ['c','e1','e2']:
            self.shares[key] = self.phi[key] / self.phi[['c','e1','e2']].sum(axis=1).sum()
        
        # share of household's periodic time endowment allocated to l, a1, a2, and s;
        for key in ['l','a1','a2','s']:
            self.shares[key] = self.phi[key] / self.phi[['l','a1','a2','s']].sum(axis=1)

###############################################################################
# "Encapsulate" stacks instants of hholds with different birth spacings;
###############################################################################

class encapsulate:
    """
    stacks instants of hhold with s={1,2,3,4}, where s is the Spacing;
    """
    
    def __init__(self,params2,params1):
        
        self.delta=pd.DataFrame()
        self.eta=pd.DataFrame()
        self.shares=pd.DataFrame()
    
        for spacing in [1,2,3,4]:
            temp = hhold(spacing,*pmap(params2,params1))
            self.delta = self.delta.append(temp.delta,ignore_index=True)
            self.eta = self.eta.append(temp.eta,ignore_index=True)
            self.shares = self.shares.append(temp.shares,ignore_index=True)
            
##############################################################################;
# "GMM" computes the GMM objective function value;
##############################################################################;

def gmm(params2,data,params1):
    """ 
    the Simulated Method of Moments objective function.
    
    inputs
    -------
    params2: a vector of parameters to be estimated; np.array;
    params1: a vector of parameters to be taken as given; np.array;
    
    data: the data; pd.DataFrame; 
    its columns are: 
        t         = period and 1st child's age;
        spacing   = age difference between children;
        inc       = household's lifetime income;
        k1, k2    = cognitive score at time t;
        n1, n2    = non-cognitive score at time t;
        k1p, k2p  = cognitive score at time t+5;
        n1p, n2p  = non-cognitive score at time t+5;
        a1, a2, s = parental time;
        
    output
    ------
    GMM objective function value
    """
    
    # stack instants of hhold
    model = encapsulate(params2,params1)
    obj = moments(model,data)
    print(obj)
    return obj
            
###############################################################################
# "pmap" maps a vector of parameters into u, f2, g2, f1, and g1;
###############################################################################

def pmap(params2,params1):
    """
    maps the parameters into the preference weights and technology elasticities;
    
    inputs
    -------
    params2: 20-element np.array;
    params1: 12-element np.array;

    outputs
    -------
    u: preferences; pd.Series
    f2: cognitive technology when both children live in hhodl; pd.DataFrame
    g2: non-cognitive technology when both children live in hhold; pd.DataFrame
    f1: cognitive technology when one child live in hhold; pd.DataFrame
    g1: non-cognitive technology when one child live in hhold; pd.DataFrame
    """
  
    f1 = pd.DataFrame(params1[0:6].reshape(3,2), index=['k','e','a'], columns=['1a','1b'])
    g1 = pd.DataFrame(params1[6:12].reshape(3,2), index=['n','e','a'], columns=['1a','1b'])
    f2 = pd.DataFrame(params2[0:8].reshape(4,2), index=['k','e','a','s'], columns=['2a','2b'])
    g2 = pd.DataFrame(params2[8:16].reshape(4,2), index=['n','e','a','s'], columns=['2a','2b'])
    u = pd.Series(np.exp(params2[16:19]),index=['alpha_2', 'alpha_3', 'alpha_5'])
    zeta = np.exp(params2[19])/(1+np.exp(params2[19]))
    
    # impose the preference restrictions;
    u /= (1+u.sum())
    u['alpha_3'] , u['alpha_4'] = u['alpha_3']*zeta, u['alpha_3']*(1-zeta)
    u['alpha_5'] , u['alpha_6'] = u['alpha_5']*zeta, u['alpha_5']*(1-zeta)
    u['alpha_1'] = 1-u.sum()
    u = u.reindex(sorted(u.index))
  
    return (u,f2,g2,f1,g1)

###############################################################################
# "moments" computes moments from data and model;      
###############################################################################

def moments(model,data):
    """
    computes moments from the model and the data;
    
    inputs
    ------
    model: a stack of instants of hholds;
    data: check gmm's docstring;
    
    outputs
    -------
    obj: gmm's objective function value;
    mom: a list of moments from the model and the data;
    """
    
    dat = data.copy()
    sim = simulate(model,dat[['id','t','spacing','inc','k1','n1','k2','n2']]) 
    
    dat['gt']=dat['t'].map(lambda x: x if x%2!=0 else x-1)
    dat['tp']=dat['t']-dat['spacing']
    dat['gp']=dat['tp'].map(lambda x: x if x%2!=0 else x-1)
    
    sim['gt']=sim['t'].map(lambda x: x if x%2!=0 else x-1)
    sim['tp']=sim['t']-sim['spacing']
    sim['gp']=sim['tp'].map(lambda x: x if x%2!=0 else x-1)
    
    # compute moments;
    mom = list()
    mom.append(dat.groupby(by=['t']).mean()[['a1','k1p','n1p']]) 
    mom.append(sim.groupby(by=['t']).mean()[['a1','k1p','n1p']])
    mom.append(dat.groupby(by=['tp']).mean()[['a2','s','k2p','n2p']])
    mom.append(sim.groupby(by=['tp']).mean()[['a2','s','k2p','n2p']])
    mom.append(dat.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['k1p']].unstack())
    mom.append(sim.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['k1p']].unstack())
    mom.append(dat.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['n1p']].unstack())
    mom.append(sim.groupby(by=['gt']).cov().loc[pd.IndexSlice[:,['inc']],['n1p']].unstack())
    mom.append(dat.groupby(by=['gp']).cov().loc[pd.IndexSlice[:,['inc']],['k2p']].unstack())
    mom.append(sim.groupby(by=['gp']).cov().loc[pd.IndexSlice[:,['inc']],['k2p']].unstack())
    mom.append(dat.groupby(by=['gp']).cov().loc[pd.IndexSlice[:,['inc']],['n2p']].unstack())
    mom.append(sim.groupby(by=['gp']).cov().loc[pd.IndexSlice[:,['inc']],['n2p']].unstack())
    
    obj  = (((mom[0] - mom[1])**2)/mom[0].max()).dropna().sum().sum()
    obj += (((mom[2] - mom[3])**2)/mom[2].max()).dropna().sum().sum()
    obj += (((mom[4] - mom[5])**2)/mom[4].max()).dropna().sum().sum()
    obj += (((mom[6] - mom[7])**2)/mom[6].max()).dropna().sum().sum()
    obj += (((mom[8] - mom[9])**2)/mom[8].max()).dropna().sum().sum()
    obj += (((mom[10] - mom[11])**2)/mom[10].max()).dropna().sum().sum()
    
    return obj

##############################################################################;

##############################################################################;

def simulate(model,data):   
    """
    simulate takes state variables at time t and returns choices at time t 
    and state values at time t+5;

    inputs
    ------
    periods: number of periods ahead
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
        sim = sim.merge(model.shares[['t','spacing','a1','e1','a2','e2','s']], how='left', on=['t','spacing'])
        sim['e1'] *= sim['inc']*10*(17+sim['spacing']) / 52
        sim['e2'] *= sim['inc']*10*(17+sim['spacing']) / 52
        sim[['a1','a2','s']] *= 7*24
        
        # get the elasticities;        
        pf = sim[['t','spacing']].merge(model.delta, how='left', on=['t','spacing'])
        pn = sim[['t','spacing']].merge(model.eta, how='left', on=['t','spacing'])

        # compute the next period's skills;
        sim['k1'] = (sim['k1']**pf['k1'])*(sim['e1']**pf['e1'])*(sim['a1']**pf['a1'])*(sim['s']**pf['s1'])
        sim['n1'] = (sim['n1']**pn['n1'])*(sim['e1']**pn['e1'])*(sim['a1']**pn['a1'])*(sim['s']**pn['s1'])
        sim['k2'] = (sim['k2']**pf['k2'])*(sim['e2']**pf['e2'])*(sim['a2']**pf['a2'])*(sim['s']**pf['s2'])
        sim['n2'] = (sim['n2']**pn['n2'])*(sim['e2']**pn['e2'])*(sim['a2']**pn['a2'])*(sim['s']**pn['s2'])
        sim.drop(['a1','e1','a2','e2','s'],axis=1,inplace=True) 
        
        
    # clean the simulated data;
    sim.rename_axis(axis=1,mapper={'k1':'k1p', 'n1':'n1p', 'k2':'k2p', 'n2':'n2p'},inplace=True)
    sim['t'] -= 4
    
    # get the choices at t;
    sim = sim.merge(model.shares[['t','spacing','a1','a2','s']], how='left',on=['t','spacing'])
    sim[['a1','a2','s']] *= 7*24
    
    # add the skills at t;
    sim = sim.merge(data[['id','k1','k2','n1','n2']],on=['id'])

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
        axes = dgp.u.plot(kind='bar', rot=0, color=['black']*6) 
    else:
        frame = pd.DataFrame({'estimate':dgp.u,'true':dgp_t.u,'initial':dgp_i.u},columns=['estimate','true','initial'])
        axes = frame.plot(kind='bar', color=['black','dimgray','darkgray'])

    mapper = dict(alpha_1 = r'$\alpha_1$', 
                  alpha_2 = r'$\alpha_2$', 
                  alpha_3 = r'$\alpha_3\zeta$', 
                  alpha_4 = r'$\alpha_3(1-\zeta)$', 
                  alpha_5 = r'$\alpha_4\zeta$', 
                  alpha_6 = r'$\alpha_4(1-\zeta)$')
    axes.set_xticklabels([mapper[x] for x in dgp.u.index], rotation='horizontal')



def fig_e(dgp,dgp_t=None,dgp_i=None,skill='k',child='1'):
    """
    Line plot of a DGP elasticities;

    inputs
    ------
    dgp:   estimated DGP; an instant of hhold;
    dgp_t: true DGP; an instant of hhold;
    dgp_i: initial DGP; an instant of hhold;
    skill: 'k' for cognitive or 'n' for non-cognitive;
    child: '1' for the first or '2' for the second child;
    
    outputs
    -------
    Line plot of elasticities;
    """
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.7)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    axes = [ax1, ax2, ax3, ax4]

    elas = 'delta' if skill == 'k' else 'eta'
    x = 't' if child == '1' else dgp.__dict__[elas]['t']-dgp.__dict__[elas]['spacing']
    y = [skill+child] + [item+child for item in ['e','a','s']]
    xlabels = ['', '', 'age', 'age']
    titles = ['Initial Skills','Monetary Investments','Alone Time Investments','Shared Time Investments']
    graph = dict(x=x, y=y, ax=axes, xlim=[0,dgp.M-1], legend=False, subplots=True, rot=0)
    if dgp_t is not None:
        dgp_t.__dict__[elas].plot(color='dimgray',ls='solid',**graph)
        dgp_i.__dict__[elas].plot(color='darkgray',ls='dotted',**graph)
    dgp.__dict__[elas].plot(color='black',ls='dashed',**graph)
    if dgp_t is not None:
        ax1.legend((ax1.lines[2],ax1.lines[0],ax1.lines[1]),('estimate','true','initial'),loc=2,framealpha=0.5)
    for ax, xlabel, title in zip(axes,xlabels,titles):
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        
