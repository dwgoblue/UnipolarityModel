##########################Packages############################

import numpy as np
import scipy
from scipy import linalg
from numpy.linalg import inv
from functools import partial
import sys
import time
import multiprocessing as mp
import _pickle as cPickle
#import matplotlib.pyplot as plt

###########################Functions###################################

# Matrix for solving CrankNicolson
def HelperFunc(D, dt, dx, U0):
    n = U0.shape[0]
    r = D*dt/(dx**2)
    n = n+2
    W = np.zeros((n, n))
    V = np.zeros((n, n+2))
    W[0, 0] = 2*(1+r/3)
    W[0, 1] = (-2*r/3)
    W[n-1, n-1] = 2*(1+r/3)
    W[n-1, n-2] = (-2*r/3)
    V[0, 0] = (r)
    V[0, 1] = 2*(1-r)
    V[0, 2] = (r)
    V[n-1, n+2-1] = (r)
    V[n-1, n+2-2] = 2*(1-r)
    V[n-1, n+2-3] = (r)
	
    for i in range(1, n-1):
        W[i, i-1] = -r
        W[i, i] = 2*(1+r)
        W[i, i+1] = -r
        V[i, i+1-1] = r
        V[i, i+1] = 2*(1-r)
        V[i, i+1+1] = r
        
    return inv(W), V

# PDE numerical tool
def CrankNicolson(intW, V, Ut):
	Utmp = np.append(np.array([Ut[0], Ut[0]]), Ut, axis=0)
	Utmp = np.append(Utmp, np.array([Ut[-1], Ut[-1]]), axis=0)
	Utt = np.matmul(V, Utmp)
	Utt = np.matmul(intW, Utt)

	return Utt[1:-1]-Ut



# Droplet constitutes
def dxdt(binding, dt, C, B, n, p, kH, ks, kp, kd, kb, L, x, u, m):
    if binding=='ON':
        # Position-dependent functions.
        if x == 0:
            udot = kp-kd*u-2*((u**2)*(1-m/p)*(B+C*m**n/(kH**n+m**n))-ks*m)
            mdot = (u**2)*(1-m/p)*(B+C*m**n/(kH**n+m**n))-ks*m
        elif x==9:
            udot = -kd*u-2*((u**2)*(1-m/p)*(B+C*m**n/(kH**n+m**n))-ks*m)
            mdot = (u**2)*(1-m/p)*(B+C*m**n/(kH**n+m**n))-ks*m
        else:
            udot = -kd*u-2*(-ks*m)
            mdot = -ks*m
    else: #OFF
    # Without membrane binding effect
        udot = kp*np.exp(-x/(L/10))-kd*u
        mdot = 0
        
    return dt*udot, dt*mdot


def main(para):
    
    # Steady state values for m and u.
    mss = para[2]
    uss = para[3]
    
    # Global parameter setting
    v = 1 #1 µm3 for ecoli
    D = 1 #um2/s, diffusion coeff
    size = 2 #2um ecoli
    L = 10 #grids.
    C = 10*15.85 # fraction of cytoplasmic constitutes.
    p = 50
    n = 6
    dx = size/L #unit um
    dt = 10**(-3) # step size, unit: s
    steps = 3600*(10**3) #total time: 60 mins
    # Reaction parameter setting
    kd = para[0]
    kp = kd*uss
    ks = para[1]
    B = ks*(mss/(uss**2))
    kH = 1
    
    # Initialize ut, mt which should be a Lx1 array.
    ut = np.zeros((L, 1))
    mt = np.zeros((L, 1))
    
    # Initialize diffusion matrix.
    intW, V = HelperFunc(D, dt, dx, ut[:, 0])
    
    # Initial condition
    ut += 0.8
    mt += 0.7
    
    # Output containers for all time points.
    ud = np.zeros(steps+1)
    md = np.zeros(steps+1)
    uU = np.zeros((L, steps+1))
    mU = np.zeros((L, steps+1))
    uU[:, 0] = ut[:, 0]
    mU[:, 0] = mt[:, 0]
    
    # Run simulation.
    for step in range(steps):
        if step%10000==0:
            # show progress.
            print('step',step)
        # Temporary container
        ut_tmp = np.zeros((L, 1))
        mt_tmp = np.zeros((L, 1))
        # Diffusion
        Diff_u = CrankNicolson(intW, V, ut[:, 0])
        for i in range(L):
            # Initiate reactions
            _RK_ = dxdt('ON', dt, C, B, n, p, kH, ks, kp, kd, kb[i], L, i, ut[i, 0], mt[i, 0])
            ut_tmp[i, 0] = ut[i, 0]+_RK_[0]+Diff_u[i]
            mt_tmp[i, 0] = mt[i, 0]+_RK_[1]
            
        # If there is any negative value in the arrays.
        if np.any(ut_tmp<0) or np.any(mt_tmp<0) or np.any(
            np.isnan(ut_tmp)) or np.any(np.isnan(mt_tmp)):
            print('wrong!')
            print('ut', ut_tmp)
            print('mt', mt_tmp)
            print(step)
            #return {'u':ut, 'm':mt} # last time point.
            #return (ut[0, 0], mt[0, 0], ut[-1, 0], mt[-1, 0], step) # only the first and the last grids.
            return {'u':uU, 'm':mU} # All time points.
            sys.exit()
        
        ut = ut_tmp
        mt = mt_tmp
        ud[step+1] = (ut[0, 0]+mt[0, 0])/(ut[-1, 0]+mt[-1, 0])
        uU[:, step+1] = ut[:, 0]
        mU[:, step+1] = mt[:, 0]
        
    #return {'u':ud}
    #return {'u':ut, 'm':mt} # last time point.
    return {'u':uU, 'm':mU} # All time points.
    #return (ut[0, 0], mt[0, 0], ut[-1, 0], mt[-1, 0], step)  # only the first and the last grids.
	
"""
The functions below are not necessary. Execute the functions only when you plan to scan parameters by parallel computating.
"""
	
def async_multicore(main, paras, size):
    pool = mp.Pool()    # Open multiprocessing pool
    result = []
    para_save = np.array([[]]).reshape(0, size)
    #do computation
    for para in paras:
        res = pool.apply_async(main, args=(para,))
        result.append(res)
        para_reshape = para.reshape(1, size)
        para_save = np.concatenate((para_save, para_reshape), axis=0)
    pool.close()
    pool.join()

    output_dict = {}
    output_dict['Results'] = result
    output_dict['Parameters'] = para_save

    return output_dict

def export_results(output_dict):
    res = output_dict['Results']
    para_save = output_dict['Parameters']
    obj = {}
    ind = 0
    for data, paras in zip(res, para_save):
        obj['Data No. {0}'.format(ind)] = {'kd':paras[0], 'ks':paras[1],
                                           'Data':data.get()}
        ind +=1
    with open('[scan][kd][ks][10+3mss]Morpho_result.pickle', 'wb') as picklefile:
        cPickle.dump(obj, picklefile, True)

if __name__ == '__main__':
    para_size = 10
    kd = np.logspace(1, 5, para_size)
    ks = np.logspace(-2, -6, para_size)
    para = np.meshgrid(kd, ks)
    tmp = np.concatenate((np.array([para[0].flatten()]),
                          np.array([para[1].flatten()])), axis=0)
    paras = np.rot90(tmp)
    
    cpus = mp.cpu_count()
    print('Opening {0} cpus for simulation...'.format(cpus))
    print('Preparing parameter sets...')
    print('Preparing simulation space...')
    print('Loading all simulations...')
    holder = partial(main,)
    start_time = time.time()
    output_dict = async_multicore(holder, paras, 2)
    end_time = time.time()
    print('Finished all simulations with {0} sec'.format(end_time - start_time))
    print('Saving data...')
    export_results(output_dict)
    print('Finishing program')



