""" Dry Particle Filters
        Loads the likelihoods from file (sim_scores.dat) to test the particle filter class

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 22nd Mar, 2017
"""
import numpy as np
import matplotlib.pyplot as plt
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore
import time
import code
from copy import deepcopy

#
import TerminalColors
tcolor = TerminalColors.bcolors()

# from ParticleBelief import ParticleBelief
from ParticleBelief import Particle

def debug( msg ):
    print tcolor.OKBLUE, '[DEBUG]',msg, tcolor.ENDC

def _dist_to_sigma(  d ):
    # when d=0   ---> sigma=0.1 (a)
    # when d=0.3 ---> sigma=5   (b)
    a = 1.5
    b = 20
    return (a + np.multiply( (b-a)/0.3 , d ))/150.0

def gaussian( x, mu, sigma_2 ):
    if sigma_2 <= 0:
        print 'ERROR : gassian : sigma is negative, not allowed'
    if abs(x-mu) > 2:
        return 0
    denom = np.sqrt( 2*np.pi*sigma_2 )
    return np.exp( -np.power(x-mu, 2.) / (2*sigma_2)  ) / denom

def print_sim_score_sigmas( sim_scores ):
    print tcolor.OKGREEN, 'print_sim_score_sigmas=[',
    for s in sim_scores:
        print _dist_to_sigma(s),
    print tcolor.ENDC

def cumulated_gaussians( sim_scores ):
    #sim_scores can be converted to sigmas. their index are the `mu`
    u = np.linspace( 0, 200, 600 )
    v = []
    for i in range(len(u)):
        #cummulate gaussian evaluated at u[i]
        s = 0
        for mu,sigmas_2 in enumerate(_dist_to_sigma(sim_scores)):
            s += 10.*(1.0-sim_scores[mu])*gaussian(u[i],mu,sigmas_2)
        v.append(s/mu)
    return u, v


## 'x' can also be a vector
def logistic( x ):
    y = np.array(x)
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)


## COnverts a space-separated string to array
def line_2_array( l ):
    l = [ float(li) for li in l.split()]
    return l

# ## Similarity score to sigma linear mapping
# def dist_to_sigma(  d ):
#     # when d=0   ---> sigma=0.1 (a)
#     # when d=0.3 ---> sigma=5   (b)
#     a = 1.5
#     b = 10
#     return a + (b-a)/0.3 * d
#
# ## Gaussian centered at mu, evaluated at x
# def gaussian( x, mu, sigma ):
#     # Approximation
#     b = sigma
#     if abs(x-mu) > 5*b:
#         return 0.0
#     c = -abs(x-mu)/b
#     E_2_2 = (1 + 1./2.*c + 1./12.*c*c) / (1 - 1./2.*c + 1./12.*c*c)
#     return 1.0/(2.0*b) * E_2_2
#
#     # Exact
#     if sigma <= 0:
#         print 'ERROR : gassian : sigma is negative, not allowed'
#     denom = np.sqrt( 2*np.pi*np.power(sigma,2.))
#     return np.exp( -np.power(x-mu, 2.) / (2*np.power(sigma,2.))  ) / denom
#
#
# def sum_of_gaussians( sim_scores, u ):
#     v = []
#     # start = time.time()
#     for loc in u:
#         j = 0
#         for i,f in enumerate(sim_scores):
#             f_f = float(f)
#             j = j + gaussian( loc, i, dist_to_sigma(f_f) )
#             # j = j + laplacian( loc, i, dist_to_sigma(f_f) )
#             # print 'mu=%f' %(i), 'sigma=%f' %(dist_to_sigma(f_f) )
#         v.append(j/i) #here i represents #of gaussian count
#     # print '%4.2f Done sum_of_gaussians' %( (time.time() - start)*1000. )
#     return v

# sim file open
fp_sim_scores = open( 'sim_scores.dat', 'r' )
F = fp_sim_scores.readlines()



# Init Particle FIlter / Monte-Carlo localization
pb = []
# w = []
# Init 50 particles
debug( 'Setting 50 particles')
for i in range(50):
    pb.append( Particle(loc=np.random.uniform(0,80), wt=1.0/50.0) )
    # w.append(1.0/50.0)
print pb



plt.ion()
for i in range(70,len(F)):
    print 'Particles : ', pb
    #F[i] is likelihoods
    sim_scores = line_2_array( F[i] )
    sim_scores_logistic = logistic( sim_scores )




    for k,p in enumerate(pb): #for each particle - `measurement_prob`. store probabilities into w[]
        p.w = p.w * p.measure_prob(sim_scores_logistic) #prior x likelihood

    sum_of_wts = sum( [p.w for p in pb])
    for p in pb:
        p.w = p.w / sum_of_wts
    debug( 'w <-- calculate probability measure' )



    for p in pb: #for each particle - move
        p.move()
    debug( 'move each particle by U=1' )



    # Plot scatter
    debug( 'plot' )
    plt.clf()
    plt.subplot(411)
    plt.axis( [0, 400, 0, 1])
    plt.bar( range(len(sim_scores)), sim_scores )
    plt.title( 'sim_scores at every frame_index' )

    plt.subplot(412)
    plt.axis( [0, 400, 0, 1])
    plt.bar( range(len(sim_scores)), sim_scores_logistic )
    plt.title( 'logistic(sim_scores) at every frame_index viz. likelihood' )

    plt.subplot(413)
    plt.axis( [0, 400, -0.1, 0.1])
    # plt.scatter( x=[p.loc for p in pb ], y=np.zeros(len(pb)), s=1000.0*np.array([p.w for p in pb ]), color='red' )
    plt.bar([p.loc for p in pb ], [p.w for p in pb ] )
    plt.title( 'Showing current particle locations as scatter plot. Size proportional to particle weights')



    #
    # plt.subplot(312)
    # # plt.set_xlim( [0,200] )
    # # plt.set_ylim( [-0.1,0.1] )
    # u, v = cumulated_gaussians( sim_scores )
    # plt.axis( [0, 200, -0.15, 0.15])
    # plt.plot( u, v )
    # plt.title( 'sim_scores as cummulated-gaussians')
    #
    # plt.subplot(313)
    # plt.bar( [p.loc for p in pb ], w )
    # plt.title( 'Current normalized-weights' )
    # plt.axis( [0, 200, -0.01, 0.1])




    #
    #
    ### AS w is now denoting pdf (posterior/prior for next iteration), need to rethink on resampling and assigning weights to new samples
    # # Re-sample particles

    debug( 'Resample')
    n_exploitation = 40
    n_exploration = 40

    p_resampled = []
    index = int( np.random.uniform()*50 )
    beta = 0.0
    mw = max([p.w for p in pb])
    for i in range(n_exploitation): # (how many new particles you want)
        beta += np.random.uniform(low=0, high=1) * 2.0 * mw
        while beta > pb[index].w:
            beta -= pb[index].w
            index = (index+1) % len(pb)
        # print 'pb[%d]' %(index)
        p_resampled.append( deepcopy(pb[index]) )
    # pb = p_resampled
    #
    sum_of_wts_resampled = sum( [p.w for p in p_resampled] )
    for px in p_resampled:
        px.w = px.w / sum_of_wts_resampled * 0.8
    #more uniform samples
    p_resampled_uniform = []
    for i in range(n_exploration):
        p_resampled_uniform.append( Particle(loc=np.random.uniform(0, len(sim_scores)), wt=.2/n_exploration) )
        p_resampled.append( Particle(loc=np.random.uniform(0, len(sim_scores)), wt=.2/n_exploration) )


    plt.subplot(414)
    plt.axis( [0, 400, -0.1, 0.1])
    plt.scatter( x=[p.loc for p in p_resampled ], y=np.zeros(len(p_resampled)), color='red' )
    plt.scatter( x=[p.loc for p in p_resampled_uniform ], y=np.zeros(len(p_resampled_uniform)), color='black' )
    # plt.scatter( x=[p.loc for p in pb ], y=np.zeros(len(pb)), s=100.0*np.array([1.0-p.w for p in pb ]), color='black'  )
    plt.title( 'resampled')



    plt.show(False)
    code.interact( local=locals() )
    pb = p_resampled
