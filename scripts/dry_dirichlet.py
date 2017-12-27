""" Composing Logistic similarity
        This model assumes if `(A)--0.90--(B)` and `(A)--0.5--(C)` then `(B)--0.9*0.5--(C)`
        This composing assumption is used to propogate prior belief with observations

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 28th Mar, 2017
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'GTKAgg' )
import time
import code
import TerminalColors
tcolor = TerminalColors.bcolors()

def debug( msg ):
    print tcolor.OKBLUE, '[DEBUG]',msg, tcolor.ENDC

## COnverts a space-separated string to array
def line_2_array( l ):
    l = [ float(li) for li in l.split()]
    return l

## 'x' can also be a vector
def logistic( x ):
    y = np.array(x)
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)


# sim file open
fp_sim_scores = open( 'sim_scores.dat', 'r' )
F = fp_sim_scores.readlines()

plt.ion()
prev_sim_scores = line_2_array( F[0] )
prev_sim_scores_logistic = logistic( prev_sim_scores )
for i in range(1, 700):
    startTime = time.time()
    sim_scores = line_2_array( F[i] )
    sim_scores_logistic = logistic( sim_scores )


    prior_ = np.hstack( (prev_sim_scores_logistic, [1]) ) * sim_scores_logistic[-1]
    posterior_ = np.multiply( 1.6*sim_scores_logistic , prior_ ) #point-wise multiply
    posterior_ = np.maximum( posterior_, 0.01 )
    plot_posterior = [1.0  if x>0.01 else 0.0 for x in posterior_ ]

    print len(prev_sim_scores_logistic), len(sim_scores)
    prev_sim_scores_logistic = posterior_# sim_scores_logistic
    print 'Time Taken : %4.2f ms' %( 1000.*(time.time() - startTime))

    # Plot observation and posterior_
    plt.clf()
    plt.subplot(311)
    plt.axis( [0,700, 0, 1])
    plt.bar( range(len(sim_scores_logistic)), sim_scores_logistic )
    plt.title( 'sim_scores_logistic')

    plt.subplot(312)
    plt.axis( [0,700, 0, 1])
    plt.bar( range(len(prior_)), prior_ )
    plt.title( 'prior by composing' )

    plt.subplot(313)
    plt.axis( [0,700, 0, 10])
    plt.bar( range(len(plot_posterior)), plot_posterior )
    plt.title( 'posterior. multiply prior with current observation')

    plt.show(False)
    plt.pause(0.001)
    # code.interact( local=locals() )
