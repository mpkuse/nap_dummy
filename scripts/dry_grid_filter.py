""" Dry Grid Filter - Efficient
        Loads the likelihoods from file (sim_scores.dat) to test the grid filter
        move() -> happens with np.roll() and np.convolve() to account for uncertainity
        weights at every locations maintained as an simple array

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 29nd Mar, 2017
"""
import numpy as np
# import matplotlib.pyplot as plt
import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore
import time
import code
from copy import deepcopy

#
import TerminalColors
tcolor = TerminalColors.bcolors()

# from ParticleBelief import ParticleBelief
# from ParticleBelief import Particle

def debug( msg ):
    print tcolor.OKBLUE, '[DEBUG]',msg, tcolor.ENDC



## 'x' can also be a vector
def logistic( x ):
    y = np.array(x)
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)


## COnverts a space-separated string to array
def line_2_array( l ):
    l = [ float(li) for li in l.split()]
    return l


# sim file open
fp_sim_scores = open( 'sim_scores.dat', 'r' )
F = fp_sim_scores.readlines()



# Init Grid FIlter
w = np.zeros( 1000 )
w[0:80] = 1
w = w / sum(w)

# plt.ion()
qapp = pg.mkQApp()
win = pg.GraphicsWindow()
plot1 = win.addPlot()
curve1 = plot1.plot()
plot2 = win.addPlot()
curve2 = plot2.plot()
plot3 = win.addPlot()
curve3 = plot3.plot()

plot1.setRange( xRange=[0,len(F)], yRange=[0,1] )
plot2.setRange( xRange=[0,len(F)], yRange=[0,1] )
plot3.setRange( xRange=[0,len(F)], yRange=[0,1] )

for i in range(70,len(F)):
    #F[i] is likelihoods
    sim_scores = line_2_array( F[i] )
    sim_scores_logistic = logistic( sim_scores )

    print 'len(sim_scores)', len(sim_scores)


    # Sense and Update
    startSenseTime = time.time()
    L = len(sim_scores_logistic )
    w[0:L] = np.multiply( w[0:L], 2.0*sim_scores_logistic[0:L]  )
    w[0:L] = np.maximum( w[0:L], 0.001 )
    w = w / sum(w)
    print 'Time for likelihood x prior : %4.2f ms' %(1000.*(time.time() - startSenseTime))



    # Move
    startMoveTime = time.time()
    w = np.roll( w, 1 )
    w[0] = w[1]
    w = np.convolve( w, [0.025,0.1,0.75,0.1,0.025], 'same' )
    print 'Time for move : %4.2f ms' %(1000. * (time.time()-startMoveTime))


    curve1.setData( range(len(sim_scores)), sim_scores )
    curve2.setData( range(len(sim_scores_logistic)), sim_scores_logistic )
    curve3.setData( range(len(w)), w )
    qapp.processEvents()
    time.sleep(0.1)


    # # Plot scatter
    # debug( 'plot' )
    # plt.clf()
    # plt.subplot(411)
    # plt.axis( [0, 700, 0, 1])
    # plt.bar( range(len(sim_scores)), sim_scores )
    # plt.title( 'sim_scores at every frame_index' )
    #
    # plt.subplot(412)
    # plt.axis( [0, 700, 0, 1])
    # plt.bar( range(len(sim_scores)), sim_scores_logistic )
    # plt.title( 'logistic(sim_scores) at every frame_index viz. likelihood' )
    #
    # plt.subplot(413)
    # plt.axis( [0, 700, 0., 0.3])
    # # plt.scatter( x=[p.loc for p in pb ], y=np.zeros(len(pb)), s=1000.0*np.array([p.w for p in pb ]), color='red' )
    # plt.bar( range(len(w)), w  )
    # plt.title( 'weights at each point (grid)')


    # plt.show(False)
    # plt.pause(0.01)
    # code.interact( local=locals() )
