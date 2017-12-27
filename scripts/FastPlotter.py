""" A pyqtgraph plotter

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 12th May, 2017
"""

import numpy as np
import pyqtgraph as pg


class FastPlotter:
    ## Init plotter with `n` subplots
    def __init__(self, n, win_width=1200, win_height=270):
        self.qapp = pg.mkQApp()
        self.win = pg.GraphicsWindow()
        self.win.resize( 1200, 270 )

        self.plot = []
        self.curve = []
        for i in range(n):
            print 'Create plot ', i
            self.plot.append( self.win.addPlot() )
            self.curve.append( self.plot[i].plot() )

    ## Set Range
    def setRange( self, plot_index, xRange=None, yRange=None):
        self.plot[plot_index].setRange( xRange=xRange, yRange=yRange )

    ## Does plot(X,Y) at the specified plot
    def set_data( self, plot_index, X, Y, title=None):
        self.curve[plot_index].setData( X, Y )
        if title is not None:
            self.plot[plot_index].setTitle( title )


    ## spin()
    def spin(self):
        self.qapp.processEvents()
