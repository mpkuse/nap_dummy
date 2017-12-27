""" Script to visualize vote_bank """

import numpy as np
import matplotlib.pyplot as plt

PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

A = np.load( PKG_PATH+'/vote_bank.npy' )
plt.imshow( A, cmap='hot' )
plt.show()
