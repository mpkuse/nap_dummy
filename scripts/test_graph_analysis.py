""" Analysis the graph created by nap_geom_node.py

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 14th May, 2017
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import code

def gps_layout2d( G, gps_t, gps_x, gps_y ):
    pos = {}
    for node_id in G.node:
        # print node_id, ': ', G.node[node_id]['time_stamp']
        time = long(G.node[node_id]['time_stamp'])
        m = abs(gps_t - time).argmin()
        # print gps_x[m], gps_y[m]
        pos[node_id] = [gps_x[m], gps_y[m]]
        #code.interact( local=locals() )
    return pos



# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

gps_t, gps_x, gps_y, gps_z = np.loadtxt( PKG_PATH+'/DUMP/GPS_track.csv', dtype={'names':('t','x','y','z'), 'formats':('i8', 'f4', 'f4', 'f4') }, delimiter=',', unpack=True)

G = nx.read_gexf( PKG_PATH+'/DUMP/Graph_head_nodes.gexf' )

pos1 = gps_layout2d( G, gps_t, gps_x, gps_y )
pos = nx.circular_layout( G )
nx.draw_networkx( G, pos1, font_size=10, width=0.5 )
plt.show()
