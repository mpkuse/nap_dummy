""" Idea from Pedro's famous graph segmentation to be applied on my problem
    Think of all the scenes as island. The task being similar to image segmentation

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 16th May, 2017
"""
import numpy as np
import cv2
import networkx as nx
import code
import time
#
import TerminalColors
tcol = TerminalColors.bcolors()

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

## 'x' can also be a vector
def logistic( x ):
    #y = np.array(x)
    #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
    # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)
    if len(x) < 3:
        return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

    y = np.convolve( np.array(x), [0.25,0.5,0.25], 'same' )
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)

def number_of_groups( all_nodes ):
    return max( [n.gid for n in all_nodes] ) + 1

def get_group_struct( all_nodes ):
    n = number_of_groups( all_nodes )
    group_struct = []
    for i in range(n):
        print 'In group %d : ' %(i), [node.uid for node in all_nodes if node.gid == i]
        group_struct.append( [node.uid for node in all_nodes if node.gid == i] )
    return group_struct

def get_external_energy( group_struct, wt, wt_filt ):
    e_ext = []
    for group in group_struct:
        e_ext.append( min( wt[group] ) )
    return e_ext

class Node:
    def __init__(self, uid, gid=-1):
        self.uid = uid #node id
        self.gid = gid #group id

    def __str__(self):
        return '(uid=%d, gid=%d)' %(self.uid,self.gid)

    def __repr__(self):
        return '(uid=%d, gid=%d)' %(self.uid,self.gid)


#----- Load Data -----#
S_char = np.load( PKG_PATH+'/DUMP/'+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH+'/DUMP/'+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH+'/DUMP/'+'S_thumbnail.npy' )#N x 96 x 128 x 3
#-------- END --------#

all_nodes = []
bit_int_e = [] #internal energy of each group. len(bit_int_e) == len( uniq(bits) )
for i in range(S_word.shape[0]):
    #assume at ith stage, all previous S are available only.
    if i==0: #no edges from 0th node to previous, ie. no previous nodes
        all_nodes.append( Node(uid=0, gid=0) )
        bit_int_e.append( 0.0+0.5/1.0 )
        continue;

    startTime = time.time()

    #now there is atleast 1 prev nodes

    #compute dot product cost
    DOT_word = np.dot( S_word[0:i,:], S_word[i,:] )
    sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) )
    wt = 1.0 - logistic( sim_scores ) #measure of dis-similarity. 0 means very similar.

    # discard definite non-connections
    wt_filt = np.where( wt < 0.30 )[0] #these are indices of all prev nodes which has some similarity to current node
    print 'Potential Connections from %d to %s' %(i, all_nodes)

    group_struct = get_group_struct( all_nodes )

    e_ext = get_external_energy( group_struct, wt, wt_filt )


    gid_of_curr = np.random.randint(0,number_of_groups(all_nodes)+1)

    bit_int_e[ gid_of_curr ]
    print 'Adding current node %d to group %d' %(i, gid_of_curr)
    all_nodes.append( Node(uid=i, gid=gid_of_curr ) )



    print tcol.OKBLUE, 'Done in (ms) : ', (time.time() - startTime )*1000., tcol.ENDC
    code.interact( banner='---End of %d---' %(i), local=locals() )
