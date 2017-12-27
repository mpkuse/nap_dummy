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

PKG_PATH = rospkg.RosPack().get_path('nap')
# PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

## 'x' can also be a vector
def logistic( x ):
    #y = np.array(x)
    #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
    # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)
    if len(x) < 3:
        return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

    y = np.convolve( np.array(x), [0.25,0.5,0.25], 'same' )
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)

class Node:
    def __init__(self, uid, parent=None):
        self.uid = uid
        self.parent = parent

    def __repr__(self):
        return '(u=%-4d, g=%-4d)' %(self.uid,get_gid(self))



def get_gid( node ):
    while node.parent is not None:
        node = node.parent
    return node.uid


def print_stats( all_nodes, internal_e, n_components ):
    S = set( [get_gid(n) for n in all_nodes] )

    print tcol.OKGREEN, 'Groups(gid) : ', len(S)
    for s in S:
        print 'In gid=%4d there are %4d nodes. int_energy=%4.4f' %(s,n_components[s], internal_e[s])
    print tcol.ENDC

    print all_nodes

#----- Load Data -----#
folder = '/DUMP/'
S_char = np.load( PKG_PATH +      folder+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH +      folder+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH +    folder+'S_thumbnail.npy' )#N x 96 x 128 x 3
S_timestamp = np.load( PKG_PATH + folder+'S_timestamp.npy' )
#-------- END --------#

all_nodes = []
internal_e = np.zeros(10000)
n_components = np.ones(10000)
for i in range(S_word.shape[0]):
    #assume at ith stage, all previous S are available only.
    if i==0: #no edges from 0th node to previous, ie. no previous nodes
        all_nodes.append( Node(uid=0) )
        internal_e[0] = 0.0
        n_components[0] = 1
        continue;

    startTime = time.time()

    #now there is atleast 1 prev nodes

    #compute dot product cost
    DOT_word = np.dot( S_word[0:i,:], S_word[i,:] )
    sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) )
    wt = 1.0 - logistic( sim_scores ) #measure of dis-similarity. 0 means very similar.
        #wt holds i<-->0, i<-->2, i<-->3, ..., i<-->i-2, i<-->i-1

    all_nodes.append( Node(uid=i) )
    for j, w in enumerate(wt):
        if w > 0.3:
            continue
        # print '%d<-- %s%f%s -->%d' %(i, tcol.OKGREEN,w,tcol.ENDC, j)
        gid_i = get_gid( all_nodes[i] )
        e_i = 0

        gid_j = get_gid( all_nodes[j] )
        e_j = internal_e[gid_j]

        if w < min(e_i+0.10/n_components[gid_i], e_j+0.15/n_components[gid_j]):
            #merge
            internal_e[gid_j] = w
            n_components[gid_j] += 1
            all_nodes[i].parent = all_nodes[j]



    # print_stats( all_nodes, internal_e, n_components )
    thumb = S_thumbs[i,:,:,:]
    cv2.putText( thumb, str(get_gid(all_nodes[i])), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )
    cv2.imshow( 'win', thumb )
    cv2.waitKey(30)
    print tcol.OKBLUE, 'Done in (ms) : ', (time.time() - startTime )*1000., tcol.ENDC
    # code.interact( banner='---End of %d---' %(i), local=locals() )
