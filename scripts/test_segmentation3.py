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
import json
import pickle
#
import TerminalColors
tcol = TerminalColors.bcolors()

from FastPlotter import FastPlotter

import matplotlib.pyplot as plt

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

class Node:
    def __init__(self, uid, parent=None):
        self.uid = uid
        self.parent = parent

    def __repr__(self):
        return '(u=%-4d, g=%-4d)' %(self.uid,get_gid(self))



def get_gid( node, verbose=False ):
    while node.parent is not None:
        node = node.parent
    return node.uid


def get_gid_path( node, verbose=False ):
    path = []
    if verbose:
        print 'Path from (%d) : ' %(node.uid),

    while node.parent is not None:
        if verbose:
            print '(%-3d)--' %(node.uid),

        path.append(node.uid)
        node = node.parent
    if verbose:
        print ''
    path.append(node.uid)
    return node.uid, path



def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()


    x, y = np.atleast_1d(x, y)
    artists = []
    i = 0
    for x0, y0 in zip(x, y):
        im = OffsetImage( cv2.cvtColor( image[i,:,:,:], cv2.COLOR_RGB2BGR ), zoom=zoom)
        i = i+1
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def show_npy( Z, title='' ):
    im = plt.imshow( Z, cmap='hot' )
    plt.title( title )
    plt.colorbar( im, orientation='horizontal' )


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

def save_json( out_file_name, data ):
    print 'Writing ', out_file_name
    with open( out_file_name, 'w') as f:
        json.dump( data, f )

def load_json( file_name ):
    print 'Loading ', file_name
    with open( file_name ) as f:
        my_dict =  json.load(f)
        return  {int(k):float(v) for k,v in my_dict.items() }


class LocationSegment:
    def __init__(self, s, e):
        self.seg_start = s
        self.seg_end   = e

    def s(self):
        return self.seg_start

    def e(self):
        return self.seg_end

    def size(self):
        return (self.seg_end - self.seg_start)

    def __repr__(self):
        return 'Seg:<%4d--(%3d)--%4d>' %(self.s(), self.size(), self.e())




#----- Load Data -----#
folder = '/DUMP/aerial_22/'
S_char = np.load( PKG_PATH+  folder+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH+  folder+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH+folder+'S_thumbnail.npy' )#N x 96 x 128 x 3
S_timestamp = np.load( PKG_PATH + folder+'S_timestamp.npy' )
#-------- END --------#


if False:
    plotter = FastPlotter(1,200,200)
    plotter.setRange( 0, yRange=[0,1] )
    all_nodes = []
    internal_e = {} #associate array
    n_components = {}
    for i in range(S_word.shape[0]):
        #assume at ith stage, all previous S are available only.
        if i==0: #no edges from 0th node to previous, ie. no previous nodes
            all_nodes.append( Node(uid=0))
            continue;

        startTime = time.time()

        #now there is atleast 1 prev nodes

        #compute dot product cost
        window_size = 50
        DOT_word = np.dot( S_word[max(0,i-window_size):i,:], S_word[i,:] )
        DOT_index = range(max(0,i-window_size),i)
        sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) )
        wt = 1.0 - logistic( sim_scores ) #measure of dis-similarity. 0 means very similar.

        all_nodes.append( Node(uid=i) )
        for j_ind,w in enumerate(wt):
            if w>0.3:
                continue

            gid_i = get_gid( all_nodes[i] )
            e_i = 0

            j = DOT_index[j_ind]
            gid_j = get_gid( all_nodes[j] )
            e_j = internal_e[gid_j] if internal_e.has_key(gid_j) else 0.0

            n_i = n_components[gid_i] if n_components.has_key(gid_i) else 1
            n_j = n_components[gid_j] if n_components.has_key(gid_j) else 1

            kappa = 0.25
            # print 'gid_i=%3d gid_j=%3d' %(gid_i, gid_j)
            # print 'w=%4.4f, ei=%4.4f, ej=%4.4f' %(w, e_i+kappa/n_i, e_j+kappa/n_j )
            if w < min(e_i+kappa/n_i, e_j+kappa/n_j):
                internal_e[gid_j] = w
                n_components[gid_j] = n_j + 1
                all_nodes[i].parent = all_nodes[j]




        #
        _past_key_frames = sorted(internal_e)
        print _past_key_frames
        _past_mid_frames = []
        for n_id,n in enumerate( _past_key_frames[0:-1] ):
            # print n_id, n
            _past_mid_frames.append( int((n+_past_key_frames[n_id+1])/2.) )
        print _past_mid_frames
        _past_DOT_word = np.dot( S_word[_past_mid_frames,:] , S_word[i,:] )
        _past_sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, _past_DOT_word ) )
        _past_wt = 1.0 - logistic( _past_sim_scores ) #measure of dis-similarity. 0 means very similar.
        print _past_wt
        if len(_past_wt) > 0 :
            plotter.set_data( 0, _past_mid_frames, _past_wt )
            plotter.spin()




        print i, 'of', S_word.shape[0], tcol.OKBLUE, 'Done in (ms) : ', np.round( (time.time() - startTime )*1000.,2 ), tcol.ENDC
        # code.interact( banner='---End of %d---' %(i), local=locals() )

        # print_stats( all_nodes, internal_e, n_components )
        thumb = S_thumbs[i,:,:,:]
        cv2.putText( thumb, str(get_gid(all_nodes[i])), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )
        cv2.imshow( 'win', thumb )
        cv2.waitKey(10)

    save_json(PKG_PATH+folder+'/internal_e.json', internal_e )
    save_json(PKG_PATH+folder+'/n_components.json', n_components )
    with open( PKG_PATH+folder+'all_nodes.pickle', 'w' ) as f:
        pickle.dump( all_nodes, f )

else:
    internal_e = load_json( PKG_PATH+folder+'/internal_e.json' )
    n_components = load_json( PKG_PATH+folder+'/n_components.json' )
    with open( PKG_PATH+folder+'all_nodes.pickle', 'r' ) as f:
        all_nodes = pickle.load( f )

#--------------- Analysis of Key Frames ---------------------#
# Vars :
#   internal_e : associate array of internal energies of each components
#   n_components : number of elements in each components



key_frames =  sorted(internal_e)
key_frames.append( S_word.shape[0] )
code.interact( local=locals() )

## Are centroids good representation of a bunch ?
## I observed yes. The histograms of dot-product of (mean of current batch) and each
## elements of current batch is usually > 0.8. histograms of (Dot of mean) with
## Rest of the images is not overlapping. Thus, I am conclusing as of 2nd June
## vector quantization might actually work.
for k in range(len(key_frames)-1):
    k1 = key_frames[k]
    k2 = key_frames[k+1]
    k_last = key_frames[-1]

    if k2-k1 < 10 :
        continue

    centeroid = S_word[k1:k2,:].mean(axis=0) / np.linalg.norm( S_word[k1:k2,:].mean(axis=0)  )
    SIM = np.dot( S_word[k1:k2,:], centeroid )
    SIM_restful = np.dot( S_word[ range(0,max(0,k1-20))+range(min(k_last,k2+20),k_last),:], centeroid )

    plt.subplot(2,1,1)
    plt.hist( SIM )
    plt.title( '%d %d; size=%d' %(k1,k2, k2-k1) )
    plt.xlim( [0.0, 1.0] )

    plt.subplot(2,1,2)
    plt.hist( SIM_restful )
    plt.title( '(%d %d) U (%d,%d)' %(0,k1-20,k2+20,k_last) )
    plt.xlim( [0.0, 1.0] )

    plt.show()


quit()


# Descriptor Distinctiveness
# Ideas from the paper :
# Arandjelovic, Relja, and Andrew Zisserman. "DisLocation: Scalable descriptor distinctiveness for location recognition." Asian Conference on Computer Vision. Springer International Publishing, 2014.
# Note : Basic idea is, distinctveness inversely proportional to local density

code.interact( local=locals() )
for k in range(len(key_frames)-1):
    k1 = key_frames[k]
    k2 = key_frames[k+1]
    H = np.dot( S_word[k1:k2,:], np.transpose( S_word[k1:k2] ) )
    # H = np.dot( S_word, np.transpose( S_word ) )
    H_tri = np.tril( H, -1 )

    H_tri_flat = H_tri.ravel()[ np.flatnonzero(H_tri) ]


    median_measure = np.median( H_tri_flat )
    min_measure = H_tri_flat.min()


    title = '%2d %4d:%4d : %2.3f %2.3f' %(k,k1,k2,median_measure,min_measure )
    print title
    if (k2-k1) < 20:
        continue
    plt.hist( H_tri_flat )
    plt.title( title )
    plt.xlim( [0.0, 1.0] )

    cv2.putText( S_thumbs[k1,:,:,:], str(k1), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )
    cv2.putText( S_thumbs[k2,:,:,:], str(k2), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA )

    cv2.imshow('k1', S_thumbs[k1,:,:,:].astype('uint8'))
    cv2.imshow('k2', S_thumbs[k2,:,:,:].astype('uint8'))
    cv2.waitKey(10)
    plt.show()


quit()

# Make collection of `LocationSegment`
LL = []
for k in range(len(key_frames)-1):
    k1 = key_frames[k]
    k2 = key_frames[k+1]
    L = LocationSegment( k1, k2 )
    L.D_SMALL_STD = set(S_word[k1:k2,:].std(axis=0).argsort()[0:1000])
    L.D_LARGE_STD = set(S_word[k1:k2,:].std(axis=0).argsort()[-1000:])
    L.D_MEAN = S_word[k1:k2,:].mean(axis=0)
    L.D_STD  = S_word[k1:k2,:].var(axis=0)
    LL.append( L )

# Entropy and KL divergence of each dim
from scipy.stats import entropy
entropy_ = []
kl_div = np.zeros( (12288 , 12288) )
for l in range(0,12288):
    a,b = np.histogram( S_word[:,l], bins='fd')
    entropy_.append( entropy(a) )

for s in range(2000000):
    if s%1000 == 0 :
        print s
    lx1 = np.random.randint(0,12288)
    lx2 = np.random.randint(0,12288)

    l1 = min( lx1, lx2)
    l2 = max( lx1, lx2 )

    a1,b1 = np.histogram( S_word[:,l1], bins=25)
    a2,b2 = np.histogram( S_word[:,l2], bins=25)
    kl_div[l1,l2] = entropy( a1, a2 )

show_npy( kl_div )

# for l1 in range(0,12288):
#     print l1
#     for l2 in range( 0, l1 ):
#         a1,b1 = np.histogram( S_word[:,l1], bins=25)
#         a2,b2 = np.histogram( S_word[:,l2], bins=25)
#         kl_div[l1,l2] = entropy( a1, a2 )
quit()



# Most distinguishing dimensions and least distingushing dimensions
D_SMALL_STD = []
D_LARGE_STD = []
for k in range(len(key_frames)-1):
    k1 = key_frames[k]
    k2 = key_frames[k+1]

    D_SMALL_STD.append( set(S_word[k1:k2,:].std(axis=0).argsort()[0:1000]) )
    D_LARGE_STD.append( set(S_word[k1:k2,:].std(axis=0).argsort()[-1000:]) )


dotp_ary = []
intr_ary = []
for k in range( len(key_frames)-1 ):
    dotp = np.dot( S_word[0,:], np.transpose( S_word[key_frames[k]] ) )
    intr = D_SMALL_STD[0].intersection( D_SMALL_STD[k] )

    print k, key_frames[k], dotp, len(intr)
    dotp_ary.append( dotp )
    intr_ary.append( len(intr) )

#
# for k1 in range(len(key_frames)):
#         for k2 in range(k1):
#             print key_frames[k1], key_frames[k2]
#
#             D1 = S_word[key_frames[k1]]

    # D = np.dot( S_word[key_frames[k]:key_frames[k+1]], np.transpose( S_word[39:110] ) )


quit()

code.interact( local=locals() )
# Draw intra graph (path to gids)
for k in range(len(key_frames)-1):
    intragraphStartTime = time.time()
    H = nx.Graph()
    for i in range( key_frames[k], key_frames[k+1] ):
        gid, path = get_gid_path( all_nodes[i] )
        H.add_path( path )
    pagerank = nx.pagerank(H)
    print 'Intragraph constructed in %4.2f ms' %(1000.*(time.time()-intragraphStartTime))

    print tcol.OKBLUE, 'Keyframe is %d with n_components=%d, internal_e=%2.4f' %(key_frames[k], n_components[key_frames[k]], internal_e[key_frames[k]]), tcol.ENDC
    print 'Number of nodes : ', H.number_of_nodes()
    if H.number_of_nodes() > 30:
        pagerank = nx.pagerank(H)
        for p in sorted( pagerank, key=pagerank.get )[-10:]:
            print 'pagerank[%d]=%4.4f' %(p, pagerank[p])

        nx.draw( H, pos=nx.spring_layout(H), with_labels=True)
        plt.show()
        code.interact( local=locals() )






quit()


#f, axarr = plt.subplots(3)
for k in range(len(key_frames)-1):
    c_with = int(np.floor( 0.5*(key_frames[k]+key_frames[k+1]) ))
    p0 = np.dot( S_word[ key_frames[k]:key_frames[k+1] ], np.transpose( S_word[ key_frames[k] ] ) )
    pn = np.dot( S_word[ key_frames[k]:key_frames[k+1] ], np.transpose( S_word[ c_with ] ) )
    pN = np.dot( S_word[ key_frames[k]:key_frames[k+1] ], np.transpose( S_word[ key_frames[k+1]-1 ] ) )

    plt.plot( range(key_frames[k],key_frames[k+1]), p0 )
    plt.plot( range(key_frames[k],key_frames[k+1]), pn )
    plt.plot( range(key_frames[k],key_frames[k+1]), pN )
    plt.show()



quit()
G = nx.Graph()
for k in key_frames:
    G.add_node( k, n_compo=n_components[k], int_e=internal_e[k], time_stamp=str(S_timestamp[k]) )


C  = np.dot( S_word[key_frames,:], np.transpose(S_word[key_frames,:]) )


gps_t, gps_x, gps_y, gps_z = np.loadtxt( PKG_PATH+'/DUMP/GPS_track.csv', dtype={'names':('t','x','y','z'), 'formats':('i8', 'f4', 'f4', 'f4') }, delimiter=',', unpack=True)
pos1 = gps_layout2d( G, gps_t, gps_x, gps_y )
nx.draw_networkx( G, pos1, font_size=10, width=0.5 )
plt.show()
