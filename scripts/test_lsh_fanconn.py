""" Testing the FALCONN library
    FALCONN provides a locality-sensitive-hashing (LSH) for
    high dimension data. It provides for 2 families of methods
    a) Hyper-plane partitioning b) Cross-polytope based methods

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 5th June, 2017
"""
import numpy as np
import cv2
import networkx as nx
import code
import time
import json
import pickle
import rospkg
#
import TerminalColors
tcol = TerminalColors.bcolors()

from FastPlotter import FastPlotter

import matplotlib.pyplot as plt

PKG_PATH = rospkg.RosPack().get_path('nap')
def load_json( file_name ):
    print 'Loading ', file_name
    with open( file_name ) as f:
        my_dict =  json.load(f)
        return  {int(k):float(v) for k,v in my_dict.items() }


#----- Load Data -----#
folder = '/DUMP/aerial_22/'
startTime = time.time()
S_char = np.load( PKG_PATH+  folder+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH+  folder+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH+folder+'S_thumbnail.npy' )#N x 96 x 128 x 3
S_full_im = np.load( PKG_PATH+folder+'S_full_images.npy' )#N x 240 x 320 x 3
S_timestamp = np.load( PKG_PATH + folder+'S_timestamp.npy' )

internal_e = load_json( PKG_PATH+folder+'/internal_e.json' )
key_frames =  sorted(internal_e)
key_frames.append( S_word.shape[0] )

print 'Done Reading in %4.2fs' %(time.time() - startTime)
#-------- END --------#


#------ Centroid of Islands Dataset -----#
S_island_word = []
S_island_start = []
S_island_end = []
for k in range(len(key_frames)-1):
    k1 = key_frames[k]
    k2 = key_frames[k+1]
    k_last = key_frames[-1]

    centeroid_of_island = S_word[k1:k2,:].mean(axis=0)
    S_island_word.append( centeroid_of_island )
    S_island_start.append( k1 )
    S_island_end.append( k2 )
S_island_word = np.array( S_island_word )
#----- END ------#


# from sklearn.neighbors import LSHForest
# #----- SKLearn LSH -----#
# startTime = time.time()
# lshf = LSHForest( random_state=42, n_estimators=50 )
# lshf.fit( S_island_word[0:5,:] )
# print 'Initial data inserted in %4.2f' %(time.time() - startTime)
#
# startTime = time.time()
# for i in range(5,S_island_word.shape[0], 10 ):
#     lshf.partial_fit( S_island_word[i:i+10,:] )
# print 'Incremental Addition in %4.2fs each' %( (time.time() - startTime)/S_island_word.shape[0] )
#
# startTime = time.time()
# for i in np.random.randint(0,1500, 100 ):
#     qppp = lshf.kneighbors( S_word[i,:].reshape(1,-1) ,n_neighbors=5 )
#     # qppp = lshf.radius_neighbors( S_word[i,:].reshape(1,-1), 0.15 )
#
# print 'Query in %4.2fs' %( (time.time() - startTime) / 100. )
# #----- END -----#
# quit()



import falconn
#----- FALCONN - LSH Params -----#
print 'Setup Hash'
params_cp = falconn.LSHConstructionParameters()
params_cp.dimension = S_word.shape[1]
params_cp.lsh_family = 'cross_polytope'
params_cp.distance_function = 'euclidean_squared'
params_cp.l = 50 #number of tables
params_cp.num_rotations = 1
params_cp.seed = 2323
params_cp.num_setup_threads = 0
params_cp.storage_hash_table = 'bit_packed_flat_hash_table'
falconn.compute_number_of_hash_functions( 5, params_cp )
#----- END ------#


#----- FALCONN - Constructing Hash Table -----#
print 'Constructing Hash Table'
startTime = time.time()
table = falconn.LSHIndex(params_cp)
# table.setup(S_word)
table.setup( np.array(S_island_word) )
print 'Done in %4.2fs' %(time.time() - startTime)
#----- END -----#



table.find_k_nearest_neighbors( S_word[1067,:] , 100 )
