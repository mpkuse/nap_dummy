""" Testing the FAISS from facebook AI research. A library for efficient
    similarity search by Herve Jegou and collegues. An implementation of
    product-quantization.

    Reference papers
    PAMI 2012 papers (Product quantization paper)
    https://arxiv.org/abs/1609.01882
    https://arxiv.org/abs/1702.08734

    https://github.com/facebookresearch/faiss

    Make sure to compile and install faiss. currently
    the python package is in ~/bin/python-packages/faiss/
    which is in PYTHONPATH.


    This script will load the S_word (ie. raw high dimensional netvlad
    vectors and do nearest neighbour search).
    Will test the performance with brute-force before pulling this code
    into the main nap node.


    export OMP_WAIT_POLICY=PASSIVE
    Is the solution to slow appends(). See github/facebookresearch/faiss issue 163.


    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 19th July, 2017
"""

import numpy as np
import cv2
from faiss import faiss
import rospkg
import time
import code

import matplotlib.pyplot as plt

import TerminalColors
tcol = TerminalColors.bcolors()

PKG_PATH = rospkg.RosPack().get_path('nap')
## 'x' can also be a vector
def logistic( x ):
    #y = np.array(x)
    #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
    # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)
    filt = [0.1,0.2,0.4,0.2,0.1]
    if len(x) < len(filt):
        return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

    y = np.convolve( np.array(x), filt, 'same' )
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)

def logistic_cwise( x ):
    return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

#--- Load Files ---#
S_word_filename = PKG_PATH + '/DUMP/S_word.npy'
print tcol.OKGREEN, 'Load : ', S_word_filename, tcol.ENDC
S_word = np.load( S_word_filename )

S_thumb_filename = PKG_PATH + '/DUMP/S_thumbnail.npy'
print tcol.OKGREEN, 'Load : ', S_thumb_filename, tcol.ENDC
S_thumb = np.load( S_thumb_filename )

S_thumb_lut_filename = PKG_PATH + '/DUMP/S_thumbnail_lut.npy'
print tcol.OKGREEN, 'Load : ', S_thumb_lut_filename, tcol.ENDC
S_thumb_lut = np.load( S_thumb_lut_filename )

S_timestamp_filename = PKG_PATH + '/DUMP/S_timestamp.npy'
print tcol.OKGREEN, 'Load : ', S_timestamp_filename, tcol.ENDC
S_timestamp = np.load( S_timestamp_filename )
#--- END ---#


# will be a 256D index
quantizer = faiss.IndexFlatL2(16384)
index = faiss.IndexIVFPQ( quantizer, 16384, 256, 8, 8 )



# index.train( S_word )
print 'Train Index'
index.train( np.random.random( (10000, 16384)  ).astype('float32') )
code.interact(local=locals())
# quit()

#--- Image Loop ---#
__time_naive = []
__time_faiss = []
for loop_index in range( 0, S_word.shape[0] ):

    #--- Naive brute-force scoring ~100ms ---#
    startTimeScoring = time.time()
    DOT_word = np.dot( S_word[0:loop_index+1], np.transpose(S_word[loop_index]) )
    sim_scores = np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) )
    sim_scores_logistic = logistic( sim_scores )
    NN_naive,  = np.where( sim_scores_logistic > 0.6 )
    naive_I = NN_naive
    naive_D = sim_scores_logistic[naive_I]
    print '%04d) Naive Scoring : %4.2fms' %( loop_index, ( time.time() - startTimeScoring)*1000. )
    __time_naive.append( (time.time() - startTimeScoring)*1000.)

    #--- End Naive ---#



    #--- Faiss Index ---#
    startTimeFaiss = time.time()
    index.add( np.expand_dims(S_word[loop_index], axis=0) ) #1x16384
    # print tcol.OKBLUE, 'Currently %d items in faissDB' %(faiss_index.ntotal), tcol.ENDC
    print '%04d) Faiss append time : %4.2fms' %( loop_index, ( time.time() - startTimeFaiss)*1000. )

    number_of_nearest_neighbors = 50
    startTimeSearch = time.time()
    # print tcol.OKBLUE, 'Search for %d nearest neighbors' %(number_of_nearest_neighbors), tcol.ENDC
    faiss_D, faiss_I = index.search( np.expand_dims(S_word[loop_index], axis=0), number_of_nearest_neighbors )
    print '%04d) Faiss Scoring : %4.2fms' %( loop_index, ( time.time() - startTimeSearch)*1000. )
    __time_faiss.append( (time.time() - startTimeFaiss)*1000.)
    faiss_D = 1.0 - faiss_D[0]*.5 #convert L2 distance to dot
    faiss_I = faiss_I[0]
    faiss_D_ = logistic_cwise( np.sqrt( 1.0 - np.minimum(1.0,faiss_D ) ) )

    o_faiss_D = []
    o_faiss_D_ = []
    o_faiss_I = []
    for _q in range( len(faiss_D) ):
        if faiss_D_[_q] > 0.6:
            o_faiss_D.append( faiss_D[_q] )
            o_faiss_D_.append( faiss_D_[_q] )
            o_faiss_I.append( faiss_I[_q] )



    #--- END ---#


plt.plot( __time_naive )
plt.plot( __time_faiss )
plt.show()
