""" Computes netvlad vectors for every image of pittsburg street view.
    I am doing this so that I can learn an voronoi on this distribution
    for fast nn-search using faiss

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 7th Dec, 2017
"""
import rospkg

PKG_PATH = rospkg.RosPack().get_path('nap')

PITS_STREETVIEW = PKG_PATH+'/tf.logs/streetview_samples/'
PARAM_MODEL = PKG_PATH+'/tf.logs/org/model-3750' # trained similar to above but with a resnet neural net

#PARAM_MODEL = PKG_PATH+'/tf2.logs/attempt_resnet6_K16_P8_N8/model-2500'

import numpy as np
import cv2
from PlaceRecognitionNetvlad import PlaceRecognitionNetvlad
import glob
import code
import time
import pickle

import TerminalColors
tcol = TerminalColors.bcolors()

from ColorLUT import ColorLUT


place_mod = PlaceRecognitionNetvlad(\
                                    PARAM_MODEL,\
                                    PARAM_CALLBACK_SKIP=2,\
                                    PARAM_K = 64
                                    )
list_of_images = glob.glob( PITS_STREETVIEW+'*/*.jpg')
list_of_netvlads = []

colorLUT = ColorLUT()
for i,file_name in enumerate(list_of_images[0:10]):
    im = cv2.imread( file_name )
    #cv2.imshow( 'im', im )
    im = cv2.cvtColor( im, cv2.COLOR_BGR2RGB )

    s = time.time()
    X = place_mod.extract_descriptor( im )
    list_of_netvlads.append( X )

    #Assgn matrix
    lut = colorLUT.lut( place_mod.Assgn_matrix[0,:,:] )
    cv2.imshow( 'lut', lut )
    

    print '%d in %4.2fms : %s' %( i, 1000.0*(time.time()-s), file_name )
    cv2.waitKey(0)



quit()
# store `list_of_images` and `list_of_netvlads`
list_of_netvlads = np.array( list_of_netvlads )

print 'Writing file: ', PITS_STREETVIEW+'/list_of_images.pickle'
with open( PITS_STREETVIEW+'/list_of_images.pickle', 'wb' ) as fp:
    pickle.dump( list_of_images, fp )

print 'Writing file: ', PITS_STREETVIEW+'/list_of_netvlads.pickle'
with open( PITS_STREETVIEW+'/list_of_netvlads.pickle', 'wb' ) as fp:
    pickle.dump( list_of_netvlads, fp )


# Read it back
# list_of_images = pickle.load( open( PITS_STREETVIEW+'/list_of_images.pickle', 'rb' ) )
# list_of_netvlads = pickle.load( open( PITS_STREETVIEW+'/list_of_netvlads.pickle', 'rb' ) )
