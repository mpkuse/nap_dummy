""" This script is used to develop guided 2-way match using Daisy

        Will load the loop closure data which includes,
        - images
        - neural net cluster assignments
        - list of loop-closure candidates (giving index of prev and curr)
        - Feature factory (per keyframe tracked-features data from VINS)

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 7th Nov, 2017
"""


import numpy as np
import cv2
import code
import time
import sys

from operator import itemgetter
cv2.ocl.setUseOpenCL(False)

from GeometricVerification import GeometricVerification
from ColorLUT import ColorLUT

from DaisyMeld.daisymeld import DaisyMeld

import TerminalColors
tcol = TerminalColors.bcolors()
from FeatureFactory import FeatureFactory

# Image, pts 2xN
def points_overlay( im_org, pts, color=(255,0,0), enable_text=False, show_index=None ):
    im = im_org.copy()
    if len(im.shape) == 2:
        im = cv2.cvtColor( im, cv2.COLOR_GRAY2BGR )

    if pts.shape[0] == 3: #if input is 3-row mat than  it is in homogeneous so perspective divide
        pts = pts / pts[2,:]

    color_com = ( 255 - color[0] , 255 - color[1], 255 - color[2] )

    if show_index is None:
        rr = range( pts.shape[1] )
    else:
        rr = show_index

    for i in rr:
        cv2.circle(  im, tuple(np.int0(pts[0:2,i])), 3, color, -1 )
        if enable_text:
            cv2.putText( im, str(i), tuple(np.int0(pts[0:2,i])), cv2.FONT_HERSHEY_SIMPLEX, .3, color_com )

    return im




#
# Specify Files
#
BASE__DUMP = '/home/mpkuse/Desktop/a/drag_nap'
KF_TIMSTAMP_FILE_NPY = BASE__DUMP+'/S_timestamp.npy'
IMAGE_FILE_NPY = BASE__DUMP+'/S_thumbnail.npy'
IMAGE_FILE_NPY_lut = BASE__DUMP+'/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = BASE__DUMP+'/S_thumbnail_lut_raw.npy'

LOOP_CANDIDATES_NPY = BASE__DUMP+'/loop_candidates.csv'

FEATURE_FACTORY = BASE__DUMP+'/FeatureFactory'

#
# Load Files
#
print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
S_timestamp = np.load(KF_TIMSTAMP_FILE_NPY)
print 'Reading : ', IMAGE_FILE_NPY_lut
S_thumbnails_lut = np.load(IMAGE_FILE_NPY_lut)
print 'Reading : ', IMAGE_FILE_NPY_lut_raw
S_thumbnails_lut_raw = np.load(IMAGE_FILE_NPY_lut_raw)
print 'S_thumbnails.shape : ', S_thumbnails.shape


print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )


feature_factory = FeatureFactory()
feature_factory.load_from_pickle( FEATURE_FACTORY )


#
# Loop Over every frame
# for i in range( S_thumbnails.shape[0] ):
#     curr_im = S_thumbnails[i, :,:,:]
#     t_curr = S_timestamp[i]
#
#     feat2d_curr_idx = feature_factory.find_index( t_curr )
#     feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
#     print feat2d_curr.shape
#
#     cv2.imshow( 'im', points_overlay( curr_im, feat2d_curr) )
#     cv2.waitKey(0)
# quit()

#
# Loop over every loop-closure
for li,l in enumerate(loop_candidates):
    curr = int(l[0])
    prev = int(l[1])

    t_curr = S_timestamp[curr]
    t_prev = S_timestamp[prev]


    print li, curr, prev
    continue
    feat2d_curr_idx = feature_factory.find_index( t_curr )
    feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords

    feat2d_prev_idx = feature_factory.find_index( t_prev )
    feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] ) #3xN in homogeneous cords
    print feat2d_curr.shape, feat2d_prev.shape


quit()

# which loop_candidates to load
if len(sys.argv) == 2:
    i=int(sys.argv[1])
else:
    i = 0

for i in range( len(loop_candidates) ):
    print '==='
    l = loop_candidates[i]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])


    print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)


    curr_im = S_thumbnails[curr, :,:,:]
    prev_im = S_thumbnails[prev, :,:,:]
    curr_lut_raw = S_thumbnails_lut_raw[curr,:,:]
    prev_lut_raw = S_thumbnails_lut_raw[prev,:,:]
    t_curr = S_timestamp[curr]
    t_prev = S_timestamp[prev]

    currm_im = S_thumbnails[curr-1, :,:,:]
    currm_lut_raw = S_thumbnails_lut_raw[curr-1,:,:]
    t_currm = S_timestamp[curr-1]

    feat2d_curr_idx = feature_factory.find_index( t_curr )
    feat2d_prev_idx = feature_factory.find_index( t_prev )

    feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
    feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] )
    print feat2d_curr.shape
    print feat2d_prev.shape

    # ------------ Data Ready now! -------------- #


    # Simulated proc
    VV = GeometricVerification()

    # Try 2 way matching
    startSet = time.time()
    VV.set_image( curr_im, 1 ) #set current image
    VV.set_image( prev_im, 2 )# set previous image (at this stage dont need lut_raw to be set as it is not used by release_candidate_match2_guided_2way() )
    print 'set_image, ch=1 and ch=2 : %4.2f (ms)' %( 1000. * (time.time() - startSet) )

    startT = time.time()
    selected_curr_i, selected_prev_i = VV.release_candidate_match2_guided_2way( feat2d_curr, feat2d_prev )
    print 'matcher.release_candidate_match2_guided_2way() : %4.2f (ms)' %(1000. * (time.time() - startT) )
    print 'guided 2way matches : ', selected_curr_i.shape[0], selected_prev_i.shape[0]
    n_guided_2way = selected_curr_i.shape[0]

    # TODO: Consider having a plot_point_sets function inside GeometricVerification
    # careful with alrady exisiting plot_point_sets in GeometricVerification though!
    code.interact( local=locals() )
    xcanvas_2way = VV.plot_2way_match( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i])  )
    cv2.imshow( 'main selected+fundamentalmatrixtest', xcanvas_2way )


    # Try 3way
    if n_guided_2way < 20:
        startSet = time.time()
        VV.set_image( currm_im, 3 )  #set curr-1 image
        print 'set_image ch=3 : %4.2f (ms)' %( 1000. * (time.time() - startSet) )
        VV.set_lut_raw( curr_lut_raw, 1 ) #set lut of curr and prev
        VV.set_lut_raw( prev_lut_raw, 2 )
        # lut for curr-1 is not set as it is not used.


        # these will be 3 co-ordinate point sets
        start3way = time.time()
        xpts_curr, xpts_prev, xpts_currm = VV.release_candidate_match3way() #this function reuses daisy for im1, and im2, just 1 daisy computation inside.
        print 'matcher.release_candidate_match3way() : %4.2f (ms)' %(1000. * (time.time() - start3way) )

        gridd = VV.plot_3way_match( curr_im, xpts_curr, prev_im, xpts_prev, currm_im, xpts_currm, enable_lines=False, enable_text=True )
        cv2.imshow( '3way', gridd )




    cv2.waitKey(0)

    VV.reset()
    # ------------------- Done --------------------- #
