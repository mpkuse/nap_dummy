""" An attempt to improve the 3way matching false positive.
    Especially looking at a simple mechanism to identify false matches
    early. Will start from current implementation of geometryverify
    and improve upon it.


        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 13th Dec, 2017
"""



import numpy as np
import cv2
import code
import time
import sys
import matplotlib.pyplot as plt
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

LOOP_CANDIDATES_NPY = BASE__DUMP+'/loop_candidates2.csv'

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

# # Loop Over every frame
# for i in range( S_thumbnails.shape[0] ):
#     curr_im = S_thumbnails[i, :,:,:]
#     t_curr = S_timestamp[i]
#     lut_raw = S_thumbnails_lut_raw[i,:,:]
#     lut = S_thumbnails_lut[i,:,:,:]
#
#     feat2d_curr_idx = feature_factory.find_index( t_curr )
#     feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
#     print feat2d_curr.shape
#     print 'lut_raw.uniq ', len( np.unique(lut_raw[:]))
#
#     cv2.imshow( 'im', points_overlay( curr_im, feat2d_curr) )
#     cv2.imshow( 'lut', lut.astype('uint8') )
#
#     # plt.hist( lut_raw[:], 64 )
#     # plt.show( False )
#     cv2.waitKey(0)
#
# quit()
###############################################################################
VV = GeometricVerification()
# for i in [10]: #range( len(loop_candidates) ):
for i in range(len(loop_candidates)):
    print '==='
    l = loop_candidates[i]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])


    print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)
    t_curr = S_timestamp[curr]
    t_prev = S_timestamp[prev]

    curr_im = S_thumbnails[curr, :,:,:]
    prev_im = S_thumbnails[prev, :,:,:]

    curr_lut = S_thumbnails_lut[curr,:,:,:]
    prev_lut = S_thumbnails_lut[prev,:,:,:]

    feat2d_curr_idx = feature_factory.find_index( t_curr )
    feat2d_prev_idx = feature_factory.find_index( t_prev )

    feat2d_curr = np.dot( feature_factory.K, feature_factory.features[feat2d_curr_idx ] ) #3xN in homogeneous cords
    feat2d_prev = np.dot( feature_factory.K, feature_factory.features[feat2d_prev_idx ] )
    print feat2d_curr.shape
    print feat2d_prev.shape

    # cv2.imshow( 'curr_im', curr_im )
    # cv2.imshow( 'prev_im', prev_im )
    cv2.imshow( 'curr_im', points_overlay( curr_im, feat2d_curr) )
    cv2.imshow( 'prev_im', points_overlay( prev_im, feat2d_prev) )

    cv2.imshow( 'curr_lut', curr_lut )
    cv2.imshow( 'prev_lut', prev_lut )


    VV.set_image( curr_im, 1 ) #set current image
    VV.set_image( prev_im, 2 )# set previous image (at this stage dont need lut_raw to be set as it is not used by release_candidate_match2_guided_2way() )
    selected_curr_i, selected_prev_i, sieve_stat = VV.release_candidate_match2_guided_2way( feat2d_curr, feat2d_prev )
    xcanvas_2way = VV.plot_2way_match( curr_im, np.int0(feat2d_curr[0:2,selected_curr_i]), prev_im, np.int0(feat2d_prev[0:2,selected_prev_i]),  enable_lines=True )


    print 'Tracked features in curr: %d' %(feat2d_curr.shape[1])
    print 'Tracked features in prev: %d' %(feat2d_prev.shape[1])
    print 'Tracked Features              :', sieve_stat[0]
    print 'Features retained post voting :', sieve_stat[1]
    print 'Features retained post f-test :', sieve_stat[2]

    # Finding-1: if atleast 20% of the tracked points of min(feat2d_curr, feat2d_prev)
    # are retained than it indicates that this one is probably a true match.

    # Finding-2: If tracked features in both images are dramatically different,
    # then, most likely it is a false match and hence need to be rejected.

    cv2.imshow( 'xcanvas_2way', xcanvas_2way )

    match2_voting_score = float(sieve_stat[1]) / sieve_stat[0] #how many remain after voting. More retained means better confident I am. However less retained doesn't mean it is wrong match. Particularly, there could be less overlaping area and the tracked features are uniformly distributed on image space.
    match2_tretained_score = float(sieve_stat[2]) / sieve_stat[0] #how many remain at the end
    match2_geometric_score  = (sieve_stat[1] - sieve_stat[2]) / sieve_stat[1]#how many were eliminated by f-test. lesser is better here. If few were eliminated means that the matching after voting is more geometrically consistent
    print 'match2_voting_score: %4.2f; ' %(match2_voting_score),
    print 'match2_tretained_score: %4.2f; ' %(match2_tretained_score),
    print 'match2_geometric_score: %4.2f' %(match2_geometric_score)



    match2_total_score = 0.
    if match2_voting_score > 0.5:
        match2_total_score += 1.0

    if match2_tretained_score > 0.25:
        match2_total_score += 2.5
    else:
        if match2_tretained_score > 0.2 and match2_tretained_score <= 0.25:
            match2_total_score += 1.0
        if match2_tretained_score > 0.15 and match2_tretained_score <= 0.2:
            match2_total_score += 0.5


    if match2_geometric_score > 0.55:
        match2_total_score -= 1.
    else:
        if match2_geometric_score < 0.4 and match2_geometric_score >= 0.3:
            match2_total_score += 1.0
        if match2_geometric_score < 0.3 and match2_geometric_score >= 0.2:
            match2_total_score += 1.5
        if match2_geometric_score < 0.2 :
            match2_total_score += 1.5

    # min/ max
    if (float(min(feat2d_curr.shape[1],feat2d_prev.shape[1])) / max(feat2d_curr.shape[1],feat2d_prev.shape[1])) < 0.70:
        match2_total_score -= 3
        print 'nTracked features are very different.'

    print '==Total_score : ', match2_total_score, '=='


    if match2_total_score > 0.0:
        cv2.waitKey(0)
