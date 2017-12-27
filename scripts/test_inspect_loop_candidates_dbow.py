""" Inspect loop candidates one at a time for DBOW.

        The dbow3_naive writes out 2 files.
        a) loop_candidates_dbow.csv
                loop_candidates.append( [L, aT, sim_scores_logistic[aT], nMatches, nInliers] )
        b) S_thumbnail_dbow.npy
                Nx240x320x3


        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 6th July, 2017
"""

import numpy as np
import cv2
import code
cv2.ocl.setUseOpenCL(False)

from GeometricVerification import GeometricVerification


DBOW_IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_dbow.npy'
DBOW_LOOP_CANDIDATES = '/home/mpkuse/catkin_ws/src/nap/DUMP/loop_candidates_dbow.csv'

print 'Reading : ', DBOW_IMAGE_FILE_NPY
S_thumbnails = np.load(DBOW_IMAGE_FILE_NPY)
print 'S_thumbnails.shape : ', S_thumbnails.shape

print 'Reading : ', DBOW_LOOP_CANDIDATES
loop_candidates = np.loadtxt( DBOW_LOOP_CANDIDATES, delimiter=',' )

__c = -1
__p = 0
flag = False
VV = GeometricVerification()
for i,l in enumerate(loop_candidates):
    # [ curr, prev, score, nMatches, nConsistentMatches]
    curr = int(l[0])
    prev = int(l[1])
    score = l[2]
    nMatches = int(l[3])
    nConsistentMatches = int(l[4])
    # if __c != curr:
    if (curr - __c) > 3 :
        print '---'
        __c = curr
        __p = -1

    if (prev - __p) > 5:
        print '.'
        flag = True
    __p = prev

    print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)
    # if nMatches > 10:
    # if score > 0.0:
    flag = True # set this to True here to show all images
    if flag is True and score>0.0:
        flag = False

        VV.set_im( S_thumbnails[curr, :,:,:] , S_thumbnails[prev, :,:,:] )
        # VV.set_im_lut( cv2.resize(S_thumbnails_lut[curr, :,:,:], (320,240), interpolation=cv2.INTER_NEAREST), cv2.resize(S_thumbnails_lut[prev, :,:,:], (320,240), interpolation=cv2.INTER_NEAREST))
        # VV.set_im_lut_raw( cv2.resize(S_thumbnails_lut_raw[curr, :,:], (320,240), interpolation=cv2.INTER_NEAREST), cv2.resize(S_thumbnails_lut_raw[prev, :,:], (320,240), interpolation=cv2.INTER_NEAREST))
        nMatches, nInliers = VV.simple_verify(features='orb')

        # VV.simple_verify(features='surf')


        cv2.imshow( 'curr', S_thumbnails[curr, :,:,:] )
        cv2.imshow( 'prev', S_thumbnails[prev, :,:,:] )
        cv2.moveWindow( 'prev', S_thumbnails[curr, :,:,:].shape[1]+20, 0 )


        if (cv2.waitKey(0) & 0xFF) == ord('q'):
            break
