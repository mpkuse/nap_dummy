""" Uses the whole image descriptor `daisy` to get a 3-way match
between (curr)--(prev)--(curr-1). Here, curr--prev is a loop-closure
pair. The process is as follows.

Step-1: Compute dense matches between curr and prev --> SetA
Step-2: Compute dense matches between curr and curr-1  --> SetB
Step-3: Find nn of each of the matches in SetA from SetB in daisy descriptor space.

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 7th Sept, 2017
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

import TerminalColors
tcol = TerminalColors.bcolors()


IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail.npy'
IMAGE_FILE_NPY_lut = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut_raw.npy'
LOOP_CANDIDATES_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/loop_candidates.csv'


print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
print 'Reading : ', IMAGE_FILE_NPY_lut
S_thumbnails_lut = np.load(IMAGE_FILE_NPY_lut)
print 'Reading : ', IMAGE_FILE_NPY_lut_raw
S_thumbnails_lut_raw = np.load(IMAGE_FILE_NPY_lut_raw)
print 'S_thumbnails.shape : ', S_thumbnails.shape


print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )

# which loop_candidates to load
if len(sys.argv) == 2:
    i=int(sys.argv[1])
else:
    i = 0



l = loop_candidates[i]
curr = int(l[0])
prev = int(l[1])
score = l[2]
nMatches = int(l[3])
nConsistentMatches = int(l[4])


print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)

# Get image triple
curr_im = S_thumbnails[curr, :,:,:]
prev_im = S_thumbnails[prev, :,:,:]
curr_m_im = S_thumbnails[curr-1, :,:,:] #curr-1

# Get lut_raw triple
lut_raw_curr_im = S_thumbnails_lut_raw[curr, :,:]
lut_raw_prev_im = S_thumbnails_lut_raw[prev, :,:]
lut_raw_curr_m_im = S_thumbnails_lut_raw[curr-1, :,:] #curr-1



VV = GeometricVerification()

#
# Step-1: Compute dense matches between curr and prev --> SetA
# curr <--> prev
#
VV.set_im( curr_im, prev_im )
VV.set_im_lut_raw( lut_raw_curr_im, lut_raw_prev_im )
nMatches, nInliners = VV.simple_verify(features='orb')
print 'Sparse Matching : nMatches=%d, nInliners=%d' %(nMatches, nInliners)


pts_curr, pts_prev, mask_c_p = VV.daisy_dense_matches()
# pts_curr, pts_prev, mask_c_p, xcanvas_array = VV.daisy_dense_matches(DEBUG=True)
xcanvas_c_p = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask_c_p)

# print 'len(xcanvas_array)', len(xcanvas_array), xcanvas_array[0].shape


#
# Step-2:
# Match expansion onto curr-1 using (curr,prev)
#
startMatchExpansion = time.time()
_pts_curr_m = VV.expand_matches_to_curr_m( pts_curr, pts_prev, mask_c_p,  curr_m_im )
print 'Time taken for match expansion : %4.2f' %( 1000.*(time.time() - startMatchExpansion) )

masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )
bigshow = VV.plot_3way_match_upscale( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )
cv2.imshow( 'gridd', gridd )
cv2.imshow( 'bigshow', bigshow )
cv2.waitKey(0)


#
# Step-3: Relative pose
# Note : At this stage 3 already have a 3-way match : ie. masked_pts_curr <--> masked_pts_prev <--> _pts_curr_m

# 3.1 Triangulate pts in curr-1 and curr.


# 3.2 pnp( 3d pts from (3.1) ,  prev )
code.interact( local=locals() )


quit()

# desc_curr = VV.daisy_im1[ ,: ]
masked_pts_curr = list( pts_curr[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )
masked_pts_prev = list( pts_prev[i] for i in np.where( mask_c_p[:,0] == 1 )[0] )


_pts_curr = np.array( masked_pts_curr )
D_pts_curr = VV.daisy_im1[ _pts_curr[:,1], _pts_curr[:,0], : ]
xcanvas_c_p = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask_c_p)
cv2.imshow( 'xcanvas_c_p', xcanvas_c_p)


#
# Alternate Step-2:: Find the matched points in curr (from step-1) in c-1. Using a
# bounding box around each point
#   INPUTS : curr_m_im, _pts_curr, D_pts_curr
#   OUTPUT : _pts_curr_m

daisy_c_m = VV.static_get_daisy_descriptor_mat(  curr_m_im  )#Daisy of (curr-1)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)



_pts_curr_m = []
for h in range(_pts_curr.shape[0]):
    print h, _pts_curr[h,:] # D_pts_curr[h,:]

    y = curr_im.copy()
    z = curr_m_im.copy()

    _W = 20
    row_range = range( max(0,_pts_curr[h,1]-_W),  min(curr_m_im.shape[0], _pts_curr[h,1]+_W) )
    col_range = range( max(0,_pts_curr[h,0]-_W),  min(curr_m_im.shape[1], _pts_curr[h,0]+_W) )
    g = np.meshgrid( row_range, col_range )
    positions = np.vstack( map(np.ravel, g) ) #2x400, these are in cartisian-indexing-convention


    # for p in range( positions.shape[1] ):
        # cv2.circle( y, (positions[1,p], positions[0,p]), 1, (255,0,0) )

    cv2.circle( y, (_pts_curr[h,0],_pts_curr[h,1]), 1, (0,0,255) )
    cv2.imshow( 'y', y)



    D_positions = daisy_c_m[positions[0,:], positions[1,:],:] #400x20
    print D_positions.shape


    matches = flann.knnMatch(np.expand_dims(D_pts_curr[h,:], axis=0).astype('float32'),D_positions.astype('float32'),k=1)
    matches = matches[0][0]
    print 'match : %d <--%4.2f--> %d' %(matches.queryIdx, matches.distance, matches.trainIdx )
    cv2.circle( z, tuple(positions[[1,0],matches.trainIdx]), 2, (0,0,255) )

    _pts_curr_m.append( tuple(positions[[1,0],matches.trainIdx]) )

    cv2.imshow( 'z', z)
    cv2.waitKey(30)
    # code.interact( local=locals() )



    # matches = flann.knnMatch(D_pts_curr.astype('float32'),D_pts_curr_m.astype('float32'),k=1) # this gives nn for each of the point in 1st desc set
    # print __U.shape


gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, _pts_curr_m )
cv2.imshow( 'gridd', gridd )
cv2.waitKey(0)

quit()
#
# Step-2: Compute dense matches between curr and curr-1  --> SetB
# curr <--> curr-1
#
VV.set_im( curr_im, curr_m_im )
VV.set_im_lut_raw( lut_raw_curr_im, lut_raw_curr_m_im )
pts_curr2, pts_curr_m, mask_c_c_m = VV.daisy_dense_matches()

masked_pts_curr2 = list( pts_curr2[i] for i in np.where( mask_c_c_m[:,0] == 1 )[0]  )
masked_pts_curr_m = list( pts_curr_m[i] for i in np.where( mask_c_c_m[:,0] == 1 )[0]  )

_pts_curr_m = np.array( masked_pts_curr_m )
D_pts_curr_m = VV.daisy_im2[ _pts_curr_m[:,1], _pts_curr_m[:,0], : ]
xcanvas_c_c_m = VV.plot_point_sets( VV.im1, pts_curr2, VV.im2, pts_curr_m, mask_c_c_m)

#
# Step-3 :
# use, `D_pts_curr` and `D_pts_curr_m`
# usem `pts_curr` and `pts_curr_m`

# Find NN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(D_pts_curr.astype('float32'),D_pts_curr_m.astype('float32'),k=1) # this gives nn for each of the point in 1st desc set

nn_curr_m = []
for m in matches:
    print m[0].queryIdx, m[0].trainIdx, m[0].distance
    nn_curr_m.append( pts_curr_m[m[0].trainIdx] )


    # print pts_curr[ m[0].queryIdx ] , '<--->', pts_curr_m[m[0].trainIdx  ]
code.interact( local=locals() )

## Make image grid : [ [curr, prev], [curr-1  X ] ]
gridd = VV.plot_3way_match( curr_im, masked_pts_curr, prev_im, masked_pts_prev, curr_m_im, nn_curr_m )
cv2.imshow( 'gridd', gridd )
# cv2.waitKey(0)
# code.interact( local=locals() )



cv2.imshow( 'xcanvas_c_p', xcanvas_c_p )
cv2.imshow( 'xcanvas_c_c_m', xcanvas_c_c_m )

cv2.waitKey(0)
