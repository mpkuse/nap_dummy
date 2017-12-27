""" Inspect a loop candidate
        In particular reader for file loop_candidates.npy which is written in
        main nap node as :

        loop_candidates.append( [L, aT, sim_scores_logistic[aT], nMatches, nInliers] )

        This script is to be used to test and develop the wide angle point feature
        matching algorithm.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 9th Aug, 2017
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

def s_overlay( im1, Z, alpha=0.5 ):
    lut = ColorLUT()
    out_imcurr = lut.lut( Z )

    out_im = alpha*im1 + (1 - alpha)*cv2.resize(out_imcurr,  (im1.shape[1], im1.shape[0])  )
    return out_im.astype('uint8')

IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail.npy'
FULL_RES_IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_full_res.npy'
IMAGE_FILE_NPY_lut = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut_raw.npy'
LOOP_CANDIDATES_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/loop_candidates.csv'

print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
S_full_res = np.load(FULL_RES_IMAGE_FILE_NPY)
S_thumbnails_lut = np.load(IMAGE_FILE_NPY_lut)
S_thumbnails_lut_raw = np.load(IMAGE_FILE_NPY_lut_raw)
print 'S_thumbnails.shape : ', S_thumbnails.shape
# for i in range(S_thumbnails.shape[0]):
#     print i, 'of', S_thumbnails.shape[0]
#     cv2.imshow( 'win', S_thumbnails[i,:,:,:] )
#     if cv2.waitKey(0) == 27:
#         break
# quit()

print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )

if len(sys.argv) == 2:
    i=int(sys.argv[1])
else:
    i = 0


VV = GeometricVerification()
#for i,l in enumerate(loop_candidates):
# [ curr, prev, score, nMatches, nConsistentMatches]


l = loop_candidates[i]
curr = int(l[0])
prev = int(l[1])
score = l[2]
nMatches = int(l[3])
nConsistentMatches = int(l[4])


print '%04d of %04d] curr=%04d, prev=%04d, score=%4.2f, nMatch=%3d, nConsistentMatch=%3d' %(i, loop_candidates.shape[0],curr,prev,score,nMatches,nConsistentMatches)

VV.set_im( S_thumbnails[curr, :,:,:] , S_thumbnails[prev, :,:,:]  )
VV.set_im_lut( S_thumbnails_lut[curr, :,:,:] , S_thumbnails_lut[prev, :,:,:]  )
VV.set_im_lut_raw( S_thumbnails_lut_raw[curr,:,:] , S_thumbnails_lut_raw[prev,:,:] )
nMatches, nInliners = VV.simple_verify(features='orb')

# TODO: If nInliers too less than attempt this. In this function set DEBUG=True to get
# debug output.
pts_curr, pts_prev, mask = VV.daisy_dense_matches()

xcanvas = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev)
xcanvas_verified = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask)

cv2.imshow( 'xcanvas', xcanvas )
cv2.imshow( 'xcanvas_verified', xcanvas_verified )
cv2.moveWindow( 'xcanvas', 800, 0)
cv2.moveWindow( 'xcanvas_verified', 800, 400)
cv2.waitKey(0)


quit()

Z_curr = VV.prominent_clusters(im_no=1)
Z_prev = VV.prominent_clusters(im_no=2)

##################################################
## Full Image Descriptors on Prominent Clusters ##
##      Input: im1, im2, Z_curr, Z_prev         ##
##################################################

# Step-1 : Get Daisy at every point
startDaisy = time.time()
D_curr = VV.get_whole_image_daisy( im_no=1 )
D_prev = VV.get_whole_image_daisy( im_no=2 )
# cv2.imshow( 'daisy_curr', D_curr[:,:,0] )
# cv2.imshow( 'daisy_prev', D_prev[:,:,0] )
print tcol.OKBLUE, 'get_whole_image_daisy (ms): %4.2f' %(1000. * (time.time() - startDaisy)), tcol.ENDC



# Step-2 : Given a k which is in both images, compare clusters with daisy. To do that do NN followd by Lowe's ratio test etc
Z_curr_uniq = np.unique( Z_curr )[1:] #from 1 to avoid 0 which is for no assigned cluster
Z_prev_uniq = np.unique( Z_prev )[1:]
print Z_curr_uniq
print Z_prev_uniq
print 'common K are '
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
for k in set(Z_curr_uniq).intersection( set(Z_prev_uniq) ):
    print k
    startk = time.time()
    H_curr = np.where( Z_curr==k ) #co-ordinates. these co-ordinates need testing
    desc_c = np.array( D_curr[ H_curr[0]*4, H_curr[1]*4 ] ) # This is since D_curr is (240,320) and H_curr is (60,80)

    H_prev = np.where( Z_prev==k ) #co-ordinates #remember , Z_prev is (80,60)
    desc_p = np.array( D_prev[ H_prev[0]*4, H_prev[1]*4 ] )

    matches = flann.knnMatch(desc_c.astype('float32'),desc_p.astype('float32'),k=2)
    canvas_dense = VV.analyze_dense_matches(  H_curr, H_prev, matches )
    print tcol.OKBLUE, 'for k=%d (ms): %4.2f' %(k, 1000. * (time.time() - startk)), tcol.ENDC

    # code.interact( local=locals() )


    overlay_curr = s_overlay(S_thumbnails[curr, :,:,:], np.int0(Z_curr==k) )
    overlay_prev = s_overlay(S_thumbnails[prev, :,:,:], np.int0(Z_prev==k) )

    cv2.circle( overlay_curr, (H_curr[1][0]*4, H_curr[0][0]*4), 4, (0,255,0), 2  )
    cv2.circle( overlay_prev, (H_prev[1][0]*4, H_prev[0][0]*4), 4, (0,255,0), 2  )

    cv2.imshow( 'overlay_curr',  overlay_curr )
    cv2.imshow( 'overlay_prev',  overlay_prev )
    cv2.imshow( 'canvas_dense', canvas_dense )
    cv2.waitKey(0)

cv2.imshow( 'overlay_curr', s_overlay(S_thumbnails[curr, :,:,:], Z_curr) )
cv2.imshow( 'overlay_prev', s_overlay(S_thumbnails[prev, :,:,:], Z_prev) )
cv2.waitKey(0)

#########
## END ##
#########
quit()


# Process Z to get rotated rectangles of Zs
startRotRect = time.time()
Q_curr = VV.get_rotated_rect( Z_curr ) # these are bounding rectangle co-ordinates (rotated) at each of the top clusters
Q_prev = VV.get_rotated_rect( Z_prev )
print 'Time taken for getting rotated rect', 1000. * (time.time() - startRotRect)
# code.interact( local=locals() )

# Visualize overlay of Z, Q on color-image. Possibly also on full resolution image
lut = ColorLUT()
out_imcurr = lut.lut( Z_curr )
out_imprev = lut.lut( Z_prev )


im_curr = 0.5*S_thumbnails[curr, :,:,:] + 0.5*cv2.resize(out_imcurr, (320,240) )#.copy()
im_prev = 0.5*S_thumbnails[prev, :,:,:] + 0.5*cv2.resize(out_imprev, (320,240) )#.copy()
im_curr = im_curr.astype('uint8')
im_prev = im_prev.astype('uint8')

im_curr_full_res = S_full_res[curr, :,:,:].copy()
im_prev_full_res = S_full_res[prev, :,:,:].copy()

im_curr_full_resf = S_full_res[curr, :,:,:].copy()
im_prev_full_resf = S_full_res[prev, :,:,:].copy()

# for Q_key in Q_curr.keys():

for Q_key in set(Q_curr.keys()).intersection( set(Q_prev.keys()) ): # loop over k which are common in both curr and prev
    print 'Q_key : ', Q_key


    contour_i = 0
    affine_patches_curr =  []
    for contour,contour_occupied_area,contour_total_area in Q_curr[Q_key]: # Loop over bounding boxes of curr and drawBox
        color = (255,0,0) #just replace this as xcolor for simple coloring
        xcolor = tuple(lut.get_color( int(Q_key) ) )
        xcolor2 = (int(xcolor[2]), int(xcolor[1]), int(xcolor[0]))
        cv2.drawContours( im_curr, [4*contour], 0, xcolor2, 2 )
        cv2.drawContours( im_curr_full_res, [ [12,10]*contour], 0, xcolor2, 2 )
        # cv2.circle( im_curr_full_res, tuple([12,10]*contour[0]), 5, (255,0,0) )
        # cv2.circle( im_curr_full_res, tuple([12,10]*contour[1]), 5, (0,255,0) )
        # cv2.circle( im_curr_full_res, tuple([12,10]*contour[2]), 5, (0,0,255) )
        _J = tuple([12,10]* np.mean(contour, axis=0, dtype='int32') )
        cv2.putText( im_curr_full_res, str(contour_i), _J, cv2.FONT_HERSHEY_SIMPLEX, 0.8, xcolor2 )
        print 'curr%02d occupied=%4.2f total=%4.2f (%2.2f %%)' %( contour_i, contour_occupied_area,contour_total_area, 100.*contour_occupied_area/contour_total_area )

        # affine warp this patch
        xout_im = VV.affine_warp( im_curr_full_resf,  [12,10]*contour )
        affine_patches_curr.append( xout_im )

        contour_i += 1


    #this 4 is because, the rotated rect is on the cluster_assignment image which is 1/4 (60x80) the original image (240,320).
    contour_i = 0
    affine_patches_prev = []
    for contour,contour_occupied_area,contour_total_area in Q_prev[Q_key]: #similar to above for but for prev image
        color = (0,0,255)
        xcolor = tuple(lut.get_color( int(Q_key) ))
        xcolor2 = (int(xcolor[2]), int(xcolor[1]), int(xcolor[0]))
        cv2.drawContours( im_prev, [4*contour], 0, xcolor2, 2 )
        cv2.drawContours( im_prev_full_res, [ [12,10]*contour], 0, xcolor2, 2 )
        cv2.circle( im_prev_full_res, tuple([12,10]*contour[0]), 5, (255,0,0) )
        cv2.circle( im_prev_full_res, tuple([12,10]*contour[1]), 5, (0,255,0) )
        cv2.circle( im_prev_full_res, tuple([12,10]*contour[2]), 5, (0,0,255) )
        # _J = tuple([12,10]*np.int32(np.mean(contour)))
        _J = tuple([12,10]* np.mean(contour, axis=0, dtype='int32') )
        cv2.putText( im_prev_full_res, str(contour_i), _J, cv2.FONT_HERSHEY_SIMPLEX, 0.8, xcolor2 )
        print 'prev%02d occupied=%4.2f total=%4.2f (%2.2f %%)' %( contour_i, contour_occupied_area,contour_total_area, 100.*contour_occupied_area/contour_total_area )

        # affine warp this patch
        xout_im = VV.affine_warp( im_prev_full_resf,  [12,10]*contour )
        affine_patches_prev.append( xout_im )

        contour_i += 1

    cv2.imshow( 'im_curr', im_curr)
    cv2.imshow( 'im_prev', im_prev)
    cv2.imshow( 'im_curr_full_res', im_curr_full_res)
    cv2.imshow( 'im_prev_full_res', im_prev_full_res)
    cv2.moveWindow( 'im_prev_full_res', 980, 0)
    cv2.moveWindow( 'im_curr', 0, 750)
    cv2.moveWindow( 'im_prev', 980, 750)
    # code.interact( local=locals() )
    cv2.imshow( 'affine_patches_curr', np.concatenate( affine_patches_curr, axis=0 ) )
    cv2.imshow( 'affine_patches_prev', np.concatenate( affine_patches_prev, axis=0 ) )


    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        # break
        pass
