""" Given candidates verify using epipolar geometry

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 7th June, 2017
"""

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
import rospy
import rospkg
import time
import code

import TerminalColors
tcol = TerminalColors.bcolors()

PKG_PATH = rospkg.RosPack().get_path('nap')

def lowe_ratio_test( kp1, kp2, matches ):
    good = []
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append( kp2[m.trainIdx].pt )
            pts1.append( kp1[m.queryIdx].pt )
    return np.int32(pts1), np.int32(pts2), good

def draw_keypts( im, ary_pts ):
    for k in ary_pts:
        im = cv2.circle( im, tuple(np.int32(k)), 2, (255,0,0), -1  )
    return im

## Load
folder = '/tpt_night_loop/'
candidates = np.load( PKG_PATH+'/DUMP/%s/loop_candidates.npy' %(folder) )
full_im = np.load( PKG_PATH+'/DUMP/%s/S_full_images.npy' %(folder) )

orb = cv2.ORB_create()
# orb = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

for l in range(len(candidates)):
    __a = int(candidates[l,0])
    __b = int(candidates[l,1])
    __c = int(candidates[l,2])
    __d = candidates[l,3]
    print '%d <--> (%d, %d) %4.2f' %(__a, __b, __c, __d)
    if __d > 0.05:
        continue

    im1 = full_im[ __a, :,:, : ].astype('uint8')
    im2 = full_im[ __b, :,:, : ].astype('uint8')
    im3 = full_im[ __c, :,:, : ].astype('uint8')
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    kp3, des3 = orb.detectAndCompute(im3, None)


    matches_12 = flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2)
    # matches_12 = sum( matches_12, [] )
    pts1, pts2, good_matches_12 = lowe_ratio_test( kp1, kp2, matches_12)

    # cv2.imshow( 'kp1', draw_keypts(im1, pts1) )
    # cv2.imshow( 'org_kp1', cv2.drawKeypoints(im1, kp1, None ) )
    # cv2.waitKey(0)
    # code.interact( local=locals() )


    matches_13 = flann.knnMatch(des1.astype('float32'),des3.astype('float32'),k=2)
    # matches_13 = sum( matches_13, [] )
    _, _, good_matches_13 = lowe_ratio_test( kp1, kp3, matches_13)




    im_matches12 = cv2.drawMatches(im1,kp1,  im2,kp2,  good_matches_12, None)
    im_matches13 = cv2.drawMatches(im1,kp1,  im3,kp3,  good_matches_13, None)

    cv2.imshow( 'match12', im_matches12 )
    cv2.imshow( 'match13', im_matches13 )


    # im1_keypts = cv2.drawKeypoints(im1, kp1, None )
    # im2_keypts = cv2.drawKeypoints(im2, kp2, None )
    # im3_keypts = cv2.drawKeypoints(im3, kp3, None )
    # cv2.imshow('keypts1', im1_keypts)
    # cv2.imshow('keypts2', im2_keypts)
    # cv2.imshow('keypts3', im3_keypts)

    # cv2.imshow( 'im1', im1)
    # cv2.imshow( 'im2', im2)
    cv2.waitKey(0)
