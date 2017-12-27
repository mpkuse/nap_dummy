""" Inspect loop candidates one at a time
        In particular reader for file loop_candidates.npy which is written in
        main nap node as :

        loop_candidates.append( [L, aT, sim_scores_logistic[aT], nMatches, nInliers] )

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 6th July, 2017
"""

import numpy as np
import cv2
import code
cv2.ocl.setUseOpenCL(False)
import time
from GeometricVerification import GeometricVerification


IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail.npy'
IMAGE_FILE_NPY_lut = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut.npy'
IMAGE_FILE_NPY_lut_raw = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_lut_raw.npy'
LOOP_CANDIDATES_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/loop_candidates.csv'

FULL_RES_IMAGE_FILE_NPY = '/home/mpkuse/catkin_ws/src/nap/DUMP/S_thumbnail_full_res.npy'



import TerminalColors
tcol = TerminalColors.bcolors()
def print_time(msg, startT, endT):
    print tcol.OKBLUE, '%8.2f :%s (ms)'  %( 1000. * (endT - startT), msg ), tcol.ENDC


print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
print 'Reading : ', IMAGE_FILE_NPY_lut
S_thumbnails_lut = np.load(IMAGE_FILE_NPY_lut)
print 'Reading : ', IMAGE_FILE_NPY_lut_raw
S_thumbnails_lut_raw = np.load(IMAGE_FILE_NPY_lut_raw)
print 'S_thumbnails.shape : ', S_thumbnails.shape
# for i in range(S_thumbnails.shape[0]):
#     print i, 'of', S_thumbnails.shape[0]
#     cv2.imshow( 'win', S_thumbnails[i,:,:,:] )
#     if cv2.waitKey(0) == 27:
#         break
# quit()


print 'Reading : ', FULL_RES_IMAGE_FILE_NPY
S_full_res = np.load(FULL_RES_IMAGE_FILE_NPY)


print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )

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
    flag = True #just remove this if u want to group matches
    if flag is True and score>0.5:
        flag = False


        #
        # alpha_curr = 0.45*S_thumbnails[curr, :,:,:] + 0.35*cv2.resize( S_thumbnails_lut[curr, :,:,:], (320,240))
        # alpha_prev = 0.45*S_thumbnails[prev, :,:,:] + 0.35*cv2.resize( S_thumbnails_lut[prev, :,:,:], (320,240))
        # cv2.imshow( 'alpha_curr', alpha_curr.astype('uint8') )
        # cv2.imshow( 'alpha_prev', alpha_prev.astype('uint8') )
        # # cv2.imwrite( '%d.png' %(curr), S_thumbnails[curr, :,:,:])
        # # cv2.imwrite( '%d.png'%(prev), S_thumbnails[prev, :,:,:])
        #
        #
        #
        VV.set_im( S_thumbnails[curr, :,:,:] , S_thumbnails[prev, :,:,:] )
        # VV.set_im_lut( cv2.resize(S_thumbnails_lut[curr, :,:,:], (320,240), interpolation=cv2.INTER_NEAREST), cv2.resize(S_thumbnails_lut[prev, :,:,:], (320,240), interpolation=cv2.INTER_NEAREST))
        # VV.set_im_lut_raw( cv2.resize(S_thumbnails_lut_raw[curr, :,:], (320,240), interpolation=cv2.INTER_NEAREST), cv2.resize(S_thumbnails_lut_raw[prev, :,:], (320,240), interpolation=cv2.INTER_NEAREST))
        VV.set_im_lut( S_thumbnails_lut[curr, :,:,:] , S_thumbnails_lut[prev, :,:,:]  )
        VV.set_im_lut_raw( S_thumbnails_lut_raw[curr,:,:] , S_thumbnails_lut_raw[prev,:,:] )
        nMatches, nInliers = VV.simple_verify(features='orb', debug_images=False)
        # VV.simple_verify(features='surf')

        if nInliers < 20:
            cv2.imshow( 'curr', S_thumbnails[curr, :,:,:] )
            cv2.imshow( 'prev', S_thumbnails[prev, :,:,:] )

            cv2.imshow( 'curr_lut', cv2.resize( S_thumbnails_lut[curr, :,:,:], (320,240)) )
            cv2.imshow( 'prev_lut', cv2.resize( S_thumbnails_lut[prev, :,:,:], (320,240)) )

            nMatches, nInliers, canvas = VV.simple_verify(features='orb', debug_images=True)

            # # Do Daisy - uncomment this to get it working
            # # !(curr <--> prev)
            # startVeri1 = time.time()
            # VV.set_im( S_thumbnails[curr, :,:,:] , S_thumbnails[prev, :,:,:] )
            # VV.set_im_lut( S_thumbnails_lut[curr, :,:,:] , S_thumbnails_lut[prev, :,:,:]  )
            # VV.set_im_lut_raw( S_thumbnails_lut_raw[curr,:,:] , S_thumbnails_lut_raw[prev,:,:] )
            # pts_curr, pts_prev, mask = VV.daisy_dense_matches()
            # xcanvas_verified1 = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask)
            # print_time( 'curr <--> prev', startVeri1, time.time()  )
            #
            # # !(curr <--> prev)
            # startVeri2 = time.time()
            # VV.set_im( S_thumbnails[curr-1, :,:,:] , S_thumbnails[curr, :,:,:] )
            # VV.set_im_lut( S_thumbnails_lut[curr-1, :,:,:] , S_thumbnails_lut[curr, :,:,:]  )
            # VV.set_im_lut_raw( S_thumbnails_lut_raw[curr-1,:,:] , S_thumbnails_lut_raw[curr,:,:] )
            # pts_curr, pts_prev, mask = VV.daisy_dense_matches()
            # xcanvas_verified2 = VV.plot_point_sets( VV.im1, pts_curr, VV.im2, pts_prev, mask)
            # print_time( 'curr-1 <--> curr', startVeri2, time.time()  )
            #
            # cv2.imshow( 'xcanvas_verified1 curr to prev', xcanvas_verified1 )
            # cv2.imshow( 'xcanvas_verified2 curr-1 to prev', xcanvas_verified2 )
            # cv2.moveWindow( 'xcanvas_verified1 curr to prev', 800, 0)
            # cv2.moveWindow( 'xcanvas_verified2 curr-1 to prev', 800, 400)
            # # END Daisy

            cv2.moveWindow( 'prev', 350, 0 )
            cv2.moveWindow( 'curr_lut', 0, 350 )
            cv2.moveWindow( 'prev_lut', 350, 350 )
            cv2.moveWindow( 'canvas', 0, 750 )

            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                break
