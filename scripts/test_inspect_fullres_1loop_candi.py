""" Try and match features from surf features or similar. Basically a old-school
    technique. 2 full resolution images are input. Try and find feature correspondences

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 15th Aug, 2017
"""
import numpy as np
import cv2
import code
import time
import sys
import rospkg
from operator import itemgetter
cv2.ocl.setUseOpenCL(False)

from GeometricVerification import GeometricVerification
from ColorLUT import ColorLUT

import TerminalColors
tcol = TerminalColors.bcolors()

PKG_PATH = rospkg.RosPack().get_path('nap')
IMAGE_FILE_NPY = PKG_PATH + '/DUMP/S_thumbnail.npy'
FULL_RES_IMAGE_FILE_NPY = PKG_PATH + '/DUMP/S_thumbnail_full_res.npy'
LOOP_CANDIDATES_NPY = PKG_PATH + '/DUMP/loop_candidates.csv'

# Read Images
print 'Reading : ', IMAGE_FILE_NPY
S_thumbnails = np.load(IMAGE_FILE_NPY)
print 'Reading : ', FULL_RES_IMAGE_FILE_NPY
S_full_res = np.load(FULL_RES_IMAGE_FILE_NPY)

# Read Loop Candidates
print 'Reading : ', LOOP_CANDIDATES_NPY
loop_candidates = np.loadtxt( LOOP_CANDIDATES_NPY, delimiter=',' )
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
