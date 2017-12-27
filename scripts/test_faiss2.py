import numpy as np
import cv2
from faiss import faiss
import rospkg
import time
import code

import matplotlib.pyplot as plt

import TerminalColors
tcol = TerminalColors.bcolors()

PKG_PATH = '/home/mpkuse/Desktop/a/drag_nap' #rospkg.RosPack().get_path('nap')


#--- Load Files ---#
S_word_filename = PKG_PATH + '/S_word.npy'
print tcol.OKGREEN, 'Load : ', S_word_filename, tcol.ENDC
S_word = np.load( S_word_filename )

S_thumb_filename = PKG_PATH + '/S_thumbnail.npy'
print tcol.OKGREEN, 'Load : ', S_thumb_filename, tcol.ENDC
S_thumb = np.load( S_thumb_filename )

S_thumb_lut_filename = PKG_PATH + '/S_thumbnail_lut.npy'
print tcol.OKGREEN, 'Load : ', S_thumb_lut_filename, tcol.ENDC
S_thumb_lut = np.load( S_thumb_lut_filename )

S_timestamp_filename = PKG_PATH + '/S_timestamp.npy'
print tcol.OKGREEN, 'Load : ', S_timestamp_filename, tcol.ENDC
S_timestamp = np.load( S_timestamp_filename )
#--- END ---#

quit()

idx = 310
index = faiss.IndexFlatL2( S_word.shape[1] )
print index.is_trained
index.add( S_word )
D,I = index.search( np.expand_dims( S_word[idx,:], 0 ), k=40 )
print I
print index.ntotal


F = np.dot( S_word, S_word[idx,:] )
F_s = np.sqrt( 1.0 - np.minimum(1.0, F ) ) #minimum is added to ensure dot product doesnt go beyond 1.0 as it sometimes happens because of numerical issues, which inturn causes issues with sqrt
F_l = (1.0 / (1.0 + np.exp( 11.0*F_s - 3.0 )) + 0.01)
plt.plot( F )
plt.figure()
plt.plot( F_l )
plt.show(False)

cv2.imshow( 'im', S_thumb[idx,:,:,:] )
cv2.waitKey(0)
