""" Tries to answer the question what the network sees as important.

    1. Read an image `A`
    2. Get image desctriptor N(A)
    3. Blackken a 50x50 patch of the image A at (i,j) to obtain `B`
    4. Get descriptor of `B`
    5. I(i,j) = dot( A, B )
    6. Repeat 3,4,5 for every (i,j) with a stride of s
    7. Visualize heatmap I

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 12th June, 2017
"""
import numpy as np
import cv2
import networkx as nx
import code
import time
import json
import pickle
import rospkg
import copy
#
import TerminalColors
tcolor = TerminalColors.bcolors()
# tcol = t

from FastPlotter import FastPlotter

import matplotlib.pyplot as plt

import tensorflow as tf
from CartWheelFlow import VGGDescriptor



PKG_PATH = rospkg.RosPack().get_path('nap')


## Normalize colors for an image - used in NetVLAD descriptor extraction
def rgbnormalize( im ):
    im_R = im[:,:,0].astype('float32')
    im_G = im[:,:,1].astype('float32')
    im_B = im[:,:,2].astype('float32')
    S = im_R + im_G + im_B
    out_im = np.zeros(im.shape)
    out_im[:,:,0] = im_R / (S+1.0)
    out_im[:,:,1] = im_G / (S+1.0)
    out_im[:,:,2] = im_B / (S+1.0)

    return out_im

## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)


## Normalize a batch of images. Calls `rgbnormalize()` - used in netvlad computation
def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        # im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )
        im_batch_normalized[b,:,:,0] = zNormalize( im_batch[b,:,:,0])
        im_batch_normalized[b,:,:,1] = zNormalize( im_batch[b,:,:,1])
        im_batch_normalized[b,:,:,2] = zNormalize( im_batch[b,:,:,2])

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized

def logistic_func( x ): #x is scalar
    return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

# Given an image, computes its netvlad word
def compute_netvlad( IM ):
    im_batch = np.zeros( (1,IM.shape[0],IM.shape[1],IM.shape[2]) )
    im_batch[0,:,:,:] = IM.astype('float32')
    im_batch_normalized = normalize_batch( im_batch )
    feed_dict = {tf_x : im_batch_normalized,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
    d_WORD = tff_vlad_word[0,:]
    # code.interact( local=locals() )
    return d_WORD


def get_importance_heatmap( IM, param_mask_halfsize, param_stride, verbose=False ):
    print 'param_mask_halfsize=%d, param_stride=%d' %(param_mask_halfsize, param_stride)

    h = param_mask_halfsize
    s = param_stride
    heatmap = np.zeros( (len(range(h,IM.shape[0]-h, s )), len(range(h,IM.shape[1]-h, s))) )

    for ei, i in enumerate( range(h,IM.shape[0]-h, s ) ):
        for ej, j in enumerate( range(h,IM.shape[1]-h, s) ):
            IM_eclipsed = copy.deepcopy( IM )
            IM_eclipsed[i-h:i+h,j-h:j+h,:] = 255

            # Compute NetVLAD for IM_eclipsed
            im_eclipsed_word = compute_netvlad( IM_eclipsed )


            # print np.dot( im_eclipsed_word, im_word )
            heatmap[ei,ej] = logistic_func( np.dot( im_eclipsed_word, im_word ) )

            if verbose:
                cv2.imshow( 'IM', IM )
                cv2.imshow( 'IM_eclipsed', IM_eclipsed )
                cv2.waitKey(10)

    return heatmap

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

#--- Load Data ---#
folder = '/DUMP/ust_walk/'
print 'Loading : ',PKG_PATH+folder+'S_full_images.npy'
S_full_im = np.load( PKG_PATH+folder+'S_full_images.npy' )#N x 240 x 320 x 3
#--- END ---#

#--- PARAMS ---#
PARAM_stride = 30
PARAM_mask_halfsize = 50 #can be even or odd
# PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k48/model-13000'
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k64_tokyoTM/model-3500' #trained with tokyo, normalization is simple '

# PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'

#--- END ---#





#--- Init NetVLAD ---#

#
# Init netvlad - def computational graph, load trained model
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=1, K=64)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )
#--- END ---#

# for idx in range(0,1600,50):
idx = 2500
# Select Image
IM = S_full_im[idx,:,:,:]


# Compute NetVLAD for IM
im_word = compute_netvlad( IM )


# ## --- Masking --- ###
# # Heatmap for Masking
# for m in range(10,60, 5):
#     # heatmap = get_importance_heatmap( IM, param_mask_halfsize, param_stride, verbose=False )
#     heatmap = get_importance_heatmap( IM, m, int(m/2), verbose=True )
#
#     cv2.imshow( 'IM', IM )
#     cv2.waitKey(1)
#     plt.clf()
#     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
#


# ### --- Masking Movie --- ###
# # Generate an importance movie of the entire sequence
# for idx in range(50,S_full_im.shape[0], 100):
#     startTime = time.time()
#     heatmap = get_importance_heatmap( S_full_im[idx,:,:,:], 10, 5, verbose=False )
#     cv2.imshow( 'win', S_full_im[idx,:,:,:])
#     cv2.waitKey(1)
#     plt.clf()
#     plt.subplot(121)
#     plt.imshow( S_full_im[idx,:,:,::-1 ]  )
#     plt.subplot(122)
#     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     # plt.show()
#     plt.savefig( PKG_PATH+folder+'/heatmap_'+str(idx)+'.png')
#     print 'Image %d done in %4.2fs' %( idx, time.time() - startTime)
#
#
# quit()


#### -- Transformations -- ###

rows,cols, _ = IM.shape

# # Planar Translation
# heatmap = np.zeros( ( len(range(-100, 100, 5 )), len(range(-100, 100, 5 )) ) )
# for ixx, xx in enumerate( range(-100, 100, 5 ) ):
#     for iyy, yy in enumerate( range( -100, 100, 5 ) ):
#         M = np.float32([[1,0,xx],[0,1,yy]])
#         dst = cv2.warpAffine(IM,M,(cols,rows))
#
#         dst_word = compute_netvlad(dst)
#         heatmap[ixx, iyy] = np.dot( im_word, dst_word )
#
#         cv2.imshow( 'IM', IM )
#         cv2.imshow( 'dst', dst )
#         cv2.waitKey(10)
#
# plt.imshow(heatmap, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
# quit()

# # Rotation
# rot_scores = []
# for irot in range(-90,90):
#     M = cv2.getRotationMatrix2D((cols*.75,rows*.75),irot,1.)
#     dst = cv2.warpAffine(IM,M,(cols,rows))
#
#     dst_word = compute_netvlad(dst)
#     print irot, np.dot( dst_word, im_word)
#     rot_scores.append(np.dot( dst_word, im_word))
#
#     cv2.imshow( 'IM', IM )
#     cv2.imshow( 'dst', dst )
#     cv2.waitKey(10)
#
# plt.plot( range(-90,90), rot_scores)
# plt.show()


# # Random Affine Transform
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
# for t in range(50):
#     a = np.float32( [ np.random.randint(-30,120 ),  np.random.randint(-30,100 ) ]   )
#     b = np.float32( [ np.random.randint(150,290 ),  np.random.randint(20,90 ) ] )
#     c = np.float32( [ np.random.randint(80,150 ), np.random.randint(190,290 ) ] )
#     pts2 = np.stack( (a,b,c), axis=0 )
#     print pts2
#
#     M = cv2.getAffineTransform(pts1,pts2)
#     dst = cv2.warpAffine(IM,M,(cols,rows))
#     dst_word = compute_netvlad(dst)
#     print np.dot( dst_word, im_word)
#     cv2.imshow( 'IM', IM )
#     cv2.imshow( 'dst', dst )
#     cv2.waitKey(0)
# quit()

# Scaling Transform `alpha*I + beta`
# Note, Netvlad is invariant to alpha, but performs very bad to variations in beta
# dst = copy.deepcopy( IM )

score_ary = []
alpha_range = np.linspace( 0.5, 1.6, 50 )
beta_range = np.linspace( -40, 40, 50 )
gamma_range = np.linspace( 0.5, 1.5, 20 )
# for alpha in alpha_range:
alpha = 1.0
# for beta in beta_range:
for gamma in gamma_range:
    # dst = cv2.convertScaleAbs( IM, alpha=alpha, beta=beta)
    dst = adjust_gamma( IM, gamma )

    dst_word = compute_netvlad(dst)
    print gamma, np.dot( dst_word, im_word)
    print gamma, np.linalg.norm( dst_word- im_word)
    score_ary.append( np.dot( dst_word, im_word) )

    cv2.imshow( 'dst', dst)
    cv2.imshow( 'IM', IM )
    cv2.waitKey(0)

# plt.plot(alpha_range, score_ary )
# plt.plot(beta_range, score_ary )
plt.plot(gamma_range, score_ary )
plt.show()
