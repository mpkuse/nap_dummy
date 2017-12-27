## class VGGDescriptor in CartWheelFlow.py
## Test the cluster assignment (vgg_obj.nl_sm)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim

# from PandaRender import NetVLADRenderer
from CartWheelFlow import VGGDescriptor

#
import TerminalColors
tcolor = TerminalColors.bcolors()




## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)

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


def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        im_batch_normalized[b,:,:,0] = zNormalize( im_batch[b,:,:,0])
        im_batch_normalized[b,:,:,1] = zNormalize( im_batch[b,:,:,1])
        im_batch_normalized[b,:,:,2] = zNormalize( im_batch[b,:,:,2])
        # im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized


# test - looking at assignment of cluster. I have a doubt that the assignment is too soft (19th June, 2017)

# np.random.seed(1)
im_batch_size = 10
XX = np.zeros( (im_batch_size,240,320,3) )
for i in range(im_batch_size):
    nx = 600+2*i
    # nx = np.random.randint(1000)

    im_file_name = '../tf.logs/netvlad_k48/db_xl/im/%d.jpg' %(nx)
    # im_file_name = 'other_seqs/FAB_MAP_IJRR2008_DATA/City_Centre/Images/%04d.jpg' %(nx)
    print 'Reading : ', im_file_name
    XX[i,:,:,:] = cv2.resize( cv2.imread( im_file_name ), (320,240) )
y = normalize_batch( XX )





tf_x = tf.placeholder( 'float', [None,240,320,3], name='conv_desc' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor(b=im_batch_size, K=64)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
# tf_vlad_word = vgg_obj.vgg16_raw_features(tf_x, is_training)
# netvlad = vgg_obj.netvlad_layer( tf_vlad_word )

sess = tf.Session()
if False: #random init
    sess.run(tf.global_variables_initializer())
else: # load from a trained model
    tensorflow_saver = tf.train.Saver()
    # PARAM_model_restore = 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
    # PARAM_model_restore = 'tf.logs/netvlad_k48/model-13000'
    # PARAM_model_restore = 'tf.logs/netvlad_k64_znormed/model-2000' # trained from 3d model z-normalize R,G,B individual,
    PARAM_model_restore = '../tf.logs/netvlad_k64_tokyoTM/model-2500'

    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver.restore( sess, PARAM_model_restore )



tff_netvlad, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm ],  feed_dict={tf_x:y,  is_training:False, vgg_obj.initial_t: 0})

code.interact( local=locals() )
# veri_sm = verify_membership_mat( tff_netvlad_conv )
# veri_netvlad = verify_vlad( tff_vlad_c, XX, tff_sm )

# reshape tff_sm (b*60*80 x K)
tff_sm_variance = tff_sm[:,0]
for h in range(tff_sm.shape[0]):
    tff_sm_variance[h] = tff_sm[h,:].var()
tff_sm_variance = np.reshape( tff_sm_variance, [im_batch_size,60,80] )
#h=2800;plt.plot( tff_sm[h,:] ); plt.title( str(tff_sm[h,:].var()) ); plt.show()


Assgn_matrix = np.reshape( tff_sm, [im_batch_size,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1

for i in range(im_batch_size):
    plt.subplot(2,2,1)
    plt.imshow( Assgn_matrix[i], cmap='hot')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow( XX[i,:,:,::-1].astype('uint8') )

    plt.subplot(2,2,3)
    plt.hist( Assgn_matrix[i].flatten(), bins=48 )

    plt.subplot(2,2,4)
    plt.imshow( -np.log(tff_sm_variance[i,:,:]), cmap='hot' )
    plt.colorbar()

    plt.show()



quit()
