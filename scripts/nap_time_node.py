#!/usr/bin/python
""" Subscribes to images topic for every key-frame (or semi key frame) images.
    Publish asynchronously time-time message when a loop is detected.
    Images are indexed by time. In the future possibly index with distance
    using an IMU. Note that this script does not know about poses (generated from SLAM system)

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 3rd Apr, 2017
"""


import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32

from nap.msg import NapMsg

import cv2
from cv_bridge import CvBridge, CvBridgeError

import Queue
import numpy as np


import time
from collections import namedtuple
import code
import os
import signal
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

from CartWheelFlow import VGGDescriptor
import DimRed
from Quaternion import Quat
# import VPTree

# import matplotlib.pyplot as plt
import pyqtgraph as pg

#
import TerminalColors
tcolor = TerminalColors.bcolors()


# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'



PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'

PARAM_FPS = 25
PARAM_FRAMES_SKIP = 2


def signal_handler(signal, frame ):
    print 'Written File : ', PKG_PATH+'/vote_bank.npy'
    np.save( PKG_PATH+'/vote_bank.npy', vote_bank )
    print 'You Pressed CTRL+C'
    sys.exit(0)


call_q = 0
def callback_image( data ):
    global call_q
    n_SKIP = PARAM_FRAMES_SKIP

    rospy.logdebug( 'Received Image : %d,%d' %( data.height, data.width ) )
    cv_image = CvBridge().imgmsg_to_cv2( data, 'bgr8' )

    if call_q%n_SKIP == 0: #only use 1 out of 10 images
        im_queue.put( cv_image )
        im_timestamp_queue.put(data.header.stamp)
    call_q = call_q + 1




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

## Normalize a batch of images. Calls `rgbnormalize()` - used in netvlad computation
def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized


## 'x' can also be a vector
def logistic( x ):
    #y = np.array(x)
    #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
    # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)

    y = np.convolve( np.array(x), [0.25,0.5,0.25], 'same' )
    return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)



def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )
#
# Setup Callbacks and Publishers
im_queue = Queue.Queue()
im_timestamp_queue = Queue.Queue()

rospy.init_node( 'nap_time_node', log_level=rospy.INFO )
# rospy.Subscriber( 'chatter', String, callback_string )
rospy.Subscriber( '/semi_keyframes', Image, callback_image )
# rospy.Subscriber( '/color_image/image_raw', Image, callback_image ) #subscribes to color image
# rospy.Subscriber( '/mv_29900616/image_raw_resized', Image, callback_image ) #subscribes to color image

pub_colocation = rospy.Publisher( '/colocation', NapMsg, queue_size=1000 )


# Timing Publishers
pub_time_queue_size = rospy.Publisher( '/time/queue_size', Float32, queue_size=1000)
pub_time_color_normalization = rospy.Publisher( '/time/color_normalization', Float32, queue_size=1000)
pub_time_netvlad = rospy.Publisher( '/time/netvlad', Float32, queue_size=1000)
pub_time_dimred = rospy.Publisher( '/time/dimred', Float32, queue_size=1000)
pub_time_total_desc_computation = rospy.Publisher( '/time/total_desc_computation', Float32, queue_size=1000)
pub_time_similarity = rospy.Publisher( '/time/similarity', Float32, queue_size=1000)

# pub_time_grid_filter = rospy.Publisher( '/time/grid_filter', Float32, queue_size=1000)
pub_time_grid_filter_sense = rospy.Publisher( '/time/grid_filter/sense', Float32, queue_size=1000)
pub_time_grid_filter_move = rospy.Publisher( '/time/grid_filter/move', Float32, queue_size=1000)

pub_time_ros_publish = rospy.Publisher( '/time/ros_publish', Float32, queue_size=1000)

pub_time_total_loop = rospy.Publisher( '/time/total_loop', Float32, queue_size=1000)


#
# Init netvlad - def computational graph, load trained model
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=1)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )


#
# Init DimRed Mapping (Dimensionality Reduction by Learning Invariant Mapping)
dm_vlad_word = tf.placeholder( 'float', [None,None], name='vlad_word' )
net = DimRed.DimRed()
dm_vlad_char = net.fc( dm_vlad_word )
tensorflow_saver2 = tf.train.Saver( net.return_vars() )
tensorflow_saver2.restore( tensorflow_session, PARAM_MODEL_DIM_RED )
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL_DIM_RED, tcolor.ENDC



#
# Init Grid Filter
w = np.zeros( 25000 ) + 1E-10
w[0:30] = 1
w = w / sum(w)

signal.signal(signal.SIGINT, signal_handler)
vote_bank = np.zeros( (3000,3000) )

#
# Vispy - Fast Plotting
qapp = pg.mkQApp()
win = pg.GraphicsWindow()
win.resize( 1200, 270 )
plot1 = win.addPlot()
curve1 = plot1.plot()
plot2 = win.addPlot()
curve2 = plot2.plot()
plot3 = win.addPlot()
curve3 = plot3.plot()
curve_thresh = plot3.plot()

# plot1.setRange( xRange=[0,1000 ], yRange=[0,1] )
# plot2.setRange( xRange=[0,1000 ], yRange=[0,1] )
# plot3.setRange( xRange=[0,1000 ], yRange=[0,15] )
plot1.setRange( yRange=[0,1] )
plot2.setRange( yRange=[0,1] )
plot3.setRange( yRange=[0,10] )


#
# Main Loop
rate = rospy.Rate(PARAM_FPS)
Likelihood = namedtuple( 'Likelihood', 'L dist')
S = np.zeros( (25000,128) ) #char
S_word = np.zeros( (25000,8192) ) #word
S_timestamp = np.zeros( 25000, dtype=rospy.Time )
loop_index = -1
while not rospy.is_shutdown():
    rate.sleep()
    rospy.logdebug( '---\nQueue Size : %d, %d' %( im_queue.qsize(), im_timestamp_queue.qsize()) )
    if im_queue.qsize() < 1 and im_timestamp_queue.qsize() < 1:
        # rospy.loginfo( 'Empty Queue...Waiting' )
        continue

    publish_time( pub_time_queue_size, im_queue.qsize() )
    startTimeTotalLoop = time.time()
    im_raw = im_queue.get()
    im_raw_timestamp = im_timestamp_queue.get()

    loop_index += 1

    # Convert to RGB format (from BGR) and resize
    im_ = cv2.resize( im_raw, (320,240) )
    A = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )


    ############# descriptors compute ##################
    # VLAD Computation
    d_compute_time_ms = []
    startTime = time.time()
    im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
    im_batch[0,:,:,:] = A.astype('float32')
    im_batch_normalized = normalize_batch( im_batch )
    d_compute_time_ms.append( (time.time() - startTime)*1000. )

    startTime = time.time()
    feed_dict = {tf_x : im_batch_normalized,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
    d_WORD = tff_vlad_word[0,:]
    d_compute_time_ms.append( (time.time() - startTime)*1000. )


    # Dim Reduction
    startTime = time.time()
    dmm_vlad_char = tensorflow_session.run( dm_vlad_char, feed_dict={dm_vlad_word: tff_vlad_word})
    dmm_vlad_char = dmm_vlad_char
    d_CHAR = dmm_vlad_char[0,:]
    d_compute_time_ms.append( (time.time() - startTime)*1000. )

    ###### END of descriptor compute : d_WORD, d_CHAR, d_compute_time_ms[] ###############
    rospy.logdebug( '[%6.2fms] Descriptor Computation' %(sum(d_compute_time_ms)) )
    publish_time( pub_time_color_normalization, d_compute_time_ms[0] )
    publish_time( pub_time_netvlad, d_compute_time_ms[1] )
    publish_time( pub_time_dimred, d_compute_time_ms[2] )
    publish_time( pub_time_total_desc_computation, sum(d_compute_time_ms) )


    ################## Array Insert (in S)
    startSimTime = time.time()
    S[loop_index,:] = d_CHAR
    S_word[loop_index,:] = d_WORD
    # S_im_index[loop_index] = int(im_indx)
    S_timestamp[loop_index] = im_raw_timestamp
    # sim_score =  1.0 - np.dot( S[0:loop_index+1,:], d_CHAR )
    # sim_score =  np.sqrt( 1.0 - np.minimum(1.0, np.dot( S[0:loop_index+1,:], d_CHAR )) ) #minimum is added to ensure dot product doesnt go beyond 1.0 as it sometimes happens because of numerical issues, which inturn causes issues with sqrt
    sim_score =  np.sqrt( 1.0 - np.minimum( 1.0, np.dot( S_word[0:loop_index+1,:], d_WORD ) ) )
    # sim_score =  np.dot( S[:loop_index,:], d_CHAR )

    sim_scores_logistic = logistic( sim_score ) #convert the raw Similarity scores above to likelihoods


    # # Clusterwise sum - Sample Code
    #X = np.random.randint(0,10, (20, 10) )
    # sanyo_mul = np.multiply( S_word[0:loop_index+1,:], 32.0*S_word[loop_index,:] )
    # sanyo_seg_sum = np.add.reduceat( sanyo_mul, range(0,sanyo_mul.shape[1],256), axis=1 )
    # sanyo_ncluster_matches = (np.sqrt(sanyo_seg_sum) > 0.95).sum( axis=1 )
    # print sanyo_seg_sum


    rospy.logdebug( '[%6.2fms] Similarity with all prev in' %( (time.time() - startSimTime)*1000. ) )
    publish_time( pub_time_similarity,   (time.time() - startSimTime)*1000. )




    ############### Grid Filter ############

    if loop_index < 30:
        continue


    # vote_bank[loop_index,0:(loop_index+1)] = sanyo_ncluster_matches/32.
    # Sense and Update Weights
    startSenseTime = time.time()

    L = len(sim_scores_logistic )
    # w[0:L] = np.multiply( np.power(w[0:L],float(L)/(L+1) ), np.power(sim_scores_logistic[0:L], 1.0/L)  )
    w[0:L] = np.multiply( w[0:L], sim_scores_logistic[0:L]  )


    rospy.logdebug( '[%4.2fms] GridFilter : Time for likelihood x prior' %(1000.*(time.time() - startSenseTime)) )
    publish_time( pub_time_grid_filter_sense, (1000.*(time.time() - startSenseTime) ) )



    # Move
    startMoveTime = time.time()
    w = np.roll( w, 1 )
    w[0] = w[1]
    w = np.convolve( w, [0.025,0.1,0.75,0.1,0.025], 'same' )

    w = w / sum(w)
    w[0:L] = np.maximum( w[0:L], 2.0/L )
    w[L:] = 1E-10
    rospy.logdebug( '[%4.2f ms] GridFilter Time for move' %(1000. * (time.time()-startMoveTime)) )
    publish_time( pub_time_grid_filter_move, (1000. * (time.time()-startMoveTime)) )


    thresh_log_scale = -np.log(1.0/L)
    thresh_log_scale -= 0.15*thresh_log_scale

    # Plot bar graphs
    curve1.setData( range(len(sim_score)), sim_score )
    # curve2.setData( range(len(sim_scores_logistic)), -np.log(sim_scores_logistic)/np.log(10) )
    curve2.setData( range(len(sim_scores_logistic)), sim_scores_logistic )
    curve3.setData( range(0,L+50), -np.log(w[0:L+50]) )
    curve_thresh.setData( range(0,L+50), thresh_log_scale*np.ones( (L+50) ) , pen=(1,2)  )
    # curve4.setData( range(len(sanyo_ncluster_matches)),  sanyo_ncluster_matches )
    qapp.processEvents()


    ################# Publish (on loop closure detection)
    # if loop_index > 50:
        # code.interact( local=locals() )

    startTimePub = time.time()
    # Publish (c_timestamp, prev_timestamp, goodness)
    # @Baysian Filter
    w_log = -np.log( w[0:L-200] ) #dont report on latest 20 frames
    argT = np.argwhere( w_log < thresh_log_scale ) #find all index of all elements in w less than 6

    # @Raw sim scores
    w_log = sim_scores_logistic[0:L-200]
    argT = np.argwhere( w_log > 0.60 )
    for aT in argT:
        nap_msg = NapMsg()
        # nap_msg.c_timestamp = rospy.Time.now()
        # nap_msg.prev_timestamp = rospy.Time.now()

        nap_msg.c_timestamp = rospy.Time.from_sec( float(S_timestamp[L-1].to_sec()) )
        nap_msg.prev_timestamp = rospy.Time.from_sec( float(S_timestamp[aT[0]].to_sec()) )
        nap_msg.goodness = w_log[aT]
        pub_colocation.publish( nap_msg )
        rospy.loginfo( 'c_time=%s ; prev_time=%s ; goodness=%f' %(str(S_timestamp[L-1]), str(S_timestamp[aT]), w_log[aT] ) )


    publish_time( pub_time_ros_publish, (1000.0*(time.time()-startTimePub)))

    # rospy.loginfo( '[%6.2fms] SPublished in ' %( (time.time() - startTimePub)*1000. ) )
    publish_time( pub_time_total_loop, (1000.0*(time.time()-startTimeTotalLoop)))

    # cv2.imshow( 'win', im_ )
    # cv2.waitKey(10)
