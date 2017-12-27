#!/usr/bin/python
""" Subscribes to images topic for every key-frame (or semi key frame) images.
    Publish asynchronously time-time message when a loop is detected.
    Images are indexed by time. In the future possibly index with distance
    using an IMU. Note that this script does not know about poses (generated from SLAM system)

    In this edition (2) of this script, there is an attempt to organize this code.
    The core netvlad place recognition system is moved into the class
    `PlaceRecognition`.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 3rd Apr, 2017
        Edition : 2 (of nap_time_node.py)
"""


import rospy
import rospkg
import time
import code

import numpy as np
import cv2


from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from nap.msg import NapMsg
from nap.msg import NapNodeMsg


from PlaceRecognitionNetvlad import PlaceRecognitionNetvlad
from FastPlotter import FastPlotter


############# PARAMS #############
# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_k48/model-13000' #PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/netvlad_k48/db2/siamese_dimred/model-400' #PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'

PARAM_NETVLAD_WORD_DIM = 12288 # If these are not compatible with tensorfloaw model files program will fail
PARAM_NETVLAD_CHAR_DIM = 256

# INPUT_IMAGE_TOPIC = '/dji_sdk/image_raw'
# INPUT_IMAGE_TOPIC = '/youtube_camera/image'
# INPUT_IMAGE_TOPIC = '/android/image'
# INPUT_IMAGE_TOPIC = '/fabmap_data/image'
INPUT_IMAGE_TOPIC = '/semi_keyframes' #this is t be used for launch
PARAM_CALLBACK_SKIP = 1

PARAM_FPS = 25

def publish_time( PUB, time_ms ):
    PUB.publish( Float32(time_ms) )

########### Init PlaceRecognitionNetvlad ##########
place_mod = PlaceRecognitionNetvlad(\
                                    PARAM_MODEL,\
                                    PARAM_CALLBACK_SKIP=PARAM_CALLBACK_SKIP\
                                    )

place_mod.load_siamese_dim_red_module( PARAM_MODEL_DIM_RED, PARAM_NETVLAD_WORD_DIM, 1024, PARAM_NETVLAD_CHAR_DIM  )

################### Init Node and Topics ############
rospy.init_node( 'nap_geom_node', log_level=rospy.INFO )
rospy.Subscriber( INPUT_IMAGE_TOPIC, Image, place_mod.callback_image )
rospy.loginfo( 'Subscribed to '+INPUT_IMAGE_TOPIC )

# raw nodes
pub_node_msg   = rospy.Publisher( '/raw_graph_node', NapNodeMsg, queue_size=1000 )
rospy.loginfo( 'Publish to /raw_graph_node' )

# raw edges
pub_edge_msg = rospy.Publisher( '/raw_graph_edge', NapMsg, queue_size=1000 )
rospy.loginfo( 'Publish to /raw_graph_edge' )


pub_time_queue_size = rospy.Publisher( '/time/queue_size', Float32, queue_size=1000)
pub_time_desc_comp = rospy.Publisher( '/time/netvlad_comp', Float32, queue_size=1000)
pub_time_publish = rospy.Publisher( '/time/publish', Float32, queue_size=1000)
pub_time_total_loop = rospy.Publisher( '/time/total', Float32, queue_size=1000)


#################### Init Plotter #####################
plotter = FastPlotter(n=5)
plotter.setRange( 0, yRange=[0,1] )
plotter.setRange( 1, yRange=[0,1] )
plotter.setRange( 2, yRange=[0,1] )
plotter.setRange( 3, yRange=[0,1] )
# plotter.setRange( 4, yRange=[0,2] )

##################### Main Loop ########################
rate = rospy.Rate(PARAM_FPS)
S_char = np.zeros( (25000,PARAM_NETVLAD_CHAR_DIM) ) #char
# S_word = np.zeros( (25000,8192) ) #word
S_word = np.zeros( (25000,PARAM_NETVLAD_WORD_DIM) ) #word-48
S_timestamp = np.zeros( 25000, dtype=rospy.Time )
S_thumbnail = []
loop_index = -1

# init grid filter
w = np.zeros( 25000 ) + 1E-10
w[0:50] = 1
# w = w / sum(w)


startTotalTime = time.time()

while not rospy.is_shutdown():
    rate.sleep()
    publish_time( pub_time_total_loop, 1000.*(time.time() - startTotalTime) ) #this has been put like to use startTotalTime from prev iteration
    startTotalTime = time.time()
    #------------------- Queue book-keeping---------------------#
    # rospy.logdebug( '---\nQueue Size : %d, %d' %( place_mod.im_queue.qsize(), place_mod.im_timestamp_queue.qsize()) )
    publish_time( pub_time_queue_size, place_mod.im_queue.qsize() )
    if place_mod.im_queue.qsize() < 1 and place_mod.im_timestamp_queue.qsize() < 1:
        rospy.logdebug( 'Empty Queue...Waiting' )
        continue

    # Get Image & TimeStamp
    im_raw = place_mod.im_queue.get()
    im_raw_timestamp = place_mod.im_timestamp_queue.get()

    loop_index += 1
    #---------------------------- END  -----------------------------#


    #---------------------- Descriptor Extractor ------------------#
    startDescComp = time.time()
    rospy.loginfo( 'NetVLAD Computation' )
    d_CHAR, d_WORD = place_mod.extract_reduced_descriptor(im_raw)
    # d_WORD = place_mod.extract_descriptor(im_raw)
    publish_time( pub_time_desc_comp, 1000.*(time.time() - startDescComp) )
    print 'Word Shape : ', d_WORD.shape
    print 'Char Shape : ', d_CHAR.shape
    #---------------------------- END  -----------------------------#

    #------------------- Storage & Score Computation ----------------#
    rospy.loginfo( 'Storage & Score Computation' )
    S_char[loop_index,:] = d_CHAR
    S_word[loop_index,:] = d_WORD
    S_timestamp[loop_index] = im_raw_timestamp
    S_thumbnail.append(  cv2.resize( im_raw.astype('uint8'), (128,96) ) )#, fx=0.2, fy=0.2 ) )

    DOT = np.dot( S_char[0:loop_index+1,:], S_char[loop_index,:] )
    DOT_word = np.dot( S_word[0:loop_index+1,:], S_word[loop_index,:] )
    sim_scores =  np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) ) #minimum is added to ensure dot product doesnt go beyond 1.0 as it sometimes happens because of numerical issues, which inturn causes issues with sqrt
    sim_scores_logistic = place_mod.logistic( sim_scores ) #convert the raw Similarity scores above to likelihoods
    #---------------------------- END  -----------------------------#


    # Plot sim_scores
    plotter.set_data( 0, range(len(DOT)), DOT, title="DOT"  )
    plotter.set_data( 1, range(len(DOT_word)), DOT_word, title="DOT_word"  )
    plotter.set_data( 2, range(len(sim_scores)), sim_scores, title="sim_scores"  )
    plotter.set_data( 3, range(len(sim_scores_logistic)), sim_scores_logistic, title="sim_scores_logistic"  )
    plotter.spin()

    #------------------- Publish Current Node  -------------------------#
    node_msg = NapNodeMsg()
    node_msg.node_timestamp = rospy.Time.from_sec( float(im_raw_timestamp.to_sec()) )
    node_msg.node_label = str(loop_index)
    node_msg.node_label_str = str(loop_index)
    node_msg.color_r = node_msg.color_g = node_msg.color_b =1.0
    pub_node_msg.publish( node_msg )

    #------------------------------ END  -------------------------------#



    #----------------- Grid Filter / Temporal Fusion -------------------#
    if loop_index < 50: #let data accumulate
        # Do nothing
        _jkghgn = 0
    else:
        # Likelihood x Prior
        L = len(sim_scores_logistic)
        w[0:L] =  np.multiply( w[0:L], 1.5*sim_scores_logistic[0:L] )
        # w[0:L] = w[0:L] + sim_scores_logistic[0:L]

        # Move
        w = np.roll(w, 1)
        w[0] = w[1]
        w = np.convolve( w, [0.05,0.1,0.7,0.1,0.05], 'same' )
        w = w / sum(w)
        w[0:L] = np.maximum( w[0:L], 0.01 )
        w[L:] = 1E-10
        plotter.set_data( 4, range(L+50), -np.log(w[0:L+50]) )

    #------------------------------ END  -------------------------------#



    #------------- Publish Edges (thresh on raw likelihoods) ---------------#
    startPublish = time.time()
    L = loop_index #alias

    argT = np.where( sim_scores_logistic[0:L-2] > 0.70 )
    print len(argT), argT
    if len(argT ) < 1:
        continue
    for aT in argT[0]:
        # print aT
        # print '%d<--->%d' %(L,aT)
        nap_msg = NapMsg()
        nap_msg.c_timestamp = rospy.Time.from_sec( float(S_timestamp[L].to_sec()) )
        nap_msg.prev_timestamp = rospy.Time.from_sec( float(S_timestamp[aT].to_sec()) )
        nap_msg.goodness = sim_scores_logistic[aT]
        nap_msg.color_r = 0.0
        nap_msg.color_g = 0.0
        nap_msg.color_b = 1.0

        if float(S_timestamp[L-1].to_sec() - S_timestamp[aT].to_sec())>10.:
            pub_edge_msg.publish( nap_msg ) #publish raw edges (all)
            # print 'Add Edge : ', loop_index-1, aT
            # G.add_edge(loop_index-1, aT, weight=sim_scores_logistic[aT])


    publish_time( pub_time_publish, 1000.*(time.time() - startPublish) )
    #-------------------------------- END  -----------------------------#


# Save S_word, S_char, S_thumbnail
print 'Quit...!'
print 'Writing ', PKG_PATH+'/DUMP/S_word.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_char.npy'
print 'Writing ', PKG_PATH+'/DUMP/S_thumbnail.npy'
np.save( PKG_PATH+'/DUMP/S_word.npy', S_word[0:loop_index+1] )
np.save( PKG_PATH+'/DUMP/S_char.npy', S_char[0:loop_index+1] )
np.save( PKG_PATH+'/DUMP/S_timestamp.npy', S_timestamp[0:loop_index+1] )
np.save( PKG_PATH+'/DUMP/S_thumbnail.npy', S_thumbnail )
