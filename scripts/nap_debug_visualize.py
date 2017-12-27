#!/usr/bin/python
""" Subscribes to images topic for every key-frame (or semi key frame) images.
    Load GT poses file. Publish a) GT poses b) Poses of Nearest neighbours of every frame processed.
    Note that, poses of neighbours is essentially GTs of frame index

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 9th Mar, 2017
"""


import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
import cv2
from cv_bridge import CvBridge, CvBridgeError

import Queue
import numpy as np


import time
from collections import namedtuple
import code
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from CartWheelFlow import VGGDescriptor
import DimRed
from Quaternion import Quat
import VPTree


#
import TerminalColors
tcolor = TerminalColors.bcolors()


# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'
PARAM_GT_POSES_FILE = PKG_PATH+'/other_seqs/kitti_dataset/poses/06.txt'

PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'



def callback_string( data ):
    rospy.logdebug( rospy.get_caller_id() + 'I heard %s', data.data )


def callback_image( data ):
    rospy.logdebug( 'Received Image : %d,%d' %( data.height, data.width ) )
    cv_image = CvBridge().imgmsg_to_cv2( data, 'bgr8' )

    im_queue.put( cv_image )
    im_indx_queue.put( int(data.header.frame_id) )
    #I am embedding the image-index in DB as frame_id string. This is for knowning the GT pose


## PUB : Publisher which publishes Marker()
## im_indx : Index of the place
## color : 3-tuple. Color of the marker
## marker_id : id of the marker (optional). If None, will use im_indx as the id
def publish_gt( PUB, im_indx, color, marker_id=None ):
    M = gt_poses[im_indx,:].reshape(3,4)
    t = M[:,3]
    R = M[0:3,0:3]
    quat = Quat( R )
    rospy.logdebug( quat.q )

    p = Marker()
    p.header.stamp = rospy.Time.now()
    p.header.frame_id = 'base_link'
    p.lifetime = rospy.Duration()
    p.id = im_indx if marker_id is None else marker_id
    p.type = Marker.CUBE
    p.action = Marker.ADD
    p.pose.position.x = M[0,3]
    p.pose.position.y = M[1,3]
    p.pose.position.z = M[2,3]
    p.pose.orientation.x = quat.q[0]
    p.pose.orientation.y = quat.q[1]
    p.pose.orientation.z = quat.q[2]
    p.pose.orientation.w = quat.q[3]
    p.scale.x = 1.5 if im_indx > 0 else 0.0001
    p.scale.y = 1.5 if im_indx > 0 else 0.0001
    p.scale.z = 1.5 if im_indx > 0 else 0.0001
    if len(color) == 4:
        p.color.a = color[3]
    else:
        p.color.a = 1.0
    p.color.r = color[0]
    p.color.g = color[1]
    p.color.b = color[2]
    PUB.publish( p )
    rospy.logdebug( 'Published marker')





## Publish locations of all the NN
MAX_IDS = 0
def publish_likelihoods(PUB, im_indx, likelihoods):
    global MAX_IDS
    # make_xerox( im_indx, likelihoods )
    print 'NNs of ', im_indx, 'are : ',
    if len(likelihoods) == 0: #no neighbours within tau-neighbourhood
        i=0
    for i, nei in enumerate(likelihoods):
        print nei.L,
        publish_gt( PUB, nei.L, color=(1,0,0), marker_id=i)
    print ''
    if i > MAX_IDS:
        MAX_IDS = i
    else:
        #empty publishes i to MAX_IDS
        for k in range(i,MAX_IDS+1):
            publish_gt( PUB, -1, color=(1,1,1), marker_id=k )


MAX_PUB = 50 #maximum 50 neighbours
def publish_likelihoods2(PUB, im_indx, likelihoods):
    global MAX_PUB
    continents, continents_start, islands = make_xerox( im_indx, likelihoods )
    #publish continents starts
    n_plotted = 0
    for i,v in enumerate(continents_start): #light-green for continents starts
        publish_gt( PUB, v, color=(0.5,1.0,0), marker_id=i)
        n_plotted += 1
    # for i,v in enumerate(islands): #dark-green for islands
    #     publish_gt( PUB, v, color=(0.0,0.4,0.0), marker_id=i+len(continents_start))
    #     n_plotted += 1


    for b in range(n_plotted, MAX_PUB):
        publish_gt( PUB, -1, color=(0.5,1.0,0), marker_id=b )


## Plot all the found neighbours, color coded with dot distances
cmap = np.loadtxt( PKG_PATH+'/scripts/CoolWarmFloat33.csv', comments='#', delimiter=',' )
def dist_to_color( dist ): #dist is in [0,1]
    for i in range(1,cmap.shape[0]):
        if cmap[i-1,0] <= dist and dist <= cmap[i,0]:
            return cmap[i-1,1:], i-1



def publish_likelihoods_all_colorcoded( PUB, im_indx, likelihoods ):
    for i, nei in enumerate(likelihoods):
        if nei.L +50 < im_indx:
            d_0_1 = min( max(0.0,nei.dist-0.1), 0.66666 )*1.49 #convert neu.dist to a number between 0 and 1
            clr, inx = dist_to_color(  d_0_1 )
            # print '%d(%3d) ' %(nei.L,inx),
            publish_gt( PUB, nei.L, color=(clr[0],clr[1],clr[2]), marker_id=None)
    # print ''




from shutil import copyfile
def _print_line( msg, Ary ):
    print msg,
    for a in Ary:
        print a, '',
    print ''

def _fp_print_line( fp, msg, Ary ):
    fp.write( msg )
    for a in Ary:
        fp.write( str(a)+' ' )
    fp.write( '\n' )

def make_xerox( im_indx, likelihoods, quite=False ):
    #mkdir im_indx


    dst_dir = PKG_PATH+'/DUMP/%06d/' %(im_indx)
    os.makedirs( dst_dir )

    # Write all neighbours in a file along with distances
    fp = open( dst_dir+'/all_nn.txt', 'w' )
    all_nn = []
    print 'NN of %d : ' %(im_indx),
    for nei in likelihoods:
        fp.write( '%06d <-- (DOT: %4.3f)(GPS_dist: %5.2f) --> %06d\n' %(im_indx, nei.dist, GPS_distance(im_indx,nei.L), nei.L) )
        all_nn.append( nei.L )
        print nei.L,
    print ''



    # Reduce Neighbours
    # eg. [975,976,1046,1047,1048,1107] --> [976,1046,1107]
    if len(all_nn) == 0:
        continents = []
        islands = []
    elif len(all_nn) == 1:
        continents = all_nn
        islands = []
    elif len(all_nn) == 2:
        all_nn.sort()
        if all_nn[1] - all_nn[0] < 5: #continent, eg. [7,9]
            continents = all_nn
            islands = []
        else: #both islands, eg. [5, 50]
            continents = []
            islands = all_nn
    else:
        all_nn.sort()

        continents = []
        islands = []
        # Think of this casss
        # 47,48,80,100,130
        # 47,48,80,81,82
        for i,val in enumerate(all_nn[1:-1]):
            n = i + 1
            back_diff = all_nn[n] - all_nn[n-1]
            fwd_diff  = all_nn[n+1] - all_nn[n]

            if back_diff <= 5 and fwd_diff < 5:
                continents.append(val)
            elif back_diff > 5 and fwd_diff > 5:
                islands.append(val)
            elif back_diff <= 5 and fwd_diff > 5:
                continents.append(val)
            elif back_diff > 5  and fwd_diff <= 5:
                continents.append(val)

        # deal with 0th val
        if all_nn[1] - all_nn[0] <=5 :
            continents.append( all_nn[0] )
        else:
            islands.append( all_nn[0] )


        # deal with last val
        if all_nn[-1] - all_nn[-2] <=5 :
            continents.append( all_nn[-1] )
        else:
            islands.append( all_nn[-1] )


        islands.sort()
        continents.sort()

        # D = np.diff( all_nn )
        # D = np.hstack( ([0], D) ) #ensure 1st element is zero and D and all_nn are of same size
        # continents = [ val for n,val in enumerate(all_nn) if D[n] < 5 and D[n] > 0 ]
        #
        # islands = []
        # for n,val in enumerate(all_nn[:-1]):
        #     if D[n] > 5 and D[n+1] > 5:
        #         islands.append(val)
        #
        # #1st element
        # if D[1] < 5:
        #     continents.append(all_nn[0])
        # else:
        #     islands.append( all_nn[0] )
        #
        # #last element
        # if D[-1] < 5:
        #     continents.append(all_nn[-1])
        # else:
        #     islands.append(all_nn[-1])
        #
        #
        # islands.sort()
        # continents.sort()

    _print_line( 'All NN     of %4d :' %(im_indx), all_nn)
    _print_line( 'Continents of %4d :' %(im_indx), continents )
    _print_line( 'Islands    of %4d :' %(im_indx), islands )
    _fp_print_line( fp, 'All NN     of %4d :' %(im_indx), all_nn)
    _fp_print_line( fp, 'Continents of %4d :' %(im_indx), continents )
    _fp_print_line( fp, 'Islands    of %4d :' %(im_indx), islands )
    fp.close()




    Dc = np.hstack( ([100], np.diff(continents)) )
    continents_start = []
    c = 0
    for i,r in enumerate(continents):
        if Dc[i] > 5:
            c += 1
            continents_start.append(r)
            src = PKG_PATH+'/other_seqs/kitti_dataset/sequences/00/image_2/%06d.png' %(r)
            copyfile( src, dst_dir+'/c_%06d.png' %(r) )
            print 'File Written ', dst_dir+'/%06d.png' %(r)

    # Also write current image `im_indx` to dump
    src = PKG_PATH+'/other_seqs/kitti_dataset/sequences/00/image_2/%06d.png' %(im_indx)
    copyfile( src, dst_dir+'/q_%06d.png' %(im_indx) )
    print 'File Written ', dst_dir+'/q_%06d.png' %(im_indx)


    #Also write islands
    for i,isl in enumerate(islands):
        src = PKG_PATH+'/other_seqs/kitti_dataset/sequences/00/image_2/%06d.png' %(isl)
        copyfile( src, dst_dir+'/isl_%06d.png' %(isl) )



    print '# of Continents : ', c
    print '# of islands    : ', len(islands)



    # # Write all neighbours in dir all_nn
    # os.makedirs( dst_dir+'/all_nn/' )
    # for nei in likelihoods:
    #     src = PKG_PATH+'/other_seqs/kitti_dataset/sequences/00/image_2/%06d.png' %(nei.L)
    #     copyfile( src, dst_dir+'/all_nn/%06d___%4.3f.png' %(nei.L, nei.dist) )
    return continents, continents_start, islands


def GPS_distance( i, j):
    Mi = gt_poses[i,:].reshape(3,4)
    ti = Mi[:,3]

    Mj = gt_poses[j,:].reshape(3,4)
    tj = Mj[:,3]

    return np.sqrt( np.linalg.norm( ti - tj ) )

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


#
# Load Ground Truth Poses
gt_poses = np.loadtxt( PARAM_GT_POSES_FILE, comments='#', delimiter=' ' )


#
# Setup Callbacks and Publishers
im_queue = Queue.Queue()
im_indx_queue = Queue.Queue()

rospy.init_node( 'listener', log_level=rospy.INFO )
rospy.Subscriber( 'chatter', String, callback_string )
rospy.Subscriber( 'semi_keyframes', Image, callback_image )

PUB_gt_path = rospy.Publisher( 'gt_path', Marker, queue_size=1000)
PUB_nn = rospy.Publisher( 'nn', Marker, queue_size=1000)



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
# Main Loop
rate = rospy.Rate(10)
treeroot = None
Likelihood = namedtuple( 'Likelihood', 'L dist')
while not rospy.is_shutdown():
    rate.sleep()
    print '---\nQueue Size : ', im_queue.qsize(), im_indx_queue.qsize()
    if im_queue.qsize() < 1 and im_indx_queue.qsize() < 1:
        rospy.loginfo( 'Empty Queue...Waiting' )
        continue


    im_raw = im_queue.get()
    im_indx = im_indx_queue.get()

    # Convert to RGB format and resize
    im_ = cv2.resize( im_raw, (320,240) )
    A = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )


    ############# descriptors compute starts
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
    rospy.loginfo( '[%6.2fms] Descriptor Computation' %(sum(d_compute_time_ms)) )


    ################## KDTREE / VPTree - Insert
    # tree.add( d_CHAR[0:3] )
    startAddTime = time.time()
    pos_index = im_indx
    node = VPTree.NDPoint( d_CHAR, pos_index )
    if treeroot is None:
        treeroot = VPTree.InnerProductTree(node, 0.27) # #PARAM this is mu
    else:
        treeroot.add_item( node )
    rospy.loginfo( '[%6.2fms] Added to InnerProductTree' %( (time.time() - startTime)*1000. ) )




    ################## Tree Nearest Neighbour Query
    startAddTime = time.time()
    q = VPTree.NDPoint( d_CHAR, -1 )
    # all_nn= VPTree.get_nearest_neighbors( treeroot, q, k=10 )
    all_nn = VPTree.get_all_in_range( treeroot, q, tau=2.13 ) # #PARAM this is tau
    # print tcolor.OKBLUE+'[%6.2fms] NN Query' %(  (time.time() - startTime)*1000. ), tcolor.ENDC
    rospy.loginfo( '[%6.2fms] NN Query' %(  (time.time() - startTime)*1000. ) )

    #collect nn
    likelihoods = []
    for nn in all_nn:
        likelihoods.append( Likelihood(L=nn[1].idx, dist=nn[0]) )


    # #print all nn obtained
    # print 'NN of ', pos_index, ":",
    # for nn in all_nn:
    #     print nn[1].idx, '(%5.3f), ' %(nn[0]), #idx, distance
    #     # if PARAM_DEBUG:
    #         # APPEARENCE_CONFUSION[ pos_index ,nn[1].idx] = 1.0 #np.exp( np.negative(nn[0]) )
    # print ''

    ################# Publish

    publish_gt( PUB_gt_path, im_indx, color=(1.0,1.0,0.5), marker_id=50000 )
    # publish_gt( PUB_gt_path, im_indx, color=(1.,0.,0.0,0.1) )
    # publish_likelihoods( PUB_nn, im_indx, likelihoods )
    publish_likelihoods_all_colorcoded( PUB_nn, im_indx, likelihoods )

    # cv2.imshow( 'win', im_ )
    # cv2.waitKey(10)
