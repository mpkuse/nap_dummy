#!/usr/bin/python
""" Reads images from file(kitti) and publish Image messages.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 9th Mar, 2017
"""

import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import glob


# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

# PARAM_IM_PATH = PKG_PATH+'/other_seqs/kitti_dataset/sequences/00/image_2/' #kitti
PARAM_IM_PATH = PKG_PATH+'/other_seqs/altizure_seq/sequences/02/' #altizure-seq
# PARAM_IM_PATH = PKG_PATH+'/other_seqs/ust_drone_seq/sequences/22/' #ust-drone-seq
PARAM_START_INDX = 0
PARAM_END_INDX = len( glob.glob(PARAM_IM_PATH+'*.png'))-1
PARAM_STEP = 5
PARAM_FPS = 10 #frames per sec


rospy.init_node( 'talker')
pub = rospy.Publisher( 'chatter', String, queue_size=1000 )
pub_im = rospy.Publisher( 'semi_keyframes', Image, queue_size=1000 )


rate = rospy.Rate(PARAM_FPS)
im_indx = PARAM_START_INDX
while not rospy.is_shutdown() and im_indx < PARAM_END_INDX:
    hello_str = "hello world %s" %( rospy.get_time() )
    rospy.logdebug( hello_str )
    pub.publish( hello_str )


    im_f_name = '%s/%06d.png' %( PARAM_IM_PATH, im_indx )
    rospy.loginfo( 'Read :  '+im_f_name+ ' of %d' %(PARAM_END_INDX) )
    im = cv2.imread( im_f_name )

    msg_frame = CvBridge().cv2_to_imgmsg(im,'bgr8')
    msg_frame.header.frame_id = str(im_indx)
    pub_im.publish( msg_frame )


    # cv2.imshow( 'win', im )
    # cv2.waitKey(10)

    im_indx += PARAM_STEP
    rate.sleep()
