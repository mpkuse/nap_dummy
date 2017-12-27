#!/usr/bin/python
""" Read images from segments dataset. For details on the exact format look at header
    of `tonybags_2_segments.py`.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 17th Mar, 2017
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


# DATA in SEGMENT format
# a) segment/00/Images -->  Images
# b) segment/00/list.txt --> List of images
# c) odometry.txt and odometry_12.txt --> time-stamps and poses. (may not be synchronized with image stream)

PARAM_DATASET_ID = '00'
PARAM_TIME_SUBSAMPLE = 10 #use every 10th frame
PARAM_FPS = 10 #frames per sec

PARAM_DATASET_PATH = PKG_PATH+'other_seqs/tony_bags/segments/%s/' %(PARAM_DATASET_ID)



# Setup rosnode
rospy.init_node( 'im_generator_segment')
# pub = rospy.Publisher( 'chatter', String, queue_size=1000 )
pub_im = rospy.Publisher( 'semi_keyframes', Image, queue_size=1000 )



# Open Files List
with open( PARAM_DATASET_PATH+'/list.txt' ) as f:
    im_list = f.readlines()


def in_between_index( t ):
    if t < ODOMETRY_poses[0,0]: #t occurs before GPS starts
        return 0
    elif t > ODOMETRY_poses[-1,0]: #t occcurs after GPS is over
        return ODOMETRY_poses.shape[0]-1
    else:
        for i in range(1,ODOMETRY_poses.shape[0]):
            if ODOMETRY_poses[i-1,0] <= t <= ODOMETRY_poses[i,0]:
                return i-1

#
# Open file with Odometry data
odometry_file_name = PARAM_DATASET_PATH+'/odometry.txt'
ODOMETRY_poses = np.loadtxt( odometry_file_name, comments='%')


# Filter file-list
im_list_time_sub_sampled = [ ffname for i,ffname in enumerate(im_list) if i%PARAM_TIME_SUBSAMPLE == 0 ]




rate = rospy.Rate(PARAM_FPS)
for i,im_fname in enumerate(im_list_time_sub_sampled):
    if rospy.is_shutdown():
        break
    im_time = float(im_fname.split( '_')[2])
    im_fname_full_path = PARAM_DATASET_PATH+'/Images/'+im_fname.strip()


    in_between_indx = in_between_index(im_time)

    im_cur = cv2.imread( im_fname_full_path )
    rospy.loginfo('i=%d, bet=%d, t=%f, READ : %s of %d' %(i, in_between_indx, im_time, im_fname, len(im_list_time_sub_sampled)) )


    msg_frame = CvBridge().cv2_to_imgmsg(im_cur,'bgr8')
    msg_frame.header.frame_id = str(in_between_indx)
    pub_im.publish( msg_frame )

    rate.sleep()


    # cv2.imshow( 'im_cur', im_cur)
    # cv2.waitKey(30)
