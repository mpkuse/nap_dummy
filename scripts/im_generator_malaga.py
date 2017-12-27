#!/usr/bin/python
""" Read images from Malaga dataset

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 14th Mar, 2017
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


# In the folder of malaga dataset following files are of importance
# a) Images
# b) malaga-urban-dataset-extract-10_all-sensors_GPS.txt (uniq-time + GPS)
# c) malaga-urban-dataset-extract-10_all-sensors_IMAGES.txt (list of images note: image file names have timestamps)
PARAM_DATASET_ID = 99
PARAM_TIME_SUBSAMPLE = 1 #use every 10th frame
PARAM_FPS = 20 #frames per sec

PARAM_DATASET_PATH = PKG_PATH+'other_seqs/malaga_dataset/malaga-urban-dataset-extract-%02d/' %(PARAM_DATASET_ID)


# Setup rosnode
rospy.init_node( 'talker')
pub = rospy.Publisher( 'chatter', String, queue_size=1000 )
# pub_im = rospy.Publisher( 'semi_keyframes', Image, queue_size=1000 )
pub_im = rospy.Publisher( '/color_image/image_raw', Image, queue_size=1000 )



# Open Files List
with open( PARAM_DATASET_PATH+'/malaga-urban-dataset-extract-%02d_all-sensors_IMAGES.txt' %( PARAM_DATASET_ID ) ) as f:
    im_list = f.readlines()


def in_between_index( t ):
    if t < GPS_poses[0,0]: #t occurs before GPS starts
        return 0
    elif t > GPS_poses[-1,0]: #t occcurs after GPS is over
        return GPS_poses.shape[0]-1
    else:
        for i in range(1,GPS_poses.shape[0]):
            if GPS_poses[i-1,0] <= t <= GPS_poses[i,0]:
                return i-1


# Open files with GPS data
gps_file_name = PARAM_DATASET_PATH+'/malaga-urban-dataset-extract-%02d_all-sensors_GPS.txt' %( PARAM_DATASET_ID )
GPS_poses = np.loadtxt( gps_file_name, comments='%')


# Filter file-list
im_list = [ ffname.strip() for ffname in im_list if ffname.find( 'left' ) > 0 ]
im_list_time_sub_sampled = [ ffname for i,ffname in enumerate(im_list) if i%PARAM_TIME_SUBSAMPLE == 0 ]




rate = rospy.Rate(PARAM_FPS)
for i,im_fname in enumerate(im_list_time_sub_sampled):
    if rospy.is_shutdown():
        break
    im_time = float(im_fname.split( '_')[2])
    im_fname_full_path = PARAM_DATASET_PATH+'/Images/'+im_fname


    in_between_indx = in_between_index(im_time)

    im_cur = cv2.imread( im_fname_full_path )
    rospy.loginfo('i=%d, bet=%d, t=%f, READ : %s of %d' %(i, in_between_indx, im_time, im_fname, len(im_list_time_sub_sampled)) )


    msg_frame = CvBridge().cv2_to_imgmsg(im_cur,'bgr8')
    msg_frame.header.frame_id = str(in_between_indx)
    pub_im.publish( msg_frame )

    rate.sleep()


    # cv2.imshow( 'im_cur', im_cur)
    # cv2.waitKey(30)
