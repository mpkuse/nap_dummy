#!/usr/bin/python
""" Tony bags to kitti format
        Takes tony's bag files. These bags contains 10Hz colors, 20Hz gray, 100Hz IMU. nav_msgs 10Hz

        rosbag info
        path:        01.bag
        version:     2.0
        duration:    10:19s (619s)
        start:       Mar 15 2017 20:15:17.55 (1489580117.55)
        end:         Mar 15 2017 20:25:36.66 (1489580736.66)
        size:        12.9 GB
        messages:    88040
        compression: none [7692/7692 chunks]
        types:       nav_msgs/Odometry [cd5e73d190d741a2f92e81eda573aca7]
                     sensor_msgs/Image [060021388200f6f0f447d0fcd9c64743]
                     sensor_msgs/Imu   [6a62c6daae103f4ff57a132d6f95cec2]
        topics:      /djiros/image              12381 msgs    : sensor_msgs/Image
                     /djiros/imu                61901 msgs    : sensor_msgs/Imu
                     /mv_29900616/image_raw      7602 msgs    : sensor_msgs/Image
                     /vins_estimator/odometry    6156 msgs    : nav_msgs/Odometry


        The output data-tree :
        segments/
            |---- 00
            	   |---- Images/
        		    |----img_bluefox_<timestamp>_color.png
        		    |----img_bluefox_<timestamp>_color.png
        		    |----img_bluefox_<timestamp>_color.png
        				 .
        				 .
        	   |---- list.txt //list of files, sorted by time
        	   |---- odometry.txt //odometry msgs 13 cols. col-0 is timestamps
               |---- odometry_12.txt //odometry msgs 12 cols. exactly same as odometry.txt but has no timestamp. for compatibility



        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 17th Mar, 2017
"""

import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import glob
import os

from Quaternion import Quat

#
import TerminalColors
tcolor = TerminalColors.bcolors()

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'


PARAM_BAG_ID = '00'

PARAM_OUT_ODOM_FILE = PKG_PATH+'/other_seqs/tony_bags/segments/'+PARAM_BAG_ID+'/odometry.txt'
PARAM_OUT_ODOM_FILE_12 = PKG_PATH+'/other_seqs/tony_bags/segments/'+PARAM_BAG_ID+'/odometry_12.txt'
PARAM_OUT_LIST_FILE = PKG_PATH+'/other_seqs/tony_bags/segments/'+PARAM_BAG_ID+'/list.txt'
PARAM_OUT_IMAGES_DIR = PKG_PATH+'/other_seqs/tony_bags/segments/'+PARAM_BAG_ID+'/Images/'

PARAM_ODOMETRY_ROSTOPIC = '/vins_estimator/odometry'
PARAM_IMAGE_ROSTOPIC    = '/mv_29900616/image_raw'


def callback_image( data ):
    # print 'image: ', data.header.stamp.secs, '%6d' %(data.header.stamp.nsecs/1000), long(data.header.stamp)
    print 'image %f' %( data.header.stamp.to_sec() )
    cv_image = CvBridge().imgmsg_to_cv2( data, 'bgr8' )

    cv2.imwrite( PARAM_OUT_IMAGES_DIR+'/img_bluefox_%f_color.png' %(data.header.stamp.to_sec()  ), cv_image  )
    fp_img_list.write( 'img_bluefox_%f_color.png\n' %(data.header.stamp.to_sec()  ) )

def callback_odometry( data ):
    # print 'odometry : ', data.header.stamp.secs, '%6d' %(data.header.stamp.nsecs/1000)
    print 'odometry %f' %( data.header.stamp.to_sec() )

    ori = data.pose.pose.orientation
    position = data.pose.pose.position

    q_w_c = [ori.x, ori.y, ori.z, ori.w]
    q = Quat(q_w_c)
    R_w_c = q.transform
    t_w_c = [[position.x, position.y, position.z]]
    T_w_c = np.hstack( [R_w_c, np.transpose(t_w_c)] )
    # print T_w_c

    fp_odom.write( '%f ' %(data.header.stamp.to_sec() ) )
    for i in range(T_w_c.shape[0] ):
        for j in range(T_w_c.shape[1] ):
            fp_odom.write( '%f ' %(T_w_c[i,j])  )
            fp_odom_12.write( '%f ' %(T_w_c[i,j])  )
    fp_odom.write('\n')
    fp_odom_12.write('\n')

print tcolor.OKGREEN, 'mkdir ', PARAM_OUT_IMAGES_DIR, tcolor.ENDC
os.makedirs( PARAM_OUT_IMAGES_DIR )

print tcolor.OKGREEN, 'open file ', PARAM_OUT_LIST_FILE, tcolor.ENDC
fp_img_list = open(PARAM_OUT_LIST_FILE, 'w' )

print tcolor.OKGREEN, 'open file ', PARAM_OUT_ODOM_FILE, tcolor.ENDC
print tcolor.OKGREEN, 'open file ', PARAM_OUT_ODOM_FILE_12, tcolor.ENDC
fp_odom = open( PARAM_OUT_ODOM_FILE, 'w' )
fp_odom_12 = open( PARAM_OUT_ODOM_FILE_12, 'w' )



rospy.init_node( 'tonybags_2_segments' )
rospy.Subscriber( PARAM_IMAGE_ROSTOPIC,  Image, callback_image )
rospy.Subscriber( PARAM_ODOMETRY_ROSTOPIC,  Odometry, callback_odometry )
rospy.loginfo( 'Subscribed to '+ PARAM_IMAGE_ROSTOPIC )
rospy.loginfo( 'Subscribed to '+ PARAM_ODOMETRY_ROSTOPIC )


rate = rospy.Rate(20)
rospy.loginfo( 'Waiting for incoming msgs' )
while not rospy.is_shutdown():
    # print 'loop'
    rate.sleep()
