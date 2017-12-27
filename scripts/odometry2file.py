#!/usr/bin/python
""" Subscribes to odometry message and write to file the 1) timestamps and 2) x,y,z
        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 15th May, 2017
"""
import rospy

import collections
import numpy as np
import code

from nav_msgs.msg import Odometry

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

def odometry_callback( data ):
    # Write to file
    global fp
    p = data.pose.pose.position
    # print data.header.stamp, float(p.x), float(p.y), float(p.z)
    fp.write( '%s, %f, %f, %f\n' %(str(data.header.stamp), float(p.x), float(p.y), float(p.z)) )




rospy.init_node( 'odometry2file', log_level=rospy.INFO )
rospy.Subscriber( '/vins_estimator/odometry', Odometry, odometry_callback, queue_size=10000 )
rospy.loginfo( 'Subscribed to /vins_estimator/odometry')

fp = open( PKG_PATH+'/DUMP/GPS_track.csv', 'w' )
fp.write( '#t, x, y, z\n#Written by odometry2file.py\n')
print 'Open file : ', PKG_PATH+'/DUMP/GPS_track.csv'
rospy.spin()
fp.close()
