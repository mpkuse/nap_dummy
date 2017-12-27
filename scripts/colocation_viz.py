#!/usr/bin/python
""" Visualization node for colocations
        Subscribes to topic `/vins_estimator/Odometry` of type `nav_msgs/Odometry`
        and `/colocation` of type `nap/NapMsg`. Publishes the odometry message as
        is. publishes the colocation messages as Marker/LINE_LIST whose poses
        will be looked up using the timestamps in colocation and odometry poses

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 4th Apr, 2017
"""


import rospy

import collections
import numpy as np
import code

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


from nap.msg import NapMsg

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'


## Callback for odometry messages (messages given out by VINS module at 50-100Hz)
def odometry_callback( data ):
    # rospy.loginfo( 'Received Odometry %s' %(data.header.stamp) )
    #TODO: Not really necesarry to store all poses that come. Can drop a few
    global seq_odom
    _d.append( data )

    # publish Marker
    marker = Marker()
    marker.header = data.header
    marker.id = seq_odom
    seq_odom += 1
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = data.pose.pose
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    pub_odometry.publish( marker )

    marker.id = 0;
    marker.color.r = 1.0;
    marker.color.g = 0.64;
    marker.color.b = 0.0;
    marker.scale.x = 2
    marker.scale.y = 2
    marker.scale.z = 2
    pub_curr_position.publish( marker )


## Binary Search on timestamps
def binary_search_on_dequeue( time_stamp ):
    # My implementation
    if len(_d) == 0:
        return -1
    left_index = 0
    right_index = len(_d) - 1
    mid_index = (left_index + right_index)//2
    while True:
        if (time_stamp - _d[left_index].header.stamp) > rospy.Duration(0) and (time_stamp - _d[right_index].header.stamp) < rospy.Duration(0):
            rospy.logdebug( 'OK' )
        else:
            rospy.logdebug( 'Exhausted' )
            return -1

        # test for equality
        if abs(time_stamp - _d[mid_index].header.stamp) < rospy.Duration(0.1):
            rospy.logdebug( 'Found' )
            return mid_index

        if (time_stamp - _d[mid_index].header.stamp) > rospy.Duration(0):
            # Search in `next half`
            rospy.logdebug( 'Continue Search in next half' )
            left_index = mid_index
            mid_index = (left_index + right_index)//2
        else:
            # search in `first half`
            rospy.logdebug( 'Continue search in 1st half' )
            right_index = mid_index
            mid_index = (left_index + right_index)//2


    #TODO in the future, replace this with a library implementation

## Plot all the found neighbours, color coded with dot distances
cmap = np.loadtxt( PKG_PATH+'/scripts/CoolWarmFloat33.csv', comments='#', delimiter=',' )
def dist_to_color( dist ): #dist is in [0,1]
    for i in range(1,cmap.shape[0]):
        if cmap[i-1,0] <= dist and dist <= cmap[i,0]:
            return cmap[i-1,1:], i-1
    return cmap[i-1,1:], i-1

## Callback for colocation messages ()
def colocation_callback( data ):
    global seq
    rospy.loginfo( 'Received Colocation' )
    # 1. do a binary search on the deque with data.c_timestamp and data.prev_timestamp
    c_indx = binary_search_on_dequeue( data.c_timestamp )
    prev_indx = binary_search_on_dequeue( data.prev_timestamp )
    print  data.c_timestamp, '<--->', data.prev_timestamp
    print  c_indx, '<--->', prev_indx
    print '(%d)' %(len(_d)), _d[-1].header.stamp, ', ', _d[-2].header.stamp, ', ', _d[-3].header.stamp

    if prev_indx < 0:
        return None
    goodness = data.goodness # currently will be between 0 and 6. 6 means less confident that it is a loop_closure. 0 means most confident that it is a loop
    goodness_color, _ = dist_to_color( 2.5*(goodness-0.6) )

    # 1.1 Make points
    pt0 = _d[c_indx].pose.pose.position
    pt1 = _d[prev_indx].pose.pose.position

    # 2. make 2 markers which represents a line using poses from part-1 (above)
    m = Marker()
    m.header = _d[c_indx].header
    m.lifetime = rospy.Duration(0) #set this to zero => colocation edges are never deleted. positive value will cause the edges to disappear after this many seconds
    m.id = seq
    seq += 1
    m.type = Marker.LINE_LIST
    m.points.append(pt0) #c_indx
    m.points.append(pt1) #prev_indx
    m.scale.x = .2
    m.scale.y = 0.5
    m.scale.z = 0.0
    m.color.a = 0.5
    m.color.r = 1.0#goodness_color[0]#1.0
    m.color.g = 0.0#goodness_color[1]#1.0
    m.color.b = 0.0#goodness_color[2]#0.0


    # 3. publish
    pub_colocation.publish( m )




rospy.init_node( 'colocation_viz', log_level=rospy.INFO )
rospy.Subscriber( '/vins_estimator/odometry', Odometry, odometry_callback )
rospy.loginfo( 'Subscribed to /vins_estimator/odometry')

rospy.Subscriber( '/colocation', NapMsg, colocation_callback )
rospy.loginfo( 'Subscribed to /colocation')


pub_odometry = rospy.Publisher( '/colocation_viz/odom_marker', Marker, queue_size=1000 )
pub_curr_position = rospy.Publisher( '/colocation_viz/cur_position', Marker, queue_size=1000 )
pub_colocation = rospy.Publisher( '/colocation_viz/colocation_marker', Marker, queue_size=1000 )
seq = 0
seq_odom = 0
_d = collections.deque()

rate = rospy.Rate(30)
while not rospy.is_shutdown():
    # print 'Length :', len(_d)
    rate.sleep()

rospy.loginfo( 'Done..!' )
