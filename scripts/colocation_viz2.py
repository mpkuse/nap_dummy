#!/usr/bin/python
""" Visualization node for colocations
        Subscribes to topic `/vins_estimator/Odometry` of type `nav_msgs/Odometry`
        and `/colocation` of type `nap/NapMsg`. Publishes the odometry message as
        is. publishes the colocation messages as Marker/LINE_LIST whose poses
        will be looked up using the timestamps in colocation and odometry poses

        This is an improvement of colocation_viz.py

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 4th Apr, 2017
"""


import rospy
import rospkg

import collections
import numpy as np
import code
import cv2
from cv_bridge import CvBridge, CvBridgeError


from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


from nap.msg import NapMsg
from nap.msg import NapNodeMsg
from nap.msg import NapVisualEdgeMsg

# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'


## Callback for odometry messages (messages given out by VINS module at 50-100Hz)
def odometry_callback( data ):
    # rospy.loginfo( 'Received Odometry %s' %(data.header.stamp) )
    #TODO: Not really necesarry to store all poses that come. Can drop a few
    global seq_odom
    global list_t, list_x, list_y, list_z

    #assuming the odometry is represented as ^wT_t
    list_t.append( long(str(data.header.stamp)) )
    list_x.append( float(data.pose.pose.position.x) )
    list_y.append( float(data.pose.pose.position.y) )
    list_z.append( float(data.pose.pose.position.z) )
    list_msg.append( data )

    # publish Marker
    marker = Marker()
    marker.header = data.header
    marker.id = seq_odom
    seq_odom += 1
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = data.pose.pose
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
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
    global list_t, list_x, list_y, list_z
    rospy.loginfo( 'Received Colocation' )
    print 'timestamp : ',list_t[-1]
    # return None

    # 1. do a binary search on the deque with data.c_timestamp and data.prev_timestamp
    c_indx = abs(np.array(list_t) - long(str(data.c_timestamp))).argmin()
    prev_indx = abs(np.array(list_t) - long(str(data.prev_timestamp))).argmin()
    print  data.c_timestamp, '<--->', data.prev_timestamp
    print  c_indx, '<--->', prev_indx


    goodness = data.goodness # currently will be between 0 and 6. 6 means less confident that it is a loop_closure. 0 means most confident that it is a loop
    goodness_color, _ = dist_to_color( 2.5*(goodness-0.6) )

    # 1.1 Make points
    pt0 = Point()
    pt0.x = list_x[c_indx]
    pt0.y = list_y[c_indx]
    pt0.z = list_z[c_indx]
    pt1 = Point()
    pt1.x = list_x[prev_indx]
    pt1.y = list_y[prev_indx]
    pt1.z = list_z[prev_indx]

    # 2. make 2 markers which represents a line using poses from part-1 (above)
    m = Marker()
    m.header = list_msg[c_indx].header
    m.lifetime = rospy.Duration(3) #set this to zero => colocation edges are never deleted. positive value will cause the edges to disappear after this many seconds
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


# Node Callback - Optional
def node_callback( data ):
    global list_t, list_x, list_y, list_z

    c_indx = abs(np.array(list_t) - long(str(data.node_timestamp))).argmin()

    marker = Marker()
    marker.header =  list_msg[c_indx].header
    marker.id = 2*int(data.node_label)
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = list_x[c_indx]
    marker.pose.position.y = list_y[c_indx]
    marker.pose.position.z = list_z[c_indx]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = .5
    marker.scale.y = .5
    marker.scale.z = .5
    marker.color.a = 0.5
    marker.color.r = data.color_r#1.0
    marker.color.g = data.color_g#1.0
    marker.color.b = data.color_b#1.0
    pub_raw_node.publish( marker )

    marker.type=Marker.TEXT_VIEW_FACING
    marker.id = 2*int(data.node_label) + 1
    marker.text = str(data.node_label_str)
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    if marker.id % 5 == 0:
        pub_raw_node.publish( marker )


# Edge Callback - Optional
def edge_callback( data ):
    global seq
    global list_t, list_x, list_y, list_z

    c_indx = abs(np.array(list_t) - long(str(data.c_timestamp))).argmin()
    prev_indx = abs(np.array(list_t) - long(str(data.prev_timestamp))).argmin()

    # goodness = data.goodness # currently will be between 0 and 6. 6 means less confident that it is a loop_closure. 0 means most confident that it is a loop
    # goodness_color, _ = dist_to_color( 2.5*(goodness-0.6) )

    # 1.1 Make points
    pt0 = Point()
    pt0.x = list_x[c_indx]
    pt0.y = list_y[c_indx]
    pt0.z = list_z[c_indx]
    pt1 = Point()
    pt1.x = list_x[prev_indx]
    pt1.y = list_y[prev_indx]
    pt1.z = list_z[prev_indx]

    # 2. make 2 markers which represents a line using poses from part-1 (above)
    m = Marker()
    m.header = list_msg[c_indx].header
    m.lifetime = rospy.Duration(0) #set this to zero => colocation edges are never deleted. positive value will cause the edges to disappear after this many seconds
    m.id = seq
    seq += 1
    m.type = Marker.LINE_LIST
    m.points.append(pt0) #c_indx
    m.points.append(pt1) #prev_indx
    m.scale.x = .05
    # m.scale.y = 0.5
    # m.scale.z = 0.0
    m.color.a = 0.3
    m.color.r = data.color_r#1.0
    m.color.g = data.color_g#0.0
    m.color.b = data.color_b#0.0


    # 3. publish
    pub_raw_edge.publish( m )


def visual_edge_callback( data ):
    curr_image = CvBridge().imgmsg_to_cv2( data.curr_image, 'bgr8' )
    prev_image = CvBridge().imgmsg_to_cv2( data.prev_image, 'bgr8' )
    curr_label = data.curr_label
    prev_label = data.prev_label
    cv2.putText( curr_image, curr_label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) )
    cv2.putText( prev_image, prev_label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) )

    catted = np.concatenate( (curr_image,prev_image), axis=1)
    msg_frame = CvBridge().cv2_to_imgmsg( catted, "bgr8" )
    pub_raw_edge_image_pair.publish( msg_frame )
    # cv2.imshow( 'curr_image', curr_image )
    # cv2.imshow( 'prev_image', prev_image )
    # cv2.imshow( 'cated', np.concatenate( (curr_image,prev_image), axis=1 ) )
    # cv2.waitKey(1)



rospy.init_node( 'colocation_viz', log_level=rospy.INFO )
rospy.Subscriber( '/vins_estimator/odometry', Odometry, odometry_callback, queue_size=10000 )
rospy.loginfo( 'Subscribed to /vins_estimator/odometry')


#rospy.Subscriber( '/colocation', NapMsg, colocation_callback, queue_size=10000 )
#rospy.loginfo( 'Subscribed to /colocation')

# Raw Graph
rospy.Subscriber( '/raw_graph_node', NapNodeMsg, node_callback, queue_size=10000 )
rospy.Subscriber( '/raw_graph_edge', NapMsg, edge_callback, queue_size=10000 )
rospy.Subscriber( '/raw_graph_visual_edge', NapVisualEdgeMsg, visual_edge_callback, queue_size=10000 )
# rospy.Subscriber( '/raw_graph_visual_edge_cluster_assgn', NapVisualEdgeMsg, visual_edge_callback, queue_size=10000 )  #uncomment this line and comment the above one to see neural cluster assgnment false color images as pairs
rospy.loginfo( 'Subscribed to Raw Graph : /raw_graph_node and /raw_graph_edge and /raw_graph_visual_edge')


pub_odometry = rospy.Publisher( '/colocation_viz/odom_marker', Marker, queue_size=1000 )
pub_curr_position = rospy.Publisher( '/colocation_viz/cur_position', Marker, queue_size=1000 )
pub_colocation = rospy.Publisher( '/colocation_viz/colocation_marker', Marker, queue_size=1000 )

pub_raw_node = rospy.Publisher( '/colocation_viz/raw/node', Marker, queue_size=1000 )
pub_raw_edge = rospy.Publisher( '/colocation_viz/raw/edge', Marker, queue_size=1000 )
pub_raw_edge_image_pair = rospy.Publisher( '/colocation_viz/raw/image_pair', Image, queue_size=1000 )

seq = 0
seq_odom = 0
# _d = collections.deque()
list_t = []
list_x = []
list_y = []
list_z = []
list_msg = []

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')

rate = rospy.Rate(30)
while not rospy.is_shutdown():
    # print 'Length :', len(_d)
    rate.sleep()

rospy.loginfo( 'Done..!' )
